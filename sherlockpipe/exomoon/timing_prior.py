"""The t0 prior of the per-epoch planet fit, measured from the planet transit times themselves.

The per-epoch mid-time fit has a Gaussian prior that keeps low-SNR epochs from running away.
Its width used to be only the TTV of the largest searched moon, which is a fraction of a minute
when the moon radius range is small, so real timing variations of another origin (other
planets, a wrong catalog ephemeris) were pulled back to the linear ephemeris and left a residual
at ingress and egress. The prior now comes from the data: every epoch is first fitted without
it, a timing model (none, linear or linear plus a sinusoid, chosen by BIC) is fitted to those
mid-times with an excess scatter on top of their precisions, and the prior of each epoch is
centred on the model with a width that adds that scatter, the ephemeris uncertainty and the
moon TTV.
"""
import copy
import logging
import numpy as np
from scipy.optimize import brentq

TimingModel = dict
"""A fitted timing model: name, period (days or None), coefficients, excess scatter (days),
BIC, the epochs used and the model evaluated at every epoch (days from the linear ephemeris)."""


class TimingPrior:
    """Data-driven prior on the per-epoch planet mid-time (see the module docstring)."""

    TTV_TREND_MIN_EPOCHS = 3
    """Fewest usable epochs to measure the timing model; with fewer, the moon TTV alone is used."""
    TTV_SINUSOID_MIN_EPOCHS = 8
    """Fewest usable epochs to try the linear plus sinusoid model."""
    TTV_OUTLIER_SIGMA = 5.0
    """Epochs further than this from the model, in units of their precision and the robust (MAD)
    excess scatter, are refitted out."""
    TTV_PERIOD_GRID_SIZE = 2000
    TTV_MAX_PRECISION_INGRESS = 0.5
    """Epochs whose timing precision exceeds this fraction of the ingress are left out of the timing
    model: Carter et al. (2008) assume a timing error small against the ingress, and beyond it a free
    fit follows the noise, so its scatter would widen the prior where it is most needed."""

    @staticmethod
    def excess_scatter(residuals, precisions, dof):
        """Return the scatter in days to add to the precisions so that chi2 equals dof (Paule & Mandel 1982).

        0 when the residuals are already consistent with the precisions.
        """
        residuals, precisions = np.asarray(residuals, float), np.asarray(precisions, float)
        dof = max(dof, 1)

        def excess_chi2(sigma):
            return float(np.sum(residuals ** 2 / (precisions ** 2 + sigma ** 2))) - dof

        if excess_chi2(0.0) <= 0:
            return 0.0
        upper = max(float(np.max(np.abs(residuals))), float(np.max(precisions)), 1e-12)
        while excess_chi2(upper) > 0:
            upper *= 2
        return float(brentq(excess_chi2, 0.0, upper, xtol=1e-12))

    @classmethod
    def fit_timing_design(cls, design, oc, precisions):
        """Weighted least squares of the mid-times on a design matrix, with the excess scatter iterated.

        Returns
        -------
        tuple
            (coefficients, excess scatter, -2 ln L).
        """
        oc, precisions = np.asarray(oc, float), np.asarray(precisions, float)
        n, k = len(oc), 0 if design is None else design.shape[1]
        sigma, coefficients = 0.0, np.zeros(k)
        for _ in range(50):
            variance = precisions ** 2 + sigma ** 2
            if k:
                weights = 1 / variance
                normal = design.T @ (design * weights[:, None])
                coefficients = np.linalg.lstsq(normal, design.T @ (weights * oc), rcond=None)[0]
                residuals = oc - design @ coefficients
            else:
                residuals = oc
            new_sigma = cls.excess_scatter(residuals, precisions, n - k)
            if abs(new_sigma - sigma) <= 1e-4 * max(new_sigma, float(np.min(precisions))):
                sigma = new_sigma
                break
            sigma = new_sigma
        variance = precisions ** 2 + sigma ** 2
        residuals = oc - (design @ coefficients if k else 0.0)
        return coefficients, sigma, float(np.sum(residuals ** 2 / variance + np.log(2 * np.pi * variance)))

    @staticmethod
    def timing_design(times, numbers, name, period=None):
        """Design matrix of a timing model at the given epochs: none, linear or sinusoid."""
        if name == "none":
            return None
        columns = [np.ones(len(numbers)), np.asarray(numbers, float)]
        if name == "sinusoid":
            phase = 2 * np.pi * np.asarray(times, float) / period
            columns += [np.sin(phase), np.cos(phase)]
        return np.column_stack(columns)

    @classmethod
    def fit_timing_model(cls, times, numbers, oc, precisions, all_times, all_numbers):
        """Fit the timing models and keep the one with the lowest BIC.

        The models are no variation, a linear ephemeris correction, and that correction plus a
        sinusoid whose period is searched between the baseline and twice the median epoch spacing.
        The excess scatter counts as one more parameter, and the period as another. Epochs further
        than TTV_OUTLIER_SIGMA from the chosen model are dropped and the models refitted once.

        Returns
        -------
        TimingModel
        """
        times, numbers = np.asarray(times, float), np.asarray(numbers, float)
        oc, precisions = np.asarray(oc, float), np.asarray(precisions, float)
        keep = np.ones(len(oc), bool)
        best = None
        for _ in range(2):
            t, n, y, s = times[keep], numbers[keep], oc[keep], precisions[keep]
            candidates = [("none", None)]
            if len(y) >= 4:
                candidates.append(("linear", None))
            if len(y) >= cls.TTV_SINUSOID_MIN_EPOCHS:
                baseline = t.max() - t.min()
                spacing = np.median(np.diff(np.sort(t)))
                if baseline > 0 and spacing > 0 and 1 / baseline < 0.5 / spacing:
                    frequencies = np.linspace(1 / baseline, 0.5 / spacing, cls.TTV_PERIOD_GRID_SIZE)
                    candidates += [("sinusoid", 1 / f) for f in frequencies]
            best = None
            for name, period in candidates:
                design = cls.timing_design(t, n, name, period)
                coefficients, sigma, m2lnl = cls.fit_timing_design(design, y, s)
                parameters = (0 if design is None else design.shape[1]) + 1 + (1 if period else 0)
                bic = m2lnl + parameters * np.log(len(y))
                if best is None or bic < best["bic"]:
                    best = {"name": name, "period": period, "coefficients": coefficients, "excess": sigma,
                            "bic": bic}
            residuals = oc - cls.evaluate_timing_model(best, times, numbers)
            # A robust scale, so that the epoch that ran away does not hide itself by inflating the excess.
            mad = 1.4826 * np.median(np.abs(residuals[keep] - np.median(residuals[keep])))
            robust_excess = np.sqrt(max(mad ** 2 - np.median(precisions[keep] ** 2), 0.0))
            outliers = np.abs(residuals) > cls.TTV_OUTLIER_SIGMA * np.sqrt(precisions ** 2 + robust_excess ** 2)
            if not np.any(outliers & keep) or np.sum(keep & ~outliers) < cls.TTV_TREND_MIN_EPOCHS:
                break
            keep &= ~outliers
        best["used"] = keep
        best["centres"] = cls.evaluate_timing_model(best, all_times, all_numbers)
        return best

    @classmethod
    def evaluate_timing_model(cls, model, times, numbers):
        design = cls.timing_design(times, numbers, model["name"], model["period"])
        return np.zeros(len(times)) if design is None else design @ model["coefficients"]

    def measure_epoch_times(self, time, flux, flux_err, params, duration_window, precisions):
        """Fit every epoch's mid-time without a prior (first pass of the data-driven t0 prior).

        Returns
        -------
        list of float or None
            Mid-time minus the linear ephemeris in days for each epoch of bary_t0s, None when the
            epoch has too few points, no precision, a fit that is not usable, or less in-transit
            coverage than min_transit_completeness.
        """
        offsets = []
        for t0, precision in zip(self.bary_t0s, precisions):
            mask = np.abs(time - t0) < duration_window
            if precision is None or np.sum(mask) < self.min_cadences_per_t0:
                offsets.append(None)
                continue
            epoch_params = copy.copy(params)
            epoch_params.t0, epoch_params.u = t0, self.ab
            theta, _, status = self.fit_single_t0(time[mask], flux[mask], flux_err[mask], t0, duration_window,
                                                  epoch_params, self.exposure_at(time[mask]), None)
            # The same coverage rule as the subtraction: an epoch short of in-transit points is dropped there,
            # and its free mid-time is poorly constrained, so it does not enter the timing model.
            epoch_time = time[mask]
            points = np.sum(np.abs(epoch_time - theta[0]) < self.planet_duration / 2)
            cadence = np.median(np.diff(epoch_time)) if len(epoch_time) > 1 else 0
            covered = cadence > 0 and points >= self.min_transit_completeness * self.planet_duration / cadence
            offsets.append(float(theta[0] - t0) if self.epoch_fit_usable(status) and covered else None)
        return offsets

    def observed_timing_prior(self, time, flux, flux_err, params, duration_window, precisions, ephemeris):
        """Return the centre shift and width of the t0 prior of each epoch, from the measured mid-times.

        The width is sqrt(ephemeris^2 + excess^2 + moon TTV^2): the moon term is the floor that
        `max_moon_ttv` (or the given ttv_prior_width) sets, not a ceiling. Without flux errors or
        with fewer than TTV_TREND_MIN_EPOCHS usable epochs, the centres stay on the linear
        ephemeris and the excess is 0. Sets the ttv_* attributes reported in the result.
        """
        moon = self.ttv_prior_width_used
        offsets = self.measure_epoch_times(time, flux, flux_err, params, duration_window, precisions) \
            if flux_err is not None else [None] * len(self.bary_t0s)
        self.ttv_measured = offsets
        self.ttv_precisions = precisions
        numbers = np.round((np.asarray(self.bary_t0s, float) - self.planet_t0) / self.planet_period)
        ingress = self.ingress_duration(params, self.planet_duration)
        usable = [i for i, (o, p) in enumerate(zip(offsets, precisions))
                  if o is not None and p is not None and 0 < p <= self.TTV_MAX_PRECISION_INGRESS * ingress]
        self.ttv_model = None
        centres, excess = np.zeros(len(self.bary_t0s)), 0.0
        if len(usable) >= self.TTV_TREND_MIN_EPOCHS:
            model = self.fit_timing_model(np.asarray(self.bary_t0s)[usable], numbers[usable],
                                          [offsets[i] for i in usable], [precisions[i] for i in usable],
                                          np.asarray(self.bary_t0s), numbers)
            limit = duration_window / 8
            centres = np.clip(model["centres"], -limit, limit)
            if np.any(centres != model["centres"]):
                logging.warning("The timing model moves some epochs by more than T14 / 4 (%.1f min); the t0 prior "
                                "centres are clipped there", limit * 1440)
            excess = model["excess"]
            self.ttv_model = {"name": model["name"], "period": model["period"],
                              "coefficients_min": [float(c) * 1440 for c in model["coefficients"]],
                              "excess_min": excess * 1440, "epochs_used": int(np.sum(model["used"])),
                              "epochs_measured": len(usable)}
            raw = np.array([offsets[i] for i in usable])
            logging.info("Measured mid-times (no prior, %d epochs): RMS %.2f min, median precision %.2f min; timing "
                         "model %s%s, excess scatter %.2f min", len(usable), np.std(raw) * 1440,
                         np.median([precisions[i] for i in usable]) * 1440, model["name"],
                         "" if model["period"] is None else " (P = %.1f d)" % model["period"], excess * 1440)
        else:
            logging.info("Too few epochs with a mid-time measured to better than %.1f of the ingress (%d of %d) for a "
                         "timing model; the t0 prior is centred on the linear ephemeris", self.TTV_MAX_PRECISION_INGRESS,
                         len(usable), len(self.bary_t0s))
        self.ttv_excess_used = excess
        self.ttv_prior_centres = [float(c) for c in centres]
        sigmas = [None if sigma is None else float(np.sqrt(sigma ** 2 + excess ** 2 + moon ** 2))
                  for sigma in ephemeris]
        self.ttv_prior_sigmas = sigmas
        known = [s for s in sigmas if s is not None]
        if known:
            logging.info("t0 prior width: median %.2f min (moon TTV %.2f, excess %.2f, ephemeris median %.2f min)",
                         np.median(known) * 1440, moon * 1440, excess * 1440,
                         np.median([s for s in ephemeris if s is not None]) * 1440)
        return list(self.ttv_prior_centres), sigmas
