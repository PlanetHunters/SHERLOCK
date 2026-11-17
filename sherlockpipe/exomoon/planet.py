"""The planet stage: fold fit of the transit shape, per-epoch mid-times, the planet
subtraction and the derivatives of the subtracted model that the joint fit corrects.
"""
import copy
import tempfile
from collections import namedtuple
import logging
import astropy.units as u
from lcbuilder.helper import LcbuilderHelper
import batman
import numpy as np
from scipy import stats
from scipy.signal import lombscargle
from sherlockpipe.exomoon.constants import AU_TO_RSUN, MIN_RP_OVER_RSTAR, M_earth_to_M_sun, NoUsableEpochsError
from sherlockpipe.exomoon.orbits import forecast_moon_mass
from sherlockpipe.exomoon.planet_derivatives import PlanetDerivatives
from sherlockpipe.exomoon.planet_figures import PlanetFigures
from sherlockpipe.exomoon.timing_prior import TimingPrior
from sherlockpipe.exomoon.transit_fits import TransitFits

Fold = namedtuple('Fold', ['time', 'flux', 'flux_err', 'exposure'])
"""The epochs of the planet folded on t0: times from t0, fluxes, errors (or None) and exposures."""


class PlanetModel(TransitFits, TimingPrior, PlanetDerivatives, PlanetFigures):
    """The planet stage: fold fit of the transit shape, per-epoch mid-times, the planet
    subtraction and the derivatives of the subtracted model that the joint fit corrects.
    """

    def check_fitted_duration(self):
        """Warn when the fold-fitted T14 differs from the input one, which sets the planet speed."""
        self._speed_logged = False  # the speed may change with the fitted geometry, so it is logged again
        if not self.duration_from_fit or self.catalog_planet_duration is None or self.catalog_planet_duration <= 0:
            return
        difference = self.planet_duration / self.catalog_planet_duration - 1
        level = logging.WARNING if abs(difference) > self.SPEED_MISMATCH else logging.INFO
        logging.log(level, "Fitted T14 %.4f d against the input %.4f d (%+.1f%%)%s", self.planet_duration,
                    self.catalog_planet_duration, 100 * difference,
                    "; the moon offsets use the input duration, so check it" if level == logging.WARNING else "")

    def epoch_timing_precision(self, epoch_time, epoch_flux, depth, ingress):
        """Return the mid-time precision of one epoch (Carter et al. 2008, Eq. 43).

        sigma = T / Q * sqrt(tau / 2T), with Q = sqrt(N_in) depth / noise and the noise measured
        as the MAD outside the transit.

        Parameters
        ----------
        epoch_time : numpy.ndarray
            Times relative to the predicted mid-transit, in days.
        epoch_flux : numpy.ndarray
            Normalised flux of the epoch.
        depth : float
            Transit depth.
        ingress : float
            Ingress duration in days.

        Returns
        -------
        float or None
            Precision in days, or None when the epoch has too few points to measure it.
        """
        t14 = self.planet_duration
        in_transit = np.abs(epoch_time) < t14 / 2
        outside = ~in_transit
        if np.sum(in_transit) < 2 or np.sum(outside) < 2 or depth <= 0 or ingress <= 0:
            return None
        noise = 1.4826 * np.median(np.abs(epoch_flux[outside] - np.median(epoch_flux[outside])))
        if not noise > 0:
            return None
        q = np.sqrt(np.sum(in_transit)) * depth / noise
        return float(t14 / q * np.sqrt(ingress / (2 * t14)))

    def ephemeris_sigmas(self, t0s, precisions):
        """Return the uncertainty of the linear ephemeris at each epoch, in days (audit item C1).

        With planet_t0_err and planet_period_err it is sqrt(sigma_T0^2 + (n sigma_P)^2), n being
        the epoch number from planet_t0. Otherwise each epoch gets the uncertainty of a weighted
        linear ephemeris fitted to the timing precisions of the other epochs (leave one out), so
        an epoch's own noise, already in its likelihood, is not counted twice. This needs at
        least two other epochs with a measured precision; with fewer, the epoch's own precision
        is used instead. Sets ephemeris_source to "fit", "other transits" or "own transit"
        (the last when any epoch fell back).

        Parameters
        ----------
        t0s : list of float
            Predicted mid-transit times.
        precisions : list of float or None
            Mid-time precision of each epoch (see `epoch_timing_precision`).

        Returns
        -------
        list of float or None
            The uncertainty per epoch, None when it cannot be estimated.
        """
        numbers = np.round((np.asarray(t0s, dtype=float) - self.planet_t0) / self.planet_period)
        if self.planet_t0_err is not None and self.planet_period_err is not None:
            self.ephemeris_source = "fit"
            return [float(np.hypot(self.planet_t0_err, n * self.planet_period_err)) for n in numbers]
        weights = np.array([0.0 if p is None or not p > 0 else 1 / p ** 2 for p in precisions])
        self.ephemeris_source = "other transits"
        sigmas = []
        for j, n in enumerate(numbers):
            w = weights.copy()
            w[j] = 0.0
            s0, s1, s2 = np.sum(w), np.sum(w * numbers), np.sum(w * numbers ** 2)
            det = s0 * s2 - s1 ** 2
            if np.count_nonzero(w) >= 2 and det > 0:
                sigmas.append(float(np.sqrt((s2 - 2 * n * s1 + n ** 2 * s0) / det)))
            else:
                self.ephemeris_source = "own transit"
                sigmas.append(precisions[j])
        return sigmas

    def mask_half_width(self, epoch_time):
        """Return the half-width in days of the mask-mode window around a planet transit (audit item E2).

        With mask_duration_factor given it is that factor times T14 / 2. By default it is
        1.1 T14 / 2 plus one cadence, so an imperfect T14 or t0, or a long exposure, leaves no
        ingress or egress points next to the mask.
        """
        if self.mask_duration_factor is not None:
            return self.planet_duration / 2 * self.mask_duration_factor
        cadence = np.median(np.diff(np.sort(epoch_time))) if len(epoch_time) > 1 else 0.0
        exposure = self.exposure_at(epoch_time) if len(epoch_time) else self.exposure_time
        if exposure is not None:
            cadence = max(cadence, float(np.max(exposure)))
        return 1.1 * self.planet_duration / 2 + cadence

    @staticmethod
    def mask_survival(max_offset, half_duration, half_mask):
        """Return the mean fraction of a moon's transit time left outside the planet mask (audit item E3).

        The moon transits max_offset sin(phase) from the planet, over a time as long as the planet
        transit (2 half_duration), and the mask covers +-half_mask; the fraction is averaged over a
        uniform orbital phase. 0.32 for a maximum offset of one half-duration with a T14 mask.
        """
        phase = np.linspace(0, np.pi / 2, 2001)
        centre = max_offset * np.sin(phase)
        overlap = np.clip(np.minimum(centre + half_duration, half_mask) - np.maximum(centre - half_duration,
                                                                                     -half_mask), 0, None)
        return float(np.mean(1 - overlap / (2 * half_duration)))

    def warn_mask_mode_survival(self, threshold=0.5):
        """Warn when mask mode removes most of the signal of the widest searched moons (audit item E3)."""
        mass = np.max(self.planet_mass_grid) * M_earth_to_M_sun
        max_period = self.max_moon_period(np.max(self.planet_mass_grid))
        max_offset = self.compute_max_moon_offset(self.compute_semimajor_axis(mass, max_period),
                                                  self.planet_semimajor_axis, self.planet_period)
        half_duration = self.planet_duration / 2
        survival = self.mask_survival(max_offset, half_duration, self.mask_half_width(self.time))
        self.mask_mode_survival = survival
        level = logging.WARNING if survival < threshold else logging.INFO
        logging.log(level, "Mask mode keeps on average %.0f%% of the transit time of the widest searched moons "
                    "(offset up to %.2f d); closer moons keep less. Subtract mode keeps them", survival * 100,
                    max_offset)

    def check_planet_shape(self, time, residual_flux, fitted_t0s, bins=20, p_threshold=1e-3, noise_factor=1.0,
                           label="Planet shape check", log=True):
        """Test the stacked post-subtraction residual for a planet shape left behind (audit item D2).

        The residuals of the kept epochs are stacked on their fitted mid-times and binned over
        +-0.75 T14, so ingress, egress and the contact points are covered. The chi2 of the bin
        means against the level between 0.75 T14 and 1.5 T14, with errors from the scatter
        there, is compared with a chi2 distribution; a p-value below p_threshold fails the
        check. A moon inside the planet transit can also fail it, so it is a flag.
        noise_factor (the red-noise factor beta) scales the bin errors, since a bin mean of
        correlated noise scatters beta times more than white noise.

        Returns
        -------
        dict or None
            {"chi2", "dof", "p_value", "ok", "noise_factor"}, or None without enough data.
        """
        if not fitted_t0s:
            return None
        phase = np.concatenate([time[np.abs(time - t0) < 1.5 * self.planet_duration] - t0 for t0 in fitted_t0s])
        values = np.concatenate([residual_flux[np.abs(time - t0) < 1.5 * self.planet_duration]
                                 for t0 in fitted_t0s])
        inner = np.abs(phase) < 0.75 * self.planet_duration
        outer = ~inner
        if np.sum(outer) < 20 or np.sum(inner) < 2 * bins:
            return None
        sigma = 1.4826 * np.median(np.abs(values[outer] - np.median(values[outer])))
        level = np.median(values[outer])
        edges = np.linspace(-0.75 * self.planet_duration, 0.75 * self.planet_duration, bins + 1)
        index = np.digitize(phase[inner], edges) - 1
        chi2, dof = 0.0, 0
        for b in range(bins):
            in_bin = values[inner][index == b]
            if len(in_bin) >= 3:
                chi2 += ((np.mean(in_bin) - level) / (noise_factor * sigma / np.sqrt(len(in_bin)))) ** 2
                dof += 1
        if dof == 0 or not sigma > 0:
            return None
        p_value = float(stats.chi2.sf(chi2, dof))
        result = {"chi2": float(chi2), "dof": dof, "p_value": p_value, "ok": bool(p_value >= p_threshold),
                  "noise_factor": float(noise_factor)}
        if not log:
            return result
        if not result["ok"]:
            logging.warning("%s failed: the stacked residual over the transit has chi2 %.1f for %d bins (p = %.1e, "
                            "red-noise factor %.2f), so the planet model leaves a residual, or a moon sits inside "
                            "the transit", label, chi2, dof, p_value, noise_factor)
        else:
            logging.info("%s passed: stacked residual chi2 %.1f for %d bins (p = %.2g, red-noise factor %.2f)",
                         label, chi2, dof, p_value, noise_factor)
        return result

    def realign_fold(self, time, flux, flux_err, params, duration_window, priors, fold_bounds):
        """Refit the planet shape on a fold aligned on the fitted epoch times (audit item B6).

        The first fold uses the linear ephemeris, so TTVs or a drifting ephemeris smear it. Each
        epoch's mid-time is fitted with the first shape and the C1 prior, the epochs are folded on
        those times and the shape is refitted. The new shape is kept when the fold chi2 drops by
        more than N + 4 + 3 sqrt(2 N) for N epochs, three standard deviations above what fitting one
        time per epoch and the shape to noise alone gives. The moon stage keeps the linear ephemeris (bary_t0s) for
        the moon phases, so a moon-induced TTV is not absorbed into them. params, the planet
        geometry attributes and fold_realigned are updated in place.
        """
        def fold(shifts):
            parts = []
            for t0, shift in shifts:
                near = np.abs(time - t0) < duration_window
                parts.append((time[near] - t0 - shift, flux[near], flux_err[near], time[near]))
            fold_time, fold_flux, fold_err, original = (np.concatenate([part[i] for part in parts]) for i in range(4))
            # The exposures are looked up at the times of the points, before they move onto the fold.
            return fold_time, fold_flux, fold_err, self.exposure_at(original)

        def chi2(theta_params, fold_time, fold_flux, fold_err, fold_exposure):
            model = self.transit_model(theta_params, fold_time, fold_exposure)
            return float(np.sum(((fold_flux - model.light_curve(theta_params)) / fold_err) ** 2))

        shifts = []
        for t0, centre, sigma in zip(self.bary_t0s, *priors):
            mask = np.abs(time - t0) < duration_window
            if np.sum(mask) < self.min_cadences_per_t0:
                continue
            epoch_params = copy.copy(params)
            epoch_params.t0, epoch_params.u = t0 + centre, self.ab
            theta, _, status = self.fit_single_t0(time[mask], flux[mask], flux_err[mask], t0 + centre,
                                                                 duration_window, epoch_params,
                                                                 self.exposure_at(time[mask]), sigma)
            usable = status is None or (status["converged"] and "t0" not in status["at_bound"])
            shifts.append((t0, float(theta[0] - t0) if usable else centre))
        if len(shifts) < 2:
            return
        current = copy.copy(params)
        current.t0 = 0.0
        linear = fold([(t0, 0.0) for t0, _ in shifts])
        aligned = fold(shifts)
        before = chi2(current, *linear)
        start_b = self.impact_parameter(params.a, params.inc, params.ecc, params.w)
        guess = [0.0, self.planet_duration, float(np.clip(start_b, fold_bounds[2][0], fold_bounds[2][1])),
                 float(np.clip(params.rp, fold_bounds[3][0], fold_bounds[3][1]))]
        try:
            theta, _, (a_new, inc_new), status = self.fit_fold_shape(
                *aligned[:3], guess, fold_bounds, copy.copy(params), aligned[3])
        except ValueError:
            return
        refitted = copy.copy(params)
        refitted.t0, refitted.a, refitted.inc, refitted.rp = theta[0], a_new, inc_new, theta[3]
        after = chi2(refitted, *aligned)
        rms = float(np.sqrt(np.mean(np.square([shift for _, shift in shifts]))))
        # Fitting one time per epoch and the four shape parameters to noise alone lowers the chi2 by
        # about N + 4, with a scatter of sqrt(2 N); three of those above it is a real improvement.
        needed = len(shifts) + 4 + 3 * np.sqrt(2 * len(shifts))
        if before - after <= needed:
            logging.info("Fold not realigned: aligning on the fitted epoch times (rms %.1f min) lowers the fold "
                         "chi2 by %.1f, not more than the %.1f noise alone gives for %d epochs", rms * 1440,
                         before - after, needed, len(shifts))
            return
        params.a, params.inc, params.rp = a_new, inc_new, theta[3]
        fitted_duration = self.model_transit_duration(refitted, duration_window)
        if fitted_duration > 0:
            self.planet_duration = fitted_duration
        self.planet_inc = float(inc_new)
        self.planet_impact_param = float(theta[2])
        self.planet_a_over_rstar = float(a_new)
        self.planet_rp_over_rstar = float(theta[3])
        self.fold_fit_status = status
        self.fold_realigned = True
        logging.info("Fold realigned on the fitted epoch times (rms %.1f min): chi2 %.1f -> %.1f, T14=%.4f d, "
                     "b=%.3f, rp=%.4f", rms * 1440, before, after, self.planet_duration, theta[2], theta[3])

    def normalise_epochs(self, time, flux, flux_err, min_points=10):
        """Divide each epoch by its out-of-window baseline, so the planet fits see a unit baseline.

        A3: the baseline is a straight line fitted to the points outside the widest moon window
        (and at least one transit duration from the epoch), so neither the planet nor a searched
        moon can shape it. A residual slope would otherwise move the fitted mid-time. With fewer
        than min_points / 2 points on either side, the median of the points outside is used.

        The planet fits assume the flux outside the transit is exactly 1. An offset would
        otherwise be absorbed into the planet depth and left as a step after subtraction.
        Each epoch owns the data within half a planet period of it, so the moon search
        never sees a step between normalised and unnormalised data. The median is taken
        outside one transit duration on each side of the predicted time, so a transit
        shifted by up to half a duration is still excluded. That is the largest shift the
        per-epoch time fit allows.

        Parameters
        ----------
        time, flux : numpy.ndarray
            Light curve.
        flux_err : numpy.ndarray or None
            Flux uncertainties, scaled with the flux.
        min_points : int, optional
            Epochs with fewer out-of-transit points are left unchanged, default 10.

        Returns
        -------
        tuple
            (flux, flux_err) as new arrays.
        """
        flux = np.array(flux, dtype=float)
        flux_err = None if flux_err is None else np.array(flux_err, dtype=float)
        exclusion = max(self.planet_duration, self.moon_window_half_width())
        self.epoch_baseline_slopes = []
        for t0 in self.bary_t0s:
            window = np.abs(time - t0) < self.planet_period / 2
            outside = window & (np.abs(time - t0) > exclusion)
            if np.sum(outside) < min_points:
                logging.warning("Epoch T0=%.4f not normalised: %d points outside the moon window, fewer than %d",
                                t0, np.sum(outside), min_points)
                continue
            before, after = np.sum(outside & (time < t0)), np.sum(outside & (time > t0))
            if min(before, after) >= min_points / 2:
                slope, intercept = np.polyfit(time[outside] - t0, flux[outside], 1)
                residual = flux[outside] - (intercept + slope * (time[outside] - t0))
                keep = np.abs(residual) < 5 * 1.4826 * np.median(np.abs(residual - np.median(residual)))
                if np.sum(keep) >= min_points:
                    slope, intercept = np.polyfit(time[outside][keep] - t0, flux[outside][keep], 1)
                level = intercept + slope * (time[window] - t0)
                self.epoch_baseline_slopes.append(float(slope))
            else:
                level = np.full(np.sum(window), np.median(flux[outside]))
            if not np.all(np.isfinite(level)) or np.any(level <= 0):
                logging.warning("Epoch T0=%.4f not normalised: invalid out-of-window baseline", t0)
                continue
            flux[window] = flux[window] / level
            if flux_err is not None:
                flux_err[window] = flux_err[window] / level
        return flux, flux_err

    def subtract_planet_transit(self, time, flux, flux_err, mode='subtract', plot=False):
        """Subtract or mask the planet transit signal from the light curve.

        The stages are: the starting transit parameters (`input_transit_params`), the fold of every
        epoch (`fold_epochs`) and its shape fit (`fold_fit_start`, `fit_planet_fold`), the t0 prior
        of each epoch (`epoch_t0_priors`), the fold realigned on the fitted epoch times
        (`realign_fold`), and the per-epoch fit (`fit_epoch`) whose model is subtracted
        (`check_subtracted_epoch`) or whose transit is masked (`masked_epoch_centre`). Epochs that
        cannot be used are dropped at the end (`drop_rejected_epochs`).

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        mode : str, optional
            'subtract' to subtract the planet model or 'mask' to remove
            in-transit points, default 'subtract'.
        plot : bool, optional
            If True, generate subtraction plots, default False.

        Returns
        -------
        tuple
            (result_time, result_model, result_flux_err) after planet removal.
        """
        params = self.input_transit_params()
        rp_err = self.planet_radius_err / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        duration_window = self.planet_duration * 2
        flux, flux_err = self.normalise_epochs(time, flux, flux_err)
        fold = self.fold_epochs(time, flux, flux_err, duration_window)
        fold_bounds = None
        if fold is not None:
            start_params, fold_guess, fold_bounds = self.fold_fit_start(params, fold, rp_err, duration_window)
        self.fold_fit_status = None
        if not self.skip_fold_fit:
            if fold is None:
                raise NoUsableEpochsError("No planet transit epoch has enough points for the fold fit")
            self.fit_planet_fold(params, start_params, fold, fold_guess, fold_bounds, duration_window, plot)
        self.ttvs = []
        self.epochs_dropped_fit = []
        self.epochs_flagged_rms = []
        self.epochs_flagged_rp = []
        self.epochs_dropped_coverage = []
        priors = self.epoch_t0_priors(time, flux, flux_err, params, duration_window)
        self.fold_realigned = False
        if not self.skip_fold_fit and flux_err is not None:
            self.realign_fold(time, flux, flux_err, params, duration_window, priors, fold_bounds)
        result_time = time.copy()
        result_flux = flux.copy()
        result_flux_err = flux_err.copy() if flux_err is not None else None
        rejected_t0s = []
        for t0, centre, t0_prior_sigma in zip(self.bary_t0s, *priors):
            window = np.abs(time - t0) < duration_window
            indexes = np.flatnonzero(window)
            best_theta, best_model, status = self.fit_epoch(time[window], flux[window], flux_err[window],
                                                            t0 + centre, t0_prior_sigma, params, rp_err,
                                                            duration_window)
            best_t0 = best_theta[0]
            self.ttvs.append(best_t0 - t0)
            # C4: a fit that did not converge, or whose t0 stopped at its bound, subtracts the planet
            # at the wrong time, so the epoch is dropped. An rp at its bound is only flagged: it
            # means extra depth, such as a moon inside the transit, which the bound keeps out of
            # the planet model (audit item B3).
            if not self.epoch_fit_usable(status):
                logging.info("  Rejecting transit T0=%.4f: epoch fit not usable (converged=%s, at bound: %s)", t0,
                             status["converged"], ", ".join(status["at_bound"]) or "none")
                rejected_t0s.append(t0)
                self.epochs_dropped_fit.append({"t0": float(t0), "converged": status["converged"],
                                                "at_bound": status["at_bound"]})
            elif status is not None and "rp" in status["at_bound"]:
                logging.info("  Flagging transit T0=%.4f: epoch rp at its bound (%.4f), kept", t0, best_theta[1])
                self.epochs_flagged_rp.append({"t0": float(t0), "rp": float(best_theta[1])})
            if mode == 'subtract':
                result_flux[indexes] = result_flux[indexes] - (best_model - 1)
                points, expected = self.check_subtracted_epoch(t0, best_t0, time[window], result_flux[indexes])
                if points < self.min_transit_completeness * expected and t0 not in rejected_t0s:
                    logging.info("  Rejecting transit T0=%.4f: incomplete in-transit coverage (%d pts vs %.0f "
                                 "expected)", best_t0, points, expected)
                    rejected_t0s.append(t0)
                    self.epochs_dropped_coverage.append({"t0": float(t0), "points": int(points),
                                                         "expected": float(expected)})
                if plot:
                    self.plot_subtracted_epoch(time[indexes], flux[indexes], best_model, result_flux[indexes],
                                               best_t0, rejected=t0 in rejected_t0s)
            else:
                centre = self.masked_epoch_centre(t0 + centre, best_t0, status, time[window])
                half_mask = self.mask_half_width(time[window])
                if plot:
                    self.plot_masked_epoch(time[window], flux[window], best_model, centre, half_mask)
                keep = np.abs(result_time - centre) > half_mask
                result_time = result_time[keep]
                result_flux = result_flux[keep]
                if result_flux_err is not None:
                    result_flux_err = result_flux_err[keep]
        self.drop_rejected_epochs(rejected_t0s)
        result_time = self.move_to_barycentre_frame(result_time)
        if mode == 'subtract':
            # Checked in run() once the red-noise factor is measured.
            self.planet_shape_check_inputs = (result_time, result_flux, self.planet_mid_times())
        else:
            self.warn_mask_mode_survival()
        self.plot_ttvs()
        self.ttv_diagnostics()
        return result_time, result_flux, result_flux_err

    def input_transit_params(self):
        """Return the batman parameters of the input planet: catalog radius, a/R* and inclination,
        with t0 left unset.
        """
        params = batman.TransitParams()
        params.per = self.planet_period
        params.ecc = self.planet_ecc
        params.w = self.planet_arg_periastron  # longitude of periastron (deg)
        params.rp = self.planet_radius / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        params.a = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.au) \
            if self.planet_a_over_rstar is None else self.planet_a_over_rstar
        params.inc = self.planet_inc
        params.u = self.ab
        params.limb_dark = "quadratic"
        return params

    def fold_epochs(self, time, flux, flux_err, duration_window):
        """Phase-fold the epochs with at least `min_cadences_per_t0` points on the linear ephemeris, plus
        the epoch offsets of the timing pre-pass when it measured them (`measure_epoch_offsets`): a fold
        smeared by tens of minutes of TTV fits a longer, more grazing transit (on KOI-94.01 a 60-min TTV
        gave T14 +30 % and b 0.88 against 0.68, and the realigned fold stopped at b 0.52 against 0.40),
        and the moon durations and offsets of the search come from that geometry.

        Returns
        -------
        Fold or None
            The folded times (from each epoch's t0), fluxes, errors (None without errors) and
            exposures, or None when no epoch has enough points.
        """
        parts = []
        for t0 in self.bary_t0s:
            centre = t0 + self.epoch_offset(t0, is_time=True)
            near = np.abs(time - centre) < duration_window
            if np.sum(near) < self.min_cadences_per_t0:
                continue
            parts.append((time[near] - centre, flux[near], None if flux_err is None else flux_err[near],
                          self.exposure_at(time[near]) if self.exposure_lookup is not None else None))
        if not parts:
            return None
        fold_time, fold_flux = (np.concatenate([part[i] for part in parts]) for i in range(2))
        fold_err = np.concatenate([part[2] for part in parts]) if flux_err is not None else None
        fold_exposure = np.concatenate([part[3] for part in parts]) if self.exposure_lookup is not None \
            else self.exposure_time
        return Fold(fold_time, fold_flux, fold_err, fold_exposure)

    def fold_fit_start(self, params, fold, rp_err, duration_window):
        """Return the start of the fold fit: the input-shape parameters, the guess and the bounds.

        B1: the fold is fitted in duration, impact parameter and radius, so any transit shape is
        reachable, starting from the shape of the input geometry. B2: the catalog radius comes from
        another stellar radius and may not match this depth, and the radius bound is only +-1 sigma
        wide, so its start and the centre of that bound are the radius that gives the observed
        central fold depth with the input shape (limb darkening included). The width stays +-1
        sigma, so the planet does not gain room to absorb a moon (audit item B3). params.rp is
        updated in place.

        Returns
        -------
        tuple
            (start_params with t0 = 0 and the catalog radius, guess [t0, T14, b, rp], bounds).
        """
        start_params = copy.copy(params)
        start_params.t0 = 0.0
        start_t14 = self.model_transit_duration(start_params, duration_window)
        if start_t14 <= 0:
            start_t14 = self.planet_duration
        start_b = self.impact_parameter(params.a, params.inc, params.ecc, params.w)
        if not self.skip_fold_fit and fold.flux_err is not None:
            centre = np.abs(fold.time) < 0.25 * start_t14
            if np.sum(centre) >= 10:
                observed = 1 - np.median(fold.flux[centre])
                model_centre = self.transit_model(
                    start_params, fold.time[centre],
                    self.exposure_subset(fold.exposure, centre)).light_curve(start_params)
                expected = 1 - np.median(model_centre)
                if observed > 0 and expected > 0:
                    data_rp = float(params.rp * np.sqrt(observed / expected))
                    if abs(data_rp / params.rp - 1) > 0.01:
                        logging.info("Planet rp start from the fold depth: %.4f (catalog %.4f, %.0f ppm observed "
                                     "against %.0f ppm expected at the centre)", data_rp, params.rp,
                                     observed * 1e6, expected * 1e6)
                    params.rp = data_rp
        guess = [0.0, start_t14, min(start_b, 1.0), params.rp]
        bounds = [(-start_t14 / 2, start_t14 / 2),
                  (start_t14 / 2, start_t14 * 2),
                  (0.0, 1.0 + params.rp + rp_err),
                  (max(params.rp - rp_err, MIN_RP_OVER_RSTAR), params.rp + rp_err)]
        if self.fix_impact_param and self.planet_impact_param is not None:
            # The input impact parameter is kept: its bound is a point (a width the optimizers accept).
            fixed_b = float(np.clip(self.planet_impact_param, 0.0, 1.0 + params.rp))
            guess[2] = fixed_b
            bounds[2] = (fixed_b, fixed_b + 1e-9)
        return start_params, guess, bounds

    def fit_planet_fold(self, params, start_params, fold, guess, bounds, duration_window, plot=False):
        """Fit the transit shape on the fold and adopt it: params (a, inc, rp), the planet duration
        and the planet geometry attributes are updated in place, and fold_fit_status is set.

        B4: the moon stage uses the fitted transit geometry (a/R*, inclination, b) for durations
        and templates. The semi-major axis in AU, from Kepler's third law, still sets the Hill
        sphere; the moon offsets a_m / v_p use the input transit duration (`planet_speed`), and the
        fitted duration only checks it.
        """
        theta, _, (fold_a, fold_inc), self.fold_fit_status = self.fit_fold_shape(
            fold.time, fold.flux, fold.flux_err, guess, bounds, params, fold.exposure)
        logging.info("Fold fit: T14=%.4f->%.4f d, b=%.3f->%.3f, rp=%.4f->%.4f (depth %.1f->%.1f ppm), "
                     "a/R*=%.2f->%.2f, inc=%.2f->%.2f", guess[1], theta[1], guess[2], theta[2], guess[3], theta[3],
                     guess[3] ** 2 * 1e6, theta[3] ** 2 * 1e6, start_params.a, fold_a, start_params.inc, fold_inc)
        if not self.fold_fit_status["converged"] or self.fold_fit_status["at_bound"]:
            logging.warning("Fold fit did not converge cleanly (converged=%s, at bound: %s); the planet "
                            "model may leave residuals", self.fold_fit_status["converged"],
                            ", ".join(self.fold_fit_status["at_bound"]) or "none")
        params.a, params.inc, params.rp = fold_a, fold_inc, theta[3]
        params_best = copy.copy(params)
        params_best.t0 = theta[0]
        fitted_duration = self.model_transit_duration(params_best, duration_window)
        if fitted_duration > 0:
            self.planet_duration = fitted_duration
        self.check_fitted_duration()
        self.planet_inc = float(fold_inc)
        self.planet_impact_param = float(theta[2])
        self.planet_a_over_rstar = float(fold_a)
        self.planet_rp_over_rstar = float(theta[3])
        if plot:
            self.plot_fold_fit(fold, start_params, params_best, guess, theta)

    def epoch_t0_priors(self, time, flux, flux_err, params, duration_window):
        """Return the centre shift and width of the t0 prior of each epoch (audit item C1).

        The moon term is the TTV the largest searched moon can cause (`ttv_prior_width`, 'auto'
        for `max_moon_ttv`), stored in ttv_prior_width_used. With ttv_prior_from_data (default),
        the prior is centred on a timing model of the mid-times measured without a prior and
        widened by their excess scatter (`observed_timing_prior`), so timing variations that no
        searched moon causes, such as those from other planets, are followed rather than pulled
        back to the linear ephemeris; otherwise it is centred on the linear ephemeris with the moon
        term and the ephemeris uncertainty only. With ttv_prior_width None there is no prior.

        Returns
        -------
        tuple
            (centre shifts in days from the linear ephemeris, prior widths in days or None).
        """
        zeros = [0.0] * len(self.bary_t0s)
        self.ttv_measured, self.ttv_precisions, self.ttv_model, self.ttv_excess_used = None, None, None, None
        self.ttv_prior_centres, self.ttv_prior_sigmas = zeros, [None] * len(self.bary_t0s)
        if self.ttv_prior_width == "auto":
            self.ttv_prior_width_used = self.max_moon_ttv()
            logging.info("Moon TTV in the t0 prior: %.1f min, the largest a searched moon can cause (%.2f R_earth, "
                         "%.1f M_earth forecast, on the widest searched orbit)", self.ttv_prior_width_used * 1440,
                         self.max_radius, forecast_moon_mass(self.max_radius, self.max_moon_density))
        else:
            self.ttv_prior_width_used = self.ttv_prior_width
            if self.ttv_prior_width is not None:
                logging.info("Moon TTV in the t0 prior: %.1f min (given)", self.ttv_prior_width * 1440)
        if self.ttv_prior_width_used is None:
            return zeros, [None] * len(self.bary_t0s)
        ingress = self.ingress_duration(params, self.planet_duration)
        precisions = []
        for t0 in self.bary_t0s:
            near = np.abs(time - t0) < duration_window
            precisions.append(self.epoch_timing_precision(time[near] - t0, flux[near], params.rp ** 2, ingress))
        ephemeris = self.ephemeris_sigmas(self.bary_t0s, precisions)
        known = [sigma for sigma in ephemeris if sigma is not None]
        if known:
            logging.info("Ephemeris uncertainty in the t0 prior (from %s): median %.1f min, largest %.1f min",
                         self.ephemeris_source, np.median(known) * 1440, np.max(known) * 1440)
        if self.ttv_prior_from_data:
            return self.observed_timing_prior(time, flux, flux_err, params, duration_window, precisions, ephemeris)
        self.ttv_precisions = precisions
        sigmas = [None if sigma is None else float(np.hypot(sigma, self.ttv_prior_width_used)) for sigma in ephemeris]
        self.ttv_prior_sigmas = sigmas
        return zeros, sigmas

    def fit_epoch(self, epoch_time, epoch_flux, epoch_err, t0, t0_prior_sigma, params, rp_err, duration_window):
        """Fit one epoch's mid-time (and radius, with `fit_rp_per_transit`) with the fold shape.

        Returns
        -------
        tuple
            (theta [t0] or [t0, rp], model at epoch_time, fit status or None).
        """
        epoch_params = copy.copy(params)
        epoch_params.t0 = t0
        epoch_params.u = self.ab
        exposure = self.exposure_at(epoch_time)
        if self.fit_rp_per_transit:
            return self.fit_single_t0_rp(epoch_time, epoch_flux, epoch_err, t0, params.rp, rp_err, duration_window,
                                         epoch_params, exposure, t0_prior_sigma)
        return self.fit_single_t0(epoch_time, epoch_flux, epoch_err, t0, duration_window, epoch_params, exposure,
                                  t0_prior_sigma)

    @staticmethod
    def epoch_fit_usable(status):
        """Whether an epoch fit gives a usable mid-time: converged, with t0 not at its bound."""
        return status is None or (status["converged"] and "t0" not in status["at_bound"])

    def check_subtracted_epoch(self, t0, best_t0, epoch_time, residual):
        """Check one epoch after the subtraction and return its in-transit coverage.

        D1: an in-transit residual scatter above `max_subtraction_rms_ratio` times the
        out-of-transit one is flagged (epochs_flagged_rms), not dropped, because a moon dip inside
        the transit also raises it. The coverage counts the points within T14, not those where the
        model dims: averaged over a long exposure it still dims half an exposure beyond T14, and an
        epoch short of in-transit points passed.

        Returns
        -------
        tuple
            (points within T14, points expected at the median cadence of the epoch).
        """
        in_transit = np.abs(epoch_time - best_t0) < self.planet_duration / 2
        points = int(np.sum(in_transit))
        cadence = np.median(np.diff(epoch_time)) if len(epoch_time) > 1 else 0
        expected = self.planet_duration / cadence if cadence > 0 else np.inf
        if points > 5 and np.sum(~in_transit) > 5:
            it_rms = np.std(residual[in_transit])
            oot_rms = np.std(residual[~in_transit])
            rms_ratio = it_rms / oot_rms if oot_rms > 0 else float('inf')
            if rms_ratio > self.max_subtraction_rms_ratio:
                logging.info("  Flagging transit T0=%.4f: post-sub RMS ratio=%.2f (it=%.6f, oot=%.6f), "
                             "kept because a moon dip also raises it", best_t0, rms_ratio, it_rms, oot_rms)
                self.epochs_flagged_rms.append({"t0": float(t0), "rms_ratio": float(rms_ratio)})
        return points, expected

    def masked_epoch_centre(self, t0, best_t0, status, epoch_time):
        """Return the centre of an epoch's mask (audit item D4): the fitted mid-time, or the ephemeris
        time when the epoch has no in-transit data or its fit cannot be used.
        """
        covered = np.sum(np.abs(epoch_time - best_t0) < self.planet_duration / 2)
        if self.epoch_fit_usable(status) and covered > 0:
            return best_t0
        logging.info("  Masking transit T0=%.4f around the ephemeris time: %s", t0,
                     "no in-transit data" if covered == 0 else "the epoch fit is not usable")
        return t0

    def move_to_barycentre_frame(self, time):
        """Shift each epoch by minus the timing model, so the barycentre transits on the linear ephemeris.

        A moon travels with the planet-moon barycentre, so a timing variation that moves the whole
        system, such as one from another planet, moves the moon transits with the planet's. The moon
        stage places its windows on the linear ephemeris (bary_t0s), so with moon_windows_follow_ttv
        each epoch of the planet-subtracted curve is shifted by minus the timing model of that epoch
        (the t0 prior centre, `observed_timing_prior`) and the moon windows land where the moon is.
        Only a fitted timing model (linear or sinusoid) moves the data; its shifts are kept in
        barycentre_shifts (days), zero otherwise. The per-point exposures follow the shifted times.

        Parameters
        ----------
        time : numpy.ndarray
            Times of the planet-subtracted curve.

        Returns
        -------
        numpy.ndarray
            The times in the frame of the linear ephemeris.
        """
        centres = getattr(self, "ttv_prior_centres", None)
        model = getattr(self, "ttv_model", None)
        follow = getattr(self, "moon_windows_follow_ttv", False) and model is not None \
            and model["name"] != "none" and centres is not None and len(centres) == len(self.bary_t0s)
        # The fitted planet mid-time of each epoch, not the smooth model: the moon offsets of the search are
        # measured from the planet (a_m), while from the barycentre they are a_m M_p / (M_p + M_m), 7 % less
        # for an 8.5 M_earth moon of a 106 M_earth planet. The per-epoch fit keeps its prior, centred on the
        # model, so a low-SNR epoch stays near the model.
        ttvs = getattr(self, "ttvs", [])
        follow = follow and len(ttvs) == len(self.bary_t0s)
        self.barycentre_shifts = [float(t) for t in ttvs] if follow else [0.0] * len(self.bary_t0s)
        if not follow or not np.any(self.barycentre_shifts) or len(time) == 0:
            return time
        moved = np.asarray(time, dtype=float) - self.nearest_epoch_values(time, self.bary_t0s, self.barycentre_shifts)
        if self.exposure_lookup is not None:
            reference_time, reference_exposure = self.exposure_lookup
            shifted = reference_time - self.nearest_epoch_values(reference_time, self.bary_t0s, self.barycentre_shifts)
            order = np.argsort(shifted, kind="stable")
            self.exposure_lookup = (shifted[order], np.asarray(reference_exposure)[order])
        logging.info("Moon windows follow the %s timing model: epochs shifted by up to %.1f min (RMS %.1f min) so the "
                     "barycentre transits on the linear ephemeris", model["name"],
                     np.max(np.abs(self.barycentre_shifts)) * 1440, np.std(self.barycentre_shifts) * 1440)
        return moved

    def measure_epoch_offsets(self, time, flux, flux_err):
        """Timing pre-pass: measure the barycentre transit offset of each epoch before the detrending.

        The detrending masks and the data kept around each epoch are centred on the epochs; when another
        planet moves the system by tens of minutes, a moon then falls partly outside them and the
        biweight takes part of it (the moon windows alone recover nothing: on KOI-94.01 a 60-min
        sinusoidal TTV cost 27 % of the moon SNR either way). So the planet stage is run once on the
        undetrended curve, each epoch normalised on its own (`normalise_epochs`), and its timing model
        (`observed_timing_prior`), when it finds one, sets epoch_offsets {epoch number: days}, which
        `epoch_distance` and `remove_non_transit_flux` add to the linear ephemeris. The offsets are
        clipped at T14 / 4 like the prior centres.
        """
        self.epoch_offsets = None
        if not self.moon_windows_follow_ttv or flux_err is None:
            return
        pre = copy.copy(self)
        pre.epoch_offsets = None
        pre.null_runs = 0
        try:
            with tempfile.TemporaryDirectory() as plots_dir:
                pre.object_dir = plots_dir
                trimmed = pre.remove_non_transit_flux(np.asarray(time), np.asarray(flux), np.asarray(flux_err),
                                                      pre.bary_t0s, np.max(pre.planet_mass_grid))
                pre.subtract_planet_transit(*trimmed, mode='subtract')
        except Exception as error:  # the main planet stage reports its own failures
            logging.info("Timing pre-pass failed (%s); the windows stay on the linear ephemeris", error)
            return
        model = pre.ttv_model
        if model is None or model["name"] == "none":
            logging.info("Timing pre-pass: no timing model, the windows stay on the linear ephemeris")
            return
        numbers = np.round((np.asarray(self.bary_t0s, dtype=float) - self.planet_t0) / self.planet_period)
        fitted = {"name": model["name"], "period": model["period"],
                  "coefficients": np.asarray(model["coefficients_min"], dtype=float) / 1440}
        limit = self.planet_duration / 4
        offsets = np.clip(self.evaluate_timing_model(fitted, np.asarray(self.bary_t0s, dtype=float), numbers),
                          -limit, limit)
        self.epoch_offsets = {int(n): float(o) for n, o in zip(numbers, offsets)}
        logging.info("Timing pre-pass: %s timing model, detrending masks and data windows follow the epochs by up to "
                     "%.1f min", model["name"], np.max(np.abs(offsets)) * 1440)

    def planet_mid_times(self):
        """Planet mid-transit times in the frame of the moon stage: the fitted times minus the
        barycentre shift of each epoch, or the linear ephemeris without fitted times."""
        ttvs = getattr(self, "ttvs", [])
        if len(ttvs) != len(self.bary_t0s):
            return list(self.bary_t0s)
        shifts = getattr(self, "barycentre_shifts", None)
        if shifts is None or len(shifts) != len(self.bary_t0s):
            shifts = [0.0] * len(self.bary_t0s)
        return [t0 + ttv - shift for t0, ttv, shift in zip(self.bary_t0s, ttvs, shifts)]

    def drop_rejected_epochs(self, rejected_t0s):
        """Log the epoch checks and drop the rejected epochs from bary_t0s and ttvs."""
        if len(rejected_t0s) > 0 or self.epochs_flagged_rms or self.epochs_flagged_rp:
            logging.info("Dropped %d/%d transits: %d epoch fits at a bound or not converged, %d with incomplete "
                         "coverage; kept and flagged: %d for the residual RMS, %d for rp at its bound",
                         len(rejected_t0s), len(self.bary_t0s), len(self.epochs_dropped_fit),
                         len(self.epochs_dropped_coverage), len(self.epochs_flagged_rms), len(self.epochs_flagged_rp))
        if len(rejected_t0s) > 0:
            self.ttvs = [ttv for t0, ttv in zip(self.bary_t0s, self.ttvs) if t0 not in rejected_t0s]
            for name in ("ttv_measured", "ttv_precisions", "ttv_prior_centres", "ttv_prior_sigmas"):
                values = getattr(self, name, None)
                if values is not None and len(values) == len(self.bary_t0s):
                    setattr(self, name, [v for t0, v in zip(self.bary_t0s, values) if t0 not in rejected_t0s])
            self.bary_t0s = [t0 for t0 in self.bary_t0s if t0 not in rejected_t0s]
            self.require_usable_epochs()

    def ttv_diagnostics(self):
        """Log the TTV scatter and periodogram peak, with more than 5 epochs, and plot the periodogram."""
        if len(self.ttvs) <= 5:
            return
        logging.info("TTV diagnostics: RMS=%.4f d, median|O-C|=%.4f d, max|O-C|=%.4f d",
                     np.std(self.ttvs), np.median(np.abs(self.ttvs)), np.max(np.abs(self.ttvs)))
        ttv_times = np.array(self.bary_t0s)
        freqs = np.linspace(1 / (ttv_times[-1] - ttv_times[0]), 0.5 / np.median(np.diff(ttv_times)), 5000)
        power = lombscargle(ttv_times, self.ttvs, 2 * np.pi * freqs)
        periods = 1 / freqs
        best_idx = np.argmax(power)
        logging.info("TTV periodogram: best P=%.4f d, power=%.4f", periods[best_idx], power[best_idx])
        self.plot_ttv_periodogram(periods, power)

    def is_skip_t0(self, time):
        """Check if a transit epoch should be skipped due to insufficient cadences.

        Parameters
        ----------
        time : array_like
            Time array around the transit.

        Returns
        -------
        bool
            True if the transit should be skipped.
        """
        if len(time) < self.min_cadences_per_t0:
            return True
        return False

    def remove_non_transit_flux(self, time, flux, flux_err, t0s, max_planet_mass):
        """Remove flux points outside the moon transit windows.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        t0s : list
            List of barycentric transit times.
        max_planet_mass : float
            Maximum planet mass in Earth masses.

        Returns
        -------
        tuple
            (time, flux, flux_err) restricted to transit windows.
        """
        moon_orbit_transit_duration = self.trim_half_window(max_planet_mass)
        mask_parts = []
        for t0 in t0s:
            centre = t0 + self.epoch_offset(t0, is_time=True)
            mask_parts.append(np.argwhere(
                (time > centre - moon_orbit_transit_duration) & (time < centre + moon_orbit_transit_duration)).flatten())
        flux_mask = np.concatenate(mask_parts).astype(int)
        time = time[flux_mask]
        flux = flux[flux_mask]
        flux_err = flux_err[flux_mask]
        return time, flux, flux_err

    def trim_half_window(self, max_planet_mass):
        """Return the half-width in days of the data kept around each epoch (see `remove_non_transit_flux`)."""
        max_period = self.au_to_period(max_planet_mass * M_earth_to_M_sun, self.compute_hill_radius(self.star_mass, max_planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis))
        moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, max_period)
        keplerian = max(1.0, self.conjunction_time_factor()) * self.compute_transit_duration(
            self.star_radius, self.planet_semimajor_axis * AU_TO_RSUN, self.planet_period,
            moon_semimajor_axis * AU_TO_RSUN)
        # The planet crosses R* + a_m on each side at the measured transit speed.
        measured = 2 * (self.star_radius + moon_semimajor_axis * AU_TO_RSUN) / self.planet_speed()
        return max(keplerian, measured)

    def require_usable_epochs(self):
        """Stop with an explicit error when no transit epoch is left to search.

        Raises
        ------
        NoUsableEpochsError
            If there are no usable epochs.
        """
        if len(self.bary_t0s) == 0:
            raise NoUsableEpochsError("No usable planet transit epochs, so there is nothing to search")
