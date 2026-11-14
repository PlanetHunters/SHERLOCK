"""The planet stage: fold fit of the transit shape, per-epoch mid-times, the planet
subtraction and the derivatives of the subtracted model that the joint fit corrects.
"""
import copy
import logging
import os
import astropy.units as u
from lcbuilder.helper import LcbuilderHelper
from scipy.optimize import least_squares, minimize
import batman
import numpy as np
from scipy import stats
from scipy.signal import lombscargle
import matplotlib.pyplot as plt
from sherlockpipe.exomoon.constants import AU_TO_RSUN, MIN_RP_OVER_RSTAR, M_earth_to_M_sun, NoUsableEpochsError
from sherlockpipe.exomoon.orbits import forecast_moon_mass


class PlanetModel:
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

    @staticmethod
    def fit_status(result, bounds, names, free_edges=(), rtol=1e-3):
        """Summarise an optimizer result: convergence, a finite cost, and the parameters left at a bound.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            Optimizer result.
        bounds : list of (float, float)
            Bounds of each parameter.
        names : list of str
            Name of each parameter.
        free_edges : iterable of (str, str), optional
            (name, 'lower' or 'upper') edges that are physical values rather than limits of the fit,
            such as a central transit at b = 0. They are not reported.
        rtol : float, optional
            Distance to a bound, as a fraction of the bound width, that counts as at the bound.

        Returns
        -------
        dict
            {"converged": bool, "at_bound": [name, ...]}.
        """
        at_bound = []
        for name, value, (lower, upper) in zip(names, np.atleast_1d(result.x), bounds):
            width = upper - lower
            if width <= 0:
                continue
            if value - lower <= rtol * width and (name, 'lower') not in free_edges:
                at_bound.append(name)
            elif upper - value <= rtol * width and (name, 'upper') not in free_edges:
                at_bound.append(name)
        return {"converged": bool(result.success) and bool(np.isfinite(result.fun)), "at_bound": at_bound}

    @classmethod
    def fit_single_t0(cls, time, flux, flux_err, t0_guess, duration_window, transit_params, exposure_time=None,
                      t0_prior_sigma=None):
        """Fit only t0 for a single transit. a, inc, rp must be pre-set on transit_params.

        t0 may move half a transit duration (a quarter of duration_window) from t0_guess. With
        t0_prior_sigma, a Gaussian prior centred on t0_guess keeps low-SNR epochs on the ephemeris.

        Returns (best_theta, best_model, status) where best_theta is [t0] and status comes from
        `fit_status`; with no data, status is None.
        """
        bounds = [(t0_guess - duration_window / 4, t0_guess + duration_window / 4)]

        def model(t0_val, t):
            transit_params.t0 = t0_val
            m = cls.transit_model(transit_params, t, exposure_time)
            return m.light_curve(transit_params)

        def cost(theta):
            m = model(theta[0], time)
            prior = 0.0 if t0_prior_sigma is None else 0.5 * ((theta[0] - t0_guess) / t0_prior_sigma) ** 2
            return 0.5 * np.sum(((flux - m) / flux_err) ** 2) + prior

        status = None
        if len(time) > 0:
            res = minimize(cost, [t0_guess], method='Powell', bounds=bounds)
            best_theta = res.x
            status = cls.fit_status(res, bounds, ['t0'])
        else:
            best_theta = np.array([t0_guess])
        best_model = model(float(best_theta[0]), time)
        return best_theta, best_model, status

    @classmethod
    def fit_single_t0_rp(cls, time, flux, flux_err, t0_guess, rp_guess, rp_err,
                         duration_window, transit_params, exposure_time=None, t0_prior_sigma=None):
        """Fit t0 and rp for a single transit. a, inc pre-set on transit_params.

        t0 has the same bound and prior as in `fit_single_t0`.

        Returns (best_theta, best_model, status) where best_theta is [t0, rp].
        """
        bounds = [(t0_guess - duration_window / 4, t0_guess + duration_window / 4),
                  (max(rp_guess - rp_err, 1e-5), rp_guess + rp_err)]

        def model(theta, t):
            transit_params.t0 = theta[0]
            transit_params.rp = theta[1]
            m = cls.transit_model(transit_params, t, exposure_time)
            return m.light_curve(transit_params)

        def cost(theta):
            m = model(theta, time)
            prior = 0.0 if t0_prior_sigma is None else 0.5 * ((theta[0] - t0_guess) / t0_prior_sigma) ** 2
            return 0.5 * np.sum(((flux - m) / flux_err) ** 2) + prior

        status = None
        if len(time) > 0:
            res = minimize(cost, [t0_guess, rp_guess], method='Powell', bounds=bounds)
            best_theta = res.x
            status = cls.fit_status(res, bounds, ['t0', 'rp'])
        else:
            best_theta = np.array([t0_guess, rp_guess])
        best_model = model(best_theta, time)
        return best_theta, best_model, status

    @staticmethod
    def geometry_from_shape(t14, impact_param, rp, period, ecc, arg_periastron):
        """Return the orbit (a/R*, inclination) that gives a transit duration and impact parameter.

        Inverts Winn (2010, Eqs. 7 and 14-16): b = a cos i (1 - e^2) / (1 + e sin w) and
        T14 = P / pi * arcsin(sqrt((1 + k)^2 - b^2) / (a sin i)) * sqrt(1 - e^2) / (1 + e sin w).

        Parameters
        ----------
        t14 : float
            First-to-fourth contact duration in days.
        impact_param : float
            Impact parameter in stellar radii.
        rp : float
            Planet radius in stellar radii.
        period : float
            Orbital period in days.
        ecc, arg_periastron : float
            Eccentricity and argument of periastron in degrees.

        Returns
        -------
        tuple or None
            (a/R*, inclination in degrees), or None when no orbit has that shape.
        """
        sin_w = np.sin(np.deg2rad(arg_periastron))
        speed_factor = np.sqrt(1 - ecc ** 2) / (1 + ecc * sin_w)
        distance_factor = (1 - ecc ** 2) / (1 + ecc * sin_w)
        chord_squared = (1 + rp) ** 2 - impact_param ** 2
        angle = np.pi * t14 / (period * speed_factor)
        if not (t14 > 0 and chord_squared > 0 and 0 < angle < np.pi / 2 and impact_param >= 0):
            return None
        a_sin_i = np.sqrt(chord_squared) / np.sin(angle)
        a_cos_i = impact_param / distance_factor
        return float(np.hypot(a_sin_i, a_cos_i)), float(np.rad2deg(np.arctan2(a_sin_i, a_cos_i)))

    @classmethod
    def fit_fold_shape(cls, time, flux, flux_err, guess_params, guess_params_bounds,
                       transit_params: batman.TransitParams, exposure_time=None):
        """Fit the folded transit in duration, impact parameter and radius (audit item B1).

        Fitting (t0, T14, b, rp) instead of (t0, a/R*, inc, rp) lets the fit reach any transit
        shape: a duration longer than the catalog a/R* allows no longer leaves a residual.

        Parameters
        ----------
        time, flux, flux_err : array_like
            Folded light curve, time relative to the transit centre.
        guess_params : list
            Starting [t0, t14, b, rp].
        guess_params_bounds : list
            Bounds for each parameter.
        transit_params : batman.TransitParams
            Period, eccentricity, argument of periastron and limb darkening are read from it; t0,
            a, inc and rp are overwritten.
        exposure_time : float, optional
            Exposure time in days for the model.

        Returns
        -------
        tuple
            (best_theta, best_model, geometry, status): best_theta is [t0, t14, b, rp], geometry
            is (a/R*, inclination) and status comes from `fit_status`.
        """
        def geometry(theta):
            return cls.geometry_from_shape(theta[1], theta[2], theta[3], transit_params.per,
                                                          transit_params.ecc, transit_params.w)

        def model(theta, t):
            transit_params.t0, transit_params.rp = theta[0], theta[3]
            transit_params.a, transit_params.inc = geometry(theta)
            return cls.transit_model(transit_params, t, exposure_time).light_curve(transit_params)

        def cost(theta):
            if geometry(theta) is None:
                return 1e30
            return 0.5 * np.sum(((flux - model(theta, time)) / flux_err) ** 2)

        guess_params = list(guess_params)
        if geometry(guess_params) is None:
            raise ValueError(f"The fold starting point {guess_params} has no transiting orbit")
        # b, rp and T14 are degenerate for grazing transits, so a single Powell run can stop in a
        # wrong valley. Start from several impact parameters and keep the lowest cost.
        b_lower, b_upper = guess_params_bounds[2]
        starts = [guess_params]
        for b_start in (0.0, 0.5, 0.8, 0.95 * (1 + guess_params[3])):
            candidate = list(guess_params)
            candidate[2] = float(np.clip(b_start, b_lower, b_upper))
            if geometry(candidate) is not None and all(abs(candidate[2] - start[2]) > 1e-3 for start in starts):
                starts.append(candidate)
        res = min((minimize(cost, start, method='Powell', bounds=guess_params_bounds) for start in starts),
                  key=lambda result: result.fun)
        # Powell stops early in narrow valleys; one restart from its result finishes the descent.
        polished = minimize(cost, res.x, method='Powell', bounds=guess_params_bounds)
        if polished.fun <= res.fun:
            res = polished
        # B8: Powell's ftol is relative to a cost that grows with the number of points, so with
        # correlated b, T14 and rp it stops early on large folds. A bounded least-squares run on
        # the residuals from its result converges on the gradient instead.
        n_points = len(flux)

        def residuals(theta):
            if geometry(theta) is None:
                return np.full(n_points, 1e6)
            return (flux - model(theta, time)) / flux_err

        lower = np.array([bound[0] for bound in guess_params_bounds], dtype=float)
        upper = np.array([bound[1] for bound in guess_params_bounds], dtype=float)
        start = np.clip(np.asarray(res.x, dtype=float), lower, upper)
        try:
            refined = least_squares(residuals, start, bounds=(lower, upper), method='trf', x_scale='jac')
            if refined.success and refined.cost <= res.fun and geometry(refined.x) is not None:
                refined.fun = refined.cost
                res = refined
        except ValueError:
            pass
        best_theta = res.x if geometry(res.x) is not None else np.asarray(guess_params, dtype=float)
        status = cls.fit_status(res, guess_params_bounds, ['t0', 't14', 'b', 'rp'],
                                                free_edges=[('b', 'lower')])
        best_model = model(best_theta, time)
        return best_theta, best_model, geometry(best_theta), status

    @classmethod
    def ingress_duration(cls, transit_params, t14):
        """Return the ingress duration (T14 - T23) / 2 in days, or T14 / 2 for a grazing transit.

        Parameters
        ----------
        transit_params : batman.TransitParams
            Planet geometry: a/R*, inclination, rp, period, eccentricity and argument of periastron.
        t14 : float
            First-to-fourth contact duration in days.

        Returns
        -------
        float
            Ingress duration in days.
        """
        sin_w = np.sin(np.deg2rad(transit_params.w))
        speed_factor = np.sqrt(1 - transit_params.ecc ** 2) / (1 + transit_params.ecc * sin_w)
        b = cls.impact_parameter(transit_params.a, transit_params.inc, transit_params.ecc,
                                                 transit_params.w)
        chord_squared = (1 - transit_params.rp) ** 2 - b ** 2
        a_sin_i = transit_params.a * np.sin(np.deg2rad(transit_params.inc))
        if chord_squared <= 0 or a_sin_i <= 0:
            return t14 / 2
        t23 = transit_params.per / np.pi * np.arcsin(min(np.sqrt(chord_squared) / a_sin_i, 1.0)) * speed_factor
        return float(np.clip((t14 - t23) / 2, 0, t14 / 2))

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

    def realign_fold(self, time, flux, flux_err, params, duration_window, prior_sigmas, fold_bounds):
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
        for t0, sigma in zip(self.bary_t0s, prior_sigmas):
            mask = np.abs(time - t0) < duration_window
            if np.sum(mask) < self.min_cadences_per_t0:
                continue
            epoch_params = copy.copy(params)
            epoch_params.t0, epoch_params.u = t0, self.ab
            theta, _, status = self.fit_single_t0(time[mask], flux[mask], flux_err[mask], t0,
                                                                 duration_window, epoch_params,
                                                                 self.exposure_at(time[mask]), sigma)
            usable = status is None or (status["converged"] and "t0" not in status["at_bound"])
            shifts.append((t0, float(theta[0] - t0) if usable else 0.0))
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
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        duration_window = self.planet_duration * 2
        result_model = flux.copy()
        result_time = time.copy()
        if flux_err is not None:
            result_flux_err = flux_err.copy()
        else:
            result_flux_err = None
        params = batman.TransitParams()
        params.per = self.planet_period
        params.ecc = self.planet_ecc  # eccentricity
        params.w = self.planet_arg_periastron  # longitude of periastron (deg)
        params.rp = self.planet_radius / star_radius_rearth  # planet radius / stellar radius
        params.a = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                u.au) \
            if self.planet_a_over_rstar is None else self.planet_a_over_rstar
        params.inc = self.planet_inc
        params.u = self.ab
        params.limb_dark = "quadratic"
        flux, flux_err = self.normalise_epochs(time, flux, flux_err)
        result_model = flux.copy()
        if flux_err is not None:
            result_flux_err = flux_err.copy()
        # Phase-fold all transits to get a reliable transit shape
        fold_time = []
        fold_flux = []
        fold_flux_err = []
        fold_exposure = []
        for t0 in self.bary_t0s:
            mask = np.abs(time - t0) < duration_window
            if len(time[mask]) < self.min_cadences_per_t0:
                continue
            if self.exposure_lookup is not None:
                fold_exposure.append(self.exposure_at(time[mask]))
            fold_time.append(time[mask] - t0)
            fold_flux.append(flux[mask])
            if flux_err is not None:
                fold_flux_err.append(flux_err[mask])
        if len(fold_time) > 0:
            fold_time = np.concatenate(fold_time)
            fold_exposure = np.concatenate(fold_exposure) if fold_exposure else self.exposure_time
            fold_flux = np.concatenate(fold_flux)
            fold_flux_err = np.concatenate(fold_flux_err) if flux_err is not None else None
            # B1: fit the fold in duration, impact parameter and radius, so any transit shape is
            # reachable. The start is the shape of the input geometry.
            rp_err = self.planet_radius_err / star_radius_rearth
            start_params = copy.copy(params)
            start_params.t0 = 0.0
            start_t14 = self.model_transit_duration(start_params, duration_window)
            if start_t14 <= 0:
                start_t14 = self.planet_duration
            start_b = self.impact_parameter(params.a, params.inc, params.ecc, params.w)
            if not self.skip_fold_fit and fold_flux_err is not None:
                # B2: the catalog radius comes from another stellar radius and may not match this
                # depth, and the bound is only +-1 sigma wide. The start, and the centre of that
                # bound, is the radius that gives the observed central fold depth with the input
                # shape (limb darkening included); the width stays +-1 sigma, so the planet does not
                # gain room to absorb a moon (audit item B3).
                centre = np.abs(fold_time) < 0.25 * start_t14
                if np.sum(centre) >= 10:
                    observed = 1 - np.median(fold_flux[centre])
                    model_centre = self.transit_model(
                        start_params, fold_time[centre],
                        self.exposure_subset(fold_exposure, centre)).light_curve(start_params)
                    expected = 1 - np.median(model_centre)
                    if observed > 0 and expected > 0:
                        data_rp = float(params.rp * np.sqrt(observed / expected))
                        if abs(data_rp / params.rp - 1) > 0.01:
                            logging.info("Planet rp start from the fold depth: %.4f (catalog %.4f, %.0f ppm observed "
                                         "against %.0f ppm expected at the centre)", data_rp, params.rp,
                                         observed * 1e6, expected * 1e6)
                        params.rp = data_rp
            fold_guess = [0.0, start_t14, min(start_b, 1.0), params.rp]
            fold_bounds = [(-start_t14 / 2, start_t14 / 2),
                           (start_t14 / 2, start_t14 * 2),
                           (0.0, 1.0 + params.rp + rp_err),
                           (max(params.rp - rp_err, MIN_RP_OVER_RSTAR), params.rp + rp_err)]
            if self.fix_impact_param and self.planet_impact_param is not None:
                # The input impact parameter is kept: its bound is a point (a width the optimizers accept).
                fixed_b = float(np.clip(self.planet_impact_param, 0.0, 1.0 + params.rp))
                fold_guess[2] = fixed_b
                fold_bounds[2] = (fixed_b, fixed_b + 1e-9)
        self.fold_fit_status = None
        if not self.skip_fold_fit:
            fold_theta, _, (fold_a, fold_inc), self.fold_fit_status = self.fit_fold_shape(
                fold_time, fold_flux, fold_flux_err, fold_guess, fold_bounds, params, fold_exposure)
            logging.info("Fold fit: T14=%.4f->%.4f d, b=%.3f->%.3f, rp=%.4f->%.4f (depth %.1f->%.1f ppm), "
                         "a/R*=%.2f->%.2f, inc=%.2f->%.2f", fold_guess[1], fold_theta[1], fold_guess[2],
                         fold_theta[2], fold_guess[3], fold_theta[3], fold_guess[3] ** 2 * 1e6,
                         fold_theta[3] ** 2 * 1e6, start_params.a, fold_a, start_params.inc, fold_inc)
            if not self.fold_fit_status["converged"] or self.fold_fit_status["at_bound"]:
                logging.warning("Fold fit did not converge cleanly (converged=%s, at bound: %s); the planet "
                                "model may leave residuals", self.fold_fit_status["converged"],
                                ", ".join(self.fold_fit_status["at_bound"]) or "none")
            params.a, params.inc, params.rp = fold_a, fold_inc, fold_theta[3]
            params_best = copy.copy(params)
            params_best.t0 = fold_theta[0]
            fitted_duration = self.model_transit_duration(params_best, duration_window)
            if fitted_duration > 0:
                self.planet_duration = fitted_duration
            # B4: the moon stage uses the fitted transit geometry (a/R*, inclination, b) for durations
            # and templates. The semi-major axis in AU, from Kepler's third law, still sets the
            # Hill sphere; the moon offsets a_m / v_p use the input transit duration (`planet_speed`),
            # and the fitted duration only checks it.
            self.check_fitted_duration()
            self.planet_inc = float(fold_inc)
            self.planet_impact_param = float(fold_theta[2])
            self.planet_a_over_rstar = float(fold_a)
            self.planet_rp_over_rstar = float(fold_theta[3])
            if plot:
                guess_params = copy.copy(start_params)
                model_guess = self.transit_model(guess_params, fold_time,
                                                                fold_exposure).light_curve(guess_params)
                model_best = self.transit_model(params_best, fold_time,
                                                               fold_exposure).light_curve(params_best)
                fig_fold, ax_fold = plt.subplots(1, 1, figsize=(10, 6))
                ax_fold.scatter(fold_time, fold_flux, color='gray', alpha=0.5, s=5)
                fold_order = np.argsort(fold_time)
                ax_fold.plot(fold_time[fold_order], model_guess[fold_order], 'b--', alpha=0.7, label='Initial guess')
                ax_fold.plot(fold_time[fold_order], model_best[fold_order], 'r-', alpha=1, label='Fold fit')
                ax_fold.set_title(f"Phase-folded fit: T14={fold_guess[1]:.4f}->{fold_theta[1]:.4f} d, "
                                  f"b={fold_guess[2]:.2f}->{fold_theta[2]:.2f}, "
                                  f"depth={fold_guess[3]**2*1e6:.0f}->{fold_theta[3]**2*1e6:.0f} ppm")
                ax_fold.set_xlabel('Time from T0 (days)')
                ax_fold.set_ylabel('Flux')
                ax_fold.legend()
                plt.savefig(os.path.join(self.object_dir, "fold_fit.png"))
                plt.close(fig_fold)
        self.ttvs = []
        ttv_t0s = []
        rejected_t0s = []
        self.epochs_dropped_fit = []
        self.epochs_flagged_rms = []
        self.epochs_flagged_rp = []
        self.epochs_dropped_coverage = []
        ingress = self.ingress_duration(params, self.planet_duration)
        if self.ttv_prior_width == "auto":
            self.ttv_prior_width_used = self.max_moon_ttv()
            logging.info("TTV allowed by the t0 prior: %.1f min, the largest a searched moon can cause (%.2f R_earth, "
                         "%.1f M_earth forecast, on the widest searched orbit)", self.ttv_prior_width_used * 1440,
                         self.max_radius, forecast_moon_mass(self.max_radius, self.max_moon_density))
        else:
            self.ttv_prior_width_used = self.ttv_prior_width
            if self.ttv_prior_width is not None:
                logging.info("TTV allowed by the t0 prior: %.1f min (given)", self.ttv_prior_width * 1440)
        prior_sigmas = [None] * len(self.bary_t0s)
        if self.ttv_prior_width_used is not None:
            precisions = [self.epoch_timing_precision(time[np.abs(time - t0) < duration_window] - t0,
                                                      flux[np.abs(time - t0) < duration_window], params.rp ** 2,
                                                      ingress) for t0 in self.bary_t0s]
            ephemeris = self.ephemeris_sigmas(self.bary_t0s, precisions)
            prior_sigmas = [None if sigma is None else float(np.hypot(sigma, self.ttv_prior_width_used))
                            for sigma in ephemeris]
            known = [sigma for sigma in ephemeris if sigma is not None]
            if known:
                logging.info("Ephemeris uncertainty in the t0 prior (from %s): median %.1f min, largest %.1f min",
                             self.ephemeris_source, np.median(known) * 1440, np.max(known) * 1440)
        self.fold_realigned = False
        if not self.skip_fold_fit and len(fold_time) > 0 and flux_err is not None:
            self.realign_fold(time, flux, flux_err, params, duration_window, prior_sigmas, fold_bounds)
        for t0, t0_prior_sigma in zip(self.bary_t0s, prior_sigmas):
            mask = np.abs(time - t0) < duration_window
            intransit_indexes = np.argwhere(np.abs(time - t0) < duration_window).flatten()
            # Always fit t0 and compute TTV
            epoch_params = copy.copy(params)
            epoch_params.t0 = t0
            epoch_params.u = self.ab
            if self.fit_rp_per_transit:
                best_theta, best_model, status = self.fit_single_t0_rp(
                    time[mask], flux[mask], flux_err[mask], t0, params.rp,
                    self.planet_radius_err / star_radius_rearth, duration_window, epoch_params, self.exposure_at(time[mask]),
                    t0_prior_sigma)
            else:
                best_theta, best_model, status = self.fit_single_t0(
                    time[mask], flux[mask], flux_err[mask], t0, duration_window, epoch_params, self.exposure_at(time[mask]),
                    t0_prior_sigma)
            best_t0 = best_theta[0]
            self.ttvs.append(best_t0 - t0)
            ttv_t0s.append(t0)
            # C4: a fit that did not converge, or whose t0 stopped at its bound, subtracts the planet
            # at the wrong time, so the epoch is dropped. An rp at its bound is only flagged: it
            # means extra depth, such as a moon inside the transit, which the bound keeps out of
            # the planet model (audit item B3).
            if status is not None and (not status["converged"] or "t0" in status["at_bound"]):
                logging.info("  Rejecting transit T0=%.4f: epoch fit not usable (converged=%s, at bound: %s)", t0,
                             status["converged"], ", ".join(status["at_bound"]) or "none")
                rejected_t0s.append(t0)
                self.epochs_dropped_fit.append({"t0": float(t0), "converged": status["converged"],
                                                "at_bound": status["at_bound"]})
            elif status is not None and "rp" in status["at_bound"]:
                logging.info("  Flagging transit T0=%.4f: epoch rp at its bound (%.4f), kept", t0, best_theta[1])
                self.epochs_flagged_rp.append({"t0": float(t0), "rp": float(best_theta[1])})
            if mode == 'subtract':
                result_model[intransit_indexes] = result_model[intransit_indexes] - (best_model - 1)
                # Quality control: post-subtraction RMS ratio
                transit_mask = np.abs(time[mask] - best_t0) < self.planet_duration / 2
                oot_mask = ~transit_mask
                # Quality control: in-transit completeness check. The points within T14 count, not
                # those where the model dims: averaged over a long exposure it still dims half an
                # exposure beyond T14, and an epoch short of in-transit points passed.
                n_in_transit = int(np.sum(transit_mask))
                cadence_d = np.median(np.diff(time[mask])) if len(time[mask]) > 1 else 0
                n_expected = self.planet_duration / cadence_d if cadence_d > 0 else np.inf
                incomplete = n_in_transit < self.min_transit_completeness * n_expected
                if np.sum(transit_mask) > 5 and np.sum(oot_mask) > 5:
                    post_res = result_model[intransit_indexes]
                    it_rms = np.std(post_res[transit_mask])
                    oot_rms = np.std(post_res[oot_mask])
                    rms_ratio = it_rms / oot_rms if oot_rms > 0 else float('inf')
                    if rms_ratio > self.max_subtraction_rms_ratio:
                        # D1: a moon dip inside the transit also raises the in-transit scatter, so the
                        # epoch is flagged, not dropped.
                        logging.info("  Flagging transit T0=%.4f: post-sub RMS ratio=%.2f (it=%.6f, oot=%.6f), "
                                     "kept because a moon dip also raises it", best_t0, rms_ratio, it_rms, oot_rms)
                        self.epochs_flagged_rms.append({"t0": float(t0), "rms_ratio": float(rms_ratio)})
                    if incomplete and t0 not in rejected_t0s:
                        logging.info("  Rejecting transit T0=%.4f: incomplete in-transit coverage (%d pts vs %.0f expected)",
                                     best_t0, n_in_transit, n_expected)
                        rejected_t0s.append(t0)
                        self.epochs_dropped_coverage.append({"t0": float(t0), "points": int(n_in_transit),
                                                             "expected": float(n_expected)})
                elif incomplete and t0 not in rejected_t0s:
                    logging.info("  Rejecting transit T0=%.4f: incomplete in-transit coverage (%d pts vs %.0f expected)",
                                 best_t0, n_in_transit, n_expected)
                    rejected_t0s.append(t0)
                    self.epochs_dropped_coverage.append({"t0": float(t0), "points": int(n_in_transit),
                                                         "expected": float(n_expected)})
                if plot:
                    suffix = "_rejected" if t0 in rejected_t0s else "_sub"
                    view_dir = os.path.join(self.object_dir, "t0s_view")
                    os.makedirs(view_dir, exist_ok=True)
                    fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
                    axs[0].scatter(time[intransit_indexes], flux[
                        intransit_indexes],
                                   color='gray', alpha=1, rasterized=True, label="Flux")
                    axs[0].plot(time[intransit_indexes], best_model)
                    axs[0].set_title("Subtracting transit in t0 " + str(best_t0))
                    axs[0].set_xlabel('Time')
                    axs[0].set_ylabel('Flux')
                    axs[1].scatter(time[intransit_indexes], result_model[intransit_indexes],
                                color='gray', alpha=1, rasterized=True, label="Flux")
                    axs[1].set_title(
                        "Subtracted transit in t0 " + str(best_t0))
                    axs[1].set_xlabel('Time')
                    axs[1].set_ylabel('Flux')
                    plt.savefig(os.path.join(view_dir, "T0_" + str(round(best_t0, 3)) + suffix + ".png"))
                    plt.close()
            else:
                # D4: the mask needs a usable mid-time. Without in-transit data, or with a fit that
                # cannot be used, it is centred on the ephemeris time.
                usable_fit = status is None or (status["converged"] and "t0" not in status["at_bound"])
                covered = np.sum(np.abs(time[mask] - best_t0) < self.planet_duration / 2)
                if not usable_fit or covered == 0:
                    logging.info("  Masking transit T0=%.4f around the ephemeris time: %s", t0,
                                 "no in-transit data" if covered == 0 else "the epoch fit is not usable")
                    best_t0 = t0
                half_mask = self.mask_half_width(time[mask])
                intransit_mask_local = np.abs(result_time - best_t0) <= half_mask
                if plot:
                    view_dir = os.path.join(self.object_dir, "t0s_view")
                    os.makedirs(view_dir, exist_ok=True)
                    fig_t, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    # Before mask: windowed flux + model
                    ax1.scatter(time[mask], flux[mask], color='gray', alpha=0.7, s=10)
                    ax1.plot(time[mask], best_model, color='orange', alpha=1, label='Planet model')
                    ax1.axvspan(best_t0 - half_mask, best_t0 + half_mask,
                                color='red', alpha=0.15, label='Masked region')
                    ax1.axvline(best_t0, color='red', linestyle='--')
                    ax1.set_title(f"T0={best_t0:.4f} (before mask)")
                    ax1.set_ylabel('Flux'); ax1.legend()
                    # After mask: remaining data at this window
                    keep_mask_window = ~(np.abs(time[mask] - best_t0) <= half_mask)
                    if np.any(keep_mask_window):
                        ax2.scatter(time[mask][keep_mask_window], flux[mask][keep_mask_window],
                                    color='gray', alpha=0.7, s=10, label='After mask')
                    ax2.set_title(f"T0={best_t0:.4f} (after mask)")
                    ax2.set_xlabel('Time'); ax2.set_ylabel('Flux')
                    ax2.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(view_dir, "T0_" + str(round(best_t0, 3)) + "_mask.png"))
                    plt.close()
                result_time = result_time[~intransit_mask_local]
                result_model = result_model[~intransit_mask_local]
                if result_flux_err is not None:
                    result_flux_err = result_flux_err[~intransit_mask_local]
        if len(rejected_t0s) > 0 or self.epochs_flagged_rms or self.epochs_flagged_rp:
            logging.info("Dropped %d/%d transits: %d epoch fits at a bound or not converged, %d with incomplete "
                         "coverage; kept and flagged: %d for the residual RMS, %d for rp at its bound",
                         len(rejected_t0s), len(self.bary_t0s), len(self.epochs_dropped_fit),
                         len(self.epochs_dropped_coverage), len(self.epochs_flagged_rms), len(self.epochs_flagged_rp))
        if len(rejected_t0s) > 0:
            self.bary_t0s = [t0 for t0 in self.bary_t0s if t0 not in rejected_t0s]
            self.ttvs = [ttv for t0, ttv in zip(ttv_t0s, self.ttvs) if t0 not in rejected_t0s]
            self.require_usable_epochs()
        if mode == 'subtract':
            # Checked in run() once the red-noise factor is measured.
            self.planet_shape_check_inputs = (result_time, result_model,
                                              [t0 + ttv for t0, ttv in zip(self.bary_t0s, self.ttvs)])
        else:
            self.warn_mask_mode_survival()
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.ttvs)), self.ttvs, marker='o')
        plt.xticks(range(len(self.ttvs)), [f"{t0:.4f}" for t0 in self.bary_t0s], rotation=45)
        plt.xlabel("Barycentric T0 Values")
        plt.ylabel("TTV (days)")
        plt.title("Transit Timing Variations")
        plt.tight_layout()
        ttv_path = os.path.join(self.object_dir, "ttvs.png") if self.object_dir else None
        if ttv_path:
            plt.savefig(ttv_path)
        plt.close()
        if len(self.ttvs) > 5:
            ttv_rms = np.std(self.ttvs)
            logging.info("TTV diagnostics: RMS=%.4f d, median|O-C|=%.4f d, max|O-C|=%.4f d",
                         ttv_rms, np.median(np.abs(self.ttvs)), np.max(np.abs(self.ttvs)))
            ttv_times = np.array(self.bary_t0s)
            freqs = np.linspace(1 / (ttv_times[-1] - ttv_times[0]),
                                0.5 / np.median(np.diff(ttv_times)), 5000)
            power = lombscargle(ttv_times, self.ttvs, 2 * np.pi * freqs)
            periods = 1 / freqs
            best_idx = np.argmax(power)
            logging.info("TTV periodogram: best P=%.4f d, power=%.4f",
                         periods[best_idx], power[best_idx])
            fig_ttv_per, ax_ttv = plt.subplots(1, 1, figsize=(10, 6))
            ax_ttv.plot(periods, power, color='gray')
            ax_ttv.set_xlabel("Period (days)")
            ax_ttv.set_ylabel("Lomb-Scargle Power")
            ax_ttv.set_title("TTV Periodogram")
            ax_ttv.set_xscale('log')
            plt.tight_layout()
            plt.savefig(self.object_dir + "/ttv_periodogram.png")
            plt.close()
        return result_time, result_model, result_flux_err

    @staticmethod
    def model_transit_duration(transit_params, half_window, samples=200001):
        """Return the first-to-fourth contact duration of a transit model.

        The model is evaluated on a fine grid independent of the data cadence, so the
        result does not depend on how the light curve was sampled.

        Parameters
        ----------
        transit_params : batman.TransitParams
            Transit parameters, with t0 at the transit centre.
        half_window : float
            Half width of the grid around t0 in days. It must contain the transit.
        samples : int, optional
            Grid size, default 200001.

        Returns
        -------
        float
            Duration in days, or 0 when the model has no transit in the window.
        """
        grid = transit_params.t0 + np.linspace(-half_window, half_window, samples)
        in_transit = batman.TransitModel(transit_params, grid).light_curve(transit_params) < 1.0
        if not np.any(in_transit):
            return 0.0
        step = grid[1] - grid[0]
        first, last = np.argmax(in_transit), len(in_transit) - 1 - np.argmax(in_transit[::-1])
        # Contacts lie between the last out-of-transit and first in-transit samples, take the midpoint.
        return (grid[last] - grid[first]) + step

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
            mask_parts.append(np.argwhere(
                (time > t0 - moon_orbit_transit_duration) & (time < t0 + moon_orbit_transit_duration)).flatten())
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

    def planet_shape_derivatives(self, time):
        """Return the derivatives of the subtracted planet model with respect to its shape.

        One column per term of PLANET_SHAPE_TERMS (squared impact parameter, u1, u2), each evaluated
        at every epoch's mid-time, so it is the same shape in every epoch and zero away from the
        transits. Like the depth, each gets one coefficient shared by all epochs (see
        `planet_fit`), so it keeps the stacked power. The geometry column is taken in b^2: the
        flux is even in b, so its b derivative vanishes at b = 0 and a near-central transit would
        get an unbounded coefficient, while the b^2 derivative stays finite. It is a central
        difference, forward when b^2 is below the step; None without a planet model to correct.
        """
        if self.subtract_mode != 'subtract' or time is None or len(time) == 0 or len(self.bary_t0s) == 0:
            return None
        params = self.null_planet_params()
        mid_times = [t0 + ttv for t0, ttv in zip(self.bary_t0s, self.ttvs)] \
            if len(getattr(self, "ttvs", [])) == len(self.bary_t0s) else list(self.bary_t0s)
        distance_factor = (1 - params.ecc ** 2) / (1 + params.ecc * np.sin(np.deg2rad(params.w)))
        impact = self.impact_parameter(params.a, params.inc, params.ecc, params.w)
        self.planet_shape_impact = float(impact)

        def with_impact(base, value):
            changed = copy.copy(base)
            changed.inc = float(np.rad2deg(np.arccos(np.clip(value / (base.a * distance_factor), -1, 1))))
            return changed

        def with_ld(base, index, delta):
            changed = copy.copy(base)
            changed.u = list(base.u)
            changed.u[index] = changed.u[index] + delta
            return changed

        step = 1e-3
        columns = np.zeros((len(time), len(self.PLANET_SHAPE_TERMS)))
        for mid_time in mid_times:
            near = np.abs(time - mid_time) < self.planet_duration
            if not np.any(near):
                continue
            nominal = copy.copy(params)
            nominal.t0 = mid_time
            high_b2, low_b2 = impact ** 2 + step, max(impact ** 2 - step, 0.0)
            pairs = [(with_impact(nominal, np.sqrt(high_b2)), with_impact(nominal, np.sqrt(low_b2)), high_b2 - low_b2),
                     (with_ld(nominal, 0, step), with_ld(nominal, 0, -step), 2 * step),
                     (with_ld(nominal, 1, step), with_ld(nominal, 1, -step), 2 * step)]
            for term, (upper, lower, width) in enumerate(pairs):
                flux_upper = self.transit_model(upper, time[near], self.exposure_at(time[near])).light_curve(upper)
                flux_lower = self.transit_model(lower, time[near], self.exposure_at(time[near])).light_curve(lower)
                columns[near, term] = (flux_upper - flux_lower) / width
        return columns if np.any(columns != 0) else None

    def planet_t0_derivative(self, time):
        """Return the derivative of the subtracted planet model with respect to each epoch's mid-time.

        Returns
        -------
        tuple
            (column, epoch_index): the derivative at `time`, zero away from the transits, and the
            index in bary_t0s of the epoch each point belongs to (-1 away from the transits); both
            None when there is no planet model to correct (see `planet_depth_derivative`).
        """
        if self.subtract_mode != 'subtract' or time is None or len(time) == 0 or len(self.bary_t0s) == 0:
            return None, None
        params = self.null_planet_params()
        ttvs = getattr(self, "ttvs", [])
        mid_times = [t0 + ttv for t0, ttv in zip(self.bary_t0s, ttvs)] if len(ttvs) == len(self.bary_t0s) \
            else list(self.bary_t0s)
        step = 1e-4
        column = np.zeros(len(time))
        epoch_index = np.full(len(time), -1, dtype=int)
        for index, mid_time in enumerate(mid_times):
            near = np.abs(time - mid_time) < self.planet_duration
            if not np.any(near):
                continue
            later, earlier = copy.copy(params), copy.copy(params)
            later.t0, earlier.t0 = mid_time + step, mid_time - step
            flux_later = self.transit_model(later, time[near], self.exposure_at(time[near])).light_curve(later)
            flux_earlier = self.transit_model(earlier, time[near],
                                                             self.exposure_at(time[near])).light_curve(earlier)
            column[near] = (flux_later - flux_earlier) / (2 * step)
            epoch_index[near] = index
        if not np.any(column != 0):
            return None, None
        return column, epoch_index

    def planet_depth_derivative(self, time):
        """Return the derivative of the subtracted planet model with respect to its depth at `time`.

        It is evaluated with the fitted planet shape at the mid-time of each kept epoch (its TTV
        included), so it is the same shape in every epoch, and it is zero away from the transits.
        `search` fits one coefficient of it for all epochs (see `planet_depth_fit`). Only subtract
        mode has a planet model to correct; mask mode and an empty curve give None.
        """
        if self.subtract_mode != 'subtract' or time is None or len(time) == 0 or len(self.bary_t0s) == 0:
            return None
        params = self.null_planet_params()
        mid_times = [t0 + ttv for t0, ttv in zip(self.bary_t0s, self.ttvs)] \
            if len(getattr(self, "ttvs", [])) == len(self.bary_t0s) else list(self.bary_t0s)
        step = 1e-3
        column = np.zeros(len(time))
        for mid_time in mid_times:
            near = np.abs(time - mid_time) < self.planet_duration
            if not np.any(near):
                continue
            nominal = copy.copy(params)
            nominal.t0 = mid_time
            deeper = copy.copy(nominal)
            deeper.rp = nominal.rp * (1 + step)
            flux = self.transit_model(nominal, time[near], self.exposure_at(time[near])).light_curve(nominal)
            flux_deeper = self.transit_model(deeper, time[near], self.exposure_at(time[near])).light_curve(deeper)
            column[near] = (flux_deeper - flux) / (deeper.rp ** 2 - nominal.rp ** 2)
        return column if np.any(column != 0) else None
