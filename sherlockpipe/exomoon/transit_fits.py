"""Transit fits of the planet stage: one epoch's mid-time (and depth), the folded transit shape,
and the transit geometry they imply. They do not depend on the search state.
"""
from scipy.optimize import least_squares, minimize
import batman
import numpy as np


class TransitFits:
    """Transit model fits used by the planet stage."""

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
