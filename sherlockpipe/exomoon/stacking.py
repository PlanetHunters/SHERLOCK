"""Stacking of the epochs of one moon hypothesis into the moon window and its surroundings,
with the upper-outlier clipping.
"""
import warnings
import numpy as np
from sherlockpipe.exomoon.joint_fit import _planet_model, _planet_normal_equations, _window_cross_terms


class MoonStacking:
    """Stacking of the epochs of one moon hypothesis into the moon window and its surroundings,
    with the upper-outlier clipping.
    """

    @classmethod
    #@njit(fastmath=True, parallel=False)
    def compute_moon_transit_scenarios(cls, time, flux, flux_err, planet_t0, moon_initial_alpha, moon_period, bary_t0s,
                                       half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration,
                                       half_moon_transit_duration, extra=None, phase_times=None):
        """Compute moon transit scenarios for all barycentric transit times.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        planet_t0 : float
            First planet transit epoch.
        moon_initial_alpha : float
            Initial moon orbital phase in radians.
        moon_period : float
            Moon orbital period in days.
        bary_t0s : list
            List of barycentric transit times.
        half_moon_orbit_transit_duration : float
            Half the moon orbit transit duration in days.
        half_moon_max_orbit_transit_duration : float
            Half the maximum moon orbit transit duration in days.
        half_moon_transit_duration : float
            Half the moon transit duration in days.
        extra : array_like, optional
            A per-point column carried with the flux (the planet depth column), appended to each
            scenario.

        Returns
        -------
        list
            List of [moon_alpha, time, flux, flux_err] (plus the extra column when given), one for
            each epoch with data. Empty if no epoch has data.
        """
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = []
        # On a sorted curve each epoch's window is one slice, found by bisection instead of a pass
        # over every point (the same points as the comparison below).
        time_sorted = len(time) < 2 or bool(np.all(time[1:] >= time[:-1]))
        for index, t0 in enumerate(bary_t0s):
            max_moon_t0_tau = cls.compute_moon_ts(t0, half_moon_orbit_transit_duration, half_moon_transit_duration)
            phase_time = t0 if phase_times is None else phase_times[index]
            moon_t0, moon_alpha = cls.compute_moon_t0(planet_t0, phase_time, moon_period, moon_initial_alpha,
                                                      max_moon_t0_tau)
            # The phase may come from another epoch; the window stays on this one.
            moon_t0 = moon_t0 - phase_time + t0
            moon_t1 = moon_t0 - half_moon_transit_duration
            # t1 = moon_orbit_range[1]
            # phase_delta = (t0 - planet_t0) % moon_period * 2 * np.pi
            # alpha = (moon_initial_alpha + phase_delta) % (2 * np.pi)
            # time_alpha = np.cos(alpha) * moon_orbit_transit_length / 2
            # moon_t1 = t1 + time_alpha
            if time_sorted:
                first = np.searchsorted(time, t0 - half_moon_max_orbit_transit_duration, side='left')
                last = np.searchsorted(time, t0 + half_moon_max_orbit_transit_duration, side='right')
                time_orbit_range_args = np.arange(first, last)
                inside_window = np.abs(time[time_orbit_range_args] - t0) <= half_moon_max_orbit_transit_duration
                time_orbit_range_args = time_orbit_range_args[inside_window]
            else:
                time_orbit_range_args = np.argwhere(np.abs(time - t0) <= half_moon_max_orbit_transit_duration).flatten()
            time_moon_transit = time[time_orbit_range_args]
            flux_moon_transit = flux[time_orbit_range_args]
            flux_err_moon_transit = flux_err[time_orbit_range_args]
            time_moon_transit = time_moon_transit - (moon_t1 + half_moon_transit_duration)
            # fig_transit, axs = plt.subplots(1, 1, figsize=(8, 8))
            # axs.scatter(time_moon_transit, flux_moon_transit, color='gray', alpha=1, rasterized=True,
            #             label="Flux Transit ")
            # axs.set_title("Residuals")
            # axs.set_xlabel('Time')
            # axs.set_ylabel('Flux')
            # fig_transit.show()
            if len(time_moon_transit) > 0:
                scenario = [moon_alpha, time_moon_transit, flux_moon_transit, flux_err_moon_transit]
                if extra is not None:
                    scenario.append(extra[time_orbit_range_args])
                orbit_scenarios.append(scenario)
        return orbit_scenarios

    def split_scenarios(self, search_input, time, flux, flux_err, bary_t0s, extra=None):
        """Stack the epochs of one hypothesis and split them into the moon window and its surroundings.

        Parameters
        ----------
        search_input : SearchInput
            Moon hypothesis.
        time, flux, flux_err : numpy.ndarray
            Planet-subtracted light curve.
        bary_t0s : list of float
            Planet epochs to stack.
        extra : numpy.ndarray, optional
            The planet columns of `search_planet_depth_column`. They are stacked with the flux and
            fitted with the moon window (see `planet_fit`), and the returned fluxes carry the
            planet correction.

        Returns
        -------
        tuple or None
            (scenario_time, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err), with
            the times relative to the moon transit, or None when no epoch has data. With `extra`,
            the planet depth correction, the uncertainty inflation of the moon depth and the whole
            planet fit (see `planet_fit`) follow.
        """
        if extra is not None and time is self.time and flux is self.flux and flux_err is self.flux_err \
                and bary_t0s is self.bary_t0s:
            # The grid search: every hypothesis stacks the same points (see `search_stack`).
            stack = self.search_stack(search_input, extra)
            if stack is not None:
                return self.split_stacked(stack, search_input)
        # Alpha is the moon phase at self.planet_t0, the same reference inject_moon uses, so it
        # does not move when the first epoch is skipped or rejected.
        transit_scenarios = self.compute_moon_transit_scenarios(time, flux, flux_err,
                                                                               self.planet_t0, search_input.moon_alpha,
                                                                search_input.moon_period, bary_t0s,
                                                                               search_input.half_moon_orbit_transit_duration,
                                                                search_input.half_moon_max_orbit_transit_duration,
                                                                               search_input.half_moon_transit_duration,
                                                                               extra=extra,
                                                                phase_times=self.moon_phase_times(bary_t0s))
        if len(transit_scenarios) == 0:
            return None
        scenario_extra = []
        scenario_time = np.concatenate([scenario[1].ravel() for scenario in transit_scenarios]).astype(float)
        scenario_flux = np.concatenate([scenario[2].ravel() for scenario in transit_scenarios]).astype(float)
        scenario_flux_err = np.concatenate([scenario[3].ravel() for scenario in transit_scenarios]).astype(float)
        if extra is not None:
            scenario_extra = [scenario[4] for scenario in transit_scenarios]
        sorted_time_args = np.argsort(scenario_time)
        scenario_time = scenario_time[sorted_time_args]
        scenario_flux = scenario_flux[sorted_time_args]
        scenario_flux_err = scenario_flux_err[sorted_time_args]
        half_window = search_input.half_moon_transit_duration
        planet_fit = None
        if extra is not None:
            # The planet correction comes before the clipping, so a planet error is not clipped as
            # upper outliers; it is fitted again on the points the clipping keeps.
            scenario_extra = np.vstack(scenario_extra)[sorted_time_args]
            raw_flux = scenario_flux
            planet_fit = self.planet_fit(raw_flux, np.abs(scenario_time) <= half_window,
                                                        scenario_extra, epoch_depth=self.fit_planet_epoch_depth,
                                                        flux_err=scenario_flux_err)
            scenario_flux = raw_flux - planet_fit["model"]
        # Clipped in units of each point's error, so points of a noisier cadence are not clipped
        # more often than the others; with equal errors it is the same clipping as on the flux.
        weights = self.point_weights(scenario_flux_err, len(scenario_flux))
        outliers_args = self.remove_outliers((scenario_flux - 1) * np.sqrt(weights),
                                                            sigma_lower=float('inf'), sigma_upper=3)
        scenario_time = scenario_time[~outliers_args].flatten()
        scenario_flux = scenario_flux[~outliers_args].flatten()
        scenario_flux_err = scenario_flux_err[~outliers_args].flatten()
        if extra is not None:
            scenario_extra = scenario_extra[~outliers_args]
            raw_flux = raw_flux[~outliers_args].flatten()
            planet_fit = self.planet_fit(raw_flux, np.abs(scenario_time) <= half_window,
                                                        scenario_extra, epoch_depth=self.fit_planet_epoch_depth,
                                                        flux_err=scenario_flux_err)
            scenario_flux = raw_flux - planet_fit["model"]
        #TODO check whether to use this mean or the real scenario_flux
        #scenario_flux = self.running_mean_equal_length(scenario_flux, len(scenario_flux) // 100)
        it_args = np.argwhere(np.abs(scenario_time) <= half_window).flatten()
        oot_args = np.argwhere(np.abs(scenario_time) > half_window).flatten()
        it_time = scenario_time[it_args]
        oot_time = scenario_time[oot_args]
        it_flux = scenario_flux[it_args]
        oot_flux = scenario_flux[oot_args]
        it_flux_err = scenario_flux_err[it_args]
        oot_flux_err = scenario_flux_err[oot_args]
        if extra is not None:
            return (scenario_time, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                    planet_fit["depth"], planet_fit["inflation"], planet_fit)
        return scenario_time, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err

    @staticmethod
    def upper_clip(values, sigma_upper=3.0, max_iterations=5):
        """Return the mask of `remove_outliers` with sigma_lower=inf: astropy's sigma_clip, without its overhead.

        Median and standard deviation of the kept values, points above median + sigma_upper std
        removed, repeated until none is removed or max_iterations; the mask uses the last bounds.
        """
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        kept = values[finite]
        upper = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lower = -np.inf
            for _ in range(max_iterations):
                size = kept.size
                # The kept values are finite, so the plain median and std equal astropy's nan-aware ones.
                centre, spread = np.median(kept), np.std(kept)
                lower, upper = centre - spread * np.inf, centre + spread * sigma_upper
                kept = kept[(kept >= lower) & (kept <= upper)]
                if size == kept.size:
                    break
            return ~finite | (values < lower) | (values > upper)

    def moon_phase_times(self, epochs):
        """Return the time that sets the moon phase at each epoch, or None when every epoch uses its own.

        moon_phase_map, set only by the scrambled nulls (see `run_scrambled_nulls`), maps each epoch
        to the epoch whose phase it takes: the stacked data stay the same, and only the moon
        positions move, so the coherence of the moon orbit between epochs is lost.
        """
        phase_map = getattr(self, "moon_phase_map", None)
        if not phase_map:
            return None
        return [float(phase_map.get(float(t0), t0)) for t0 in epochs]

    def search_stack(self, search_input, extra):
        """Return the points every grid hypothesis stacks, with their fixed planet normal equations, or None.

        The data window of each epoch is the widest moon orbit searched, the same for every
        hypothesis, so the stacked points, their weights and the planet columns are fixed: only
        their times relative to the moon and the moon window move. The normal equations of the
        planet terms without the window are summed once here, and `split_stacked` adds the
        window and removes the clipped points per hypothesis. None when the errors are not usable.
        """
        half_window = search_input.half_moon_max_orbit_transit_duration
        key = (id(self.time), len(self.time), id(self.flux), id(self.flux_err), id(extra), tuple(self.bary_t0s),
               float(half_window), bool(self.fit_planet_epoch_depth),
               tuple(self.moon_phase_times(self.bary_t0s) or ()))
        cached = getattr(self, "_search_stack_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        time = np.asarray(self.time, dtype=float)
        time_sorted = len(time) < 2 or bool(np.all(time[1:] >= time[:-1]))
        indices, slots, epochs = [], [], []
        for t0 in self.bary_t0s:
            if time_sorted:
                window = np.arange(np.searchsorted(time, t0 - half_window, side='left'),
                                   np.searchsorted(time, t0 + half_window, side='right'))
                window = window[np.abs(time[window] - t0) <= half_window]
            else:
                window = np.argwhere(np.abs(time - t0) <= half_window).flatten()
            if len(window) > 0:
                indices.append(window)
                slots.append(np.full(len(window), len(epochs), dtype=np.int64))
                epochs.append(t0)
        stack = None
        if indices:
            index = np.concatenate(indices)
            flux_err = np.asarray(self.flux_err, dtype=float)[index]
            if self.usable_errors(flux_err, len(index)):
                columns = np.asarray(extra)[index]
                names, n_columns, terms = self.planet_design(columns, self.fit_planet_epoch_depth)
                weights = 1 / flux_err ** 2
                flux = np.ascontiguousarray(np.asarray(self.flux, dtype=float)[index])
                outside = np.zeros(len(index), dtype=np.bool_)
                normal, right = _planet_normal_equations(terms[0], outside, *terms[1:], weights, flux, n_columns) \
                    if n_columns > 2 else (None, None)
                stack = {"time": time[index], "flux": flux, "flux_err": flux_err, "weights": weights,
                         "root_weights": np.sqrt(weights), "slot": np.concatenate(slots), "epochs": np.asarray(epochs),
                         "phase_epochs": np.asarray(self.moon_phase_times(epochs) or epochs, dtype=float),
                         "names": names, "n_columns": n_columns, "terms": terms, "normal": normal, "right": right,
                         "total_weight": float(np.sum(weights))}
        self._search_stack_cache = (key, stack)
        return stack

    def stacked_planet_fit(self, stack, inside, normal, right, selection=None):
        """Solve the planet fit of the stacked points (all, or the selected ones) given their normal equations."""
        weights = stack["weights"] if selection is None else stack["weights"][selection]
        inside_points = inside if selection is None else inside[selection]
        n_points, n_inside = len(weights), int(np.count_nonzero(inside_points))
        unchanged = {"model": np.zeros(n_points), "depth": 0.0, "t0_shifts": {}, "epoch_depths": {}, "shape": {},
                     "inflation": 1.0}
        if n_inside == 0 or n_points - n_inside < 2 or stack["n_columns"] == 2:
            return unchanged
        inside_weight = float(np.sum(weights[inside_points]))
        n_columns = stack["n_columns"]
        solved = self.solve_planet_normal(normal, right, list(stack["names"]), n_columns,
                                                         inside_weight, float(np.sum(weights)) - inside_weight)
        if solved is None:
            return unchanged
        fit, coefficients = solved
        model = _planet_model(stack["terms"][0], inside, *stack["terms"][1:], coefficients)
        fit["model"] = model if selection is None else model[selection]
        return fit

    def split_stacked(self, stack, search_input):
        """`split_scenarios` of one grid hypothesis on the fixed stack of `search_stack`.

        The same steps on the same points: moon-relative times, planet fit with the moon window,
        upper clipping in units of each point's error, planet fit again on the kept points,
        sorted by time; the normal equations are updated (window added, clipped points removed)
        instead of summed again.
        """
        tau = self.compute_moon_ts(0.0, search_input.half_moon_orbit_transit_duration,
                                                  search_input.half_moon_transit_duration)
        epochs = stack["epochs"]
        period = search_input.moon_period
        phase = search_input.moon_alpha + (((stack["phase_epochs"] - self.planet_t0) % period) / period) * 2 * np.pi
        moon_t0 = epochs + np.cos(phase) * tau
        half = search_input.half_moon_transit_duration
        centre = (moon_t0 - half) + half
        scenario_time = stack["time"] - centre[stack["slot"]]
        inside = np.abs(scenario_time) <= half
        terms, n_columns = stack["terms"], stack["n_columns"]
        flux, weights = stack["flux"], stack["weights"]
        if n_columns > 2:
            cross_normal, cross_right = _window_cross_terms(np.flatnonzero(inside), terms[0], inside, *terms[1:],
                                                            weights, flux, n_columns)
            normal, right = stack["normal"] + cross_normal, stack["right"] + cross_right
        else:
            normal = right = None
        first = self.stacked_planet_fit(stack, inside, normal, right)
        corrected = flux - first["model"]
        clipped = self.upper_clip((corrected - 1) * stack["root_weights"])
        if np.any(clipped):
            removed = np.flatnonzero(clipped)
            kept = np.flatnonzero(~clipped)
            if n_columns > 2:
                removed_normal, removed_right = _planet_normal_equations(
                    terms[0][removed], inside[removed], terms[1][removed], terms[2][removed], terms[3][removed],
                    *terms[4:], weights[removed], flux[removed], n_columns)
                normal, right = normal - removed_normal, right - removed_right
            fit = self.stacked_planet_fit(stack, inside, normal, right, selection=kept)
        else:
            kept = np.arange(len(flux))
            fit = first
        order = np.argsort(scenario_time[kept])
        kept = kept[order]
        fit["model"] = fit["model"][order]
        scenario_time = scenario_time[kept]
        scenario_flux = flux[kept] - fit["model"]
        scenario_flux_err = stack["flux_err"][kept]
        it_args = np.abs(scenario_time) <= half
        oot_args = ~it_args
        return (scenario_time, scenario_time[it_args], scenario_flux[it_args], scenario_flux_err[it_args],
                scenario_time[oot_args], scenario_flux[oot_args], scenario_flux_err[oot_args],
                fit["depth"], fit["inflation"], fit)

    def window_depth(self, search_input, time, flux, flux_err, bary_t0s):
        """Return the mean depth of the moon window against its surroundings, as `search` measures it."""
        split = self.split_scenarios(search_input, time, flux, flux_err, bary_t0s)
        if split is None or len(split[2]) == 0 or len(split[5]) < 2:
            return np.nan
        return self.weighted_mean(split[5], split[6]) - self.weighted_mean(split[2], split[3])

    @staticmethod
    def running_mean_equal_length(data, width_signal):
        """Returns the running mean in a given window"""
        cumsum = np.cumsum(np.insert(data, 0, 0))
        med = (cumsum[width_signal:] - cumsum[:-width_signal]) / float(width_signal)
        # Append the first/last value at the beginning/end to match the length of
        # data and returned median
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values))
        return med

    @staticmethod
    def running_median(data, kernel):
        """Compute sliding median of width kernel and same length as data.

        Parameters
        ----------
        data : array_like
            Input data array.
        kernel : int
            Window width.

        Returns
        -------
        numpy.ndarray
            Running median array with same length as input.
        """
        idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None]
        med = np.median(data[idx], axis=1)

        # Append the first/last value at the beginning/end to match the length of
        # data and returned median
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values))
        return med

    @staticmethod
    def remove_outliers(
        flux, sigma=5.0, sigma_lower=None, sigma_upper=None, **kwargs
    ):
        """Remove outliers from a light curve using sigma clipping.

        Parameters
        ----------
        flux : array_like
            Flux array to clip.
        sigma : float, optional
            Sigma threshold for clipping, default 5.0.
        sigma_lower : float or None, optional
            Lower sigma threshold, default None.
        sigma_upper : float or None, optional
            Upper sigma threshold, default None.
        \*\*kwargs
            Additional arguments passed to astropy.stats.sigma_clip.

        Returns
        -------
        numpy.ndarray
            Boolean outlier mask (True where outliers are found).
        """
        # The import time for `sigma_clip` is somehow very slow, so we use
        # a local import here.
        from astropy.stats.sigma_clipping import sigma_clip

        # First, we create the outlier mask using AstroPy's sigma_clip function
        with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
            warnings.simplefilter("ignore")
            outlier_mask = sigma_clip(
                data=flux,
                sigma=sigma,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                **kwargs,
            ).mask
        # Second, we return the masked light curve and optionally the mask itself
        return outlier_mask
