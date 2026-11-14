"""Derivatives of the subtracted planet model with respect to its depth, mid-time and shape, which the
joint fit uses as extra columns to correct the planet model under each moon hypothesis.
"""
import copy
import numpy as np


class PlanetDerivatives:
    """Derivatives of the planet model that the joint fit corrects."""

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
