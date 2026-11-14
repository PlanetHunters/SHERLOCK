"""Detection statistics: the weighted window SNR, the template residuals and radius, the
red-noise factor, the SDE spectrum, the phase refinement and the single-event check.
"""
import logging
import sys
import numpy as np
from sherlockpipe.exomoon.constants import R_earth_to_R_sun


class SearchStatistics:
    """Detection statistics: the weighted window SNR, the template residuals and radius, the
    red-noise factor, the SDE spectrum, the phase refinement and the single-event check.
    """

    MAX_RESIDUAL = sys.maxsize
    SPECTRUM_KERNEL = 31
    SINGLE_EVENT_SNR_FRACTION = 0.5

    @classmethod
    def weighted_mean(cls, flux, flux_err=None):
        """Return the inverse-variance weighted mean (the plain mean without usable errors)."""
        weights = cls.point_weights(flux_err, len(flux))
        return float(np.sum(weights * flux) / np.sum(weights))

    @classmethod
    def box_snr(cls, it_flux, oot_flux, it_err=None, oot_err=None):
        """Return the SNR of the moon-window depth against its surroundings, as `search` scores it.

        The depth is a difference of two inverse-variance weighted means, so both contribute noise.
        The noise scale is measured from the scatter outside the window, sqrt(mean(w (f - mean)^2)),
        so errors that are all off by one factor do not change the SNR; with equal errors this is
        the plain difference of means over std(oot) sqrt(1 / n_in + 1 / n_out). A light curve that
        mixes cadences needs the weights: a 1-minute point carries 1/30 of the information of a
        30-minute point. Without errors the weights are equal. NaN without in-window data or with
        fewer than two points outside it.
        """
        if len(it_flux) == 0 or len(oot_flux) < 2:
            return np.nan
        if cls.usable_errors(it_err, len(it_flux)) and \
                cls.usable_errors(oot_err, len(oot_flux)):
            it_weights, oot_weights = 1 / np.asarray(it_err) ** 2, 1 / np.asarray(oot_err) ** 2
        else:
            it_weights, oot_weights = np.ones(len(it_flux)), np.ones(len(oot_flux))
        oot_mean = np.sum(oot_weights * oot_flux) / np.sum(oot_weights)
        depth = oot_mean - np.sum(it_weights * it_flux) / np.sum(it_weights)
        scale = np.sqrt(np.mean(oot_weights * (oot_flux - oot_mean) ** 2))
        return float(depth / (scale * np.sqrt(1 / np.sum(it_weights) + 1 / np.sum(oot_weights))))

    @classmethod
    def spectra(cls, chi2, oversampling_factor=1, kernel_size=30):
        """Compute SDE from chi-squared residuals (standard TLS formula).

        SDE = max(SR_detrended) / std(SR_detrended), where
        SR = min(chi²) / chi² and detrending is via running-median.

        No global normalisation; avoids instability from flat baselines.
        """
        SR = np.nan_to_num(np.min(chi2) / chi2, nan=1.0)
        kernel = oversampling_factor * kernel_size
        if kernel % 2 == 0:
            kernel += 1
        if len(SR) > 2 * kernel:
            baseline = cls.running_median(SR, kernel)
            power = SR - baseline
        else:
            power = SR - np.mean(SR)
        power = power - np.mean(power)
        SDE_raw = np.max(power / np.std(power)) if np.std(power) > 0 else 0
        power = power * SDE_raw / np.max(power) if np.max(power) > 0 else power
        SDE = float(np.max(power))
        return SR, np.zeros_like(SR), power, SDE_raw, SDE

    @staticmethod
    def refine_alpha(alpha_grid, residuals_matrix):
        """Refine the best alpha per period via parabolic interpolation.

        For each period (column of residuals_matrix), fits a quadratic
        through the three alpha values bracketing the grid minimum, then
        returns the alpha at the parabola's vertex. This gives sub-grid
        alpha resolution at zero extra search cost.

        Parameters
        ----------
        alpha_grid : array_like, shape (N_alpha,)
            Grid of alpha values in radians.
        residuals_matrix : array_like, shape (N_alpha, N_periods)
            Residuals matrix where rows index alpha and columns index period.

        Returns
        -------
        numpy.ndarray, shape (N_periods, 2)
            Each row is [refined_alpha, min_residual] for that period.
        """
        n_alpha = len(alpha_grid)
        refined = np.empty((residuals_matrix.shape[1], 2))
        for col in range(residuals_matrix.shape[1]):
            r_col = residuals_matrix[:, col]
            if np.all(np.isnan(r_col)):
                refined[col] = [np.nan, np.nan]
                continue
            best_j = np.nanargmin(r_col)
            j_left = (best_j - 1) % n_alpha
            j_right = (best_j + 1) % n_alpha
            a_center = alpha_grid[best_j]
            step_left = (alpha_grid[j_left] - a_center + np.pi) % (2.0 * np.pi) - np.pi
            step_right = (alpha_grid[j_right] - a_center + np.pi) % (2.0 * np.pi) - np.pi
            a_local = np.array([step_left, 0.0, step_right])
            r_local = np.array([r_col[j_left], r_col[best_j], r_col[j_right]])
            # np.polyfit returns the highest power first: r = c[0]*a^2 + c[1]*a + c[2]
            delta = 0.0
            coeffs = np.polyfit(a_local, r_local, 2) if np.all(np.isfinite(r_local)) else [0.0]
            if coeffs[0] > 0:
                delta = np.clip(-coeffs[1] / (2.0 * coeffs[0]), step_left, step_right)
            refined[col, 0] = (a_center + delta) % (2.0 * np.pi)
            refined[col, 1] = r_col[best_j]
        return refined

    @classmethod
    #@njit(fastmath=True, parallel=False)
    def calculate_residuals(cls, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                            t0_flux_mean_value, datapoints, models, min_radius, max_radius,
                            star_radius, radius_grid_size):
        """Calculate residuals between model and data for a transit scenario.

        Parameters
        ----------
        it_time : array_like
            In-transit time array in days.
        it_flux : array_like
            In-transit normalized flux array.
        it_flux_err : array_like
            In-transit flux uncertainty array.
        oot_time : array_like
            Out-of-transit time array in days.
        oot_flux : array_like
            Out-of-transit normalized flux array.
        oot_flux_err : array_like
            Out-of-transit flux uncertainty array.
        t0_flux_mean_value : float
            Mean flux at the transit center.
        datapoints : int
            Number of data points.
        models : list of (float, array_like, array_like)
            Pre-computed model transits as
            [(radius_Rearth, time_days_from_transit_center, flux), ...].
        min_radius : float
            Minimum moon radius in Earth radii. Unused since every positive dip is fitted
            (audit item H3); kept so positional callers do not break.
        max_radius : float
            Maximum moon radius in Earth radii.
        star_radius : float
            Stellar radius in solar radii.
        radius_grid_size : int
            Number of grid points for radius search.

        Returns
        -------
        tuple
            (best_residual, residuals_baseline, best_radius, best_model,
            template_unusable). template_unusable is True when the scenario
            has a dip but the template has no depth at the observed times, so
            it can be told apart from a scenario without a dip.

        Raises
        ------
        ValueError
            If the closest template's time and flux arrays are misaligned or
            have fewer than two samples.
        """
        oot_radius_residuals = np.sum(((oot_flux - 1) / oot_flux_err) ** 2)
        n_it = len(it_flux)
        model_baseline = np.ones(n_it)
        residuals_baseline = np.sum(((it_flux - model_baseline) / it_flux_err) ** 2) + oot_radius_residuals
        # H1: a hypothesis without a dip, or with a brightening, scores exactly as the null (a moon
        # only dims the star), so no placeholder value reaches the spectrum.
        # H3: every positive dip is fitted, including those shallower than min_radius; the caller
        # flags a below-range radius.
        best_radius = 0.0
        best_residual = residuals_baseline
        best_model = model_baseline
        if t0_flux_mean_value < 1:
            # H13: the template amplitude is the weighted least-squares value against the deficit
            # below the surrounding level, not the template peak forced to the mean depth.
            # H14: the radius follows from that amplitude through the template set (depth scales
            # with r^2 for a template of the same geometry and limb darkening), refined by moving
            # to the template closest to the estimate, so impact parameter and limb darkening are
            # inverted with the template, not assumed away by r = sqrt(mean depth).
            deficit = (1 - t0_flux_mean_value) + cls.weighted_mean(it_flux, it_flux_err) - it_flux
            weights = 1 / it_flux_err ** 2
            radius = np.sqrt(1 - t0_flux_mean_value) * star_radius / R_earth_to_R_sun
            amplitude, shape = None, None
            for _ in range(4):
                template_radius, template_time, closest_model = min(models, key=lambda x: abs(x[0] - radius))
                if len(closest_model) < 2 or len(template_time) != len(closest_model):
                    raise ValueError(f"Template time and flux must be aligned with at least two samples, got "
                                     f"{len(template_time)} and {len(closest_model)}")
                shape = 1.0 - np.interp(it_time, template_time, closest_model, left=1.0, right=1.0)
                norm = np.sum(weights * shape ** 2)
                if not norm > 0:
                    return residuals_baseline, residuals_baseline, 0.0, model_baseline, True
                amplitude = np.sum(weights * deficit * shape) / norm
                if not amplitude > 0:
                    return residuals_baseline, residuals_baseline, 0.0, model_baseline, False
                new_radius = template_radius * np.sqrt(amplitude)
                if abs(new_radius - radius) < 1e-3 * radius:
                    radius = new_radius
                    break
                radius = new_radius
            model_sample_scaled = 1 - amplitude * shape
            it_radius_residuals = np.sum(((it_flux - model_sample_scaled) / it_flux_err) ** 2)
            best_residual = it_radius_residuals + oot_radius_residuals
            best_radius = float(radius)
            best_model = model_sample_scaled
        return best_residual, residuals_baseline, best_radius, best_model, False

    @staticmethod
    def depth_to_radius(depth, star_radius):
        """Convert a transit depth to a body radius.

        Parameters
        ----------
        depth : float
            Transit depth (fractional flux decrease).
        star_radius : float
            Stellar radius in solar radii.

        Returns
        -------
        float
            Body radius in Earth radii.
        """
        return np.sqrt(depth * ((star_radius / R_earth_to_R_sun) ** 2))

    @staticmethod
    def snr_matrix(all_residuals, alpha_grid_size, period_count):
        """Arrange the grid SNRs as an (alpha, period) matrix."""
        matrix = np.full((alpha_grid_size, period_count), np.nan)
        for i, residual in enumerate(all_residuals[:alpha_grid_size * period_count]):
            matrix[i % alpha_grid_size][i // alpha_grid_size] = residual[3]
        return matrix

    def measure_red_noise(self, time, flux, exclusion, bin_width, flux_err=None):
        """Return the red-noise factor beta on the moon-window time scale (audit item H11).

        The search SNR divides the depth by the point scatter over sqrt(N), which assumes
        independent points. Following Pont, Zucker & Queloz (2006), the data away from every
        planet epoch (by more than exclusion, so neither the planet nor a searched moon is
        there) are binned at the moon transit duration, and beta is the scatter of the
        differences between adjacent bins, over sqrt(2), divided by the white-noise expectation
        sigma / sqrt(n). Adjacent differences ignore variability slower than two windows, which
        the comparison of the moon window with its surroundings also cancels.

        With flux errors the bins are inverse-variance weighted means, each difference is divided
        by its own expected noise sqrt(1 / W_k + 1 / W_k+1) for bin weights W, and the white-noise
        scale is sqrt(mean(w (f - mean)^2)), as in `box_snr`; a bin is full when it holds at least
        half the median bin weight. Bins of different cadences then compare on one scale. With
        equal errors and equal bin counts this is the unweighted beta.

        Parameters
        ----------
        time, flux : numpy.ndarray
            Light curve, normalised to one.
        exclusion : float
            Half-width in days excluded around each planet epoch.
        bin_width : float
            Bin width in days, the moon transit duration.

        flux_err : numpy.ndarray, optional
            Flux errors; the bins are then weighted (see above).

        Returns
        -------
        float or None
            beta, or None when fewer than 10 pairs of adjacent bins are available.
        """
        order = np.argsort(time)
        weighted = self.usable_errors(flux_err, len(time))
        time, flux = np.asarray(time, dtype=float)[order], np.asarray(flux, dtype=float)[order]
        weights = 1 / np.asarray(flux_err, dtype=float)[order] ** 2 if weighted else np.ones(len(time))
        epochs = self.planet_t0 + self.planet_period * np.arange(
            np.floor((time[0] - self.planet_t0) / self.planet_period) - 1,
            np.ceil((time[-1] - self.planet_t0) / self.planet_period) + 2)
        nearest = np.min(np.abs(time[:, None] - epochs[None, :]), axis=1) if len(time) else np.array([])
        keep = (nearest >= exclusion) & np.isfinite(flux)
        time, flux, weights = time[keep], flux[keep], weights[keep]
        if len(time) < 20 or not bin_width > 0:
            return None
        index = np.floor((time - time[0]) / bin_width).astype(int)
        bins, start, counts = np.unique(index, return_index=True, return_counts=True)
        if weighted:
            bin_weights = np.add.reduceat(weights, start)
            means = np.add.reduceat(weights * flux, start) / bin_weights
            full = (counts >= 2) & (bin_weights >= 0.5 * np.median(bin_weights))
            adjacent = full[:-1] & full[1:] & (np.diff(bins) == 1)
            if np.count_nonzero(adjacent) < 10:
                return None
            expected = np.sqrt(1 / bin_weights[:-1] + 1 / bin_weights[1:])
            mean = np.sum(weights * flux) / np.sum(weights)
            scale = np.sqrt(np.mean(weights * (flux - mean) ** 2))
            if not scale > 0:
                return None
            return float(np.std((np.diff(means) / expected)[adjacent]) / scale)
        sums = np.add.reduceat(flux, start)
        full = counts >= max(2, 0.5 * np.median(counts))
        means = sums / counts
        adjacent = full[:-1] & full[1:] & (np.diff(bins) == 1)
        if np.count_nonzero(adjacent) < 10:
            return None
        binned_scatter = np.std(np.diff(means)[adjacent]) / np.sqrt(2)
        sigma = np.std(flux)
        if not sigma > 0:
            return None
        white = sigma / np.sqrt(np.mean(counts[full]))
        return float(binned_scatter / white)

    def set_red_noise_factor(self, before_trim):
        """Measure beta (see `measure_red_noise`) on the light curve before trimming and set red_noise_factor.

        The data within the widest moon window of every epoch are excluded. beta below one is
        set to one, so the SNR is never raised. Without enough data the factor stays one.
        """
        max_period = self.max_moon_period(np.max(self.planet_mass_grid))
        window = self.search_input(max_period, 0.0, self.planet_mass, max_period, self.planet_impact_param)
        time, flux, flux_err = before_trim.time, before_trim.flux, before_trim.flux_err
        if self.detrend_window is not None:
            time, flux, flux_err = self.detrend(time, flux, flux_err)
        self.red_noise_measured = self.measure_red_noise(time, flux, window.half_moon_max_orbit_transit_duration,
                                                         window.moon_transit_duration, flux_err)
        if self.red_noise_measured is None:
            self.red_noise_factor = 1.0
            logging.warning("Red-noise factor not measured: too little data away from the transits; the SNR "
                            "assumes white noise")
        else:
            self.red_noise_factor = max(1.0, self.red_noise_measured)
            logging.info("Red-noise factor beta = %.2f on the %.2f h moon-window scale (measured %.2f); the SNR "
                         "is divided by it", self.red_noise_factor, window.moon_transit_duration * 24,
                         self.red_noise_measured)

    def assess_single_event(self, search_input):
        """Measure each epoch's depth and SNR and the leave-one-out SNR at the best hypothesis (audit item H12).

        Sets epoch_stats, loo_snr_min and single_event. single_event is True when one epoch
        carries the signal: with one epoch, or when leaving one epoch out keeps less than
        SINGLE_EVENT_SNR_FRACTION of the SNR. A signal spread over N epochs keeps about
        sqrt((N - 1) / N) of it. The flag does not change the detection.
        """
        column = self.search_planet_depth_column()

        def measure(epochs):
            # The same statistic as `search`, planet depth correction included.
            split = self.split_scenarios(search_input, self.time, self.flux, self.flux_err, epochs,
                                         extra=column) if epochs else None
            if split is None or len(split[2]) == 0 or len(split[5]) < 2:
                return None, None
            inflation = split[8] if column is not None else 1.0
            return (self.weighted_mean(split[5], split[6])
                    - self.weighted_mean(split[2], split[3]),
                    self.box_snr(split[2], split[5], split[3], split[6]) / inflation
                    / self.red_noise_factor)

        self.epoch_stats = []
        for t0 in self.bary_t0s:
            depth, snr = measure([t0])
            _, loo = measure([epoch for epoch in self.bary_t0s if epoch != t0])
            self.epoch_stats.append({"t0": float(t0), "depth": depth, "snr": snr, "loo_snr": loo})
        loos = [stat["loo_snr"] for stat in self.epoch_stats if stat["loo_snr"] is not None
                and np.isfinite(stat["loo_snr"])]
        self.loo_snr_min = float(min(loos)) if loos else None
        if not (np.isfinite(self.max_snr) and self.max_snr > 0):
            self.single_event = None
            return
        self.single_event = bool(len(self.bary_t0s) < 2 or self.loo_snr_min is None
                                 or self.loo_snr_min < self.SINGLE_EVENT_SNR_FRACTION * self.max_snr)
        if self.single_event:
            logging.warning("Single-event flag: leaving one transit out drops the SNR from %.2f to %s, so one epoch "
                            "carries the signal; see epoch_stats", self.max_snr,
                            "nothing" if self.loo_snr_min is None else f"{self.loo_snr_min:.2f}")
