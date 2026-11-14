"""The null distribution from pseudo-epoch searches and the false-alarm probability.
"""
import copy
import logging
import tempfile
import batman
import numpy as np
from scipy import stats
from sherlockpipe.exomoon.constants import AU_TO_RSUN, R_earth_to_R_sun


MIN_NULL_RUNS = 20


FAP_BOOTSTRAP_SAMPLES = 1000


FAP_UPPER_PERCENTILE = 95


MIN_INDEPENDENT_OFFSETS = 3


class NullDistribution:
    """The null distribution from pseudo-epoch searches and the false-alarm probability.
    """

    def null_offsets(self, trim_half_window, exclusion):
        """Offsets from each planet epoch, in days, at which a pseudo-epoch holds neither planet nor moon."""
        step = trim_half_window / 4
        return np.arange(exclusion, self.planet_period - exclusion + step / 2, step)

    def null_valid_offsets(self, before_trim, trim_half_window, offsets):
        """Return, per epoch, the offsets whose window holds at least half the points of the real epoch.

        Returns
        -------
        list of numpy.ndarray
            One array of usable offsets per epoch of bary_t0s.
        """
        time = np.sort(before_trim.time)

        def counts(centres):
            centres = np.asarray(centres, dtype=float)
            return np.searchsorted(time, centres + trim_half_window) - np.searchsorted(time, centres - trim_half_window)

        valid = []
        for t0 in self.bary_t0s:
            needed = max(self.min_cadences_per_t0, 0.5 * counts([t0])[0])
            valid.append(offsets[counts(t0 + offsets) >= needed])
        return valid

    @staticmethod
    def independent_offsets(valid, trim_half_window):
        """Return how many of the offsets are at least one window (2 trim_half_window) apart, so share no data."""
        count, last = 0, -np.inf
        for offset in np.sort(valid):
            if offset - last >= 2 * trim_half_window:
                count, last = count + 1, offset
        return count

    def null_curve(self, before_trim, rng, trim_half_window, valid_offsets):
        """Build one pseudo-epoch light curve for the null distribution (audit item H10).

        For each epoch, the data around a random offset from it, where neither the planet nor any
        searched moon can transit, are moved onto the epoch, and the fitted planet is injected
        there. The search therefore sees the same epochs, geometry and noise, with no moon.
        valid_offsets holds the usable offsets of each epoch (see `null_valid_offsets`).

        Returns
        -------
        tuple or None
            (time, flux, flux_err, epochs, exposures), or None when fewer than half the epochs have
            a usable offset. exposures holds each moved point's own exposure when the light curve
            has exposures per point, else None.
        """
        order = np.argsort(before_trim.time)
        time, flux, flux_err = before_trim.time[order], before_trim.flux[order], before_trim.flux_err[order]
        params = self.null_planet_params()
        pieces, epochs = [], []
        for t0, valid in zip(self.bary_t0s, valid_offsets):
            if len(valid) == 0:
                continue
            offset = rng.choice(valid)
            window = np.abs(time - (t0 + offset)) < trim_half_window
            piece_time = time[window] - offset
            piece_exposure = self.exposure_at(time[window])
            params.t0 = t0
            planet = self.transit_model(params, piece_time, piece_exposure).light_curve(params)
            pieces.append((piece_time, flux[window] * planet, flux_err[window],
                           np.broadcast_to(np.asarray(piece_exposure if piece_exposure is not None else 0.0,
                                                      dtype=float), piece_time.shape)))
            epochs.append(t0)
        if len(epochs) == 0 or len(epochs) < len(self.bary_t0s) / 2:
            return None
        null_time = np.concatenate([piece[0] for piece in pieces])
        order = np.argsort(null_time)
        exposures = np.concatenate([piece[3] for piece in pieces])[order] if self.exposure_lookup is not None \
            else None
        return (null_time[order], np.concatenate([piece[1] for piece in pieces])[order],
                np.concatenate([piece[2] for piece in pieces])[order], epochs, exposures)

    def null_planet_params(self):
        """Return the fitted planet model that the null searches inject and subtract."""
        params = batman.TransitParams()
        params.t0 = 0.0
        params.per = self.planet_period
        params.rp = self.planet_rp_over_rstar if self.planet_rp_over_rstar is not None \
            else self.planet_radius * R_earth_to_R_sun / self.star_radius
        params.a = self.planet_a_over_rstar if self.planet_a_over_rstar is not None \
            else self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius
        params.inc = self.planet_inc
        params.ecc = self.planet_ecc
        params.w = self.planet_arg_periastron
        params.u = self.ab
        params.limb_dark = "quadratic"
        return params

    def null_statistic(self, before_trim, time, flux, flux_err, epochs, exposures=None):
        """Return the detection SNR of the full search on one pseudo-epoch light curve."""
        null = copy.copy(before_trim)
        null.time, null.flux, null.flux_err = time, flux, flux_err
        if exposures is not None:
            # The moved points keep their own exposures, looked up now at their new times.
            order = np.argsort(time, kind="stable")
            null.exposure_lookup = (np.asarray(time, dtype=float)[order], np.asarray(exposures, dtype=float)[order])
        null.bary_t0s = list(epochs)
        null.null_runs = 0
        null.skip_fold_fit = True
        null.red_noise_factor = self.red_noise_factor
        for name in ("planet_inc", "planet_impact_param", "planet_a_over_rstar", "planet_rp_over_rstar",
                     "planet_duration"):
            setattr(null, name, getattr(self, name))
        params = self.null_planet_params()
        null.planet_radius = params.rp * self.star_radius / R_earth_to_R_sun
        with tempfile.TemporaryDirectory() as plots_dir:
            null.object_dir = plots_dir
            if null.detrend_window is not None:
                null.time, null.flux, null.flux_err = null.detrend(null.time, null.flux, null.flux_err)
            null.time, null.flux, null.flux_err = null.remove_non_transit_flux(
                null.time, null.flux, null.flux_err, null.bary_t0s, np.max(null.planet_mass_grid))
            null.time, null.flux, null.flux_err = null.subtract_planet_transit(null.time, null.flux, null.flux_err,
                                                                               mode=self.subtract_mode)
            null.build_search_templates()
            period_grid, alpha_grid, _, residuals = null.grid_search(progress=False)
            best = null.select_best(period_grid, alpha_grid,
                                    self.snr_matrix(residuals, len(alpha_grid), len(period_grid)))
        return np.nan if best is None else best[3]

    def run_nulls(self, before_trim, period_grid):
        """Return the null distribution of the detection SNR from null_runs pseudo-epoch searches (audit item H10).

        Each null search uses windows of this light curve placed away from the planet transits
        (see `null_curve`), so it keeps the target's own noise, gaps and correlated noise (audit
        item H11), and goes through the same planet fit, grid and selection as the real search.
        The windows of different null runs overlap, so they are not independent; the number of
        independent offsets per epoch is recorded in null_independent_offsets.

        Returns
        -------
        list of float
            The SNR of each completed null search.
        """
        trim_half_window = self.trim_half_window(np.max(self.planet_mass_grid))
        search_half_window = self.search_input(period_grid[-1], 0.0, self.planet_mass, period_grid[-1],
                                               self.planet_impact_param).half_moon_max_orbit_transit_duration
        exclusion = trim_half_window + search_half_window
        offsets = self.null_offsets(trim_half_window, exclusion)
        valid_offsets = self.null_valid_offsets(before_trim, trim_half_window, offsets) if len(offsets) else []
        # Independent positions where the data exist, the median over the epochs.
        self.null_independent_offsets = int(np.median([self.independent_offsets(valid, trim_half_window)
                                                       for valid in valid_offsets])) if valid_offsets else 0
        if len(offsets) == 0:
            logging.warning("No null runs: the data kept around each transit (+-%.2f d) plus the widest moon window "
                            "(+-%.2f d) leave no room within the %.2f d period for windows without the planet or a "
                            "moon, so no false-alarm probability is given", trim_half_window, search_half_window,
                            self.planet_period)
            return []
        logging.info("Null distribution: %d searches on pseudo-epochs offset %.2f-%.2f d from each transit "
                     "(%d independent offsets with data per epoch)", self.null_runs, offsets[0], offsets[-1],
                     self.null_independent_offsets)
        if self.null_independent_offsets < MIN_INDEPENDENT_OFFSETS:
            logging.warning("No null runs: only %d independent pseudo-epoch offsets with data per epoch (at least %d "
                            "needed), so the null searches would reuse the same data and no false-alarm "
                            "probability is given", self.null_independent_offsets, MIN_INDEPENDENT_OFFSETS)
            return []
        rng = np.random.default_rng(self.null_seed)
        snrs, failed = [], 0
        previous_disable = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            for _ in range(self.null_runs):
                curve = self.null_curve(before_trim, rng, trim_half_window, valid_offsets)
                if curve is None:
                    failed += 1
                    continue
                try:
                    snr = self.null_statistic(before_trim, *curve)
                except Exception:
                    snr = np.nan
                if np.isfinite(snr):
                    snrs.append(float(snr))
                else:
                    failed += 1
        finally:
            logging.disable(previous_disable)
        if failed:
            logging.warning("%d/%d null runs failed: too few pseudo-epochs with data, or the search gave no SNR",
                            failed, self.null_runs)
        if snrs:
            logging.info("Null SNR over %d runs: median %.2f, 95th percentile %.2f, largest %.2f", len(snrs),
                         np.median(snrs), np.percentile(snrs, 95), np.max(snrs))
        return snrs

    def false_alarm(self, snr):
        """Return (Gumbel, empirical) false-alarm probabilities of an SNR against the null runs.

        The Gumbel distribution, fitted to the null maxima, gives the probability; the empirical
        value (k + 1) / (N + 1) is a check that cannot go below 1 / (N + 1). Both are None with
        fewer than MIN_NULL_RUNS nulls.
        """
        nulls = np.array([value for value in (self.null_snrs or []) if np.isfinite(value)])
        if len(nulls) < MIN_NULL_RUNS or not np.isfinite(snr):
            if self.null_snrs:
                logging.warning("Only %d null runs completed, fewer than %d: no false-alarm probability",
                                len(nulls), MIN_NULL_RUNS)
            return None, None
        empirical = float((np.sum(nulls >= snr) + 1) / (len(nulls) + 1))
        try:
            loc, scale = stats.gumbel_r.fit(nulls)
            fap = float(stats.gumbel_r.sf(snr, loc, scale)) if scale > 0 else empirical
        except Exception:
            fap = empirical
        return fap, empirical

    def false_alarm_upper(self, snr):
        """Return the FAP_UPPER_PERCENTILE percentile of the Gumbel false-alarm probability over bootstrap
        resamples of the null runs, and whether an Anderson-Darling test rejects the Gumbel fit at 5 %.

        With N nulls the measured probability cannot go below 1 / (N + 1), so a detection at a 0.01 threshold
        with 100 nulls is always the Gumbel tail extrapolated past the largest null. The bootstrap measures how
        much that extrapolation depends on which nulls were drawn, and the detection asks the upper value to be
        below the threshold. The Gumbel (GEV shape 0) is kept rather than a free-shape GEV: the free shape
        fitted to 100 maxima is poorly constrained, and on real nulls it comes out negative (a shorter tail),
        so the Gumbel is the conservative tail. Returns (None, None) without enough nulls.
        """
        nulls = np.array([value for value in (self.null_snrs or []) if np.isfinite(value)])
        if len(nulls) < MIN_NULL_RUNS or not np.isfinite(snr):
            return None, None
        rng = np.random.default_rng(self.null_seed)
        faps = []
        for _ in range(FAP_BOOTSTRAP_SAMPLES):
            sample = rng.choice(nulls, len(nulls))
            try:
                loc, scale = stats.gumbel_r.fit(sample)
                faps.append(float(stats.gumbel_r.sf(snr, loc, scale)) if scale > 0 else np.nan)
            except Exception:
                faps.append(np.nan)
        faps = np.array(faps)
        upper = float(np.percentile(faps[np.isfinite(faps)], FAP_UPPER_PERCENTILE)) if np.any(np.isfinite(faps)) \
            else None
        try:
            test = stats.anderson(nulls, dist='gumbel_r')
            rejected = bool(test.statistic > test.critical_values[list(test.significance_level).index(5.0)])
        except Exception:
            rejected = None
        return upper, rejected

    def assess_false_alarm(self, snr):
        """Set fap, fap_empirical, fap_upper, fap_extrapolated, null_fit_rejected and detected for an SNR.

        detected is fap_upper < fap_threshold, so it holds whichever nulls the bootstrap draws; all are
        None without enough null runs.
        """
        self.fap, self.fap_empirical = self.false_alarm(snr)
        if self.fap is None:
            self.fap_upper = self.fap_extrapolated = self.null_fit_rejected = self.detected = None
            return
        self.fap_upper, self.null_fit_rejected = self.false_alarm_upper(snr)
        if self.fap_upper is None:
            self.fap_upper = self.fap
        completed = int(np.sum(np.isfinite(np.asarray(self.null_snrs, dtype=float))))
        self.fap_extrapolated = bool(self.fap < 1 / (completed + 1))
        self.detected = bool(self.fap_upper < self.fap_threshold)
