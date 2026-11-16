"""ExoMoonLeastSquares, the EMLS exomoon search: setup, grid search over moon hypotheses, selection,
false-alarm assessment and figures, built from the stage mixins of this package.
"""
import copy
import logging
import os
import time
from multiprocessing import Pool
import batman
from tqdm import tqdm
import numpy as np
from scipy import stats
from sherlockpipe.exomoon import plots as exomoon_plots
from sherlockpipe.exomoon.constants import AU_TO_RSUN, M_earth_to_M_sun, R_earth_to_R_sun
from sherlockpipe.exomoon.detrending import Detrending
from sherlockpipe.exomoon.exposure import ExposureHandling
from sherlockpipe.exomoon.injection import MoonInjection
from sherlockpipe.exomoon.joint_fit import JointPlanetFit
from sherlockpipe.exomoon.nulls import FAP_UPPER_PERCENTILE, NullDistribution
from sherlockpipe.exomoon.orbits import MoonOrbits
from sherlockpipe.exomoon.planet import PlanetModel
from sherlockpipe.exomoon.stacking import MoonStacking
from sherlockpipe.exomoon.statistics import SearchStatistics
from sherlockpipe.exomoon.templates import MoonTemplates
from sherlockpipe.exomoon.tides import TIDAL_SURVIVAL_AGE_GYR, TIDAL_SURVIVAL_K2_Q_FACTOR, max_surviving_moon_radius


_WORKER_SEARCH = None


def _set_worker_search(emls):
    """Pool initializer: keep the searching EMLS object in the worker process."""
    global _WORKER_SEARCH
    _WORKER_SEARCH = emls


def _worker_search(search_input):
    """Score one hypothesis with the worker's EMLS object (see `ExoMoonLeastSquares.grid_search`)."""
    return _WORKER_SEARCH.search(search_input)


class ExoMoonLeastSquares(MoonOrbits, ExposureHandling, PlanetModel, JointPlanetFit, MoonStacking, SearchStatistics, MoonTemplates, Detrending, MoonInjection, NullDistribution):
    """Performs exomoon least-squares search and fitting."""

    # Hypothesis figures written: the selected one and the next SNR peaks at other periods.
    HYPOTHESIS_FIGURES = 3

    def __init__(self, object_dir, cpus, star_mass, star_radius, ab, planet_radius, planet_radius_err,
                 planet_mass, planet_period,
                 planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc,
                 planet_ecc_err, planet_arg_periastron, planet_impact_param, min_radius, max_radius, t0s, time, flux,
                 flux_err, max_moon_density, period_grid_size=10000, radius_grid_size=10,
                 min_cadences_per_t0: bool = 20, min_orbit_window: float = 0,
                  subtract_mode: str = "subtract", mask_duration_factor: float = None,
                  skip_fold_fit: bool = False, fit_rp_per_transit: bool = True,
                  min_transit_completeness: float = 0.5,
                  max_subtraction_rms_ratio: float = 3.0, exposure_time="auto",
                  ttv_prior_width="auto", planet_t0_err=None, planet_period_err=None, null_runs=0,
                  fap_threshold=0.01, null_seed=0, detrend_window=None, duration_from_fit=True,
                 detrend_mask='auto', fix_impact_param=False, retrograde=True, null_stop_fap=0.1,
                 scramble_runs=0, scramble_trigger_fap=0.1):
        """Initialize the exomoon least-squares search.

        Parameters
        ----------
        object_dir : str
            Directory for output files.
        cpus : int
            Number of CPU cores to use.
        star_mass : float
            Stellar mass in solar masses.
        star_radius : float
            Stellar radius in solar radii.
        ab : list
            Limb darkening coefficients [u1, u2].
        planet_radius : float
            Planet radius in Earth radii.
        planet_radius_err : float
            Planet radius uncertainty in Earth radii.
        planet_mass : float
            Planet mass in Earth masses.
        planet_period : float
            Planet orbital period in days.
        planet_t0 : float
            Planet transit epoch in BJD.
        planet_duration : float
            Planet transit duration in days.
        planet_semimajor_axis : float
            Planet semimajor axis in AU.
        planet_inc : float
            Planet orbital inclination in degrees.
        planet_inc_err : float
            Planet inclination uncertainty in degrees.
        planet_ecc : float
            Planet orbital eccentricity.
        planet_ecc_err : float
            Planet eccentricity uncertainty.
        planet_arg_periastron : float
            Planet argument of periastron in degrees.
        planet_impact_param : float
            Planet impact parameter.
        min_radius : float
            Minimum moon radius to search in Earth radii.
        max_radius : float
            Maximum moon radius to search in Earth radii.
        t0s : array_like
            Barycentric transit times.
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        max_moon_density : float
            Maximum moon density in g/cm^3.
        period_grid_size : int, optional
            Number of grid points for period search, default 10000, which captured at least 0.96
            of moons with tau >= T at their true period in the audit fixtures (audit item G19;
            2000 gave 0.90-0.93).
        radius_grid_size : int, optional
            Number of grid points for radius search, default 10.
        min_cadences_per_t0 : bool, optional
            Minimum cadences required per transit, default 20.
        min_orbit_window : float, optional
            Minimum moon orbit transit duration as multiple of planet
            transit duration. Periods with smaller orbit transit duration
            are excluded from the search grid. Default 0 (no constraint).
        subtract_mode : str, optional
            Planet subtraction mode: 'subtract' (remove model) or 'mask'
            (cut in-transit points). Default 'subtract'.
        mask_duration_factor : float, optional
            In mask mode, the mask half-width in units of T14 / 2. Default None: 1.1 T14 / 2
            plus one cadence (see `mask_half_width`).
        max_subtraction_rms_ratio : float, optional
            In-transit to out-of-transit residual RMS ratio above which an epoch is
            flagged in epochs_flagged_rms. It is kept, because a moon dip inside the
            transit also raises the ratio. Default 3.0.
        exposure_time : float, "auto" or None, optional
            Exposure time of each data point in days. The planet fits and the moon
            templates are averaged over it. Default "auto": the median spacing between
            consecutive points, which is the exposure of mission data. None gives an
            instantaneous model, and a warning when the points are spaced by more than
            SUBEXPOSURE_DAYS, because a long exposure smears the planet ingress and the
            planet residual can then look like a moon.
        ttv_prior_width : float, "auto" or None, optional
            Timing variation in days allowed around the linear ephemeris, on top of the
            ephemeris uncertainty at each epoch, in the Gaussian prior on the per-epoch t0.
            Default "auto": the largest planet TTV that a searched moon can cause (see
            `max_moon_ttv`). None fits t0 without a prior.
        planet_t0_err, planet_period_err : float, optional
            1-sigma uncertainties in days of planet_t0 and planet_period, such as those of a
            transit fit. They set the ephemeris uncertainty in the t0 prior. When either is
            missing, it is measured from the timing precision of the other transits (see
            `ephemeris_sigmas`).
        null_runs : int, optional
            Number of searches on pseudo-epochs, windows of the same light curve away from the
            planet transits with the fitted planet injected, that form the null distribution of
            the detection SNR (audit item H10, see `run_nulls`). Default 0, no false-alarm
            probability.
        fap_threshold : float, optional
            False-alarm probability below which the best hypothesis counts as a detection,
            default 0.01. The detection uses the FAP_UPPER_PERCENTILE percentile of the Gumbel
            probability over bootstrap resamples of the nulls (see `false_alarm_upper`).
        null_seed : int, optional
            Seed of the pseudo-epoch offsets, default 0.
        null_stop_fap : float or None, optional
            Stop the null searches once the false-alarm probability is certain to end above this
            value, default 0.1 (see `run_nulls`). It must be above fap_threshold. None runs all
            null_runs searches.
        scramble_runs : int, optional
            Number of searches on the real epochs with the moon phases permuted between epochs
            (see `run_scrambled_nulls`), which vet a promising hypothesis against the real
            planet residuals. Default 0, none. They set scramble_fap and scramble_passed and do
            not change detected.
        scramble_trigger_fap : float or None, optional
            Run the scrambled nulls only when the pseudo-epoch false-alarm probability is below
            this value, default 0.1. None runs them whenever scramble_runs > 0, also without
            pseudo-epoch nulls.
        detrend_window : float, optional
            Biweight window in days. When given, the light curve is taken as undetrended and
            `run` detrends it with every moon window masked (audit item A1), after any injected
            moon is added (audit item A2). Default None: the curve is already detrended.
        detrend_mask : str, optional
            What the detrending keeps out of the biweight (see `detrend`): 'moon_window' masks the
            widest searched moon window of every epoch, 'planet' only the planet transit, and 'none'
            masks nothing and divides the planet model out before the biweight. 'auto' (default)
            picks 'planet' when the moon window is more than AUTO_PLANET_MASK_RATIO times the
            biweight window, and 'moon_window' otherwise (see `resolved_detrend_mask`).
        duration_from_fit : bool, optional
            True (default) when planet_duration and planet_impact_param come from a transit fit,
            such as a catalog: their speed across the star sets the moon offsets. False for a
            search duration, such as a TLS one: the offsets use 2 pi a / P with e and w until the
            fold fit, then its T14 and b.
        retrograde : bool, optional
            True (default) extends the moon period grid to the widest stable retrograde orbit,
            0.93 r_H against 0.49 r_H prograde (Domingos et al. 2006). The moon model only uses the
            projected offset cos(alpha + 2 pi t / P), which is the same for a retrograde moon at
            (P, alpha) and a prograde one at (P, -alpha), so the phase grid already holds the
            retrograde moons; only their wider stable region needs the longer periods. False keeps
            the prograde limit.
        """
        self.object_dir = object_dir
        self.cpus = cpus
        self.star_mass = star_mass
        self.star_radius = star_radius
        self.ab = ab
        self.planet_radius = planet_radius
        if not np.isfinite(planet_radius_err):
            planet_radius_err = 0.1 * planet_radius
            logging.warning("Planet radius error is not finite, using 10%% of the radius: %.3f R_earth",
                            planet_radius_err)
        self.planet_radius_err = planet_radius_err
        self.planet_mass = planet_mass
        self.planet_period = planet_period
        self.planet_t0 = planet_t0
        self.planet_duration = planet_duration
        self.planet_semimajor_axis = planet_semimajor_axis
        self.planet_inc = planet_inc
        self.planet_inc_err = planet_inc_err
        self.planet_ecc = planet_ecc
        self.planet_ecc_err = planet_ecc_err
        self.planet_arg_periastron = planet_arg_periastron
        self.planet_impact_param = planet_impact_param
        # The input (catalog) transit fit sets the planet speed across the star for the moon offsets.
        self.catalog_planet_duration = planet_duration
        self.catalog_impact_param = planet_impact_param
        self.duration_from_fit = bool(duration_from_fit)
        self._speed_logged = False
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.max_moon_density = max_moon_density
        self.min_cadences_per_t0 = min_cadences_per_t0
        self.min_orbit_window = min_orbit_window
        self.subtract_mode = subtract_mode
        self.mask_duration_factor = mask_duration_factor
        self.skip_fold_fit = skip_fold_fit
        # Keep the input impact parameter in the fold fit instead of fitting it.
        self.fix_impact_param = bool(fix_impact_param)
        self.search_retrograde = bool(retrograde)
        self.fit_rp_per_transit = fit_rp_per_transit
        self.min_transit_completeness = min_transit_completeness
        self.max_subtraction_rms_ratio = max_subtraction_rms_ratio
        self.setup_point_exposures(exposure_time, time)
        if not (ttv_prior_width is None or ttv_prior_width == "auto"
                or (isinstance(ttv_prior_width, (int, float)) and np.isfinite(ttv_prior_width) and ttv_prior_width > 0)):
            raise ValueError(f"ttv_prior_width must be a positive number of days, 'auto' or None, got "
                             f"{ttv_prior_width}")
        self.ttv_prior_width = ttv_prior_width
        self.ttv_prior_width_used = None
        for name, value in (("planet_t0_err", planet_t0_err), ("planet_period_err", planet_period_err)):
            if value is not None and not (np.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a non-negative number of days or None, got {value}")
        self.planet_t0_err = planet_t0_err
        self.planet_period_err = planet_period_err
        self.ephemeris_source = None
        if not (isinstance(null_runs, (int, np.integer)) and null_runs >= 0):
            raise ValueError(f"null_runs must be a non-negative integer, got {null_runs}")
        if not (np.isfinite(fap_threshold) and 0 < fap_threshold < 1):
            raise ValueError(f"fap_threshold must be between 0 and 1, got {fap_threshold}")
        if detrend_window is not None and not (np.isfinite(detrend_window) and detrend_window > 0):
            raise ValueError(f"detrend_window must be a positive number of days or None, got {detrend_window}")
        self.detrend_window = detrend_window
        if detrend_mask not in ExoMoonLeastSquares.DETREND_MASKS:
            raise ValueError(f"detrend_mask must be one of {ExoMoonLeastSquares.DETREND_MASKS}, got {detrend_mask}")
        self.detrend_mask = detrend_mask
        self._detrend_mask_logged = False
        self.null_runs = int(null_runs)
        self.fap_threshold = float(fap_threshold)
        if null_stop_fap is not None and not (np.isfinite(null_stop_fap) and self.fap_threshold < null_stop_fap < 1):
            raise ValueError(f"null_stop_fap must be None or between fap_threshold ({self.fap_threshold}) and 1, "
                             f"got {null_stop_fap}")
        self.null_stop_fap = None if null_stop_fap is None else float(null_stop_fap)
        self.null_stopped_early = None
        if int(scramble_runs) < 0:
            raise ValueError(f"scramble_runs must be zero or positive, got {scramble_runs}")
        self.scramble_runs = int(scramble_runs)
        self.scramble_trigger_fap = None if scramble_trigger_fap is None else float(scramble_trigger_fap)
        self.moon_phase_map = None
        self.scramble_snrs = None
        self.scramble_stopped_early = None
        self.scramble_fap = None
        self.scramble_fap_empirical = None
        self.scramble_fap_upper = None
        self.scramble_passed = None
        self.null_seed = null_seed
        self.null_snrs = None
        self.null_independent_offsets = None
        self.fap = None
        self.fap_empirical = None
        self.fap_upper = None
        self.fap_extrapolated = None
        self.null_fit_rejected = None
        self.detected = None
        self.sde = None
        self.epoch_stats = None
        self.loo_snr_min = None
        self.single_event = None
        self.recovered = None
        self.recovered_alias = None
        self.injection_alias_k = None
        self.injection_alias_mirror = None
        self.planet_rp_over_rstar = None
        self.fold_realigned = False
        self.model_validity = None
        self.planet_overlap_fraction = None
        self.planet_overlap_mean = None
        self.best_tau = None
        self.period_constrained = None
        self.planet_shape_check = None
        self.planet_shape_check_inputs = None
        self.planet_shape_correction_large = None
        self.mask_mode_survival = None
        self.aliases = None
        self.red_noise_factor = 1.0
        self.red_noise_measured = None
        self.fold_fit_status = None
        self.planet_a_over_rstar = None
        self.absorbed_fraction = None
        # Derivative of the subtracted planet model with respect to its depth, at each kept epoch,
        # and the depth correction the search measures with it at the best moon (see `search`).
        self.planet_depth_column = None
        self.planet_depth_correction = None
        self.last_planet_depth_correction = None
        # Per-epoch derivative of the subtracted planet model with respect to its mid-time, and the
        # epoch each point belongs to (-1 away from the transits). The search fits one mid-time
        # shift per epoch with them (see `planet_fit`); fit_planet_t0 turns that off.
        self.planet_t0_column = None
        self.planet_epoch_index = None
        self.fit_planet_t0 = True
        self.last_planet_fit = None
        self.planet_t0_shifts = None
        # Derivatives of the subtracted planet model with respect to its squared impact parameter and
        # its two limb darkening coefficients, at every epoch. Like the depth, each is one correction
        # shared by all epochs, fitted with each moon; fit_planet_shape turns them off.
        self.planet_shape_columns = None
        # One planet depth correction per epoch instead of one shared by all, fitted with each moon
        # like the mid-times: real transits change depth from epoch to epoch by more than the noise
        # (spots, dilution), which the pseudo-epoch nulls do not have. fit_planet_epoch_depth turns
        # it off.
        self.fit_planet_epoch_depth = True
        self.planet_epoch_depths = None
        self.epoch_depth_comparison = None
        # Impact parameter the shape columns are taken at.
        self.planet_shape_impact = None
        self.fit_planet_shape = True
        self.planet_shape_corrections = None
        self.epochs_dropped_fit = []
        self.epochs_flagged_rms = []
        self.epochs_flagged_rp = []
        self.epochs_dropped_coverage = []
        self.bary_t0s = []
        for i, bary_t0 in enumerate(t0s):
            mask = np.abs(time - bary_t0) < self.planet_duration * 2
            if not self.is_skip_t0(time[mask]):
                self.bary_t0s.append(bary_t0)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.best_snr_radius = np.nan
        self.best_snr_radius_above_range = False
        self.best_snr_radius_below_range = False
        # Tidal survival (Barnes & O'Brien 2002, generous margin of `tides`): a warning, not a cut. The
        # retrograde bound (their Eq. 12, a captured moon migrating inwards) applies when retrograde
        # moons are searched.
        self.max_surviving_moon_radius = max_surviving_moon_radius(star_mass, planet_mass, planet_radius,
                                                                   planet_period, planet_ecc)
        self.max_surviving_moon_radius_retrograde = max_surviving_moon_radius(star_mass, planet_mass, planet_radius,
                                                                              planet_period, planet_ecc,
                                                                              retrograde=True)
        self.best_snr_radius_above_tidal_survival = False
        self.best_snr_radius_above_tidal_survival_retrograde = False
        if self.max_surviving_moon_radius < max_radius:
            retrograde_note = (f"; retrograde (captured, inward-migrating) moons above "
                               f"{self.max_surviving_moon_radius_retrograde:.2f} R_earth"
                               if self.search_retrograde else "")
            logging.info("Prograde moons above %.2f R_earth%s are unlikely to survive tidal migration (Barnes & O'Brien "
                         "2002 with a generous margin: %.0f Gyr, k2/Q x %.1f); they are still searched",
                         self.max_surviving_moon_radius, retrograde_note, TIDAL_SURVIVAL_AGE_GYR,
                         TIDAL_SURVIVAL_K2_Q_FACTOR)
        if period_grid_size < ExoMoonLeastSquares.SPECTRUM_KERNEL:
            raise ValueError(f"period_grid_size must be at least {ExoMoonLeastSquares.SPECTRUM_KERNEL}, "
                             f"got {period_grid_size}")
        self.period_grid_size = period_grid_size
        self.radius_grid_size = radius_grid_size
        self.planet_mass_grid = [planet_mass]

    def search(self, search_input, return_lc=False):
        """Search for a moon signal for a given set of input parameters.

        Parameters
        ----------
        search_input : SearchInput
            Search input parameters.
        return_lc : bool, optional
            If True, return light curve data in addition to residuals,
            default False.

        Returns
        -------
        tuple
            (residual, baseline_residual, radius, model_snr,
            template_unusable) if return_lc is False, or (residual,
            baseline_residual, radius, model_snr, it_time, it_flux, oot_time,
            oot_flux, residual_model) if return_lc is True.
            template_unusable is True when a dip was seen but no template
            had depth at the observed times.
        """
        column = self.search_planet_depth_column()
        split = self.split_scenarios(search_input, self.time, self.flux, self.flux_err, self.bary_t0s, extra=column)
        if split is None:
            no_data = np.array([])
            scores = (ExoMoonLeastSquares.MAX_RESIDUAL, ExoMoonLeastSquares.MAX_RESIDUAL, self.min_radius, np.nan)
            return scores + (no_data, no_data, no_data, no_data, no_data) if return_lc else scores + (False,)
        scenario_time, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err = split[:7]
        if len(it_flux) == 0 or len(oot_flux) < 2:
            # No in-window or out-of-window data: the depth and the noise are undefined.
            no_data = np.array([])
            scores = (ExoMoonLeastSquares.MAX_RESIDUAL, ExoMoonLeastSquares.MAX_RESIDUAL, self.min_radius, np.nan)
            return scores + (it_time, it_flux, oot_time, oot_flux, no_data) if return_lc else scores + (False,)
        # Depth is measured against the surrounding out-of-window level, so a normalisation offset
        # or the upper-only outlier clipping does not count as signal. With the planet depth column,
        # one planet depth correction shared by all epochs is fitted with it (see planet_depth_fit).
        inflation = 1.0
        if column is not None:
            self.last_planet_depth_correction, inflation, self.last_planet_fit = split[7], split[8], split[9]
        depth = ExoMoonLeastSquares.weighted_mean(oot_flux, oot_flux_err) - \
            ExoMoonLeastSquares.weighted_mean(it_flux, it_flux_err)
        t0_flux_mean_value = 1 - depth
        if len(self.models) == 0:
            residual_baseline = np.sum(((it_flux - 1) / it_flux_err) ** 2)
            residual_calculation = residual_baseline
            residual_radius = 0.0
            residual_model = np.ones_like(it_flux)
            template_unusable = True
        else:
            residual_calculation, residual_baseline, residual_radius, residual_model, template_unusable = (
                self.calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                                         t0_flux_mean_value, len(scenario_time), self.models, self.min_radius,
                                         self.max_radius, self.star_radius, self.radius_grid_size))
        if inflation > 1 and residual_calculation < residual_baseline:
            # The moon amplitude fitted with the planet corrections varies inflation times more than
            # a plain difference of means, and its chi-squared gain inflation^2 times more, so the
            # gain is divided by inflation^2, as the SNR is divided by inflation: on noise the
            # residual spectrum then keeps the spread it has without the corrections.
            residual_calculation = residual_baseline - (residual_baseline - residual_calculation) / inflation ** 2
        #oot_noise = np.std(oot_flux)
        #oot_noise = np.sqrt(np.std(oot_flux) * (np.sqrt(np.sum(oot_flux_err ** 2) / len(oot_flux_err))))
        model_snr = ExoMoonLeastSquares.box_snr(it_flux, oot_flux, it_flux_err, oot_flux_err) / inflation \
            / self.red_noise_factor
        if return_lc:
            return residual_calculation, residual_baseline, residual_radius, model_snr, it_time, it_flux, oot_time, oot_flux, residual_model
        else:
            return residual_calculation, residual_baseline, residual_radius, model_snr, template_unusable

    def write_figures(self, period_grid, alpha_grid, snr_matrix, snr_spectrum, residual_spectrum, sde_residuals,
                      sde_snr, best_period, best_alpha, best_snr, best_input, recovered_t0s, injected_t0s=None,
                      inject_params=None):
        """Write the figures of the search to the object directory (see `sherlockpipe.exomoon.plots`).

        search_overview.png: the SNR spectrum and the diagnostic spectra over the moon period grid.
        snr_map.png: the SNR over the moon period and initial phase grid.
        hypothesis_<n>.png: the stacked light curve of the selected hypothesis (1) and of the next
        SNR peaks at other periods.
        null_distribution.png: the null SNRs, their Gumbel fit and the selected SNR, with null runs.
        planet_epoch_depths.png: the per-epoch planet depth changes against the share of the moon.
        epochs/T0_<t0>.png: each planet transit after the planet subtraction, with the moon.
        """
        injected_period = inject_params['moon_period'] if inject_params is not None else None
        snr_threshold = None
        if self.null_snrs is not None and len(self.null_snrs) > 2:
            nulls = np.asarray([value for value in self.null_snrs if np.isfinite(value)], dtype=float)
            snr_threshold = float(stats.gumbel_r.isf(self.fap_threshold, *stats.gumbel_r.fit(nulls)))
        exomoon_plots.search_overview_figure(
            os.path.join(self.object_dir, "search_overview.png"), period_grid, snr_spectrum, residual_spectrum,
            sde_snr, sde_residuals, best_period, best_snr, snr_threshold=snr_threshold,
            fap_threshold=self.fap_threshold, injected_period=injected_period,
            aliases=exomoon_plots.alias_periods(injected_period if injected_period is not None else best_period,
                                                self.planet_period, period_grid[0], period_grid[-1]),
            sde_selected=self.sde, sde_peak=self.max_sde)
        exomoon_plots.grid_map_figure(
            os.path.join(self.object_dir, "snr_map.png"), period_grid, alpha_grid, snr_matrix, "SNR",
            "SNR over the moon period and initial phase grid", best=(best_period, best_alpha),
            injected=(injected_period, inject_params['initial_alpha']) if inject_params is not None else None)
        moon_duration = 2 * best_input.half_moon_transit_duration
        hypotheses = [(best_period, best_alpha)]
        for index in exomoon_plots.spectrum_peaks(snr_spectrum, period_grid, self.HYPOTHESIS_FIGURES,
                                                  exclude=[best_period]):
            best_alpha_index = int(np.nanargmax(np.nan_to_num(snr_matrix[:, index], nan=-np.inf)))
            hypotheses.append((period_grid[index], alpha_grid[best_alpha_index]))
        for rank, (period, alpha) in enumerate(hypotheses[:self.HYPOTHESIS_FIGURES], start=1):
            search_input = self.search_input(period, alpha, self.planet_mass, period_grid[-1],
                                             self.planet_impact_param)
            _, _, radius, snr, it_time, it_flux, oot_time, oot_flux, model = self.search(search_input, return_lc=True)
            if len(model) == 0:
                continue
            label = "Selected hypothesis" if rank == 1 else f"SNR peak {rank}"
            exomoon_plots.hypothesis_figure(
                os.path.join(self.object_dir, f"hypothesis_{rank}.png"),
                exomoon_plots.hypothesis_heading(label, period, alpha, snr, radius, np.mean(oot_flux) - np.mean(it_flux)),
                it_time, it_flux, oot_time, oot_flux, model, moon_duration, len(self.bary_t0s))
        if self.fap is not None and self.null_snrs:
            exomoon_plots.null_distribution_figure(
                os.path.join(self.object_dir, "null_distribution.png"), self.null_snrs, best_snr, self.fap,
                self.fap_upper, self.fap_empirical, self.fap_threshold, self.detected, self.fap_extrapolated)
        # The epoch figures show the points with the planet corrections fitted with the selected moon.
        self.last_planet_depth_correction = None
        self.last_planet_fit = None
        _, _, radius, _, _, _, _, _, _ = self.search(best_input, return_lc=True)
        columns = self.search_planet_depth_column()
        planet_fit = self.last_planet_fit if columns is not None else None
        shown_flux = self.flux - ExoMoonLeastSquares.planet_correction(planet_fit, columns) \
            if planet_fit is not None else self.flux
        half_orbit = best_input.half_moon_max_orbit_transit_duration
        half_moon = best_input.half_moon_transit_duration
        moon_params = batman.TransitParams()
        moon_params.per = self.planet_period
        moon_params.rp = max(radius, self.min_radius) * R_earth_to_R_sun / self.star_radius
        moon_params.a = self.transit_semimajor_axis() / self.star_radius
        moon_params.inc = self.planet_inc
        moon_params.ecc = self.planet_ecc
        moon_params.w = 0
        moon_params.u = self.ab
        moon_params.limb_dark = "quadratic"
        injected = injected_t0s is not None and hasattr(self, 'injected_flux_moon_pandora')
        epoch_dir = os.path.join(self.object_dir, "epochs")
        os.makedirs(epoch_dir, exist_ok=True)
        moon_flux = np.ones(len(self.time))
        # The window shows the moon orbit and at least the planet transit with as much baseline on
        # each side; a narrow moon orbit window can hold fewer points than the search minimum.
        half_window = max(half_orbit, 1.5 * self.planet_duration)
        for i, t0 in enumerate(self.bary_t0s):
            window = np.abs(self.time - t0) < half_window
            if np.sum(window) < 3:
                continue
            moon_params.t0 = recovered_t0s[i]
            local_model = ExoMoonLeastSquares.transit_model(moon_params, self.time[window], self.exposure_at(self.time[window])) \
                .light_curve(moon_params)
            moon_flux[window] = local_model
            heading = (f"Planet transit at {t0:.3f} d, moon P = {best_period:.3f} d, alpha = {best_alpha:.2f} rad, "
                       f"radius {radius:.2f} R_earth")
            if injected:
                heading += (f"\nInjected P = {self.injected_moon_period:.3f} d, "
                            f"alpha = {self.injected_moon_alpha:.2f} rad")
            injected_window = np.abs(self.inject_flux_time - t0) < half_window if injected else None
            exomoon_plots.epoch_figure(
                os.path.join(epoch_dir, f"T0_{t0:.3f}.png"), t0, self.planet_duration, self.time[window],
                shown_flux[window], self.time[window], local_model, recovered_t0s[i], heading,
                injected_time=self.inject_flux_time[injected_window] if injected else None,
                injected_flux=self.injected_flux_moon_pandora[injected_window] if injected else None,
                injected_t0=injected_t0s[i] if injected else None, timing_tolerance=half_moon if injected else None)
        if planet_fit is not None and planet_fit["epoch_depths"]:
            # Each epoch's depth is measured without a moon: the planet corrections except the depth
            # are applied, then a local constant and the epoch's depth column are fitted. The moon
            # share is the same fit on the moon model alone. Their constant parts are the planet
            # depth, so the test is whether their changes between epochs agree.
            no_depth_fit = dict(planet_fit, epoch_depths={}, depth=0.0)
            flux = self.flux - ExoMoonLeastSquares.planet_correction(no_depth_fit, columns)
            depths, moon_share, errors, epoch_t0s, overlaps = [], [], [], [], []
            # With the coplanar orbit of the template, a moon within a planet radius plus a moon radius
            # of the planet on the sky could be over the planet disk, where its dip would be hidden.
            # Whether it is depends on the moon orbit inclination, which is not known: a few degrees
            # move it above or below the disk. Those epochs are marked and left out of the comparison,
            # not modelled. The sky speed is the planet's.
            rp = self.null_planet_params().rp
            rm = max(radius, self.min_radius) * R_earth_to_R_sun / self.star_radius
            chord = 2 * np.sqrt(max((1 + rp) ** 2 - self.planet_impact_param ** 2, 1e-6))
            overlap_time = (rp + rm) / chord * self.planet_duration
            for index in sorted(planet_fit["epoch_depths"]):
                if index >= len(self.bary_t0s):
                    continue
                window = np.abs(self.time - self.bary_t0s[index]) < half_window
                column = np.where(columns[:, 2] == index, columns[:, 0], 0.0)[window]
                if np.sum(column != 0) < 3 or np.sum(column == 0) < 3:
                    continue
                design = np.column_stack([np.ones(len(column)), column])
                (_, depth), _, _, _ = np.linalg.lstsq(design, flux[window] - 1, rcond=None)
                (_, share), _, _, _ = np.linalg.lstsq(design, moon_flux[window] - 1, rcond=None)
                residual = flux[window] - 1 - design @ np.linalg.lstsq(design, flux[window] - 1, rcond=None)[0]
                noise = np.std(residual) * np.sqrt(len(column) / max(len(column) - 2, 1))
                errors.append(noise * np.sqrt(np.linalg.inv(design.T @ design)[1, 1]))
                epoch_t0s.append(self.bary_t0s[index])
                overlaps.append(bool(abs(recovered_t0s[index] - self.bary_t0s[index]) < overlap_time))
                depths.append(depth)
                moon_share.append(share)
            self.epoch_depth_comparison = {"t0": [float(t0) for t0 in epoch_t0s],
                                           "depth_ppm": [float(value * 1e6) for value in depths],
                                           "moon_share_ppm": [float(value * 1e6) for value in moon_share],
                                           "error_ppm": [float(value * 1e6) for value in errors],
                                           "moon_possibly_over_planet": overlaps}
            if len(depths) > 2:
                exomoon_plots.epoch_depth_figure(os.path.join(self.object_dir, "planet_epoch_depths.png"),
                                                 epoch_t0s, depths, moon_share, errors, best_period, best_alpha,
                                                 overlaps=overlaps)

    def select_best(self, period_grid, alpha_grid, snr_matrix):
        """Select the hypothesis that every reported statistic describes (audit items H7 and H8).

        The SNR is the detection statistic. The best grid point is taken, and its phase is refined
        with a parabola over alpha when the refined phase scores a higher SNR in `search`. The same
        selection scores the nulls, so the SNR and its false-alarm probability are comparable.

        Returns
        -------
        tuple or None
            (period index, period, alpha, SNR), or None when no hypothesis has an SNR.
        """
        finite = np.where(np.isfinite(snr_matrix), snr_matrix, -np.inf)
        if not np.any(np.isfinite(snr_matrix)):
            return None
        alpha_index, period_index = np.unravel_index(np.argmax(finite), finite.shape)
        period, alpha = float(period_grid[period_index]), float(alpha_grid[alpha_index])
        snr = float(snr_matrix[alpha_index, period_index])
        try:
            refined = float(ExoMoonLeastSquares.refine_alpha(alpha_grid, -finite)[period_index][0])
        except Exception:
            refined = np.nan
        if np.isfinite(refined) and refined != alpha:
            refined_snr = self.search(self.search_input(period, refined, self.planet_mass, period_grid[-1],
                                                        self.planet_impact_param))[3]
            if np.isfinite(refined_snr) and refined_snr > snr:
                alpha, snr = refined, float(refined_snr)
        return int(period_index), period, alpha, snr

    def alias_family(self, period_grid, snr_spectrum, best_period):
        """Return the sampling and mirror aliases of the selected period with their SNR (audit item H9).

        With one look per planet orbit, the moon phase at epoch n is alpha + 2 pi n P_p / P, so
        every P' with 1 / P' = 1 / P + k / P_p gives the same phases, and every P' with
        1 / P' = k / P_p - 1 / P the same phases negated, which alpha' = pi - alpha turns into the
        same positions (find_period_alias). They differ only through the orbit size and the
        timing offsets; in close-in geometry the data barely prefer one of them. Each alias inside
        the grid (alias_frequencies) is listed with the best SNR over alpha at the nearest grid
        period and its SNR below the selected one, highest SNR first.

        Returns
        -------
        list of dict
            {"k", "mirror", "period", "grid_period", "snr", "snr_below_best"} per alias, the
            selected period (k = 0, not mirror) included.
        """
        period_grid = np.asarray(period_grid, dtype=float)
        snr_spectrum = np.asarray(snr_spectrum, dtype=float)
        best_index = int(np.argmin(np.abs(np.log(period_grid / best_period))))
        best_snr = snr_spectrum[best_index]
        family = []
        for mirror in (False, True):
            for k, frequency in ExoMoonLeastSquares.alias_frequencies(best_period, self.planet_period, mirror):
                period = 1 / frequency
                if not period_grid[0] <= period <= period_grid[-1]:
                    continue
                index = int(np.argmin(np.abs(np.log(period_grid / period))))
                snr = snr_spectrum[index]
                family.append({"k": k, "mirror": mirror, "period": float(period),
                               "grid_period": float(period_grid[index]),
                               "snr": float(snr) if np.isfinite(snr) else None,
                               "snr_below_best": float(best_snr - snr) if np.isfinite(snr) and np.isfinite(best_snr)
                               else None})
        family.sort(key=lambda alias: -np.inf if alias["snr"] is None else -alias["snr"])
        return family

    def grid_search(self, progress=True):
        """Score every moon hypothesis of the grid.

        Parameters
        ----------
        progress : bool, optional
            Show a progress bar, default True.

        Returns
        -------
        tuple
            (period_grid, alpha_grid, search_inputs, all_residuals), with the alpha index varying
            fastest in search_inputs and all_residuals.
        """
        planet_mass_grid = self.planet_mass_grid
        search_inputs = []
        if len(self.models) == 0:
            logging.warning("No moon radius produced a transit template, using a flat fallback template")
            self.models = [(self.min_radius, np.array([-1.0, 1.0]), np.array([0.9999, 0.9999]))]
        alpha_grid_size = 25
        alpha_grid = np.linspace(0, np.pi * 2 - np.pi * 2 / 25, alpha_grid_size)
        #alpha_grid = [0.0]
        impact_params_grid = [self.planet_impact_param]
        for planet_mass in planet_mass_grid:
            # CITE Édouard Roche (1849, Académie des Sciences de Montpellier). 2.44 is the fluid-body coefficient.
            # planet_volume_m3 = (4/3) * np.pi * ((LcbuilderHelper.convert_from_to(self.planet_radius, u.R_earth, u.m)) ** 3)
            # planet_mass_kg = LcbuilderHelper.convert_from_to(self.planet_mass, u.M_earth, u.kg)
            # planet_density = planet_mass_kg / planet_volume_m3
            # roche_lobe = 2.44 * self.planet_radius * (planet_density / self.max_moon_density) ** (1/3)
            min_period = ExoMoonLeastSquares.compute_min_moon_period(self.max_moon_density, planet_mass,
                                                                     self.planet_radius, self.max_radius)
            if self.min_orbit_window > 0:
                widest = self.max_moon_period(planet_mass)
                moon_transit = self.search_input(widest, 0.0, planet_mass, widest,
                                                 self.planet_impact_param).moon_transit_duration
                a_moon_rsun = (self.planet_duration * self.min_orbit_window - moon_transit) / 2 * self.planet_speed()
                p_min_win = 0.0 if a_moon_rsun <= 0 else ExoMoonLeastSquares.au_to_period(
                    planet_mass * M_earth_to_M_sun, a_moon_rsun / AU_TO_RSUN)
                min_period = max(min_period, p_min_win)
            # O3: the minimum is no longer only the Roche limit.
            logging.info("Minimum moon period: %s d (largest of Roche limit, planet contact%s)", min_period,
                         " and orbit window" if self.min_orbit_window > 0 else "")
            max_period = self.max_moon_period(planet_mass)
            a_max = self.max_moon_semimajor_axis(self.star_mass, planet_mass * M_earth_to_M_sun,
                                                 self.planet_semimajor_axis, self.planet_ecc,
                                                 retrograde=self.search_retrograde)
            hill = self.compute_hill_radius(self.star_mass, planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis)
            logging.info("Outer moon limit: a = %.3f r_H (Domingos et al. 2006, %s, e_p = %.2f), P = %.4f d",
                         a_max / hill, "retrograde" if self.search_retrograde else "prograde", self.planet_ecc,
                         max_period)
            period_grid = self.compute_moon_period_grid(min_period, max_period, samples=self.period_grid_size, mode="log")
            #period_grid = [2.5, 50]
            for impact_param in impact_params_grid:
                for moon_period in period_grid:
                    search_inputs.extend(self.search_inputs(moon_period, alpha_grid, planet_mass, max_period,
                                                            impact_param))
        # Each worker gets this object once (see `_set_worker_search`): passing the bound method
        # sent a copy of it, about 20 MB on a 1-minute Kepler curve, with every chunk of 100.
        with Pool(processes=self.cpus, maxtasksperchild=500, initializer=_set_worker_search,
                  initargs=(self,)) as pool:
            all_residuals = list(
                tqdm(pool.imap(_worker_search, search_inputs, chunksize=100),
                     total=len(search_inputs), disable=not progress)
            )
        return period_grid, alpha_grid, search_inputs, all_residuals

    def run(self, plot=False, inject_params=None, null_snrs=None):
        """Run the full exomoon search over the parameter grid.

        Parameters
        ----------
        plot : bool, optional
            If True, generate diagnostic plots, default False.
        inject_params : dict or None, optional
            If a dict with keys 'moon_radius', 'moon_period', 'initial_alpha',
            'moon_mass', injects a moon signal into the flux before the grid
            search to test injection-recovery. Default None (no injection).
        null_snrs : array_like, optional
            Null SNR maxima to use instead of running null_runs searches, for example from an
            earlier run on the same light curve, as injection campaigns do.
        """
        self.require_usable_epochs()
        planet_mass_grid = self.planet_mass_grid
        self.inject_flux_time = self.time.copy()  # save full time before trimming
        before_trim = copy.copy(self)  # the whole light curve, without an injected moon, for the nulls
        if self.detrend_window is not None:
            # A2: the moon is injected into the undetrended curve, so recovery includes the detrending loss.
            if inject_params is not None:
                self.flux = self.inject_moon(self.time, self.flux, self.flux_err, self.bary_t0s,
                                             self.planet_semimajor_axis, self.planet_ecc, inject_params['moon_radius'],
                                             inject_params['moon_period'], inject_params['moon_mass'],
                                             initial_alpha=inject_params['initial_alpha'], inject_planet=False,
                                             inject_moon=True, retrograde=inject_params.get('retrograde', False))
                logging.info("Injected moon before detrending: P=%.4f d, R=%.2f R_e, alpha=%.4f rad, %s",
                             inject_params['moon_period'], inject_params['moon_radius'], inject_params['initial_alpha'],
                             "retrograde" if inject_params.get('retrograde', False) else "prograde")
                self.inject_flux_time = self.time.copy()
            self.time, self.flux, self.flux_err = self.detrend(self.time, self.flux, self.flux_err)
            if inject_params is not None:
                order = np.argsort(self.inject_flux_time)
                self.inject_flux_time = self.inject_flux_time[order]
                self.injected_flux_moon_pandora = np.asarray(self.injected_flux_moon_pandora)[order]
            logging.info("Detrended with a %.2f d biweight, masking +-%.2f d around every epoch", self.detrend_window,
                         self.moon_window_half_width())
        self.time, self.flux, self.flux_err = self.remove_non_transit_flux(self.time, self.flux, self.flux_err, self.bary_t0s, np.max(planet_mass_grid))
        rms = np.std(self.flux)
        if inject_params is not None and self.detrend_window is None:
            self.flux = self.inject_moon(self.time, self.flux, self.flux_err, self.bary_t0s, self.planet_semimajor_axis,
                                         self.planet_ecc, inject_params['moon_radius'], inject_params['moon_period'],
                                         inject_params['moon_mass'], initial_alpha=inject_params['initial_alpha'],
                                         inject_planet=False, inject_moon=True,
                                         retrograde=inject_params.get('retrograde', False))
            logging.info("Injected moon: P=%.4f d, R=%.2f R_e, alpha=%.4f rad",
                         inject_params['moon_period'], inject_params['moon_radius'], inject_params['initial_alpha'])
            self.inject_flux_time = self.time.copy()
        before_planet_fit = copy.copy(self)
        self.time, self.flux, self.flux_err = self.subtract_planet_transit(self.time, self.flux, self.flux_err, mode=self.subtract_mode, plot=plot)
        self.build_search_templates()
        self.check_model_validity()
        self.set_red_noise_factor(before_trim)
        if self.planet_shape_check_inputs is not None:
            self.planet_shape_check = self.check_planet_shape(*self.planet_shape_check_inputs,
                                                              noise_factor=self.red_noise_factor)
        period_grid, alpha_grid, search_inputs, all_residuals = self.grid_search()
        alpha_grid_size = len(alpha_grid)
        no_data_count = sum(bool(np.isnan(all_residual[3])) for all_residual in all_residuals)
        if no_data_count > 0:
            logging.warning("%d/%d scenarios had no in-window or out-of-window data and were scored as no detection",
                            no_data_count, len(all_residuals))
        unusable_template_count = sum(all_residual[4] for all_residual in all_residuals)
        if unusable_template_count > 0:
            logging.warning("%d/%d scenarios had no usable template (depth 0 at observed times)",
                            unusable_template_count, len(all_residuals))
        best_residuals_per_scenarios = []
        best_residuals_spectra = []
        best_residuals_spectra_values = []
        best_residual_per_period = np.inf
        best_residual_per_period_values = []
        best_alpha_per_period = []
        best_snr_per_period = []
        residuals_matrix = np.full((alpha_grid_size, len(period_grid)), np.nan)
        snr_matrix = np.full((alpha_grid_size, len(period_grid)), np.nan)
        max_residual = 0
        for i in np.arange(0, len(search_inputs)):
            if i % alpha_grid_size == 0 and i != 0:
                best_residuals_spectra = best_residuals_spectra + [best_residual_per_period]
                best_residuals_spectra_values = best_residuals_spectra_values + best_residual_per_period_values
                best_residual_per_period = np.inf
                best_residual_per_period_values = []
            all_residual = all_residuals[i]
            residuals = all_residual[0]
            residuals_baseline = all_residual[1]
            radius = all_residual[2]
            model_snr = all_residual[3]
            moon_period = search_inputs[i].moon_period
            moon_initial_alpha = search_inputs[i].moon_alpha
            corrected_residual = residuals / residuals_baseline if residuals_baseline > 1e-10 else 1.0
            if not (0 < corrected_residual < ExoMoonLeastSquares.MAX_RESIDUAL / 2):
                corrected_residual = 1.0
            if corrected_residual > max_residual and corrected_residual < ExoMoonLeastSquares.MAX_RESIDUAL / 2:
                max_residual = corrected_residual
            best_residuals_per_scenarios.append(
                [moon_period, moon_initial_alpha, corrected_residual, radius])
            residuals_matrix[i % alpha_grid_size][
                i // alpha_grid_size] = corrected_residual
            snr_matrix[i % alpha_grid_size][
                i // alpha_grid_size] = model_snr
            if i % alpha_grid_size == 0:
                best_alpha_per_period.append([moon_initial_alpha, corrected_residual])
            elif best_alpha_per_period[i // alpha_grid_size][1] > corrected_residual:
                best_alpha_per_period[i // alpha_grid_size] = [moon_initial_alpha, corrected_residual]
            if best_residual_per_period > corrected_residual:
                best_residual_per_period = corrected_residual
                best_residual_per_period_values = [residuals_baseline, radius, moon_period, moon_initial_alpha]
            if i % alpha_grid_size == 0:
                best_snr_per_period.append([moon_initial_alpha, model_snr])
            elif not np.isnan(model_snr) and not best_snr_per_period[i // alpha_grid_size][1] >= model_snr:
                # "not >=" also replaces a NaN kept from the first alpha of this period.
                best_snr_per_period[i // alpha_grid_size] = [moon_initial_alpha, model_snr]
        best_residuals_spectra = best_residuals_spectra + [best_residual_per_period]
        best_residuals_spectra = np.array(best_residuals_spectra)
        best_residuals_spectra[np.isnan(best_residuals_spectra)] = np.nanmax(best_residuals_spectra)
        best_residuals_spectra_values = best_residuals_spectra_values + best_residual_per_period_values
        residuals_matrix[np.isnan(residuals_matrix)] = np.nanmax(residuals_matrix)
        residuals_matrix[residuals_matrix > ExoMoonLeastSquares.MAX_RESIDUAL / 2] = max_residual
        best_residuals_spectra[np.isnan(best_residuals_spectra)] = np.nanmax(best_residuals_spectra)
        residuals_per_scenario_baseline_args = np.argwhere(best_residuals_spectra > ExoMoonLeastSquares.MAX_RESIDUAL / 2).flatten()
        best_residuals_spectra[residuals_per_scenario_baseline_args] = max_residual
        snr_spectrum = np.transpose(best_snr_per_period)[1]
        kernel = ExoMoonLeastSquares.SPECTRUM_KERNEL
        SR, power_raw, power, SDE_raw, SDE = ExoMoonLeastSquares.spectra(best_residuals_spectra, 1, kernel)
        snr_median = ExoMoonLeastSquares.running_median(snr_spectrum, kernel)
        snr_norm = snr_spectrum - snr_median
        snr_norm_std = np.nanstd(snr_norm)
        snr_sde_spectrum = snr_norm / snr_norm_std if snr_norm_std > 0 else np.zeros_like(snr_norm)
        best_scenario_per_snr_args = np.argsort(np.nan_to_num(np.transpose(best_snr_per_period)[1], nan=-np.inf))
        # --- H8: one hypothesis for every reported statistic ---
        best = self.select_best(period_grid, alpha_grid, snr_matrix)
        if best is None:
            best_snr_idx = best_scenario_per_snr_args[-1]
            best_snr_period, best_snr_alpha, selected_snr = period_grid[best_snr_idx], alpha_grid[0], np.nan
        else:
            best_snr_idx, best_snr_period, best_snr_alpha, selected_snr = best
        best_input = self.search_input(best_snr_period, best_snr_alpha, self.planet_mass, period_grid[-1],
                                       self.planet_impact_param)
        # The radius comes from the same hypothesis as the SNR and the SDE (audit item H8).
        # H14: it is the template-inverted radius of the least-squares amplitude.
        self.last_planet_depth_correction = None
        self.last_planet_fit = None
        best_result = self.search(best_input, return_lc=True)
        if self.last_planet_fit is not None and self.last_planet_fit["t0_shifts"]:
            self.planet_t0_shifts = [{"t0": float(self.bary_t0s[index]), "shift_minutes": float(shift * 1440)}
                                     for index, shift in sorted(self.last_planet_fit["t0_shifts"].items())
                                     if index < len(self.bary_t0s)]
            logging.info("Planet mid-times fitted with the best moon: shifts of %.2f min rms over %d epochs",
                         np.sqrt(np.mean(np.square([shift["shift_minutes"] for shift in self.planet_t0_shifts]))),
                         len(self.planet_t0_shifts))
        if self.last_planet_fit is not None and self.last_planet_fit.get("epoch_depths"):
            self.planet_epoch_depths = [{"t0": float(self.bary_t0s[index]), "depth_ppm": float(value * 1e6)}
                                        for index, value in sorted(self.last_planet_fit["epoch_depths"].items())
                                        if index < len(self.bary_t0s)]
            values = [entry["depth_ppm"] for entry in self.planet_epoch_depths]
            logging.info("Planet depth fitted per epoch with the best moon: mean %+.0f ppm, scatter %.0f ppm over %d "
                         "epochs", np.mean(values), np.std(values), len(values))
        if self.last_planet_fit is not None and self.last_planet_fit.get("shape"):
            self.planet_shape_corrections = dict(self.last_planet_fit["shape"])
            if "impact_squared" in self.planet_shape_corrections and self.planet_shape_impact is not None:
                b0 = self.planet_shape_impact
                b2 = b0 ** 2 + self.planet_shape_corrections["impact_squared"]
                self.planet_shape_corrections["impact"] = float(np.sqrt(max(b2, 0.0)) - b0)
            logging.info("Planet shape fitted with the best moon: %s (changes to the subtracted impact parameter "
                         "and limb darkening)", ", ".join(f"{name} {value:+.4f}" for name, value
                                                         in self.planet_shape_corrections.items()))
            # The quadratic limb darkening enters the flux linearly, so its columns are exact (and u1, u2,
            # correlated by about 0.98, can drift together along their degenerate direction); the impact
            # parameter column is a first-order derivative in b^2, so its range is a change of b^2.
            impact_squared = self.planet_shape_corrections.get("impact_squared", 0.0)
            self.planet_shape_correction_large = bool(abs(impact_squared) > self.MAX_LINEAR_IMPACT_SQUARED_CORRECTION)
            if self.planet_shape_correction_large:
                logging.warning("Planet impact parameter corrected by %+.3f (b^2 %+.4f) with the best moon, beyond the "
                                "%.2f b^2 first-order range of its column: the subtracted planet geometry differs much "
                                "from the data, so the correction is approximate",
                                self.planet_shape_corrections.get("impact", 0.0), impact_squared,
                                self.MAX_LINEAR_IMPACT_SQUARED_CORRECTION)
        if self.planet_depth_column is not None and self.last_planet_depth_correction is not None:
            self.planet_depth_correction = float(self.last_planet_depth_correction * 1e6)
            logging.info("Planet depth fitted with the best moon: %+.0f ppm on the fold-fit depth %.0f ppm (negative "
                         "when the fold fit had taken part of the moon as planet depth)",
                         self.planet_depth_correction, self.null_planet_params().rp ** 2 * 1e6)
        moon_radius = float(best_result[2]) if len(best_result[-1]) else 0.0
        self.best_snr_radius = moon_radius
        # H15: the radius comes from the amplitude, so a dip deeper than the searched range is
        # reported as measured and flagged, not limited to max_radius.
        self.best_snr_radius_above_range = bool(moon_radius > self.max_radius)
        self.best_snr_radius_below_range = bool(0 < moon_radius < self.min_radius)
        if self.best_snr_radius_below_range:
            logging.warning("Best SNR radius %.2f R_earth is below the searched range (min %.2f R_earth): the "
                            "dip is shallower than the smallest moon searched for", moon_radius, self.min_radius)
        self.best_snr_radius_above_tidal_survival = bool(moon_radius > self.max_surviving_moon_radius)
        self.best_snr_radius_above_tidal_survival_retrograde = bool(moon_radius >
                                                                    self.max_surviving_moon_radius_retrograde)
        margin = f"Barnes & O'Brien 2002, {TIDAL_SURVIVAL_AGE_GYR:.0f} Gyr, k2/Q x {TIDAL_SURVIVAL_K2_Q_FACTOR:.1f}"
        if self.best_snr_radius_above_tidal_survival and (not self.search_retrograde or
                                                          self.best_snr_radius_above_tidal_survival_retrograde):
            logging.warning("Best SNR radius %.2f R_earth is above the largest moon that survives tidal migration, "
                            "%.2f R_earth prograde%s even with a generous margin (%s): such a moon is unlikely, though "
                            "the bound is an order-of-magnitude estimate", moon_radius, self.max_surviving_moon_radius,
                            f" and {self.max_surviving_moon_radius_retrograde:.2f} R_earth retrograde"
                            if self.search_retrograde else "", margin)
        elif self.best_snr_radius_above_tidal_survival:
            logging.warning("Best SNR radius %.2f R_earth is above the largest prograde moon that survives tidal "
                            "migration, %.2f R_earth with a generous margin (%s), but below the retrograde bound "
                            "%.2f R_earth: it is plausible as a captured retrograde moon, whose orbit sense EMLS "
                            "cannot tell", moon_radius, self.max_surviving_moon_radius, margin,
                            self.max_surviving_moon_radius_retrograde)
        if self.best_snr_radius_above_range:
            logging.warning("Best SNR radius %.2f R_earth is above the searched range (max %.2f R_earth): the "
                            "dip is deeper than any moon searched for, for example a planet residual or a "
                            "stellar companion", moon_radius, self.max_radius)
        # The SDE of the SNR spectrum at the selected period, a diagnostic only (audit item H7).
        snr_spectrum_std = np.nanstd(snr_spectrum - snr_median)
        self.sde = float((snr_spectrum[best_snr_idx] - snr_median[best_snr_idx]) / snr_spectrum_std) \
            if snr_spectrum_std > 0 and np.isfinite(snr_spectrum[best_snr_idx]) else None
        half_orbit_dur_max = best_input.half_moon_max_orbit_transit_duration
        half_orbit_dur_rec = best_input.half_moon_orbit_transit_duration
        half_moon_dur = best_input.half_moon_transit_duration
        rec_t0s, inj_t0s = [], []
        has_injected = hasattr(self, 'injected_flux_moon_pandora')
        for t0_i in self.bary_t0s:
            max_tau = ExoMoonLeastSquares.compute_moon_ts(t0_i, half_orbit_dur_rec, half_moon_dur)
            rec_t0, _ = ExoMoonLeastSquares.compute_moon_t0(
                self.planet_t0, t0_i, best_snr_period, best_snr_alpha, max_tau)
            rec_t0s.append(rec_t0)
            if has_injected:
                mask_w = np.abs(self.inject_flux_time - t0_i) < half_orbit_dur_max
                inj_flux = self.injected_flux_moon_pandora[mask_w]
                # One entry per epoch, so injected and recovered times stay paired by epoch.
                inj_t0s.append(self.inject_flux_time[mask_w][np.argmin(inj_flux)] if np.any(inj_flux < 1.0) else None)
        self.recovered_t0s = rec_t0s
        self.assess_planet_overlap(rec_t0s, half_orbit_dur_rec, half_moon_dur)
        self.injected_t0s = inj_t0s if any(t0 is not None for t0 in inj_t0s) else None
        self.recovered_period = float(best_snr_period)
        self.recovered_alpha = float(best_snr_alpha)

        if inject_params is not None:
            period_err = abs(best_snr_period - inject_params['moon_period']) / inject_params['moon_period']
            alpha_err = abs(np.cos(best_snr_alpha) - np.cos(inject_params['initial_alpha']))
            period_match, alias_match, alias_k, alias_period, min_alias_err, alias_mirror = \
                ExoMoonLeastSquares.classify_recovered_period(inject_params['moon_period'], self.planet_period,
                                                              best_snr_period)
            self.injection_alias_k = alias_k
            self.injection_alias_mirror = alias_mirror
            logging.info("INJECTION RECOVERY (SNR): injected P=%.4f d, alpha=%.4f rad -> recovered P=%.4f d, alpha=%.4f rad (delta_P=%.2f%%, delta_cos-alpha=%.4f)",
                         inject_params['moon_period'], inject_params['initial_alpha'],
                         best_snr_period, best_snr_alpha, period_err * 100, alpha_err)
            if alias_k is not None and (alias_k != 0 or alias_mirror):
                logging.info("  Best alias match: %s k=%d, P_alias=%.4f d (delta=%.2f%%)",
                             "mirror" if alias_mirror else "sampling", alias_k, alias_period, min_alias_err * 100)
            self.recovered_period = float(best_snr_period)
            self.recovered_alpha = float(best_snr_alpha)
            # I1: recovered needs a detection, the moon times within half a moon transit, and the
            # period; a sampling alias with matching times is reported apart. The period counts as
            # recovered only when the injected period itself (k = 0) is the nearest alias.
            self.injection_period_error = float(period_err)
            self.injection_timing_error = None
            timing_match = None
            if self.injected_t0s and len(rec_t0s) > 0:
                diffs = np.abs([rec - inj for rec, inj in zip(rec_t0s, self.injected_t0s) if inj is not None])
                self.injection_timing_error = float(np.median(diffs))
                timing_match = bool(self.injection_timing_error < half_moon_dur)
            pending_recovery = (timing_match, period_match, alias_match)

        self.max_snr = float(selected_snr)
        self.max_sde = float(SDE)
        self.aliases = self.alias_family(period_grid, snr_spectrum, best_snr_period)
        close = [alias for alias in self.aliases if (alias["k"] != 0 or alias["mirror"])
                 and alias["snr_below_best"] is not None and alias["snr_below_best"] < 1]
        if close:
            logging.info("Period aliases within 1 SNR of the best (P=%.4f d): %s; the period is not identified "
                         "among them (audit item H9)", best_snr_period,
                         ", ".join(f"{'mirror ' if alias['mirror'] else ''}k={alias['k']} P={alias['period']:.4f} d "
                                   f"SNR {alias['snr']:.2f}"
                                   for alias in close[:5]))
        self.absorbed_fraction = self.measure_absorbed_fraction(before_planet_fit, best_input, rec_t0s)
        self.assess_single_event(best_input)
        if null_snrs is not None:
            self.null_snrs = [float(value) for value in null_snrs]
        elif self.null_runs > 0:
            self.null_snrs = self.run_nulls(before_trim, period_grid, self.max_snr)
        self.assess_false_alarm(self.max_snr)
        if self.fap is None:
            logging.info("No false-alarm probability: SNR %.2f at P=%.4f d, alpha=%.3f rad is a search statistic "
                         "without a calibrated significance%s", self.max_snr, best_snr_period, best_snr_alpha,
                         "" if self.null_runs or null_snrs is not None else " (no null runs requested)")
        else:
            logging.info("SNR %.2f at P=%.4f d, alpha=%.3f rad: false-alarm probability %.2g (Gumbel fit to %d "
                         "nulls; %d%% bootstrap upper value %.2g; empirical %.2g), %s at the %.2g threshold%s",
                         self.max_snr, best_snr_period, best_snr_alpha, self.fap, len(self.null_snrs),
                         FAP_UPPER_PERCENTILE, self.fap_upper, self.fap_empirical,
                         "detected" if self.detected else "not detected", self.fap_threshold,
                         "; below 1/(N+1), so extrapolated past the largest null: confirm with more null runs "
                         "(--null_runs 1000 measures 0.01 directly)" if self.fap_extrapolated else "")
            if self.null_fit_rejected:
                logging.warning("An Anderson-Darling test rejects the Gumbel fit to the %d null SNRs at 5 %%, so "
                                "the extrapolated false-alarm probability is less reliable", len(self.null_snrs))
        if self.scramble_runs > 0 and np.isfinite(self.max_snr) and (
                self.scramble_trigger_fap is None or (self.fap is not None and self.fap < self.scramble_trigger_fap)):
            self.scramble_snrs = self.run_scrambled_nulls(self.max_snr)
            self.assess_scrambled_false_alarm(self.max_snr)
            if self.scramble_fap is not None:
                logging.log(logging.INFO if self.scramble_passed else logging.WARNING,
                            "Scrambled-phase nulls: false-alarm probability %.2g (%d%% bootstrap upper value %.2g; "
                            "empirical %.2g over %d runs) -> %s. They keep the real planet residuals, so a signal "
                            "they reproduce does not need a moon%s", self.scramble_fap, FAP_UPPER_PERCENTILE,
                            self.scramble_fap_upper, self.scramble_fap_empirical, len(self.scramble_snrs),
                            "passes" if self.scramble_passed else "fails",
                            "" if self.period_constrained is not False else "; the period is not constrained "
                            "(audit item G17), where even a real moon tends to fail")
        elif self.scramble_runs > 0:
            logging.info("Scrambled-phase nulls not run: false-alarm probability %s is not below the trigger %s",
                         None if self.fap is None else f"{self.fap:.2g}", self.scramble_trigger_fap)
        if inject_params is not None:
            timing_match, period_match, alias_match = pending_recovery
            if self.detected is None:
                self.recovered = self.recovered_alias = None
                logging.info("Injection recovery undetermined: recovery needs a false-alarm probability (null runs)")
            else:
                times_ok = timing_match if timing_match is not None else True
                self.recovered = bool(self.detected and times_ok and period_match)
                self.recovered_alias = bool(self.detected and times_ok and not period_match and alias_match)
                logging.info("Injection recovery: detected=%s, moon times %s (median error %s d, tolerance %.4f d), "
                             "period match=%s, alias=%s (nearest alias k=%s) -> recovered=%s", self.detected,
                             "match" if timing_match else ("not checked" if timing_match is None else "differ"),
                             None if self.injection_timing_error is None else round(self.injection_timing_error, 4),
                             half_moon_dur, period_match, alias_match, self.injection_alias_k, self.recovered)

        try:
            self.write_figures(period_grid, alpha_grid, snr_matrix, snr_spectrum, best_residuals_spectra, power,
                               snr_sde_spectrum, best_snr_period, best_snr_alpha, selected_snr, best_input, rec_t0s,
                               inj_t0s if has_injected else None, inject_params)
        except Exception as e:
            logging.warning("Could not write the EMLS figures: %s", e)
