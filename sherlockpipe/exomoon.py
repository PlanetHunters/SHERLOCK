import copy
import itertools
import logging
import math
import os
import sys
import time
import tempfile
import warnings
from multiprocessing import Pool
import astropy.units as u
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from scipy.optimize import least_squares, minimize
from numba import njit
from pytransit import QuadraticModel
import batman
from tqdm import tqdm
import pandoramoon as pandora
import ellc
import numpy as np
import astropy.constants as ac
import astropy.units as u
import wotan
from lcbuilder.lcbuilder_class import LcBuilder
from scipy import stats
from scipy.signal import lombscargle, argrelextrema
import matplotlib.pyplot as plt
import pandas as pd
from sherlockpipe.search.sherlock import Sherlock
from sherlockpipe.system_stability.mr_forecast import MrForecast

G = ac.G.value  # m3 kg-1 s-2
AU_TO_M = ac.au.value
AU_TO_RSUN = (ac.au / ac.R_sun).value
Msolar_to_kg = ac.M_sun.value
Mearth_to_kg = ac.M_earth.value
M_earth_to_M_sun = Mearth_to_kg / Msolar_to_kg
R_earth_to_R_sun = (ac.R_earth / ac.R_sun).value
EARTH_DENSITY = Mearth_to_kg / (4 / 3 * np.pi * ac.R_earth.value ** 3) / 1000  # g/cm3
MIN_RP_OVER_RSTAR = 1e-5
MIN_NULL_RUNS = 20
MIN_INDEPENDENT_OFFSETS = 3
MOON_MASS_FORECAST_SEED = 0
_moon_mass_cache = {}


def forecast_moon_mass(radius, max_density):
    """Return a realistic mass in Earth masses for a moon of the given radius in Earth radii.

    The median of the Chen & Kipping (2017) mass-radius forecast, capped at the mass of a body of
    that radius at max_density in g/cm3. The forecast uses a fixed seed and the global numpy random
    state is restored. A radius outside the forecaster range falls back to the density cap.
    """
    cap = max_density / EARTH_DENSITY * radius ** 3
    key = float(radius)
    if key not in _moon_mass_cache:
        state = np.random.get_state()
        try:
            np.random.seed(MOON_MASS_FORECAST_SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                forecast = MrForecast.Rstat2M(key, 1e-3 * key)
        finally:
            np.random.set_state(state)
        _moon_mass_cache[key] = None if forecast is None else float(forecast[0])
    forecast = _moon_mass_cache[key]
    return cap if forecast is None else min(forecast, cap)


class ExposureAveragedModel:
    """batman transit model averaged over each exposure with the midpoint rule.

    batman's own supersample_factor converges slowly at long cadence (about 80 ppm off at 15
    sub-exposures over 30 minutes for a 0.25 % transit), while midpoint averaging is within about
    5 ppm. It exposes the same light_curve(params) interface as batman.TransitModel.
    """

    def __init__(self, transit_params, time, exposure_time, sub_exposures):
        time = np.asarray(time, dtype=float)
        offsets = ((np.arange(sub_exposures) + 0.5) / sub_exposures - 0.5) * exposure_time
        self.shape = (len(time), sub_exposures)
        self.model = batman.TransitModel(transit_params, (time[:, None] + offsets[None, :]).ravel())

    def light_curve(self, transit_params):
        return self.model.light_curve(transit_params).reshape(self.shape).mean(axis=1)


class NoUsableEpochsError(ValueError):
    """No planet transit epoch is left to search for a moon."""


class ExoMoonLeastSquares:
    RECOVERY_PERIOD_TOLERANCE = 0.05
    MAX_ALIAS_ORDER = 30
    SPEED_MISMATCH = 0.1
    SINGLE_EVENT_SNR_FRACTION = 0.5
    """Performs exomoon least-squares search and fitting."""
    MAX_RESIDUAL = sys.maxsize
    SPECTRUM_KERNEL = 31
    TEMPLATE_MIN_SAMPLES = 1000
    SUBEXPOSURE_DAYS = 2 / 1440

    def __init__(self, object_dir, cpus, star_mass, star_radius, ab, planet_radius, planet_radius_err,
                 planet_mass, planet_period,
                 planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc,
                 planet_ecc_err, planet_arg_periastron, planet_impact_param, min_radius, max_radius, t0s, time, flux,
                 flux_err, max_moon_density, period_grid_size=10000, radius_grid_size=10,
                 min_cadences_per_t0: bool = 20, min_orbit_window: float = 0,
                  subtract_mode: str = "subtract", mask_duration_factor: float = None,
                  skip_fold_fit: bool = False, fit_rp_per_transit: bool = True,
                  min_transit_completeness: float = 0.5,
                  max_subtraction_rms_ratio: float = 3.0, exposure_time: float = None,
                  ttv_prior_width="auto", planet_t0_err=None, planet_period_err=None, null_runs=0,
                  fap_threshold=0.01, null_seed=0, detrend_window=None, duration_from_fit=True):
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
        exposure_time : float, optional
            Exposure time of each data point in days. The planet fits and the moon
            templates are averaged over it. Default None, an instantaneous model.
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
            default 0.01.
        null_seed : int, optional
            Seed of the pseudo-epoch offsets, default 0.
        detrend_window : float, optional
            Biweight window in days. When given, the light curve is taken as undetrended and
            `run` detrends it with every moon window masked (audit item A1), after any injected
            moon is added (audit item A2). Default None: the curve is already detrended.
        duration_from_fit : bool, optional
            True (default) when planet_duration and planet_impact_param come from a transit fit,
            such as a catalog: their speed across the star sets the moon offsets. False for a
            search duration, such as a TLS one: the offsets use 2 pi a / P with e and w until the
            fold fit, then its T14 and b.
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
        self.fit_rp_per_transit = fit_rp_per_transit
        self.min_transit_completeness = min_transit_completeness
        self.max_subtraction_rms_ratio = max_subtraction_rms_ratio
        if exposure_time is not None and not (np.isfinite(exposure_time) and 0 < exposure_time <= 1):
            raise ValueError(f"exposure_time must be a positive number of days up to 1, got {exposure_time}")
        self.exposure_time = exposure_time
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
        self.null_runs = int(null_runs)
        self.fap_threshold = float(fap_threshold)
        self.null_seed = null_seed
        self.null_snrs = None
        self.null_independent_offsets = None
        self.fap = None
        self.fap_empirical = None
        self.detected = None
        self.sde = None
        self.epoch_stats = None
        self.loo_snr_min = None
        self.single_event = None
        self.recovered = None
        self.recovered_alias = None
        self.injection_alias_k = None
        self.planet_rp_over_rstar = None
        self.fold_realigned = False
        self.model_validity = None
        self.planet_overlap_fraction = None
        self.planet_overlap_mean = None
        self.best_tau = None
        self.period_constrained = None
        self.planet_shape_check = None
        self.mask_mode_survival = None
        self.aliases = None
        self.red_noise_factor = 1.0
        self.red_noise_measured = None
        self.fold_fit_status = None
        self.planet_a_over_rstar = None
        self.absorbed_fraction = None
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
        if period_grid_size < ExoMoonLeastSquares.SPECTRUM_KERNEL:
            raise ValueError(f"period_grid_size must be at least {ExoMoonLeastSquares.SPECTRUM_KERNEL}, "
                             f"got {period_grid_size}")
        self.period_grid_size = period_grid_size
        self.radius_grid_size = radius_grid_size
        self.planet_mass_grid = [planet_mass]

    @staticmethod
    def compute_semimajor_axis(major_mass, minor_period):
        """Compute the semimajor axis from mass and orbital period.

        Parameters
        ----------
        major_mass : float
            Mass of the central body in solar masses.
        minor_period : float
            Orbital period in days.

        Returns
        -------
        float
            Semimajor axis in AU.
        """
        period_seconds = minor_period * 24. * 3600.
        mass_kg = major_mass * Msolar_to_kg
        a1 = (G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / AU_TO_M

    @staticmethod
    def compute_roche_period(moon_density):
        """Compute the Roche limit orbital period for a given moon density.

        Parameters
        ----------
        moon_density : float
            Moon density in g/cm^3.

        Returns
        -------
        float
            Roche limit period in days.
        """
        return np.sqrt((3 * np.pi) / (G * moon_density * 1000)) * 2.44 ** (3 / 2) / 3600 / 24

    @staticmethod
    def compute_min_moon_period(moon_density, planet_mass, planet_radius, max_moon_radius):
        """Return the shortest moon period that is outside both the Roche limit and the planet.

        Parameters
        ----------
        moon_density : float
            Moon density in g/cm^3.
        planet_mass : float
            Planet mass in Earth masses.
        planet_radius : float
            Planet radius in Earth radii.
        max_moon_radius : float
            Largest moon radius searched in Earth radii.

        Returns
        -------
        float
            Minimum moon period in days.
        """
        contact_au = (planet_radius + max_moon_radius) * R_earth_to_R_sun / AU_TO_RSUN
        contact_period = ExoMoonLeastSquares.au_to_period(planet_mass * M_earth_to_M_sun, contact_au)
        return max(ExoMoonLeastSquares.compute_roche_period(moon_density), contact_period)

    @staticmethod
    def compute_roche_radius(planet_radius, planet_mass, moon_density):
        """Return the fluid Roche limit in the units of planet_radius.

        Parameters
        ----------
        planet_radius : float
            Planet radius in Earth radii.
        planet_mass : float
            Planet mass in Earth masses.
        moon_density : float
            Moon density in g/cm^3.

        Returns
        -------
        float
            Roche radius in Earth radii.
        """
        planet_density = EARTH_DENSITY * planet_mass / planet_radius ** 3
        return 2.44 * planet_radius * (planet_density / moon_density) ** (1 / 3)

    @staticmethod
    def compute_hill_radius(major_mass, minor_mass, semimajor_axis, eccentricity=0):
        """
        :param major_mass: The main body mass
        :param minor_mass: The minor body mass
        :param semimajor_axis: The minor body semimajor axis in AU.
        :param eccentricity: the planet eccentricity
        :return: the hill radius of the minor body in the same units than the semimajor_axis
        """
        return semimajor_axis * (1 - eccentricity) * ((minor_mass / (3 * major_mass)) ** (1 / 3))

    # Domingos, Winter & Yokoyama (2006), prograde: a_max = 0.4895 (1 - 1.0305 e_p) r_H, with r_H
    # without the (1 - e_p) factor, as quoted by Rosario-Franco et al. (2020, Table 1) and
    # Heller et al. (2014).
    DOMINGOS_PROGRADE = (0.4895, 1.0305)

    @staticmethod
    def max_moon_semimajor_axis(star_mass, planet_mass, planet_semimajor_axis, planet_ecc=0.0):
        """Return the widest stable prograde moon orbit in AU (audit items G5, G6).

        a_max = 0.4895 (1 - 1.0305 e_p) r_H (Domingos et al. 2006), with the Hill radius of a
        circular orbit, so the eccentricity is counted once: 0.49 r_H for a circular planet orbit
        and 0.34 r_H at e_p = 0.3.

        Parameters
        ----------
        star_mass, planet_mass : float
            Masses in solar masses.
        planet_semimajor_axis : float
            Planet semi-major axis in AU.
        planet_ecc : float, optional
            Planet eccentricity, default 0.

        Returns
        -------
        float
            Semi-major axis in AU.
        """
        coefficient, slope = ExoMoonLeastSquares.DOMINGOS_PROGRADE
        factor = coefficient * (1 - slope * planet_ecc)
        if not factor > 0:
            raise ValueError(f"No stable prograde moon orbit for a planet eccentricity of {planet_ecc}")
        return factor * ExoMoonLeastSquares.compute_hill_radius(star_mass, planet_mass, planet_semimajor_axis)

    def check_model_validity(self):
        """Log where the screening moon model stops holding (audit items G3, G4, G14, G16).

        The moon model is coplanar, circular and prograde, with the moon phase frozen at the
        planet mid-transit, one transit duration, a planet-only (not barycentric) moon orbit and
        no mutual planet-moon events. A photodynamical fit (Pandora) should confirm and refine a
        candidate. This logs, for the widest searched moons:

        - epsilon = tau v_m / (v_p T), the scale of the timing and duration errors in units of the
          planet duration T (G3, G4): small up to Neptune mass, about 0.5 for wide moons of
          Jupiter at 300 d, where the audit measured median timing errors of 0.15-0.19 T;
        - q = M_m / M_p for the Chen & Kipping mass of the largest searched radius (G14): the
          offsets are off by 1 - (1 + q)^(-2/3), negligible below q = 0.01.

        Sets model_validity to {"epsilon", "q", "offset_error"}.
        """
        planet_mass = np.max(self.planet_mass_grid)
        mass = planet_mass * M_earth_to_M_sun
        widest = self.max_moon_period(planet_mass)
        moon_a = self.compute_semimajor_axis(mass, widest)
        tau = self.moon_offset(moon_a)
        speed_ratio = 2 * np.pi * moon_a * AU_TO_RSUN / widest / self.planet_speed()
        epsilon = float(tau * speed_ratio / self.planet_duration)
        q = forecast_moon_mass(self.max_radius, self.max_moon_density) / planet_mass
        offset_error = float(1 - (1 + q) ** (-2 / 3))
        self.model_validity = {"epsilon": epsilon, "q": float(q), "offset_error": offset_error}
        if epsilon > 0.1:
            logging.warning("Moon timing model: for the widest searched moons (P = %.2f d) the moon moves during the "
                            "planet-moon offset, so moon transit times and durations can be off by up to about %.2f "
                            "planet durations (tau v_m / (v_p T), audit items G3, G4). Confirm candidates with a "
                            "photodynamical fit", widest, epsilon)
        else:
            logging.info("Moon timing model valid for the searched moons: tau v_m / (v_p T) = %.3f", epsilon)
        if q > 0.01:
            logging.warning("The largest searched moons have q = M_m / M_p = %.3f (Chen & Kipping mass), so the "
                            "planet-only moon orbit puts their offsets %.1f%% too far (audit item G14)", q,
                            offset_error * 100)

    def assess_planet_overlap(self, moon_t0s, half_orbit_duration, half_moon_duration):
        """Measure how much the best moon window sits on the planet transit (audit item G17).

        Sets planet_overlap_fraction, the share of epochs whose moon window overlaps the planet
        transit at all, planet_overlap_mean, the mean fraction of the moon window inside it, best_tau, the largest moon offset of the best period, and
        period_constrained, False when that offset is below T/4: then the moon window barely
        moves between periods, the signal rests on the planet subtraction, and the period is not
        constrained by the data even if the dip is detected.
        """
        overlaps = []
        for t0, moon_t0 in zip(self.bary_t0s, moon_t0s):
            start = max(moon_t0 - half_moon_duration, t0 - self.planet_duration / 2)
            end = min(moon_t0 + half_moon_duration, t0 + self.planet_duration / 2)
            overlaps.append(max(0.0, end - start) / (2 * half_moon_duration) if half_moon_duration > 0 else 1.0)
        self.planet_overlap_fraction = float(np.mean(np.array(overlaps) > 0)) if overlaps else None
        self.planet_overlap_mean = float(np.mean(overlaps)) if overlaps else None
        self.best_tau = float(half_orbit_duration - half_moon_duration)
        self.period_constrained = bool(self.best_tau >= self.planet_duration / 4)
        logging.log(logging.INFO if self.period_constrained else logging.WARNING,
                    "Best moon window: overlaps the planet transit in %.0f%% of the epochs (%.0f%% of the window on "
                    "average), offset up to %.3f d (T14 %.3f d)%s", 100 * (self.planet_overlap_fraction or 0),
                    100 * (self.planet_overlap_mean or 0), self.best_tau, self.planet_duration,
                    "" if self.period_constrained else "; below T14/4, so the period is not constrained by the "
                    "data and the signal rests on the planet subtraction (audit item G17)")

    def conjunction_time_factor(self):
        """Return sqrt(1 - e^2) / (1 + e sin w), the transit-time scale of an eccentric planet orbit (audit item G11).

        The planet moves faster than on a circular orbit by the inverse of this factor at
        conjunction, so the moon transit duration and the moon offset a_m / v_p scale with it.
        """
        ecc = self.planet_ecc or 0.0
        return float(np.sqrt(1 - ecc ** 2) / (1 + ecc * np.sin(np.deg2rad(self.planet_arg_periastron or 0.0))))

    def planet_speed(self):
        """Return the planet speed across the star at mid-transit, in solar radii per day.

        It comes from the input (catalog) transit fit: the planet crosses the chord
        2 R* sqrt((1 + Rp/R*)^2 - b^2) in the transit duration T14. With a search duration
        (duration_from_fit False) the fold fit's T14 and b are used once it has run. That speed is what the
        catalog fit measured, whatever the eccentricity, while 2 pi a / P times
        `conjunction_time_factor` needs e and w, which catalogs rarely fit. If the inputs give no
        chord or duration, the Keplerian speed with e and w is used instead.
        """
        k = self.planet_radius * R_earth_to_R_sun / self.star_radius
        if self.duration_from_fit:
            b, duration = self.catalog_impact_param or 0.0, self.catalog_planet_duration
        elif self.planet_a_over_rstar is not None:
            # A search duration is no T14: after the fold fit, its T14 and b measure the speed.
            b, duration = self.planet_impact_param or 0.0, self.planet_duration
            k = self.planet_rp_over_rstar or k
        else:
            return self.keplerian_planet_speed()
        chord_sq = (1 + k) ** 2 - b ** 2
        if duration is not None and duration > 0 and chord_sq > 0:
            return 2 * self.star_radius * np.sqrt(chord_sq) / duration
        return self.keplerian_planet_speed()

    def keplerian_planet_speed(self):
        """Return 2 pi a / P over `conjunction_time_factor`, the speed the input a, e and w imply, in R_sun per day."""
        return 2 * np.pi * self.planet_semimajor_axis * AU_TO_RSUN / self.planet_period / \
            self.conjunction_time_factor()

    def moon_offset(self, moon_semimajor_axis):
        """Return the largest moon-planet transit time offset a_m / v_p in days, a_m in AU."""
        return moon_semimajor_axis * AU_TO_RSUN / self.planet_speed()

    def log_planet_speed(self):
        """Log the planet speed from the transit duration and how far the input a, e and w are from it."""
        if self._speed_logged:
            return
        self._speed_logged = True
        speed, keplerian = self.planet_speed(), self.keplerian_planet_speed()
        ratio = keplerian / speed
        level = logging.WARNING if abs(ratio - 1) > ExoMoonLeastSquares.SPEED_MISMATCH else logging.INFO
        if self.duration_from_fit:
            source = "the input T14 %.4f d and b %.3f" % (self.catalog_planet_duration, self.catalog_impact_param or 0.0)
        elif self.planet_a_over_rstar is not None:
            source = "the fold-fitted T14 %.4f d and b %.3f (the input duration is a search duration)" % (
                self.planet_duration, self.planet_impact_param or 0.0)
        else:
            source = "2 pi a / P with e and w until the fold fit (the input duration is a search duration)"
        logging.log(level, "Planet speed across the star from %s: %.3f R_sun/d; the input a, e = %.2f and w = %.0f deg "
                    "imply %.3f R_sun/d (ratio %.2f)%s. Moon offsets a_m / v_p use the transit speed", source, speed,
                    self.planet_ecc or 0.0, self.planet_arg_periastron or 0.0, keplerian, ratio,
                    ", so the input eccentricity or periastron does not match the transit" if level == logging.WARNING
                    else "")

    def injection_a_over_rstar(self):
        """Return the a / R* for the Pandora injection, so that its planet crosses the star at `planet_speed`.

        Pandora moves the planet at 2 pi a / P over `conjunction_time_factor` at conjunction (same e and w
        convention), so a = v_p P ctf / (2 pi) gives the injected planet the input T14 and b, and the
        injected moon the offsets a_m / v_p that the search assumes. With the input a, e and w the injected
        moon would orbit a planet faster or slower than the one in the light curve. Without a transit speed
        (no chord or duration) this is the input a.
        """
        input_a = self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius
        a_over_rstar = self.planet_speed() / self.star_radius * self.planet_period * \
            self.conjunction_time_factor() / (2 * np.pi)
        ratio = a_over_rstar / input_a
        level = logging.WARNING if abs(ratio - 1) > ExoMoonLeastSquares.SPEED_MISMATCH else logging.INFO
        logging.log(level, "Pandora injection a/R* %.2f from the transit planet speed; the input a gives %.2f "
                    "(ratio %.2f)%s", a_over_rstar, input_a, ratio,
                    ", so the input a, e or w does not match the transit" if level == logging.WARNING else "")
        return float(a_over_rstar)

    def check_fitted_duration(self):
        """Warn when the fold-fitted T14 differs from the input one, which sets the planet speed."""
        self._speed_logged = False  # the speed may change with the fitted geometry, so it is logged again
        if not self.duration_from_fit or self.catalog_planet_duration is None or self.catalog_planet_duration <= 0:
            return
        difference = self.planet_duration / self.catalog_planet_duration - 1
        level = logging.WARNING if abs(difference) > ExoMoonLeastSquares.SPEED_MISMATCH else logging.INFO
        logging.log(level, "Fitted T14 %.4f d against the input %.4f d (%+.1f%%)%s", self.planet_duration,
                    self.catalog_planet_duration, 100 * difference,
                    "; the moon offsets use the input duration, so check it" if level == logging.WARNING else "")

    def max_moon_period(self, planet_mass):
        """Return the period in days of the widest stable prograde moon orbit for a planet mass in Earth masses."""
        mass = planet_mass * M_earth_to_M_sun
        return self.au_to_period(mass, self.max_moon_semimajor_axis(self.star_mass, mass, self.planet_semimajor_axis,
                                                                    self.planet_ecc))

    @staticmethod
    def au_to_period(mass, au):
        """
        Calculates the orbital period for the semi-major axis assuming a circular orbit.
        :param mass: the stellar mass
        :param au: the semi-major axis in astronomical units.
        :return: the period in days
        """
        mass_kg = mass * Msolar_to_kg
        a = au * AU_TO_M
        return ((a ** 3) * 4 * (np.pi ** 2) / G / mass_kg) ** (1. / 2.) / 3600 / 24

    @staticmethod
    def min_period_for_orbit_duration(star_radius, planet_semimajor_axis,
                                      planet_period, planet_mass,
                                      orbit_duration_days, impact_parameter=0, moon_radius=0):
        """Minimum moon period with orbit transit duration >= orbit_duration_days.

        Inverts `compute_moon_orbit_transit_duration`: the span is twice the maximum
        offset a_m / v_p plus one moon transit, so a_m = (D - T_moon) * pi * a_p / P_p.

        Parameters
        ----------
        star_radius : float
            Stellar radius in solar radii.
        planet_semimajor_axis : float
            Planet semi-major axis in solar radii (a/R_star).
        planet_period : float
            Planet orbital period in days.
        planet_mass : float
            Planet mass in solar masses.
        orbit_duration_days : float
            Desired minimum moon orbit transit duration in days.
        impact_parameter : float, optional
            Planet impact parameter, default 0.
        moon_radius : float, optional
            Moon radius in solar radii, default 0.

        Returns
        -------
        float
            Minimum moon orbital period in days. Returns 0 when the moon transit alone
            already covers orbit_duration_days.
        """
        planet_a_rsun = planet_semimajor_axis * star_radius
        moon_transit = ExoMoonLeastSquares.compute_transit_duration(star_radius, planet_a_rsun, planet_period,
                                                                    moon_radius, impact_parameter)
        a_moon_rsun = (orbit_duration_days - moon_transit) * np.pi * planet_a_rsun / planet_period
        if a_moon_rsun <= 0:
            return 0.0
        return ExoMoonLeastSquares.au_to_period(planet_mass, a_moon_rsun / AU_TO_RSUN)

    @staticmethod
    def compute_transit_duration(star_radius,
                                 transiting_body_semimajor_axis, transit_period, transiting_body_radius,
                                 impact_parameter=0):
        """

        :param star_radius: star radius
        :param transiting_body_semimajor_axis: orbit semimajor axis
        :param transit_period: in days
        :param transiting_body_radius: transiting body radius
        :param impact_parameter:
        :return:
        """
        argument = (
            np.sqrt(max(0, (star_radius + transiting_body_radius) ** 2
                        - (impact_parameter * star_radius) ** 2))
            / transiting_body_semimajor_axis
        )
        argument = np.clip(argument, -1.0, 1.0)
        return transit_period / np.pi * np.arcsin(argument)
        #return 2 * moon_semimajor_axis / (planet_semimajor_axis * 2 * np.pi) * planet_period

    @staticmethod
    def compute_max_moon_offset(moon_semimajor_axis, planet_semimajor_axis, planet_period):
        """Return the largest time offset between the moon and planet transits.

        At greatest elongation the moon sits a_m ahead of or behind the planet along its
        path, so it crosses the star a_m / v_p earlier or later, whatever the impact
        parameter. v_p = 2 pi a_p / P_p is the circular planet velocity.

        Parameters
        ----------
        moon_semimajor_axis : float
            Moon semimajor axis, in the same length unit as planet_semimajor_axis.
        planet_semimajor_axis : float
            Planet semimajor axis.
        planet_period : float
            Planet period in days.

        Returns
        -------
        float
            Maximum offset in days.
        """
        return moon_semimajor_axis * planet_period / (2 * np.pi * planet_semimajor_axis)

    @staticmethod
    def compute_moon_orbit_transit_duration(star_radius, planet_semimajor_axis, planet_period, moon_semimajor_axis,
                                            moon_radius, impact_parameter=0, transit_semimajor_axis=None):
        """Return the time span that moon transits can cover around the planet transit.

        It is twice the maximum offset plus one moon transit, so half of it minus half
        a moon transit gives back the maximum offset (see `compute_moon_ts`).

        Parameters
        ----------
        star_radius : float
            Stellar radius in solar radii.
        planet_semimajor_axis : float
            Planet semimajor axis in solar radii.
        planet_period : float
            Planet period in days.
        moon_semimajor_axis : float
            Moon semimajor axis in solar radii.
        moon_radius : float
            Moon radius in solar radii.
        impact_parameter : float, optional
            Planet impact parameter, default 0.
        transit_semimajor_axis : float, optional
            Semi-major axis in solar radii for the moon transit duration, such as the fitted
            a/R* times R*. Defaults to planet_semimajor_axis, which always sets the offset.

        Returns
        -------
        float
            Duration in days.
        """
        moon_transit_duration = ExoMoonLeastSquares.compute_transit_duration(
            star_radius, planet_semimajor_axis if transit_semimajor_axis is None else transit_semimajor_axis,
            planet_period, moon_radius, impact_parameter)
        max_offset = ExoMoonLeastSquares.compute_max_moon_offset(moon_semimajor_axis, planet_semimajor_axis,
                                                                 planet_period)
        return 2 * max_offset + moon_transit_duration

    @staticmethod
    def compute_moon_period_grid(min, max, mode="lin", samples=10000):
        """Generate a grid of moon orbital periods.

        Parameters
        ----------
        min : float
            Minimum period in days.
        max : float
            Maximum period in days.
        mode : str, optional
            Grid spacing mode, "lin" or "log", default "lin".
        samples : int, optional
            Number of grid samples, default 10000.

        Returns
        -------
        numpy.ndarray
            Array of period values.
        """
        if min >= max:
            raise ValueError(f"No valid moon period range: minimum {min} d is not below maximum {max} d")
        if "log" == mode:
            return np.logspace(math.log(min, 10), math.log(max, 10), samples, base=10)
        else:
            return np.linspace(min, max, samples)

    @staticmethod
    def transit_model(transit_params, time, exposure_time=None):
        """Build a batman model, averaged over the exposure when one is given.

        The exposure is split into sub-exposures of at most SUBEXPOSURE_DAYS and averaged at their
        midpoints, so 2-minute data stay instantaneous and 30-minute data use 15 sub-exposures
        (Kipping 2010, MNRAS 408, 1758).

        Parameters
        ----------
        transit_params : batman.TransitParams
            Transit parameters.
        time : array_like
            Mid-exposure times in days.
        exposure_time : float, optional
            Exposure time in days. None or a value up to SUBEXPOSURE_DAYS gives an instantaneous model.

        Returns
        -------
        batman.TransitModel or ExposureAveragedModel
            An object with a light_curve(params) method.
        """
        if exposure_time is None or exposure_time <= ExoMoonLeastSquares.SUBEXPOSURE_DAYS:
            return batman.TransitModel(transit_params, time)
        sub_exposures = int(np.ceil(exposure_time / ExoMoonLeastSquares.SUBEXPOSURE_DAYS))
        return ExposureAveragedModel(transit_params, time, exposure_time, sub_exposures)

    @staticmethod
    def pandora_light_curve(model, time, exposure_time=None):
        """Evaluate a Pandora moon model, averaged over the exposure like `transit_model`.

        Pandora's own supersampling expects supersampled input times, so the exposure is
        averaged here at the midpoints of sub-exposures of at most SUBEXPOSURE_DAYS.

        Parameters
        ----------
        model : pandoramoon.moon_model
            Pandora model.
        time : array_like
            Mid-exposure times in days.
        exposure_time : float, optional
            Exposure time in days. None or a value up to SUBEXPOSURE_DAYS gives instantaneous fluxes.

        Returns
        -------
        tuple of numpy.ndarray
            (total, planet, moon) fluxes at the given times.
        """
        time = np.asarray(time, dtype=float)
        if exposure_time is None or exposure_time <= ExoMoonLeastSquares.SUBEXPOSURE_DAYS:
            return tuple(np.asarray(flux, dtype=float) for flux in model.light_curve(time))
        sub_exposures = int(np.ceil(exposure_time / ExoMoonLeastSquares.SUBEXPOSURE_DAYS))
        offsets = ((np.arange(sub_exposures) + 0.5) / sub_exposures - 0.5) * exposure_time
        fine = (time[:, None] + offsets[None, :]).ravel()
        order = np.argsort(fine, kind='stable')
        fluxes = model.light_curve(fine[order])
        averaged = []
        for flux in fluxes:
            unsorted = np.empty(len(fine))
            unsorted[order] = np.asarray(flux, dtype=float)
            averaged.append(unsorted.reshape(len(time), sub_exposures).mean(axis=1))
        return tuple(averaged)

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

    @staticmethod
    def fit_single_t0(time, flux, flux_err, t0_guess, duration_window, transit_params, exposure_time=None,
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
            m = ExoMoonLeastSquares.transit_model(transit_params, t, exposure_time)
            return m.light_curve(transit_params)

        def cost(theta):
            m = model(theta[0], time)
            prior = 0.0 if t0_prior_sigma is None else 0.5 * ((theta[0] - t0_guess) / t0_prior_sigma) ** 2
            return 0.5 * np.sum(((flux - m) / flux_err) ** 2) + prior

        status = None
        if len(time) > 0:
            res = minimize(cost, [t0_guess], method='Powell', bounds=bounds)
            best_theta = res.x
            status = ExoMoonLeastSquares.fit_status(res, bounds, ['t0'])
        else:
            best_theta = np.array([t0_guess])
        best_model = model(float(best_theta[0]), time)
        return best_theta, best_model, status

    @staticmethod
    def fit_single_t0_rp(time, flux, flux_err, t0_guess, rp_guess, rp_err,
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
            m = ExoMoonLeastSquares.transit_model(transit_params, t, exposure_time)
            return m.light_curve(transit_params)

        def cost(theta):
            m = model(theta, time)
            prior = 0.0 if t0_prior_sigma is None else 0.5 * ((theta[0] - t0_guess) / t0_prior_sigma) ** 2
            return 0.5 * np.sum(((flux - m) / flux_err) ** 2) + prior

        status = None
        if len(time) > 0:
            res = minimize(cost, [t0_guess, rp_guess], method='Powell', bounds=bounds)
            best_theta = res.x
            status = ExoMoonLeastSquares.fit_status(res, bounds, ['t0', 'rp'])
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

    @staticmethod
    def fit_fold_shape(time, flux, flux_err, guess_params, guess_params_bounds,
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
            return ExoMoonLeastSquares.geometry_from_shape(theta[1], theta[2], theta[3], transit_params.per,
                                                          transit_params.ecc, transit_params.w)

        def model(theta, t):
            transit_params.t0, transit_params.rp = theta[0], theta[3]
            transit_params.a, transit_params.inc = geometry(theta)
            return ExoMoonLeastSquares.transit_model(transit_params, t, exposure_time).light_curve(transit_params)

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
        status = ExoMoonLeastSquares.fit_status(res, guess_params_bounds, ['t0', 't14', 'b', 'rp'],
                                                free_edges=[('b', 'lower')])
        best_model = model(best_theta, time)
        return best_theta, best_model, geometry(best_theta), status

    @staticmethod
    def impact_parameter(a_over_rstar, inc, ecc, arg_periastron):
        """Return the impact parameter at conjunction.

        Parameters
        ----------
        a_over_rstar : float
            Semimajor axis in stellar radii.
        inc : float
            Orbital inclination in degrees.
        ecc : float
            Orbital eccentricity.
        arg_periastron : float
            Argument of periastron in degrees.

        Returns
        -------
        float
            Impact parameter in stellar radii.
        """
        distance_factor = (1 - ecc ** 2) / (1 + ecc * np.sin(np.deg2rad(arg_periastron)))
        return a_over_rstar * np.cos(np.deg2rad(inc)) * distance_factor

    @staticmethod
    def min_transit_inclination(a_over_rstar, rp_over_rstar, ecc, arg_periastron):
        """Return the lowest inclination in degrees that still produces a transit.

        Parameters
        ----------
        a_over_rstar : float
            Semimajor axis in stellar radii.
        rp_over_rstar : float
            Planet radius in stellar radii.
        ecc : float
            Orbital eccentricity.
        arg_periastron : float
            Argument of periastron in degrees.

        Returns
        -------
        float
            Inclination in degrees.
        """
        distance_factor = (1 - ecc ** 2) / (1 + ecc * np.sin(np.deg2rad(arg_periastron)))
        return np.rad2deg(np.arccos(np.clip((1 + rp_over_rstar) / (a_over_rstar * distance_factor), 0, 1)))

    def max_moon_ttv(self, planet_mass=None):
        """Return the largest planet TTV, in days, that a moon in this search can cause.

        The planet orbits the planet-moon barycentre, a_m M_m / (M_p + M_m) away, so it crosses
        the star that distance divided by its orbital velocity v_p early or late. The largest
        moon mass is the largest searched radius at the largest moon density, and the widest
        orbit is the outer limit of the period grid, half the Hill period. The moon mass is the
        Chen & Kipping forecast for the largest searched radius, capped by that radius at the
        largest moon density (see `forecast_moon_mass`).

        Parameters
        ----------
        planet_mass : float, optional
            Planet mass in Earth masses. Defaults to the largest of the planet mass grid.

        Returns
        -------
        float
            TTV amplitude in days.
        """
        planet_mass = np.max(self.planet_mass_grid) if planet_mass is None else planet_mass
        mass = planet_mass * M_earth_to_M_sun
        moon_mass = forecast_moon_mass(self.max_radius, self.max_moon_density)  # Earth masses
        widest_period = self.max_moon_period(planet_mass)
        moon_a = self.compute_semimajor_axis(mass, widest_period)
        offset = self.compute_max_moon_offset(moon_a, self.planet_semimajor_axis, self.planet_period)
        return float(offset * moon_mass / (planet_mass + moon_mass))

    @staticmethod
    def ingress_duration(transit_params, t14):
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
        b = ExoMoonLeastSquares.impact_parameter(transit_params.a, transit_params.inc, transit_params.ecc,
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
        if self.exposure_time is not None:
            cadence = max(cadence, self.exposure_time)
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

    def check_planet_shape(self, time, residual_flux, fitted_t0s, bins=20, p_threshold=1e-3):
        """Test the stacked post-subtraction residual for a planet shape left behind (audit item D2).

        The residuals of the kept epochs are stacked on their fitted mid-times and binned over
        +-0.75 T14, so ingress, egress and the contact points are covered. The chi2 of the bin
        means against the level between 0.75 T14 and 1.5 T14, with errors from the scatter
        there, is compared with a chi2 distribution; a p-value below p_threshold fails the
        check. A moon inside the planet transit can also fail it, so it is a flag.

        Returns
        -------
        dict or None
            {"chi2", "dof", "p_value", "ok"}, or None without enough data.
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
                chi2 += ((np.mean(in_bin) - level) / (sigma / np.sqrt(len(in_bin)))) ** 2
                dof += 1
        if dof == 0 or not sigma > 0:
            return None
        p_value = float(stats.chi2.sf(chi2, dof))
        result = {"chi2": float(chi2), "dof": dof, "p_value": p_value, "ok": bool(p_value >= p_threshold)}
        if not result["ok"]:
            logging.warning("Planet shape check failed: the stacked residual over the transit has chi2 %.1f for %d "
                            "bins (p = %.1e), so the planet model leaves a residual, or a moon sits inside the "
                            "transit", chi2, dof, p_value)
        else:
            logging.info("Planet shape check passed: stacked residual chi2 %.1f for %d bins (p = %.2f)", chi2, dof,
                         p_value)
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
            parts = [(time[np.abs(time - t0) < duration_window] - t0 - shift,
                      flux[np.abs(time - t0) < duration_window], flux_err[np.abs(time - t0) < duration_window])
                     for t0, shift in shifts]
            return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))

        def chi2(theta_params, fold_time, fold_flux, fold_err):
            model = ExoMoonLeastSquares.transit_model(theta_params, fold_time, self.exposure_time)
            return float(np.sum(((fold_flux - model.light_curve(theta_params)) / fold_err) ** 2))

        shifts = []
        for t0, sigma in zip(self.bary_t0s, prior_sigmas):
            mask = np.abs(time - t0) < duration_window
            if np.sum(mask) < self.min_cadences_per_t0:
                continue
            epoch_params = copy.copy(params)
            epoch_params.t0, epoch_params.u = t0, self.ab
            theta, _, status = ExoMoonLeastSquares.fit_single_t0(time[mask], flux[mask], flux_err[mask], t0,
                                                                 duration_window, epoch_params, self.exposure_time,
                                                                 sigma)
            usable = status is None or (status["converged"] and "t0" not in status["at_bound"])
            shifts.append((t0, float(theta[0] - t0) if usable else 0.0))
        if len(shifts) < 2:
            return
        current = copy.copy(params)
        current.t0 = 0.0
        linear = fold([(t0, 0.0) for t0, _ in shifts])
        aligned = fold(shifts)
        before = chi2(current, *linear)
        start_b = ExoMoonLeastSquares.impact_parameter(params.a, params.inc, params.ecc, params.w)
        guess = [0.0, self.planet_duration, float(np.clip(start_b, fold_bounds[2][0], fold_bounds[2][1])),
                 float(np.clip(params.rp, fold_bounds[3][0], fold_bounds[3][1]))]
        try:
            theta, _, (a_new, inc_new), status = ExoMoonLeastSquares.fit_fold_shape(
                *aligned, guess, fold_bounds, copy.copy(params), self.exposure_time)
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
        fitted_duration = ExoMoonLeastSquares.model_transit_duration(refitted, duration_window)
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
        for t0 in self.bary_t0s:
            mask = np.abs(time - t0) < duration_window
            if len(time[mask]) < self.min_cadences_per_t0:
                continue
            fold_time.append(time[mask] - t0)
            fold_flux.append(flux[mask])
            if flux_err is not None:
                fold_flux_err.append(flux_err[mask])
        if len(fold_time) > 0:
            fold_time = np.concatenate(fold_time)
            fold_flux = np.concatenate(fold_flux)
            fold_flux_err = np.concatenate(fold_flux_err) if flux_err is not None else None
            # B1: fit the fold in duration, impact parameter and radius, so any transit shape is
            # reachable. The start is the shape of the input geometry.
            rp_err = self.planet_radius_err / star_radius_rearth
            start_params = copy.copy(params)
            start_params.t0 = 0.0
            start_t14 = ExoMoonLeastSquares.model_transit_duration(start_params, duration_window)
            if start_t14 <= 0:
                start_t14 = self.planet_duration
            start_b = ExoMoonLeastSquares.impact_parameter(params.a, params.inc, params.ecc, params.w)
            if not self.skip_fold_fit and fold_flux_err is not None:
                # B2: the catalog radius comes from another stellar radius and may not match this
                # depth, and the bound is only +-1 sigma wide. The start, and the centre of that
                # bound, is the radius that gives the observed central fold depth with the input
                # shape (limb darkening included); the width stays +-1 sigma, so the planet does not
                # gain room to absorb a moon (audit item B3).
                centre = np.abs(fold_time) < 0.25 * start_t14
                if np.sum(centre) >= 10:
                    observed = 1 - np.median(fold_flux[centre])
                    model_centre = ExoMoonLeastSquares.transit_model(start_params, fold_time[centre],
                                                                     self.exposure_time).light_curve(start_params)
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
        self.fold_fit_status = None
        if not self.skip_fold_fit:
            fold_theta, _, (fold_a, fold_inc), self.fold_fit_status = ExoMoonLeastSquares.fit_fold_shape(
                fold_time, fold_flux, fold_flux_err, fold_guess, fold_bounds, params, self.exposure_time)
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
            fitted_duration = ExoMoonLeastSquares.model_transit_duration(params_best, duration_window)
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
                model_guess = ExoMoonLeastSquares.transit_model(guess_params, fold_time,
                                                                self.exposure_time).light_curve(guess_params)
                model_best = ExoMoonLeastSquares.transit_model(params_best, fold_time,
                                                               self.exposure_time).light_curve(params_best)
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
        ingress = ExoMoonLeastSquares.ingress_duration(params, self.planet_duration)
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
                best_theta, best_model, status = ExoMoonLeastSquares.fit_single_t0_rp(
                    time[mask], flux[mask], flux_err[mask], t0, params.rp,
                    self.planet_radius_err / star_radius_rearth, duration_window, epoch_params, self.exposure_time,
                    t0_prior_sigma)
            else:
                best_theta, best_model, status = ExoMoonLeastSquares.fit_single_t0(
                    time[mask], flux[mask], flux_err[mask], t0, duration_window, epoch_params, self.exposure_time,
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
                # Quality control: in-transit completeness check
                n_in_transit = np.sum(best_model < 1.0)
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
            self.planet_shape_check = self.check_planet_shape(result_time, result_model,
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

    @staticmethod
    def compute_moon_ts(planet_t0, half_moon_orbit_transit_duration, half_moon_transit_duration):
        """Compute the maximum moon T0 offset (tau) for a given transit.

        Parameters
        ----------
        planet_t0 : float
            Planet transit epoch.
        half_moon_orbit_transit_duration : float
            Half the moon orbit transit duration in days.
        half_moon_transit_duration : float
            Half the moon transit duration in days.

        Returns
        -------
        float
            Maximum moon T0 tau offset in days.
        """
        # Half the orbit span is the maximum offset plus half a moon transit, see
        # compute_moon_orbit_transit_duration.
        return half_moon_orbit_transit_duration - half_moon_transit_duration

    @staticmethod
    def compute_moon_t0(planet_first_t0, planet_t0, moon_period, moon_initial_alpha, max_moon_t0_tau):
        """Compute the moon transit time and orbital phase.

        Parameters
        ----------
        planet_first_t0 : float
            First planet transit epoch.
        planet_t0 : float
            Current planet transit epoch.
        moon_period : float
            Moon orbital period in days.
        moon_initial_alpha : float
            Initial moon orbital phase in radians.
        max_moon_t0_tau : float
            Maximum moon T0 offset in days.

        Returns
        -------
        tuple
            (moon_t0, moon_phase) where moon_t0 is the transit time in days
            and moon_phase is the orbital phase in radians.
        """
        moon_phase = moon_initial_alpha + (((planet_t0 - planet_first_t0) % moon_period) / moon_period) * 2 * np.pi
        moon_tau = np.cos(moon_phase)
        moon_t0 = planet_t0 + moon_tau * max_moon_t0_tau
        return moon_t0, moon_phase


    @staticmethod
    #@njit(fastmath=True, parallel=False)
    def compute_moon_transit_scenarios(time, flux, flux_err, planet_t0, moon_initial_alpha, moon_period, bary_t0s,
                                       half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration,
                                       half_moon_transit_duration):
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

        Returns
        -------
        list
            List of [moon_alpha, time, flux, flux_err], one for each epoch
            with data. Empty if no epoch has data.
        """
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = []
        for t0 in bary_t0s:
            max_moon_t0_tau = ExoMoonLeastSquares.compute_moon_ts(t0, half_moon_orbit_transit_duration, half_moon_transit_duration)
            moon_t0, moon_alpha = ExoMoonLeastSquares.compute_moon_t0(planet_t0, t0, moon_period, moon_initial_alpha, max_moon_t0_tau)
            moon_t1 = moon_t0 - half_moon_transit_duration
            # t1 = moon_orbit_range[1]
            # phase_delta = (t0 - planet_t0) % moon_period * 2 * np.pi
            # alpha = (moon_initial_alpha + phase_delta) % (2 * np.pi)
            # time_alpha = np.cos(alpha) * moon_orbit_transit_length / 2
            # moon_t1 = t1 + time_alpha
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
                orbit_scenarios.append([moon_alpha, time_moon_transit, flux_moon_transit, flux_err_moon_transit])
        return orbit_scenarios

    def split_scenarios(self, search_input, time, flux, flux_err, bary_t0s):
        """Stack the epochs of one hypothesis and split them into the moon window and its surroundings.

        Parameters
        ----------
        search_input : SearchInput
            Moon hypothesis.
        time, flux, flux_err : numpy.ndarray
            Planet-subtracted light curve.
        bary_t0s : list of float
            Planet epochs to stack.

        Returns
        -------
        tuple or None
            (scenario_time, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err), with
            the times relative to the moon transit, or None when no epoch has data.
        """
        # Alpha is the moon phase at self.planet_t0, the same reference inject_moon uses, so it
        # does not move when the first epoch is skipped or rejected.
        transit_scenarios = ExoMoonLeastSquares.compute_moon_transit_scenarios(time, flux, flux_err,
                                                                               self.planet_t0, search_input.moon_alpha,
                                                                search_input.moon_period, bary_t0s,
                                                                               search_input.half_moon_orbit_transit_duration,
                                                                search_input.half_moon_max_orbit_transit_duration,
                                                                               search_input.half_moon_transit_duration)
        if len(transit_scenarios) == 0:
            return None
        scenario_time = []
        scenario_flux = []
        scenario_flux_err = []
        for normalized_moon_transit_scenario in transit_scenarios:
            scenario_time = np.concatenate((scenario_time, normalized_moon_transit_scenario[1].flatten()))
            scenario_flux = np.concatenate((scenario_flux, normalized_moon_transit_scenario[2].flatten()))
            scenario_flux_err = np.concatenate((scenario_flux_err, normalized_moon_transit_scenario[3].flatten()))
        sorted_time_args = np.argsort(scenario_time)
        scenario_time = scenario_time[sorted_time_args]
        scenario_flux = scenario_flux[sorted_time_args]
        scenario_flux_err = scenario_flux_err[sorted_time_args]
        outliers_args = ExoMoonLeastSquares.remove_outliers(scenario_flux, sigma_lower=float('inf'), sigma_upper=3)
        scenario_time = scenario_time[~outliers_args].flatten()
        scenario_flux = scenario_flux[~outliers_args].flatten()
        scenario_flux_err = scenario_flux_err[~outliers_args].flatten()
        #TODO check whether to use this mean or the real scenario_flux
        #scenario_flux = ExoMoonLeastSquares.running_mean_equal_length(scenario_flux, len(scenario_flux) // 100)
        it_args = np.argwhere(np.abs(scenario_time) <= search_input.half_moon_transit_duration).flatten()
        oot_args = np.argwhere(np.abs(scenario_time) > search_input.half_moon_transit_duration).flatten()
        it_time = scenario_time[it_args]
        oot_time = scenario_time[oot_args]
        it_flux = scenario_flux[it_args]
        oot_flux = scenario_flux[oot_args]
        it_flux_err = scenario_flux_err[it_args]
        oot_flux_err = scenario_flux_err[oot_args]
        return scenario_time, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err

    def window_depth(self, search_input, time, flux, flux_err, bary_t0s):
        """Return the mean depth of the moon window against its surroundings, as `search` measures it."""
        split = self.split_scenarios(search_input, time, flux, flux_err, bary_t0s)
        if split is None or len(split[2]) == 0 or len(split[5]) < 2:
            return np.nan
        return float(np.mean(split[5]) - np.mean(split[2]))

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
        split = self.split_scenarios(search_input, self.time, self.flux, self.flux_err, self.bary_t0s)
        if split is None:
            no_data = np.array([])
            scores = (ExoMoonLeastSquares.MAX_RESIDUAL, ExoMoonLeastSquares.MAX_RESIDUAL, self.min_radius, np.nan)
            return scores + (no_data, no_data, no_data, no_data, no_data) if return_lc else scores + (False,)
        scenario_time, it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err = split
        if len(it_flux) == 0 or len(oot_flux) < 2:
            # No in-window or out-of-window data: the depth and the noise are undefined.
            no_data = np.array([])
            scores = (ExoMoonLeastSquares.MAX_RESIDUAL, ExoMoonLeastSquares.MAX_RESIDUAL, self.min_radius, np.nan)
            return scores + (it_time, it_flux, oot_time, oot_flux, no_data) if return_lc else scores + (False,)
        # Depth is measured against the surrounding out-of-window level, so a normalisation offset
        # or the upper-only outlier clipping does not count as signal.
        depth = np.mean(oot_flux) - np.mean(it_flux)
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
        #oot_noise = np.std(oot_flux)
        #oot_noise = np.sqrt(np.std(oot_flux) * (np.sqrt(np.sum(oot_flux_err ** 2) / len(oot_flux_err))))
        model_snr = ExoMoonLeastSquares.box_snr(it_flux, oot_flux) / self.red_noise_factor
        if return_lc:
            return residual_calculation, residual_baseline, residual_radius, model_snr, it_time, it_flux, oot_time, oot_flux, residual_model
        else:
            return residual_calculation, residual_baseline, residual_radius, model_snr, template_unusable

    @staticmethod
    def box_snr(it_flux, oot_flux):
        """Return the SNR of the moon-window depth against its surroundings, as `search` scores it.

        The depth is a difference of two means, so both contribute noise. NaN without in-window
        data or with fewer than two points outside it.
        """
        if len(it_flux) == 0 or len(oot_flux) < 2:
            return np.nan
        depth = np.mean(oot_flux) - np.mean(it_flux)
        return float(depth / (np.std(oot_flux) * np.sqrt(1 / len(it_flux) + 1 / len(oot_flux))))

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
    def spectra(chi2, oversampling_factor=1, kernel_size=30):
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
            baseline = ExoMoonLeastSquares.running_median(SR, kernel)
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

    @staticmethod
    #@njit(fastmath=True, parallel=False)
    def calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
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
            deficit = (1 - t0_flux_mean_value) + np.mean(it_flux) - it_flux
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
    def build_moon_templates(transit_params, radii, star_radius, min_half_window, exposure_time=None):
        """Build moon transit templates that reach unity inside their support.

        Parameters
        ----------
        transit_params : batman.TransitParams
            Transit geometry. Its rp value is overwritten.
        radii : array_like
            Moon radii in Earth radii, in increasing order.
        star_radius : float
            Stellar radius in solar radii.
        min_half_window : float
            Smallest template half-width in days.
        exposure_time : float, optional
            Exposure time in days. The templates are averaged over it, so a template
            sampled at a data time matches an exposure centred there. Default None.

        Returns
        -------
        list of (float, numpy.ndarray, numpy.ndarray)
            [(radius_Rearth, time_days_from_transit_center, flux), ...] for
            the radii that produce a transit.
        """
        cadence = 2 * min_half_window / (ExoMoonLeastSquares.TEMPLATE_MIN_SAMPLES - 1)
        max_half_window = transit_params.per / 2
        half_window = min(min_half_window, max_half_window)
        transit_params.rp = radii[-1] * R_earth_to_R_sun / star_radius
        while True:
            samples = max(ExoMoonLeastSquares.TEMPLATE_MIN_SAMPLES, int(np.ceil(2 * half_window / cadence)) + 1)
            template_time = np.linspace(-half_window, half_window, samples)
            time_model = transit_params.t0 + template_time
            widest_model = ExoMoonLeastSquares.transit_model(transit_params, time_model, exposure_time).light_curve(
                transit_params)
            if (widest_model[0] >= 1 and widest_model[-1] >= 1) or half_window >= max_half_window:
                break
            half_window = min(half_window * 1.25, max_half_window)
        models = []
        for radius in radii:
            transit_params.rp = radius * R_earth_to_R_sun / star_radius
            model_r = ExoMoonLeastSquares.transit_model(transit_params, time_model, exposure_time).light_curve(
                transit_params)
            if np.count_nonzero(model_r < 1) > 1:
                models.append((radius, template_time, model_r))
        return models

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

    def inject_moon(self, time, flux, flux_err, t0s, planet_semimajor_axis, planet_ecc, moon_radius,
                    moon_period, moon_mass, initial_alpha=0, plot=False, inject_planet=False, inject_moon=True):
        """Inject a moon transit signal into the light curve.

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
        planet_semimajor_axis : float
            Planet semimajor axis in AU.
        planet_ecc : float
            Planet orbital eccentricity.
        moon_radius : float
            Moon radius in Earth radii.
        moon_period : float
            Moon orbital period in days.
        moon_mass : float
            Moon mass in Earth masses.
        initial_alpha : float, optional
            Initial moon orbital phase in radians, default 0.
        plot : bool, optional
            If True, generate injection plots, default False.
        inject_planet : bool, optional
            If True, inject planet transit as well, default False.
        inject_moon : bool, optional
            If True, inject moon transit signal, default True.

        Returns
        -------
        numpy.ndarray
            Flux array with injected moon signal.
        """
        logging.info("Injecting moon with radius of  %.2fR_e, %.2fdays and %.2frad", moon_radius, moon_period, initial_alpha)
        star_radius_m = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.m)
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        params = pandora.model_params()
        params.R_star = star_radius_m  # [m]
        params.u1 = self.ab[0]
        params.u2 = self.ab[1]
        # planet parameters
        params.per_bary = self.planet_period  # [days]
        params.a_bary = self.injection_a_over_rstar()  # [R_star]
        params.r_planet = self.planet_radius / star_radius_rearth  # [R_star]
        params.b_bary = self.planet_impact_param  # [0..1]
        params.t0_bary = self.planet_t0  # [days]
        params.t0_bary_offset = 0.001  # [days]
        params.M_planet = LcbuilderHelper.convert_from_to(self.planet_mass, u.M_earth, u.kg)  # [kg]
        params.w_bary = self.planet_arg_periastron  # [deg]
        params.ecc_bary = self.planet_ecc  # [0..1]
        # moon parameters
        params.r_moon = moon_radius / star_radius_rearth  # [R_star]
        params.per_moon = moon_period  # [days]
        params.tau_moon = (params.t0_bary / moon_period - (initial_alpha + np.pi) / (2 * np.pi)) % 1
        params.Omega_moon = 0  # [0..180]
        params.i_moon = self.planet_inc  # [0..180]
        params.e_moon = 0  # [0..1]
        params.w_moon = self.planet_arg_periastron  # [deg]
        params.M_moon = LcbuilderHelper.convert_from_to(moon_mass, u.M_earth, u.kg)  # [kg]
        # times params
        # params.epochs = 3  # [int]
        # params.epoch_duration = 2  # [days]
        # params.cadences_per_day = 48  # [int]
        params.epoch_distance = self.planet_period  # [days]
        params.supersampling_factor = 1  # [int]
        params.occult_small_threshold = 0.1  # [0..1]
        params.hill_sphere_threshold = 1
        model = pandora.moon_model(params)
        injected_flux_total, injected_flux_planet, injected_flux_moon = \
            ExoMoonLeastSquares.pandora_light_curve(model, time, self.exposure_time)
        self.injected_flux_moon_pandora = injected_flux_moon
        self.injected_flux_planet_pandora = injected_flux_planet
        self.injected_moon_period = moon_period
        self.injected_moon_alpha = initial_alpha
        # model.video(
        #     time=pandora_time,
        #     limb_darkening=True,
        #     teff=3200,
        #     planet_color="black",
        #     moon_color="black",
        #     ld_circles=100
        # )
        if inject_moon:
            flux = flux - (1 - injected_flux_moon)
        if inject_planet:
            flux = flux - (1 - injected_flux_planet)
        # for t0 in t0s:
        #     max_moon_t0_tau = ExoMoonLeastSquares.compute_moon_ts(t0, moon_orbit_transit_duration / 2, moon_transit_duration / 2)
        #     moon_t0, moon_phase = ExoMoonLeastSquares.compute_moon_t0(self.planet_t0, t0, moon_period, initial_alpha, max_moon_t0_tau)
        #     #time_transit_mask = (moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)
        #     time_transit_mask = (t0 - moon_orbit_transit_duration < time) & (time < t0 + moon_orbit_transit_duration)
        #     time_transit = time[time_transit_mask]
        #     ma = batman.TransitParams()
        #     ma.t0 = moon_t0  # time of inferior conjunction
        #     ma.per = self.planet_period  # orbital period, use Earth as a reference
        #     ma.rp = moon_radius * R_earth_to_R_sun / self.star_radius  # planet radius (in units of stellar radii)
        #     ma.a = planet_semimajor_axis * AU_TO_RSUN / self.star_radius  # semi-major axis (in units of stellar radii)
        #     ma.inc = self.planet_inc  # orbital inclination (in degrees)
        #     ma.ecc = planet_ecc  # eccentricity
        #     ma.w = 0  # longitude of periastron (in degrees)
        #     ma.u = self.ab  # limb darkening coefficients
        #     ma.limb_dark = "quadratic"  # limb darkening model
        #     m = batman.TransitModel(ma, time_transit)  # initializes model
        #     model_moon = m.light_curve(ma)  # calculates light curve
        #     # flux[(moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)] = \
        #     #     flux[(moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)] - (1 - model_moon)
        #     subtracted_planet_flux[time_transit_mask] = \
        #         subtracted_planet_flux[time_transit_mask] - (1 - injected_flux_moon[time_transit_mask])
        #     if plot:
        #         fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
        #         axs[0].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[0].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[0].set_xlabel('Time')
        #         axs[0].set_ylabel('Flux')
        #         axs[0].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[0].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[1].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[1].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[1].set_xlabel('Time')
        #         axs[1].set_ylabel('Flux')
        #         axs[1].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[1].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[2].scatter(time_transit, subtracted_planet_flux[time_transit_mask])
        #         axs[2].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[2].set_xlabel('Time')
        #         axs[2].set_ylabel('Flux')
        #         axs[2].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[2].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon.png")
        #         plt.close()
        #         time_transit_mask = (moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)
        #         time_transit = time[time_transit_mask]
        #         # limb darkening model
        #         m = batman.TransitModel(ma, time_transit)  # initializes model
        #         model_moon = m.light_curve(ma)
        #         fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
        #         axs[0].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[0].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[0].set_xlabel('Time')
        #         axs[0].set_ylabel('Flux')
        #         axs[0].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[0].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[1].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[1].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[1].set_xlabel('Time')
        #         axs[1].set_ylabel('Flux')
        #         axs[1].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[1].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[2].scatter(time_transit, subtracted_planet_flux[time_transit_mask])
        #         axs[2].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[2].set_xlabel('Time')
        #         axs[2].set_ylabel('Flux')
        #         axs[2].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[2].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon_focus.png")
        #         plt.close()
        return flux

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

    def plot_signal(self, axs, scenario_time, scenario_flux, it_time, oot_time, model):
        """Plot the moon signal detection results.

        Parameters
        ----------
        axs : matplotlib.axes.Axes
            Matplotlib axes to plot on.
        scenario_time : array_like
            Time array for the full scenario.
        scenario_flux : array_like
            Flux array for the full scenario.
        it_time : array_like
            In-transit time array.
        oot_time : array_like
            Out-of-transit time array.
        model : array_like
            Model light curve.
        """
        bin_means, bin_edges, binnumber = stats.binned_statistic(scenario_time,
                                                                 scenario_flux,
                                                                 statistic='mean', bins=60)
        bin_stds, _, _ = stats.binned_statistic(scenario_time,
                                                scenario_flux, statistic='std', bins=60)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        axs.scatter(scenario_time, scenario_flux,
                       color='gray', alpha=0.4, rasterized=True, label="Flux")
        axs.errorbar(bin_centers, bin_means, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=4,
                        color='darkorange', alpha=1, linestyle='none')
        axs.scatter(it_time, model,
                       color='red', alpha=1, rasterized=True, label="Model")
        axs.scatter(oot_time, np.ones(len(oot_time)),
                       color='red', alpha=1, rasterized=True, label="Model")
        axs.set_xlabel('Time')
        axs.set_ylabel('Flux')

    @staticmethod
    def find_period_alias(moon_period, planet_period, found_period, start_error):
        """Find the alias of a moon period closest to a found period.

        Parameters
        ----------
        moon_period : float
            True moon period in days.
        planet_period : float
            Planet period in days.
        found_period : float
            Detected period in days.
        start_error : float
            Relative error an alias must beat.

        Returns
        -------
        tuple
            (alias_period, k, relative_error). alias_period and k are None if
            no alias beats start_error, and relative_error is then start_error.
        """
        alias_period, alias_k, min_error = None, None, start_error
        for k in range(-ExoMoonLeastSquares.MAX_ALIAS_ORDER, ExoMoonLeastSquares.MAX_ALIAS_ORDER + 1):
            frequency = 1 / moon_period + k / planet_period
            if frequency <= 0:
                continue
            error = abs(1 / frequency - found_period) / moon_period
            if error < min_error:
                alias_period, alias_k, min_error = 1 / frequency, k, error
        return alias_period, alias_k, min_error

    @staticmethod
    def classify_recovered_period(moon_period, planet_period, found_period):
        """Classify a found period as the injected moon period, one of its sampling aliases, or neither.

        The found period is assigned to the nearest member of the alias family
        1/P' = 1/P + k/P_planet, |k| up to MAX_ALIAS_ORDER, and must lie within
        RECOVERY_PERIOD_TOLERANCE of it. A fixed tolerance around the injected period alone
        counts a neighbouring alias as the period whenever the alias spacing, about
        P^2 / P_planet, is below the tolerance: a 10 d moon of a 218 d planet has its k = 1
        alias 4.4 % away.

        Parameters
        ----------
        moon_period : float
            Injected moon period in days.
        planet_period : float
            Planet period in days.
        found_period : float
            Recovered period in days.

        Returns
        -------
        tuple
            (period_match, alias_match, k, alias_period, relative_error): k is the order of the
            nearest alias (0 for the injected period) and relative_error its distance to the
            found period over the moon period. period_match needs k = 0, alias_match k != 0.
        """
        alias_period, k, error = ExoMoonLeastSquares.find_period_alias(moon_period, planet_period, found_period,
                                                                       np.inf)
        close = k is not None and error < ExoMoonLeastSquares.RECOVERY_PERIOD_TOLERANCE
        return bool(close and k == 0), bool(close and k != 0), k, alias_period, error

    def require_usable_epochs(self):
        """Stop with an explicit error when no transit epoch is left to search.

        Raises
        ------
        NoUsableEpochsError
            If there are no usable epochs.
        """
        if len(self.bary_t0s) == 0:
            raise NoUsableEpochsError("No usable planet transit epochs, so there is nothing to search")

    def moon_signal(self, time, moon_t0s, radius):
        """Return the flux of a moon of the given radius transiting at moon_t0s, with the fitted geometry."""
        flux = np.ones(len(time))
        params = self.moon_template_params()
        params.rp = radius * R_earth_to_R_sun / self.star_radius
        for moon_t0 in moon_t0s:
            window = np.abs(time - moon_t0) < self.planet_period / 2
            if np.any(window):
                params.t0 = moon_t0
                flux[window] *= ExoMoonLeastSquares.transit_model(params, time[window],
                                                                  self.exposure_time).light_curve(params)
        return flux

    def measure_absorbed_fraction(self, before_planet_fit, best_input, moon_t0s):
        """Measure the fraction of the best moon signal that the planet fit absorbs (audit item B3).

        A moon that stays inside the planet transit looks like extra planet depth, so the planet
        fit takes part of it. The best hypothesis, with the fitted transit geometry, is injected into
        the data before the planet fit and the fit is repeated. The absorbed fraction is one minus
        the depth gained in the moon window after the fit, over the depth gained when the moon is
        added after the fit, over the epochs both fits kept. It assumes the absorption is linear.
        When the data already contain this moon, it measures the absorption of a second copy, which
        is a lower bound once the first copy has pushed the planet radius to its bound.

        Parameters
        ----------
        before_planet_fit : ExoMoonLeastSquares
            Copy of this search taken just before the planet fit.
        best_input : SearchInput
            Best hypothesis.
        moon_t0s : list of float
            Moon mid-transit times of the best hypothesis, one per epoch.

        Returns
        -------
        float or None
            Absorbed fraction, or None when it cannot be measured.
        """
        radius = self.best_snr_radius if np.isfinite(self.best_snr_radius) and self.best_snr_radius > 0 \
            else self.min_radius
        radius = max(radius, self.min_radius)
        try:
            injected = copy.copy(before_planet_fit)
            logging.info("Absorbed-fraction test: repeating the planet fit with the best moon (%.2f R_earth) "
                         "injected; its own fit messages are not logged", radius)
            previous_disable = logging.root.manager.disable
            logging.disable(logging.WARNING)
            try:
                with tempfile.TemporaryDirectory() as plots_dir:
                    injected.object_dir = plots_dir
                    # The injected and the reference moon both use the fitted geometry, so they are
                    # the same signal; the copy still holds the input geometry.
                    flux = before_planet_fit.flux + self.moon_signal(before_planet_fit.time, moon_t0s, radius) - 1
                    time, flux, flux_err = injected.subtract_planet_transit(before_planet_fit.time, flux,
                                                                            before_planet_fit.flux_err,
                                                                            mode=self.subtract_mode)
            finally:
                logging.disable(previous_disable)
            logging.info("Absorbed-fraction test: fold fit converged=%s, at bound: %s; %d of %d transits kept",
                         injected.fold_fit_status["converged"] if injected.fold_fit_status else None,
                         (", ".join(injected.fold_fit_status["at_bound"]) or "none") if injected.fold_fit_status
                         else "-",
                         len(injected.bary_t0s), len(before_planet_fit.bary_t0s))
            # All three depths cover the epochs both fits kept.
            epochs = [t0 for t0 in self.bary_t0s if t0 in set(injected.bary_t0s)]
            original = self.window_depth(best_input, self.time, self.flux, self.flux_err, epochs)
            added = self.window_depth(best_input, self.time, self.flux + self.moon_signal(self.time, moon_t0s, radius)
                                      - 1, self.flux_err, epochs)
            fitted = self.window_depth(best_input, time, flux, flux_err, epochs)
        except Exception as error:
            logging.warning("Could not measure the moon fraction absorbed by the planet fit: %s", error)
            return None
        if not (np.isfinite(original) and np.isfinite(added) and np.isfinite(fitted)) or added - original <= 0:
            logging.warning("Could not measure the moon fraction absorbed by the planet fit: no depth gained")
            return None
        absorbed = float(1 - (fitted - original) / (added - original))
        logging.info("The planet fit absorbs %.0f%% of the best moon signal (%.2f R_earth, injected at the best "
                     "period and phase); the reported radius and SNR are biased low by that much in depth. If the "
                     "data already hold this moon, the value is a lower bound, because the planet radius may "
                     "already sit at its bound", 100 * absorbed, radius)
        return absorbed

    def transit_semimajor_axis(self):
        """Return the planet semi-major axis in solar radii that sets transit durations and shapes.

        After the fold fit this is the fitted a/R* times the stellar radius (audit item B4);
        before it, or without it, the semi-major axis from the inputs.
        """
        if self.planet_a_over_rstar is not None:
            return self.planet_a_over_rstar * self.star_radius
        return self.planet_semimajor_axis * AU_TO_RSUN

    def moon_template_params(self):
        """Return the batman parameters that define the moon transit templates.

        Returns
        -------
        batman.TransitParams
            Planet geometry with no radius set.
        """
        params = batman.TransitParams()
        params.t0 = 10  # time of inferior conjunction
        params.per = self.planet_period
        params.a = self.transit_semimajor_axis() / self.star_radius  # in stellar radii
        params.inc = self.planet_inc
        params.ecc = self.planet_ecc
        params.w = self.planet_arg_periastron
        params.u = self.ab
        params.limb_dark = "quadratic"
        return params

    def search_input(self, moon_period, moon_alpha, planet_mass, max_moon_period, impact_param):
        """Build the search hypothesis for one moon period and initial phase.

        See `search_inputs` for the parameters.
        """
        return self.search_inputs(moon_period, [moon_alpha], planet_mass, max_moon_period, impact_param)[0]

    def search_inputs(self, moon_period, moon_alphas, planet_mass, max_moon_period, impact_param):
        """Build the search hypotheses for one moon period and several initial phases.

        The durations do not depend on the phase, so they are computed once.

        Parameters
        ----------
        moon_period : float
            Moon period in days.
        moon_alphas : iterable of float
            Initial moon phases in radians.
        planet_mass : float
            Planet mass in Earth masses.
        max_moon_period : float
            Largest moon period searched, which sets the data window.
        impact_param : float
            Planet impact parameter.

        Returns
        -------
        list of SearchInput
            One hypothesis per phase, with the moon offset limited to a_m / v_p.
        """
        transit_a = self.transit_semimajor_axis()
        moon_radius = self.max_radius * R_earth_to_R_sun
        mass = planet_mass * M_earth_to_M_sun
        # G11: the moon transit duration follows the transit geometry (the fitted one after the fold
        # fit). The offsets a_m / v_p use the planet speed measured by the input transit duration.
        moon_transit_duration = self.conjunction_time_factor() * self.compute_transit_duration(
            self.star_radius, transit_a, self.planet_period, moon_radius, impact_param)
        self.log_planet_speed()
        orbit_duration = moon_transit_duration + 2 * self.moon_offset(self.compute_semimajor_axis(mass, moon_period))
        max_orbit_duration = moon_transit_duration + 2 * self.moon_offset(
            self.compute_semimajor_axis(mass, max_moon_period))
        return [SearchInput(moon_period, moon_alpha, self.min_radius,
                            impact_param, max_orbit_duration, moon_transit_duration, orbit_duration,
                            moon_transit_duration / 2, orbit_duration / 2, max_orbit_duration / 2)
                for moon_alpha in moon_alphas]

    @staticmethod
    def snr_matrix(all_residuals, alpha_grid_size, period_count):
        """Arrange the grid SNRs as an (alpha, period) matrix."""
        matrix = np.full((alpha_grid_size, period_count), np.nan)
        for i, residual in enumerate(all_residuals[:alpha_grid_size * period_count]):
            matrix[i % alpha_grid_size][i // alpha_grid_size] = residual[3]
        return matrix

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

    def measure_red_noise(self, time, flux, exclusion, bin_width):
        """Return the red-noise factor beta on the moon-window time scale (audit item H11).

        The search SNR divides the depth by the point scatter over sqrt(N), which assumes
        independent points. Following Pont, Zucker & Queloz (2006), the data away from every
        planet epoch (by more than exclusion, so neither the planet nor a searched moon is
        there) are binned at the moon transit duration, and beta is the scatter of the
        differences between adjacent bins, over sqrt(2), divided by the white-noise expectation
        sigma / sqrt(n). Adjacent differences ignore variability slower than two windows, which
        the comparison of the moon window with its surroundings also cancels.

        Parameters
        ----------
        time, flux : numpy.ndarray
            Light curve, normalised to one.
        exclusion : float
            Half-width in days excluded around each planet epoch.
        bin_width : float
            Bin width in days, the moon transit duration.

        Returns
        -------
        float or None
            beta, or None when fewer than 10 pairs of adjacent bins are available.
        """
        order = np.argsort(time)
        time, flux = np.asarray(time, dtype=float)[order], np.asarray(flux, dtype=float)[order]
        epochs = self.planet_t0 + self.planet_period * np.arange(
            np.floor((time[0] - self.planet_t0) / self.planet_period) - 1,
            np.ceil((time[-1] - self.planet_t0) / self.planet_period) + 2)
        nearest = np.min(np.abs(time[:, None] - epochs[None, :]), axis=1) if len(time) else np.array([])
        keep = (nearest >= exclusion) & np.isfinite(flux)
        time, flux = time[keep], flux[keep]
        if len(time) < 20 or not bin_width > 0:
            return None
        index = np.floor((time - time[0]) / bin_width).astype(int)
        bins, start, counts = np.unique(index, return_index=True, return_counts=True)
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

    def moon_window_half_width(self):
        """Return the half-width in days of the widest searched moon window around each epoch."""
        mass = np.max(self.planet_mass_grid) * M_earth_to_M_sun
        max_period = self.max_moon_period(np.max(self.planet_mass_grid))
        return self.search_input(max_period, 0.0, self.planet_mass, max_period,
                                 self.planet_impact_param).half_moon_max_orbit_transit_duration

    def epoch_distance(self, time):
        """Return the distance in days from each time to the nearest planet epoch of the linear ephemeris."""
        time = np.asarray(time, dtype=float)
        if len(time) == 0:
            return time
        phase = (time - self.planet_t0) / self.planet_period
        return np.abs(phase - np.round(phase)) * self.planet_period

    def detrend(self, time, flux, flux_err):
        """Detrend an undetrended light curve without letting the planet or a moon shape the trend (audit item A1).

        A biweight of detrend_window days is computed with the points within the widest searched
        moon window of every epoch masked, so they do not shape it (Hippke et al. 2019, AJ 158,
        143, Sect. V.4). Every point, the masked ones included, is then divided by the trend;
        where a masked window is wider than the biweight window, the trend is joined linearly
        across it.

        Returns
        -------
        tuple
            (time, flux, flux_err), sorted by time, with the flux and errors divided by the trend.
        """
        order = np.argsort(time)
        time, flux = np.asarray(time, dtype=float)[order], np.asarray(flux, dtype=float)[order]
        flux_err = None if flux_err is None else np.asarray(flux_err, dtype=float)[order]
        mask = self.epoch_distance(time) < self.moon_window_half_width()
        _, trend = wotan.flatten(time, flux, method='biweight', window_length=self.detrend_window,
                                 return_trend=True, mask=mask)
        # Inside a mask the biweight only sees one side of the window, which is noisy; the
        # trend there is the straight line between the mask edges instead.
        finite = np.isfinite(trend) & (trend > 0) & ~mask
        if np.count_nonzero(finite) < 2:
            raise ValueError("The masked detrending left no trend: too few points outside the moon windows")
        trend = np.interp(time, time[finite], trend[finite])
        return time, flux / trend, None if flux_err is None else flux_err / trend

    def set_red_noise_factor(self, before_trim):
        """Measure beta (see `measure_red_noise`) on the light curve before trimming and set red_noise_factor.

        The data within the widest moon window of every epoch are excluded. beta below one is
        set to one, so the SNR is never raised. Without enough data the factor stays one.
        """
        max_period = self.max_moon_period(np.max(self.planet_mass_grid))
        window = self.search_input(max_period, 0.0, self.planet_mass, max_period, self.planet_impact_param)
        time, flux = before_trim.time, before_trim.flux
        if self.detrend_window is not None:
            time, flux, _ = self.detrend(time, flux, None)
        self.red_noise_measured = self.measure_red_noise(time, flux, window.half_moon_max_orbit_transit_duration,
                                                         window.moon_transit_duration)
        if self.red_noise_measured is None:
            self.red_noise_factor = 1.0
            logging.warning("Red-noise factor not measured: too little data away from the transits; the SNR "
                            "assumes white noise")
        else:
            self.red_noise_factor = max(1.0, self.red_noise_measured)
            logging.info("Red-noise factor beta = %.2f on the %.2f h moon-window scale (measured %.2f); the SNR "
                         "is divided by it", self.red_noise_factor, window.moon_transit_duration * 24,
                         self.red_noise_measured)

    def alias_family(self, period_grid, snr_spectrum, best_period):
        """Return the sampling aliases of the selected period with their SNR (audit item H9).

        With one look per planet orbit, the moon phase at epoch n is alpha + 2 pi n P_p / P, so
        every P' with 1 / P' = 1 / P + k / P_p gives the same phases and differs only through
        the timing offsets; in close-in geometry the data barely prefer one of them. Each alias
        inside the grid, |k| up to MAX_ALIAS_ORDER, is listed with the best SNR over alpha at
        the nearest grid period and its SNR below the selected one, highest SNR first.

        Returns
        -------
        list of dict
            {"k", "period", "grid_period", "snr", "snr_below_best"} per alias, k = 0 included.
        """
        period_grid = np.asarray(period_grid, dtype=float)
        snr_spectrum = np.asarray(snr_spectrum, dtype=float)
        best_index = int(np.argmin(np.abs(np.log(period_grid / best_period))))
        best_snr = snr_spectrum[best_index]
        family = []
        for k in range(-ExoMoonLeastSquares.MAX_ALIAS_ORDER, ExoMoonLeastSquares.MAX_ALIAS_ORDER + 1):
            frequency = 1 / best_period + k / self.planet_period
            if frequency <= 0:
                continue
            period = 1 / frequency
            if not period_grid[0] <= period <= period_grid[-1]:
                continue
            index = int(np.argmin(np.abs(np.log(period_grid / period))))
            snr = snr_spectrum[index]
            family.append({"k": k, "period": float(period), "grid_period": float(period_grid[index]),
                           "snr": float(snr) if np.isfinite(snr) else None,
                           "snr_below_best": float(best_snr - snr) if np.isfinite(snr) and np.isfinite(best_snr)
                           else None})
        family.sort(key=lambda alias: -np.inf if alias["snr"] is None else -alias["snr"])
        return family

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
            (time, flux, flux_err, epochs), or None when fewer than half the epochs have a usable
            offset.
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
            params.t0 = t0
            planet = ExoMoonLeastSquares.transit_model(params, piece_time, self.exposure_time).light_curve(params)
            pieces.append((piece_time, flux[window] * planet, flux_err[window]))
            epochs.append(t0)
        if len(epochs) == 0 or len(epochs) < len(self.bary_t0s) / 2:
            return None
        null_time = np.concatenate([piece[0] for piece in pieces])
        order = np.argsort(null_time)
        return (null_time[order], np.concatenate([piece[1] for piece in pieces])[order],
                np.concatenate([piece[2] for piece in pieces])[order], epochs)

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

    def null_statistic(self, before_trim, time, flux, flux_err, epochs):
        """Return the detection SNR of the full search on one pseudo-epoch light curve."""
        null = copy.copy(before_trim)
        null.time, null.flux, null.flux_err = time, flux, flux_err
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
                                    ExoMoonLeastSquares.snr_matrix(residuals, len(alpha_grid), len(period_grid)))
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

    def assess_single_event(self, search_input):
        """Measure each epoch's depth and SNR and the leave-one-out SNR at the best hypothesis (audit item H12).

        Sets epoch_stats, loo_snr_min and single_event. single_event is True when one epoch
        carries the signal: with one epoch, or when leaving one epoch out keeps less than
        SINGLE_EVENT_SNR_FRACTION of the SNR. A signal spread over N epochs keeps about
        sqrt((N - 1) / N) of it. The flag does not change the detection.
        """
        def measure(epochs):
            split = self.split_scenarios(search_input, self.time, self.flux, self.flux_err, epochs) if epochs else None
            if split is None or len(split[2]) == 0 or len(split[5]) < 2:
                return None, None
            return (float(np.mean(split[5]) - np.mean(split[2])),
                    ExoMoonLeastSquares.box_snr(split[2], split[5]) / self.red_noise_factor)

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
                                 or self.loo_snr_min < ExoMoonLeastSquares.SINGLE_EVENT_SNR_FRACTION * self.max_snr)
        if self.single_event:
            logging.warning("Single-event flag: leaving one transit out drops the SNR from %.2f to %s, so one epoch "
                            "carries the signal; see epoch_stats", self.max_snr,
                            "nothing" if self.loo_snr_min is None else f"{self.loo_snr_min:.2f}")

    def build_search_templates(self):
        """Build the moon transit templates for the radius grid with the current (fitted) geometry."""
        ma = self.moon_template_params()
        search_half_window = self.compute_transit_duration(self.star_radius, self.transit_semimajor_axis(),
                                                           self.planet_period, self.max_radius * R_earth_to_R_sun,
                                                           self.planet_impact_param) / 2
        self.models = ExoMoonLeastSquares.build_moon_templates(
            ma, np.linspace(self.min_radius, self.max_radius, self.radius_grid_size), self.star_radius,
            max(self.planet_duration * 0.7, search_half_window), self.exposure_time)

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
                                                 self.planet_semimajor_axis, self.planet_ecc)
            hill = self.compute_hill_radius(self.star_mass, planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis)
            logging.info("Outer moon limit: a = %.3f r_H (Domingos et al. 2006, prograde, e_p = %.2f), P = %.4f d",
                         a_max / hill, self.planet_ecc, max_period)
            period_grid = self.compute_moon_period_grid(min_period, max_period, samples=self.period_grid_size, mode="log")
            #period_grid = [2.5, 50]
            for impact_param in impact_params_grid:
                for moon_period in period_grid:
                    search_inputs.extend(self.search_inputs(moon_period, alpha_grid, planet_mass, max_period,
                                                            impact_param))
        with Pool(processes=self.cpus, maxtasksperchild=500) as pool:
            all_residuals = list(
                tqdm(pool.imap(self.search, search_inputs, chunksize=100),
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
                                             inject_moon=True)
                logging.info("Injected moon before detrending: P=%.4f d, R=%.2f R_e, alpha=%.4f rad",
                             inject_params['moon_period'], inject_params['moon_radius'], inject_params['initial_alpha'])
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
                                         inject_planet=False, inject_moon=True)
            logging.info("Injected moon: P=%.4f d, R=%.2f R_e, alpha=%.4f rad",
                         inject_params['moon_period'], inject_params['moon_radius'], inject_params['initial_alpha'])
            self.inject_flux_time = self.time.copy()
        before_planet_fit = copy.copy(self)
        self.time, self.flux, self.flux_err = self.subtract_planet_transit(self.time, self.flux, self.flux_err, mode=self.subtract_mode, plot=plot)
        self.build_search_templates()
        self.check_model_validity()
        self.set_red_noise_factor(before_trim)
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
        refined_alpha_per_period = ExoMoonLeastSquares.refine_alpha(alpha_grid, residuals_matrix)
        snr_refined_alpha_per_period = ExoMoonLeastSquares.refine_alpha(alpha_grid, -snr_matrix)
        best_residuals_per_scenarios = np.array(best_residuals_per_scenarios)
        transposed_best_residuals_per_scenarios = np.transpose(best_residuals_per_scenarios)
        residuals_per_scenario_baseline_args = np.argwhere(transposed_best_residuals_per_scenarios[2] > ExoMoonLeastSquares.MAX_RESIDUAL / 2 ).flatten()
        transposed_best_residuals_per_scenarios[2][residuals_per_scenario_baseline_args] = max_residual
        best_residuals_per_scenarios = np.transpose(transposed_best_residuals_per_scenarios)
        best_residuals_per_scenarios_plot = best_residuals_per_scenarios[np.argsort(np.array([best_residual_per_scenarios[2] for best_residual_per_scenarios in best_residuals_per_scenarios]).flatten())]
        fig_transit, axs = plt.subplots(5, 1, figsize=(20, 35))
        axs[0].plot(period_grid, best_residuals_spectra, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        axs[0].set_title("Residuals Raw")
        axs[0].set_xlabel('P (days)')
        axs[0].set_ylabel('Residuals')
        residuals_per_scenario_baseline_args = np.argwhere(best_residuals_spectra > ExoMoonLeastSquares.MAX_RESIDUAL / 2).flatten()
        best_residuals_spectra[residuals_per_scenario_baseline_args] = max_residual
        axs[1].plot(period_grid, best_residuals_spectra, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        axs[1].set_title("Residuals Reduced")
        axs[1].set_xlabel('P (days)')
        axs[1].set_ylabel('Residuals')
        snr_spectrum = np.transpose(best_snr_per_period)[1]
        axs[2].plot(period_grid, snr_spectrum)
        axs[2].set_title("SNRs")
        axs[2].set_xlabel('P (days)')
        axs[2].set_ylabel('SNR')
        kernel = ExoMoonLeastSquares.SPECTRUM_KERNEL
        SR, power_raw, power, SDE_raw, SDE = ExoMoonLeastSquares.spectra(best_residuals_spectra, 1, kernel)
        axs[3].plot(period_grid, power, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        axs[3].set_title("SDE (residuals)")
        axs[3].set_xlabel('P (days)')
        axs[3].set_ylabel('Power')
        snr_median = ExoMoonLeastSquares.running_median(snr_spectrum, kernel)
        snr_norm = snr_spectrum - snr_median
        snr_sde = np.max(snr_norm / np.std(snr_norm))
        snr_norm = snr_norm * snr_sde / np.max(snr_norm)
        axs[4].plot(period_grid, snr_norm, color='gray', alpha=1)
        axs[4].set_title("SDE (SNR)")
        axs[4].set_xlabel('P (days)')
        axs[4].set_ylabel('Power')
        plt.savefig(self.object_dir + "/stats.png")
        plt.close(fig_transit)
        residuals_score = np.nanmax(residuals_matrix) / residuals_matrix - 1
        plt.imshow(residuals_score, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.savefig(self.object_dir + "/residuals_map.png")
        plt.close()
        for i in np.arange(0, 5):
            moon_period = best_residuals_per_scenarios_plot[i][0]
            moon_initial_alpha = best_residuals_per_scenarios_plot[i][1]
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.transit_semimajor_axis(),
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                self.search_input(moon_period, moon_initial_alpha, self.planet_mass, period_grid[-1],
                                  self.planet_impact_param),
                return_lc=True)
            if len(model) == 0:
                continue
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
            logging.info("Best residual for period %s, alpha %s: Residual->%s, Radius->%s, Depth->%s, SNR->%s",
                         moon_period,
                         moon_initial_alpha, best_residuals_per_scenarios_plot[i][2],
                         moon_radius, moon_depth, model_snr)
            scenario_time = np.append(it_time, oot_time)
            scenario_flux = np.append(it_flux, oot_flux)
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            fig_transit.suptitle(f"Period {moon_period}, alpha {moon_initial_alpha}\n"
                f"Residual->{best_residuals_per_scenarios_plot[i][2]}, Radius->{moon_radius}, Depth->{moon_depth}, "
                f"SNR->{model_snr}")
            self.plot_signal(axs[0], scenario_time, scenario_flux, it_time, oot_time, model)
            args_in_moon_transit_duration = np.argwhere(np.abs(oot_time) < moon_transit_duration).flatten()
            scenario_time = np.append(it_time, oot_time[args_in_moon_transit_duration])
            scenario_flux = np.append(it_flux, oot_flux[args_in_moon_transit_duration])
            self.plot_signal(axs[1], scenario_time, scenario_flux, it_time, oot_time[args_in_moon_transit_duration], model)
            plt.savefig(self.object_dir + "/Residuals " + str(i) + ".png")
            plt.close(fig_transit)
        top_sde_args = np.flip(np.argsort(power))
        for i, top_sde_index in enumerate(top_sde_args[0:5]):
            moon_period = period_grid[top_sde_index]
            moon_initial_alpha = refined_alpha_per_period[top_sde_index][0]
            sde = power[top_sde_index]
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.transit_semimajor_axis(),
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                self.search_input(moon_period, moon_initial_alpha, self.planet_mass, period_grid[-1],
                                  self.planet_impact_param),
                return_lc=True)
            if len(model) == 0:
                continue
            scenario_time = np.append(it_time, oot_time)
            scenario_flux = np.append(it_flux, oot_flux)
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
            fig_transit.suptitle(f"Period {moon_period}, alpha {moon_initial_alpha}\n"
                f"SDE->{sde}, Radius->{moon_radius}, Depth->{moon_depth}, "
                f"SNR->{model_snr}")
            self.plot_signal(axs[0], scenario_time, scenario_flux, it_time, oot_time, model)
            args_in_moon_transit_duration = np.argwhere(np.abs(oot_time) < moon_transit_duration).flatten()
            scenario_time = np.append(it_time, oot_time[args_in_moon_transit_duration])
            scenario_flux = np.append(it_flux, oot_flux[args_in_moon_transit_duration])
            self.plot_signal(axs[1], scenario_time, scenario_flux, it_time, oot_time[args_in_moon_transit_duration], model)
            plt.savefig(self.object_dir + "/SDE " + str(i) + ".png")
            plt.close(fig_transit)
        best_scenario_per_snr_args = np.argsort(np.nan_to_num(np.transpose(best_snr_per_period)[1], nan=-np.inf))
        for i in np.arange(0, 5):
            index = best_scenario_per_snr_args[-1 -i]
            moon_period = period_grid[index]
            moon_initial_alpha = best_snr_per_period[index][0]
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.transit_semimajor_axis(),
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                self.search_input(moon_period, moon_initial_alpha, self.planet_mass, period_grid[-1],
                                  self.planet_impact_param),
                return_lc=True)
            if len(model) == 0:
                logging.info("Best SNR for period %s, alpha %s: no in-transit data, skipping",
                             moon_period, moon_initial_alpha)
                continue
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            moon_depth = 1 - np.min(model)
            moon_radius = ExoMoonLeastSquares.depth_to_radius(moon_depth, self.star_radius)
            logging.info("Best SNR for period %s, alpha %s: Radius->%s, Depth->%s, SNR->%s",
                         moon_period,
                         moon_initial_alpha,
                         moon_radius, moon_depth, model_snr)
            scenario_time = np.append(it_time, oot_time)
            scenario_flux = np.append(it_flux, oot_flux)
            self.plot_signal(axs[0], scenario_time, scenario_flux, it_time, oot_time, model)
            fig_transit.suptitle(
                f"Period {moon_period}, alpha {moon_initial_alpha}\n"
                f"Radius->{moon_radius}, Depth->{moon_depth}, "
                f"SNR->{model_snr}")
            args_in_moon_transit_duration = np.argwhere(np.abs(oot_time) < moon_transit_duration).flatten()
            scenario_time = np.append(it_time, oot_time[args_in_moon_transit_duration])
            scenario_flux = np.append(it_flux, oot_flux[args_in_moon_transit_duration])
            self.plot_signal(axs[1], scenario_time, scenario_flux, it_time, oot_time[args_in_moon_transit_duration], model)
            plt.savefig(self.object_dir + "/SNR " + str(i) + ".png")
            plt.close(fig_transit)
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
        best_result = self.search(best_input, return_lc=True)
        moon_radius = float(best_result[2]) if len(best_result[-1]) else 0.0
        self.best_snr_radius = moon_radius
        # H15: the radius comes from the amplitude, so a dip deeper than the searched range is
        # reported as measured and flagged, not limited to max_radius.
        self.best_snr_radius_above_range = bool(moon_radius > self.max_radius)
        self.best_snr_radius_below_range = bool(0 < moon_radius < self.min_radius)
        if self.best_snr_radius_below_range:
            logging.warning("Best SNR radius %.2f R_earth is below the searched range (min %.2f R_earth): the "
                            "dip is shallower than the smallest moon searched for", moon_radius, self.min_radius)
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
            period_match, alias_match, alias_k, alias_period, min_alias_err = \
                ExoMoonLeastSquares.classify_recovered_period(inject_params['moon_period'], self.planet_period,
                                                              best_snr_period)
            self.injection_alias_k = alias_k
            logging.info("INJECTION RECOVERY (SNR): injected P=%.4f d, alpha=%.4f rad -> recovered P=%.4f d, alpha=%.4f rad (delta_P=%.2f%%, delta_cos-alpha=%.4f)",
                         inject_params['moon_period'], inject_params['initial_alpha'],
                         best_snr_period, best_snr_alpha, period_err * 100, alpha_err)
            if alias_k is not None and alias_k != 0:
                logging.info("  Best alias match: k=%d, P_alias=%.4f d (delta=%.2f%%)",
                             alias_k, alias_period, min_alias_err * 100)
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
        # Plot best moon model on each single transit
        try:
            has_injected_plot = hasattr(self, 'injected_flux_moon_pandora')
            # Get recovered radius from search at best (P, alpha)
            _, _, recovered_radius, _, _, _, _, _, _ = self.search(best_input, return_lc=True)
            logging.info("Recovered moon radius: %.4f R_earth", recovered_radius)
            for i, t0 in enumerate(self.bary_t0s):
                mask_orbit = np.abs(self.time - t0) < half_orbit_dur_max
                if len(self.time[mask_orbit]) < self.min_cadences_per_t0:
                    continue
                t_local = self.time[mask_orbit]
                f_local = self.flux[mask_orbit]
                # Use pre-computed T0s
                moon_t0_t = rec_t0s[i]
                moon_t0_t_inj = inj_t0s[i] if has_injected else None
                ma_moon = batman.TransitParams()
                ma_moon.t0 = moon_t0_t
                ma_moon.per = self.planet_period
                ma_moon.rp = max(recovered_radius, self.min_radius) * R_earth_to_R_sun / self.star_radius
                ma_moon.a = self.transit_semimajor_axis() / self.star_radius
                ma_moon.inc = self.planet_inc
                ma_moon.ecc = self.planet_ecc
                ma_moon.w = 0
                ma_moon.u = self.ab
                ma_moon.limb_dark = "quadratic"
                m_moon = ExoMoonLeastSquares.transit_model(ma_moon, t_local, self.exposure_time)
                model_moon = m_moon.light_curve(ma_moon)
                fig_t, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.axvspan(t0 - self.planet_duration / 2, t0 + self.planet_duration / 2,
                           color='blue', alpha=0.1, label='Planet transit window')
                if has_injected_plot:
                    mask_orbit_full = np.abs(self.inject_flux_time - t0) < half_orbit_dur_max
                    inj_local = self.injected_flux_moon_pandora[mask_orbit_full]
                    ax.plot(self.inject_flux_time[mask_orbit_full], inj_local,
                            color='green', alpha=0.6, label='Pandora injected moon')
                    if moon_t0_t_inj is not None:
                        ax.axvspan(moon_t0_t_inj - half_moon_dur, moon_t0_t_inj + half_moon_dur,
                                   color='gray', alpha=0.2)
                        ax.axvline(moon_t0_t_inj, color='red', linestyle='--',
                                   label='Injected moon T0')
                ax.scatter(t_local, f_local, color='gray', alpha=0.7, s=10,
                           label='Subtracted flux')
                ax.plot(t_local, model_moon, color='darkorange', alpha=1,
                        label='Recovered batman model')
                ax.axvline(moon_t0_t, color='darkorange', linestyle=':',
                           label='Recovered moon T0')
                title = f"T0={t0:.4f}\nRecovered P={best_snr_period:.3f}d alpha={best_snr_alpha:.3f}rad"
                if has_injected_plot:
                    title += f"\nInjected P={self.injected_moon_period:.3f}d alpha={self.injected_moon_alpha:.3f}rad"
                ax.set_title(title)
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Flux')
                ax.legend(fontsize=7)
                plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon_model.png")
                plt.close(fig_t)
        except Exception as e:
            logging.warning("Could not plot moon model transits: %s", e)

        self.max_snr = float(selected_snr)
        self.max_sde = float(SDE)
        self.aliases = self.alias_family(period_grid, snr_spectrum, best_snr_period)
        close = [alias for alias in self.aliases if alias["k"] != 0 and alias["snr_below_best"] is not None
                 and alias["snr_below_best"] < 1]
        if close:
            logging.info("Period aliases within 1 SNR of the best (P=%.4f d): %s; the period is not identified "
                         "among them (audit item H9)", best_snr_period,
                         ", ".join(f"k={alias['k']} P={alias['period']:.4f} d SNR {alias['snr']:.2f}"
                                   for alias in close[:5]))
        self.absorbed_fraction = self.measure_absorbed_fraction(before_planet_fit, best_input, rec_t0s)
        self.assess_single_event(best_input)
        if null_snrs is not None:
            self.null_snrs = [float(value) for value in null_snrs]
        elif self.null_runs > 0:
            self.null_snrs = self.run_nulls(before_trim, period_grid)
        self.fap, self.fap_empirical = self.false_alarm(self.max_snr)
        self.detected = None if self.fap is None else bool(self.fap < self.fap_threshold)
        if self.fap is None:
            logging.info("No false-alarm probability: SNR %.2f at P=%.4f d, alpha=%.3f rad is a search statistic "
                         "without a calibrated significance%s", self.max_snr, best_snr_period, best_snr_alpha,
                         "" if self.null_runs or null_snrs is not None else " (no null runs requested)")
        else:
            logging.info("SNR %.2f at P=%.4f d, alpha=%.3f rad: false-alarm probability %.2g (Gumbel fit to %d "
                         "nulls; empirical %.2g), %s at the %.2g threshold", self.max_snr, best_snr_period,
                         best_snr_alpha, self.fap, len(self.null_snrs), self.fap_empirical,
                         "detected" if self.detected else "not detected", self.fap_threshold)
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


class SearchInput:
    """Container for moon search input parameters.

    EMLS searches coplanar, circular, prograde moon orbits (audit item G12): only the moon period
    and its phase at the reference epoch vary.
    """
    def __init__(self, moon_period, moon_alpha, min_radius, impact_param,
                 moon_max_orbit_transit_duration, moon_transit_duration, moon_orbit_transit_duration,
                 half_moon_transit_duration, half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration) -> None:
        """Initialize search input parameters for a single moon search scenario.

        Parameters
        ----------
        moon_period : float
            Moon orbital period in days.
        moon_alpha : float
            Moon initial orbital phase in radians.
        min_radius : float
            Minimum moon radius in Earth radii.
        impact_param : float
            Impact parameter.
        moon_max_orbit_transit_duration : float
            Maximum moon orbit transit duration in days.
        moon_transit_duration : float
            Moon transit duration in days.
        moon_orbit_transit_duration : float
            Moon orbit transit duration in days.
        half_moon_transit_duration : float
            Half the moon transit duration in days.
        half_moon_orbit_transit_duration : float
            Half the moon orbit transit duration in days.
        half_moon_max_orbit_transit_duration : float
            Half the maximum moon orbit transit duration in days.
        """
        self.moon_period = moon_period
        self.moon_alpha = moon_alpha
        self.min_radius = min_radius
        self.impact_param = impact_param
        self.moon_max_orbit_transit_duration = moon_max_orbit_transit_duration
        self.moon_transit_duration = moon_transit_duration
        self.moon_orbit_transit_duration = moon_orbit_transit_duration
        self.half_moon_transit_duration = half_moon_transit_duration
        self.half_moon_orbit_transit_duration = half_moon_orbit_transit_duration
        self.half_moon_max_orbit_transit_duration = half_moon_max_orbit_transit_duration

def catalog_csv_path(catalog):
    """Return the path of a local catalog table.

    Parameters
    ----------
    catalog : str
        Catalog name, such as "tois" or "kois".

    Returns
    -------
    str
        Path inside the SHERLOCK directory of the user.
    """
    return os.path.join(os.path.expanduser('~'), '.sherlockpipe', f"{catalog}.csv")


def emls_output_dir(target_name, object_of_interest):
    """Return the output directory of one planet.

    Parameters
    ----------
    target_name : str
        Name of the star.
    object_of_interest : str
        Identifier of the planet, which keeps planets of one star apart.

    Returns
    -------
    str
        Directory path.
    """
    return f"{target_name}_{object_of_interest}_EMLS/"


def prepare_koi_table(tois_df):
    """Add the columns the EMLS driver needs to a KOI table.

    Parameters
    ----------
    tois_df : pandas.DataFrame
        KOI table.

    Returns
    -------
    pandas.DataFrame
        The table with the derived columns.
    """
    if "Predicted Mass (M_Earth)" not in tois_df.columns:
        tois_df["Predicted Mass (M_Earth)"] = tois_df["Planet Radius (R_Earth)"] ** 1.5
    if "Stellar Mass (M_Sun)" not in tois_df.columns:
        g = 10 ** tois_df["Stellar log(g) (cm/s^2)"] / 100
        r_m = tois_df["koi_srad"] * ac.R_sun.value
        tois_df["Stellar Mass (M_Sun)"] = g * r_m ** 2 / (G * Msolar_to_kg)
    if "Stellar Radius (R_Sun)" not in tois_df.columns:
        tois_df["Stellar Radius (R_Sun)"] = tois_df["koi_srad"]
    tois_df["Planet SNR"] = tois_df["koi_model_snr"]
    tois_df["Planet Radius (R_Earth) err"] = tois_df["koi_prad_err1"].fillna(0.1 * tois_df["Planet Radius (R_Earth)"])
    tois_df["planet_impact"] = tois_df["koi_impact"].fillna(0)
    return tois_df


def koi_planet_inclination(impact_param, star_mass, star_radius, planet_period):
    """Return the inclination of a KOI in degrees from its impact parameter.

    Parameters
    ----------
    impact_param : float
        Impact parameter in stellar radii.
    star_mass : float
        Stellar mass in solar masses.
    star_radius : float
        Stellar radius in solar radii.
    planet_period : float
        Planet period in days.

    Returns
    -------
    float
        Inclination in degrees, at most 90.
    """
    a_rsun = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period) * AU_TO_RSUN
    return min(np.rad2deg(np.arccos(np.clip(impact_param * star_radius / a_rsun, 0, 1))), 90)


def catalog_planet_params(row, catalog, star_mass, star_radius, first_time, bjd_offset):
    """Return the planet and moon-radius inputs of EMLS for one catalog row.

    Parameters
    ----------
    row : pandas.Series
        Catalog row, after `prepare_koi_table` for KOIs.
    catalog : str
        Catalog name, ``"kois"`` or a TOI-like catalog.
    star_mass : float
        Stellar mass in solar masses.
    star_radius : float
        Stellar radius in solar radii.
    first_time : float
        First time of the light curve, used to move the epoch into the data.
    bjd_offset : float
        Offset between the catalog epoch and the light curve time.

    Returns
    -------
    dict
        Keyword values named as the `ExoMoonLeastSquares` arguments. The moon radius
        range is Rp/20 to Rp/2 for every catalog.
    """
    planet_radius = row['Planet Radius (R_Earth)']
    planet_period = row['Period (days)']
    planet_t0 = row['Epoch (BJD)'] - bjd_offset
    planet_t0 = first_time + (planet_t0 - first_time) % planet_period
    planet_mass = row['Predicted Mass (M_Earth)']
    if planet_mass is None or np.isnan(planet_mass):
        planet_mass = planet_radius ** 1.5
    planet_semimajor_axis = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period)
    if catalog == "kois":
        planet_impact_param = float(row["planet_impact"])
        planet_inc = koi_planet_inclination(planet_impact_param, star_mass, star_radius, planet_period)
        planet_inc_err = 0.5
    else:
        planet_inc = 90
        planet_inc_err = 3
        planet_impact_param = LcbuilderHelper.convert_from_to(planet_semimajor_axis, u.au, u.R_sun) / star_radius * \
            np.cos(np.deg2rad(planet_inc))
    return {
        'planet_radius': planet_radius,
        'planet_radius_err': row['Planet Radius (R_Earth) err'],
        'planet_period': planet_period,
        'planet_t0': planet_t0,
        'planet_duration': row['Duration (hours)'] / 24,
        'planet_inc': planet_inc,
        'planet_inc_err': planet_inc_err,
        'planet_ecc': 0.15,
        'planet_ecc_err': 0.15,
        'planet_arg_periastron': 0,
        'planet_mass': planet_mass,
        'planet_semimajor_axis': planet_semimajor_axis,
        'planet_impact_param': planet_impact_param,
        'min_radius': planet_radius / 20,
        'max_radius': planet_radius / 2,
    }


if __name__ == '__main__':

    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger()
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    
    lc_builder = LcBuilder()
    max_moon_density = 5.5
    catalog = "tois"
    if "--catalog" in sys.argv:
        idx = sys.argv.index("--catalog")
        if idx + 1 < len(sys.argv):
            catalog = sys.argv[idx + 1]
    csv_path = catalog_csv_path(catalog)
    tois_df = pd.read_csv(csv_path)
    cadence = [120]
    author = 'SPOC'
    if catalog == "kois":
        author = 'Kepler'
        cadence = [60]
        tois_df = prepare_koi_table(tois_df)
    confirmed_tois_df = tois_df.loc[(tois_df['Disposition'] == 'CP') | (tois_df['Disposition'] == 'KP') | (tois_df['Disposition'] == 'CONFIRMED') | (tois_df['Disposition'] == 'CANDIDATE')]
    confirmed_tois_df = confirmed_tois_df.loc[confirmed_tois_df['Planet SNR'] > 15]
    confirmed_tois_df = confirmed_tois_df.sort_values(by='Planet SNR', ascending=False)
    # Moon dynamical stability filter
    def hill_roche_stable(period, p_radius, p_mass, star_mass):
        """Check if Hill radius exceeds Roche limit: r_Hill > 2 * r_Roche.

        The Hill sphere (Hill 1878, Am. J. Math. 1, 5) defines the region
        where the planet's gravity dominates over the star:
            r_Hill = a_p × (M_p / 3M_star)^(1/3)

        The Roche limit (Roche 1849, Acad. Sci. Montpellier) for a fluid
        satellite is the closest orbit before tidal disruption:
            r_Roche = 2.44 × R_p × (ρ_p / ρ_m)^(1/3)

        A moon can only exist if r_Hill > r_Roche. The factor 2 margin for
        long-term stability is a heuristic."""
        a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, period)
        r_hill = a * (p_mass * M_earth_to_M_sun / (3 * star_mass)) ** (1 / 3)
        moon_density = 5.5
        r_roche = ExoMoonLeastSquares.compute_roche_radius(p_radius, p_mass, moon_density)
        r_roche_au = r_roche * R_earth_to_R_sun / AU_TO_RSUN
        return r_hill > r_roche_au * 2

    def planet_planet_stable(a1, m1_earth, a2, m2_earth, star_mass, n_hill=5):
        """Check if two planets are separated by n_hill mutual Hill radii.

        Two planets are dynamically packed (and moons can be ejected) if
        their separation is less than K × R_mHill where K = 3-5
        (Gladman 1993, Icarus 106, 247; Chambers et al. 1996, Icarus 119, 261).

        The mutual Hill radius is:
            R_mHill = (a1 + a2)/2 × ((M1 + M2) / 3M_star)^(1/3)

        Masses in Earth masses, semi-major axes in AU, star_mass in M_sun."""
        delta_a = abs(a1 - a2)
        m_total = (m1_earth + m2_earth) * M_earth_to_M_sun
        r_mhill = (a1 + a2) / 2 * (m_total / (3 * star_mass)) ** (1 / 3)
        return delta_a >= n_hill * r_mhill

    def moon_tidal_stable(period, p_mass, p_radius, star_mass, t_age_gyr=5):
        """Check if a moon can survive tidal migration over the system age.

        Heller et al. (2014, Astrobiology 14, 798) derived an approximate
        threshold for moon survival against planet-moon tidal migration:
            P_planet > P_crit × (M_p/M_⊕)^(1/6) × (age/1 Gyr)^(1/4)

        with P_crit ≈ 10 days for rocky planets (Q_p ≈ 100, k2p ≈ 0.3).
        Below this period, tides cause the moon to spiral into the planet
        within the system's age (Barnes & O'Brien 2002, ApJ 575, 1087)."""
        P_crit = 10.0 * (p_mass ** (1.0 / 6.0)) * (t_age_gyr ** (1.0 / 4.0))
        return period > P_crit

    keep_mask = []
    n_hillroche, n_tidal, n_companion = 0, 0, 0
    for _, r in confirmed_tois_df.iterrows():
        star_mass = r['Stellar Mass (M_Sun)']
        p_mass = r['Predicted Mass (M_Earth)']
        p_radius = r['Planet Radius (R_Earth)']
        period = r['Period (days)']
        if np.isnan(star_mass) or star_mass <= 0:
            keep_mask.append(False)
            continue
        if np.isnan(p_mass) or p_mass <= 0:
            p_mass = p_radius ** 1.5
        a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, period)
        p_mass_sun = p_mass * M_earth_to_M_sun
        moon_stable = hill_roche_stable(period, p_radius, p_mass, star_mass)
        if not moon_stable:
            n_hillroche += 1
            keep_mask.append(False)
            continue
        if not moon_tidal_stable(period, p_mass, p_radius, star_mass):
            n_tidal += 1
            keep_mask.append(False)
            continue
        sisters = confirmed_tois_df[confirmed_tois_df['Object Id'] == r['Object Id']]
        companion_ok = True
        for _, sis in sisters.iterrows():
            if sis['OI'] == r['OI']:
                continue
            sis_mass = sis['Predicted Mass (M_Earth)']
            if np.isnan(sis_mass) or sis_mass <= 0:
                sis_mass = sis['Planet Radius (R_Earth)'] ** 1.5
            sis_a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, sis['Period (days)'])
            if not planet_planet_stable(a, p_mass, sis_a, sis_mass, star_mass):
                companion_ok = False
                break
        if not companion_ok:
            n_companion += 1
        keep_mask.append(companion_ok)
    confirmed_tois_df = confirmed_tois_df[keep_mask]
    logging.info("Dynamical filter: Hill-Roche=%d, Tidal=%d, Companion=%d removed, %d remain",
                 n_hillroche, n_tidal, n_companion, len(confirmed_tois_df))
    #confirmed_tois_df = confirmed_tois_df.loc[confirmed_tois_df['Object Id'] == 'TIC 309792357']
    confirmed_tois_df = confirmed_tois_df.iloc[0:100]
    for index, row in confirmed_tois_df.iterrows():
        target_name = row['Object Id']
        logging.info("Processing target: %s", target_name)
        object_dir = emls_output_dir(target_name, row['OI'])
        if not os.path.exists(object_dir):
            os.mkdir(object_dir)
        try:
            object_info = lc_builder.build_object_info(target_name=target_name, author=[author], sectors="all", file=None, cadence=cadence,
                                          initial_mask=[], initial_transit_mask=None, star_info=None, aperture=None,
                                          eleanor_corr_flux="pdcsap_flux", outliers_sigma=3, high_rms_enabled=False,
                                          high_rms_threshold=1.5, high_rms_bin_hours=4, smooth_enabled=False, binning=0,
                                          auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                                          auto_detrend_period=None, prepare_algorithm=None, reduce_simple_oscillations=False,
                                          oscillation_snr_threshold=4, oscillation_amplitude_threshold=0.1, oscillation_ws_scale=60,
                                          oscillation_min_period=0.002, oscillation_max_period=0.2)
            lc_build = lc_builder.build(object_info, object_dir)
            star_mass = lc_build.star_info.mass
            star_radius = lc_build.star_info.radius
            times = lc_build.lc.time.value
            flux = lc_build.lc.flux.value
            flux_err = lc_build.lc.flux_err.value
            ld_coeffs = lc_build.star_info.ld_coefficients
            bjd_offset = 2454833 if catalog == "kois" else 2457000
            companions_df = tois_df.loc[(tois_df['Object Id'] == target_name) & (tois_df['OI'] != row['OI'])]
            for companion_index, companion_row in companions_df.iterrows():
                logging.info(f"Masking companion {companion_row['OI']}")
                companion_t0 = companion_row['Epoch (BJD)'] - bjd_offset
                companion_period = companion_row['Period (days)']
                companion_duration = companion_row['Duration (hours)'] / 24 * 2
                times, flux, flux_err = LcbuilderHelper.mask_transits(times, flux, companion_period, companion_duration,
                                                                     companion_t0, flux_err=flux_err)
            # star_mass = 0.88
            # star_radius = 0.878413
            # times = np.linspace(planet_t0 - 5, planet_t0 + 1000, 1005 * 60 * 24 // 2)
            # flux = np.random.normal(1, 0.001, len(times))
            # flux_err = np.random.normal(0.001, 0.0001, len(times))
            # ld_coeffs = [0.4136, 0.1999]
            planet = catalog_planet_params(row, catalog, star_mass, star_radius, times[0], bjd_offset)
            planet_radius = planet['planet_radius']
            planet_radius_err = planet['planet_radius_err']
            planet_period = planet['planet_period']
            planet_t0 = planet['planet_t0']
            planet_duration = planet['planet_duration']
            planet_inc = planet['planet_inc']
            planet_inc_err = planet['planet_inc_err']
            planet_ecc = planet['planet_ecc']
            planet_ecc_err = planet['planet_ecc_err']
            planet_arg_periastron = planet['planet_arg_periastron']
            planet_mass = planet['planet_mass']
            planet_semimajor_axis = planet['planet_semimajor_axis']
            planet_impact_param = planet['planet_impact_param']
            min_radius = planet['min_radius']
            max_radius = planet['max_radius']
            # planet_radius = 11.8142656578
            # planet_radius_err = 1.00
            # planet_period = 141.834025
            # planet_duration = 8.681 / 24
            # planet_inc = 89.903
            # planet_inc_err = 0.5
            # planet_ecc = 0.212
            # planet_ecc_err = 0.022
            # planet_arg_periastron = 0
            # planet_mass = 408.727
            # min_radius = 1
            # max_radius = 3
            flux_err = np.ma.filled(flux_err, np.ma.median(flux_err))
            flux = wotan.flatten(times, flux, method="biweight", window_length=planet_duration * 5)
            t0s = [i for i in np.arange(planet_t0, np.max(times), planet_period)]
            emls = ExoMoonLeastSquares(object_dir, 4, star_mass, star_radius, ld_coeffs,
                                       planet_radius, planet_radius_err, planet_mass, planet_period, planet_t0, planet_duration,
                                       planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc, planet_ecc_err, planet_arg_periastron, planet_impact_param,
                                       min_radius, max_radius, t0s, times, flux, flux_err, max_moon_density,
                                       period_grid_size=10000, radius_grid_size=20, subtract_mode='subtract', fit_rp_per_transit=False,
                                       min_transit_completeness=0.75, max_subtraction_rms_ratio=1.5)
            # emls.flux = emls.inject_moon(emls.time, emls.flux, emls.flux_err, emls.bary_t0s, planet_semimajor_axis,
            #                             planet_ecc, moon_radius, moon_period, moon_mass, plot=True, initial_alpha=moon_alpha,
            #                              inject_planet=False, inject_moon=False)
            p_m_relationship = 5
            emls.run(plot=True,
                    #  inject_params={
                    #     'moon_radius': planet_radius / p_m_relationship,
                    #     'moon_period': planet_duration * 20,
                    #     'initial_alpha': 1.0,
                    #     'moon_mass': (planet_radius / p_m_relationship) ** 1.5,
                    # }
            )
        except Exception as e:
            logging.exception(f"Error in run for target {object_dir}")
