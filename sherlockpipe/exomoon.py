import itertools
import copy
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
import astropy.units as u
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from scipy.optimize import minimize, least_squares, brentq, minimize_scalar
from batman import TransitModel
from numba import njit
from pytransit import QuadraticModel
import batman
from tqdm import tqdm
import numpy as np
import astropy.constants as ac
import astropy.units as u
import wotan
from lcbuilder.lcbuilder_class import LcBuilder
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import lombscargle, argrelextrema
import matplotlib.pyplot as plt
import pandas as pd
from os import path

G = 6.674e-11  # m3 kg-1 s-2
AU_TO_RSUN = 215.032
Msolar_to_kg = 2.e30
Mearth_to_kg = 5.972e24
M_earth_to_M_sun = Mearth_to_kg / Msolar_to_kg
R_earth_to_R_sun = 0.009175
resources_dir = path.join(path.dirname(__file__))
# Optional injection adapter. It stays unloaded unless injection is requested,
# while allowing callers/tests to provide a compatible implementation.
pandora = None


@dataclass
class TransitFitResult:
    """Validated internal result for a local or shared planet fit."""

    theta: np.ndarray | None
    model: np.ndarray | None
    success: bool
    status: str
    message: str
    nfev: int
    objective: float
    initial_objective: float
    coverage: dict
    active_bounds: np.ndarray


@dataclass(frozen=True)
class InjectionTruthEvent:
    """Immutable moon-injection truth for one original planetary epoch."""

    epoch_id: int
    epoch_time: float
    cadence_ids: tuple
    moon_t0: float | None
    moon_t0_cadence_id: int | None
    coverage: str


class ExoMoonLeastSquares:
    """Performs exomoon least-squares search and fitting."""
    MAX_RESIDUAL = sys.maxsize

    def __init__(self, object_dir, cpus, star_mass, star_radius, ab, planet_radius, planet_radius_err,
                 planet_mass, planet_period,
                 planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc,
                 planet_ecc_err, planet_arg_periastron, planet_impact_param, min_radius, max_radius, t0s, time, flux,
                 flux_err, max_moon_density, period_grid_size=2000, radius_grid_size=10,
                 min_cadences_per_t0: bool = 20, min_orbit_window: float = 0,
                  subtract_mode: str = "subtract", mask_duration_factor: float = 1.0,
                  skip_fold_fit: bool = False, fit_rp_per_transit: bool = True,
                  min_transit_completeness: float = 0.5,
                  max_subtraction_rms_ratio: float = 3.0,
                  exposure_times_seconds=None):
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
            Positive, finite flux uncertainty array. ``None`` is not supported
            because EMLS uses uncertainty-weighted fitting and scoring.
        max_moon_density : float
            Maximum moon density in g/cm^3.
        period_grid_size : int, optional
            Number of grid points for period search, default 2000.
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
        """
        self.object_dir = object_dir
        self.cpus = cpus
        self.star_mass = star_mass
        self.star_radius = star_radius
        self.ab = ab
        self.planet_radius = planet_radius
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
        time, flux, flux_err = self._validate_light_curve(time, flux, flux_err)
        self.time = time.copy()
        self.flux = flux.copy()
        self.flux_err = flux_err.copy()
        self.exposure_times_seconds = self._validate_exposure_times(
            exposure_times_seconds, len(time))
        self.cadence_ids = np.arange(len(time))
        # Epoch IDs belong to the supplied ephemeris, never a filtered search
        # list.  They remain stable when an epoch is later rejected.
        self.epoch_ids_by_time = {float(epoch): index for index, epoch in enumerate(sorted(set(t0s)))}
        if (not (np.isfinite(min_radius) and np.isfinite(max_radius) and 0 < min_radius <= max_radius)
                or isinstance(radius_grid_size, bool) or not isinstance(radius_grid_size, (int, np.integer))
                or radius_grid_size < 1):
            raise ValueError("Moon radii must satisfy 0 < min_radius <= max_radius")
        if (isinstance(period_grid_size, bool) or not isinstance(period_grid_size, (int, np.integer))
                or period_grid_size < 2):
            raise ValueError("Moon period grid requires at least two samples")
        if not np.isfinite(max_moon_density) or max_moon_density <= 0:
            raise ValueError("Maximum moon density must be finite and positive")
        self.max_moon_density = max_moon_density
        self.min_cadences_per_t0 = min_cadences_per_t0
        self.min_orbit_window = min_orbit_window
        self.subtract_mode = subtract_mode
        self.mask_duration_factor = mask_duration_factor
        self.skip_fold_fit = skip_fold_fit
        self.fit_rp_per_transit = fit_rp_per_transit
        # Retained only for constructor compatibility. B5 rejects on validated
        # model/coverage failure, never an empirical residual-scatter heuristic.
        self.min_transit_completeness = min_transit_completeness
        self.max_subtraction_rms_ratio = max_subtraction_rms_ratio
        self.original_bary_t0s = sorted(set(t0s))
        self.bary_t0s = []
        for i, bary_t0 in enumerate(self.original_bary_t0s):
            mask = np.abs(time - bary_t0) < self.planet_duration * 2
            if not self.is_skip_t0(time[mask]):
                self.bary_t0s.append(bary_t0)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.period_grid_size = int(period_grid_size)
        self.radius_grid_size = int(radius_grid_size)
        self.planet_mass_grid = [planet_mass]
        self.moon_inc_grid = [90]
        self.moon_ecc_grid = [0]
        self.moon_arg_periastron_grid = [0]

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
        return a1 / 1.496e11

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
    def compute_hill_radius(major_mass, minor_mass, semimajor_axis, eccentricity=0):
        """
        :param major_mass: The main body mass
        :param minor_mass: The minor body mass
        :param semimajor_axis: The minor body semimajor axis in AU.
        :param eccentricity: the planet eccentricity
        :return: the hill radius of the minor body in the same units than the semimajor_axis
        """
        return semimajor_axis * (1 - eccentricity) * ((minor_mass / (3 * major_mass)) ** (1 / 3))

    @staticmethod
    def catalog_planet_geometry(catalog, star_mass, star_radius, planet_period,
                                planet_radius, planet_impact_param, planet_ecc=0.15,
                                planet_arg_periastron=0.0):
        """Validate catalog geometry before executing catalog/network work."""
        if catalog not in ('tois', 'kois'):
            raise ValueError("Catalog must be 'tois' or 'kois'")
        values = (star_mass, star_radius, planet_period, planet_radius,
                  planet_impact_param, planet_ecc, planet_arg_periastron)
        if not np.all(np.isfinite(values)):
            raise ValueError("Catalog planet geometry requires finite values")
        if star_mass <= 0 or star_radius <= 0 or planet_period <= 0 or planet_radius <= 0:
            raise ValueError("Catalog planet mass, radius, and period must be positive")
        if not 0 <= planet_ecc < 1 or planet_impact_param < 0:
            raise ValueError("Catalog eccentricity and impact parameter are invalid")

        planet_semimajor_axis = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period)
        a_over_rstar = (planet_semimajor_axis * AU_TO_RSUN) / star_radius
        if catalog == 'kois':
            planet_inc = ExoMoonLeastSquares.inclination_from_impact_parameter(
                a_over_rstar, planet_impact_param, planet_ecc, planet_arg_periastron)
            planet_inc_err = 0.5
        else:
            planet_inc = 90.0
            planet_inc_err = 3.0
            planet_impact_param = ExoMoonLeastSquares.impact_parameter(
                a_over_rstar, planet_inc, planet_ecc, planet_arg_periastron)
        return {
            'planet_semimajor_axis': planet_semimajor_axis,
            'planet_inc': planet_inc,
            'planet_inc_err': planet_inc_err,
            'planet_impact_param': planet_impact_param,
            'min_radius': planet_radius / 20.0,
            'max_radius': planet_radius / 2.0,
        }

    @staticmethod
    def au_to_period(mass, au):
        """
        Calculates the orbital period for the semi-major axis assuming a circular orbit.
        :param mass: the stellar mass
        :param au: the semi-major axis in astronomical units.
        :return: the period in days
        """
        mass_kg = mass * 2.e30
        a = au * 1.496e11
        return ((a ** 3) * 4 * (np.pi ** 2) / G / mass_kg) ** (1. / 2.) / 3600 / 24

    @staticmethod
    def min_period_for_orbit_duration(star_radius, planet_semimajor_axis,
                                      planet_period, planet_mass,
                                      orbit_duration_days, impact_parameter=0):
        """Minimum moon period with orbit transit duration >= orbit_duration_days.

        Parameters
        ----------
        star_radius : float
            Stellar radius in solar radii.
        planet_semimajor_axis : float
            Dimensionless planet semi-major axis (a/R_star).
        planet_period : float
            Planet orbital period in days.
        planet_mass : float
            Planet mass in solar masses.
        orbit_duration_days : float
            Desired minimum moon orbit transit duration in days.
        impact_parameter : float, optional
            Planet impact parameter, default 0.

        Returns
        -------
        float
            Minimum moon orbital period in days. Returns 0 if no such
            moon can exist (i.e., even at infeasibly large orbit).
        """
        # Invert compute_transit_duration using its circular-orbit geometry.
        # ``planet_semimajor_axis`` is a/R_star, so multiply only the final
        # dimensionless moon offset by star_radius to obtain solar radii.
        if not (star_radius > 0 and planet_semimajor_axis > 0 and planet_period > 0
                and 0 <= impact_parameter and 0 <= orbit_duration_days < planet_period / 2):
            raise ValueError("Orbit window requires positive geometry and duration below half a period")
        sin_contact_angle = np.sin(np.pi * orbit_duration_days / planet_period)
        moon_a_over_rstar = np.sqrt(
            sin_contact_angle ** 2 * (planet_semimajor_axis ** 2 - impact_parameter ** 2)
            + impact_parameter ** 2
        ) - 1
        a_moon_rsun = star_radius * moon_a_over_rstar
        if a_moon_rsun <= 0:
            return 0.0
        return ExoMoonLeastSquares.au_to_period(planet_mass, a_moon_rsun / AU_TO_RSUN)

    @staticmethod
    def compute_transit_duration(star_radius,
                                 transiting_body_semimajor_axis, transit_period, transiting_body_radius,
                                 impact_parameter=0):
        """Circular first-to-fourth contact duration in days.

        ``star_radius``, ``transiting_body_semimajor_axis``, and
        ``transiting_body_radius`` must use the same length unit.  This
        helper intentionally remains circular; eccentric planet contacts are
        computed by :meth:`contact_duration` from projected separation.
        """
        if not (star_radius > 0 and transiting_body_semimajor_axis > 0
                and transit_period > 0 and transiting_body_radius >= 0
                and impact_parameter >= 0):
            raise ValueError("Transit duration requires positive circular-orbit geometry")
        a_over_rstar = transiting_body_semimajor_axis / star_radius
        radius_ratio = transiting_body_radius / star_radius
        numerator = (1 + radius_ratio) ** 2 - impact_parameter ** 2
        denominator = a_over_rstar ** 2 - impact_parameter ** 2
        if numerator <= 0 or denominator <= 0:
            return 0.0
        argument = np.sqrt(numerator / denominator)
        if argument >= 1:
            return 0.0
        return transit_period / np.pi * np.arcsin(argument)

    @staticmethod
    def impact_parameter(a_over_rstar, inclination_degrees, eccentricity=0, argument_of_periastron=90):
        """Return the BATMAN-convention impact parameter at conjunction."""
        if not (a_over_rstar > 0 and 0 <= eccentricity < 1
                and 0 <= inclination_degrees <= 180):
            raise ValueError("Impact parameter requires a valid elliptic orbit and inclination")
        conjunction_scale = (1 - eccentricity ** 2) / (
            1 + eccentricity * np.sin(np.deg2rad(argument_of_periastron)))
        return abs(a_over_rstar * conjunction_scale * np.cos(np.deg2rad(inclination_degrees)))

    @staticmethod
    def inclination_from_impact_parameter(a_over_rstar, impact_parameter,
                                          eccentricity=0, argument_of_periastron=90):
        """Convert a nonnegative conjunction impact parameter to inclination."""
        if not (a_over_rstar > 0 and impact_parameter >= 0 and 0 <= eccentricity < 1):
            raise ValueError("Inclination conversion requires a valid elliptic orbit and impact parameter")
        conjunction_scale = (1 - eccentricity ** 2) / (
            1 + eccentricity * np.sin(np.deg2rad(argument_of_periastron)))
        cosine = impact_parameter / (a_over_rstar * conjunction_scale)
        if cosine > 1:
            raise ValueError("Impact parameter is not reachable for the supplied orbit")
        return np.rad2deg(np.arccos(np.clip(cosine, 0, 1)))

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
        if not (np.isfinite(min) and np.isfinite(max) and 0 < min < max) or samples < 1:
            raise ValueError("Moon period grid requires 0 < min < max and at least one sample")
        if "log" == mode:
            return np.logspace(math.log(min, 10), math.log(max, 10), samples, base=10)
        else:
            return np.linspace(min, max, samples)

    @staticmethod
    def _validate_light_curve(time, flux, flux_err):
        """Require aligned, finite observations with positive measured errors."""
        if flux_err is None:
            raise ValueError("flux_err is required for uncertainty-weighted transit fitting/search")
        arrays = tuple(np.asarray(x, dtype=float) for x in (time, flux, flux_err))
        if any(x.ndim != 1 for x in arrays) or len({len(x) for x in arrays}) != 1:
            raise ValueError("time, flux and flux_err must be aligned one-dimensional arrays")
        if not len(arrays[0]):
            raise ValueError("No observations available for transit fitting/search")
        if any(not np.all(np.isfinite(x)) for x in arrays) or np.any(arrays[2] <= 0):
            raise ValueError("Observations must be finite and flux_err strictly positive")
        return arrays

    @staticmethod
    def _validate_exposure_times(exposure_times_seconds, size):
        """Return aligned exposure durations in seconds, or ``None`` for instantaneous models."""
        if exposure_times_seconds is None:
            return None
        exposure_times_seconds = np.asarray(exposure_times_seconds, dtype=float)
        if exposure_times_seconds.ndim == 0:
            exposure_times_seconds = np.full(size, float(exposure_times_seconds))
        if exposure_times_seconds.ndim != 1 or len(exposure_times_seconds) != size:
            raise ValueError("exposure_times_seconds must be a scalar or align with time")
        if not np.all(np.isfinite(exposure_times_seconds)) or np.any(exposure_times_seconds < 0):
            raise ValueError("exposure_times_seconds must be finite and nonnegative")
        return exposure_times_seconds.copy()

    @staticmethod
    def _transit_model(params, time, exposure_times_seconds=None):
        """Evaluate BATMAN at cadence centers, integrating supplied finite exposures."""
        time = np.asarray(time, dtype=float)
        exposures = ExoMoonLeastSquares._validate_exposure_times(exposure_times_seconds, len(time))
        if exposures is None or not np.any(exposures):
            return TransitModel(params, time).light_curve(params)
        model = np.empty(len(time), dtype=float)
        # BATMAN accepts one exposure duration per model. Group exact metadata
        # values so mixed-cadence observations retain their own durations.
        for exposure in np.unique(exposures):
            mask = exposures == exposure
            if exposure == 0:
                model[mask] = TransitModel(params, time[mask]).light_curve(params)
            else:
                # Resolve integrations at ten seconds or finer. This remains
                # substantially coarser than the test reference while keeping
                # ingress/egress smearing accurate for long cadence data.
                supersample_factor = max(5, int(np.ceil(exposure / 10.0)))
                model[mask] = TransitModel(
                    params, time[mask], supersample_factor=supersample_factor,
                    exp_time=exposure / 86400.0).light_curve(params)
        return model

    @staticmethod
    def contact_duration(params):
        """First-to-fourth geometric contact duration in days.

        BATMAN defines ``t0`` at inferior conjunction, corresponding to
        true anomaly ``pi/2 - w``. Contacts are projected-separation roots
        around that conjunction, so this duration is independent of observed
        ingress/egress samples. It is a geometric, instantaneous-exposure
        calculation, not a claim about integrated-flux contact timing.
        """
        e, w = params.ecc, np.deg2rad(params.w)
        if not (0 <= e < 1 and params.a > 0 and params.rp >= 0
                and params.per > 0 and 0 <= params.inc <= 180):
            raise ValueError("Contact duration requires a valid elliptic orbit")
        center = np.pi / 2 - w
        left, right = center - np.pi / 2, center + np.pi / 2

        def separation(f):
            r = params.a * (1 - e * e) / (1 + e * np.cos(f))
            return r * np.sqrt(1 - np.sin(np.deg2rad(params.inc)) ** 2 * np.sin(f + w) ** 2)

        minimum = minimize_scalar(separation, bounds=(left, right), method='bounded').x
        if separation(minimum) >= 1 + params.rp:
            return 0.0
        try:
            f1 = brentq(lambda f: separation(f) - 1 - params.rp, left, minimum)
            f4 = brentq(lambda f: separation(f) - 1 - params.rp, minimum, right)
        except ValueError:
            return 0.0

        def mean_anomaly(f):
            eccentric_anomaly = 2 * np.arctan2(
                np.sqrt(1 - e) * np.sin(f / 2),
                np.sqrt(1 + e) * np.cos(f / 2))
            return eccentric_anomaly - e * np.sin(eccentric_anomaly)

        delta_mean_anomaly = (mean_anomaly(f4) - mean_anomaly(f1)) % (2 * np.pi)
        return delta_mean_anomaly * params.per / (2 * np.pi)

    @staticmethod
    def _fit_transit_result(time, flux, flux_err, guess, bounds, transit_params, fields,
                            expected_duration=None, exposure_times_seconds=None):
        """Fit in locally centered, unit-box coordinates and validate support."""
        try:
            time, flux, flux_err = ExoMoonLeastSquares._validate_light_curve(time, flux, flux_err)
        except ValueError as exc:
            return TransitFitResult(None, None, False, 'invalid_input', str(exc), 0,
                                    np.inf, np.inf, {}, np.array([], dtype=bool))
        if len(guess) != len(fields):
            return TransitFitResult(None, None, False, 'invalid_bounds',
                                    'Transit guess must have one value per fitted field.', 0,
                                    np.inf, np.inf, {}, np.array([], dtype=bool))
        params = copy.deepcopy(transit_params)
        try:
            exposure_times_seconds = ExoMoonLeastSquares._validate_exposure_times(
                exposure_times_seconds, len(time))
        except ValueError as exc:
            return TransitFitResult(None, None, False, 'invalid_input', str(exc), 0,
                                    np.inf, np.inf, {}, np.array([], dtype=bool))
        # Optimize offsets around a local origin, never an absolute BJD.
        origin = float(guess[0])
        time = np.ascontiguousarray(time - origin)
        guess = np.asarray(guess, dtype=float).copy()
        guess[0] = 0.0
        bounds = np.asarray(bounds, dtype=float).copy()
        if bounds.shape != (len(fields), 2) or len(guess) != len(fields):
            return TransitFitResult(None, None, False, 'invalid_bounds',
                                    'Transit bounds must have one finite pair per fitted field.', 0,
                                    np.inf, np.inf, {}, np.array([], dtype=bool))
        bounds[0] -= origin
        if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] < bounds[:, 0]):
            return TransitFitResult(None, None, False, 'invalid_bounds',
                                    'Transit parameter bounds must be finite and ordered.', 0,
                                    np.inf, np.inf, {}, np.array([], dtype=bool))
        scale = bounds[:, 1] - bounds[:, 0]
        active = scale > 0
        scale[~active] = 1
        guess = np.clip(guess, bounds[:, 0], bounds[:, 1])

        def unpack(x):
            theta = bounds[:, 0].copy()
            theta[active] += x * scale[active]
            return theta

        def model(theta):
            for field, value in zip(fields, theta):
                setattr(params, field, float(value))
            return ExoMoonLeastSquares._transit_model(params, time, exposure_times_seconds)

        def residuals(x):
            theta = unpack(x)
            m = model(theta)
            if not np.all(np.isfinite(m)):
                return np.full(len(flux) + (expected_duration is not None), np.inf)
            residual = (flux - m) / flux_err
            if expected_duration is not None:
                duration = ExoMoonLeastSquares.contact_duration(params)
                residual = np.append(residual, np.sqrt(2e4) * max(0, expected_duration - duration))
            return residual

        def objective(x):
            return 0.5 * np.sum(residuals(x) ** 2)

        x0 = ((guess - bounds[:, 0]) / scale)[active]
        initial_objective = objective(x0)
        if not np.isfinite(initial_objective):
            return TransitFitResult(None, None, False, 'nonfinite_initial_model',
                                    'Initial transit model/objective is not finite.', 0,
                                    np.inf, np.inf, {}, np.array([], dtype=bool))
        starts = [x0]
        if active[0]:
            for fraction in (0.25, 0.75):
                start = x0.copy()
                start[0] = fraction
                starts.append(start)
        candidates = []
        for start in starts:
            if not np.any(active):
                candidates.append((objective(start), unpack(start), 0, 'fixed parameters'))
                break
            try:
                res = least_squares(residuals, start, bounds=(0, 1), x_scale='jac',
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000)
            except Exception:
                continue
            value = objective(res.x)
            if res.success and np.isfinite(value) and np.all(np.isfinite(res.x)):
                candidates.append((value, unpack(res.x), res.nfev, res.message))
        if not candidates:
            return TransitFitResult(None, None, False, 'optimizer_failure',
                                    'Transit optimizer did not converge to a finite solution.', 0,
                                    np.inf, initial_objective, {}, np.array([], dtype=bool))
        value, theta, nfev, message = min(candidates, key=lambda item: item[0])
        if value > initial_objective:
            return TransitFitResult(None, None, False, 'objective_worsened',
                                    'Transit optimization worsened the initial objective.', nfev,
                                    value, initial_objective, {}, np.array([], dtype=bool))
        best_model = model(theta)
        if not np.all(np.isfinite(best_model)):
            return TransitFitResult(None, None, False, 'nonfinite_model',
                                    'Optimized transit model is not finite.', nfev, value,
                                    initial_objective, {}, np.array([], dtype=bool))
        depth = 1 - np.min(best_model)
        duration = ExoMoonLeastSquares.contact_duration(params)
        # BATMAN's limb-darkened numerical tail need not be bitwise one, so
        # use geometric contacts rather than a floating-point flux comparison.
        in_transit = np.abs(time - theta[0]) < duration / 2
        local_t0 = theta[0]
        coverage = {'in_transit': int(np.sum(in_transit)), 'out_of_transit': int(np.sum(~in_transit)),
                    'before_midpoint': bool(np.any(in_transit & (time < local_t0))),
                    'after_midpoint': bool(np.any(in_transit & (time > local_t0))),
                    'oot_left': bool(np.any((~in_transit) & (time < local_t0))),
                    'oot_right': bool(np.any((~in_transit) & (time > local_t0)))}
        # Three in-transit points constrain local timing/depth; OOT samples on
        # both sides identify the local unit baseline rather than extrapolating it.
        if (depth <= np.sqrt(np.finfo(float).eps) or coverage['in_transit'] < 3
                or not coverage['before_midpoint'] or not coverage['after_midpoint']
                or not coverage['oot_left'] or not coverage['oot_right']):
            return TransitFitResult(None, None, False, 'insufficient_coverage',
                                    'Fit lacks transit and two-sided OOT coverage.', nfev, value,
                                    initial_objective, coverage, np.array([], dtype=bool))
        data_objective = 0.5 * np.sum(((flux - best_model) / flux_err) ** 2)
        null_objective = 0.5 * np.sum(((flux - 1) / flux_err) ** 2)
        if not np.isfinite(data_objective) or data_objective >= null_objective:
            return TransitFitResult(None, None, False, 'no_improvement',
                                    'Fitted transit does not improve the unit-flux null.', nfev, value,
                                    initial_objective, coverage, np.array([], dtype=bool))
        x_best = ((theta - bounds[:, 0]) / scale)[active]
        active_bounds = np.isclose(x_best, 0, atol=np.sqrt(np.finfo(float).eps)) | np.isclose(
            x_best, 1, atol=np.sqrt(np.finfo(float).eps))
        if np.any(active_bounds):
            return TransitFitResult(None, None, False, 'active_bound',
                                    'Optimized transit is pinned to a fitted parameter bound.', nfev, value,
                                    initial_objective, coverage, active_bounds)
        theta[0] += origin
        return TransitFitResult(theta, best_model, True, 'ok', str(message), nfev, value,
                                initial_objective, coverage, active_bounds)

    @staticmethod
    def _fit_transit(time, flux, flux_err, guess, bounds, transit_params, fields,
                     expected_duration=None, exposure_times_seconds=None):
        """Legacy tuple adapter for the validated structured fit result."""
        result = ExoMoonLeastSquares._fit_transit_result(
            time, flux, flux_err, guess, bounds, transit_params, fields, expected_duration,
            exposure_times_seconds)
        if not result.success:
            raise RuntimeError(f"Invalid transit fit [{result.status}]: {result.message}")
        return result.theta, result.model

    @staticmethod
    def fit_single_t0(time, flux, flux_err, t0_guess, duration_window, transit_params,
                      exposure_times_seconds=None):
        """Fit only t0 for a single transit. a, inc, rp must be pre-set on transit_params.

        Returns (best_theta, best_model) where best_theta is [t0].
        """
        return ExoMoonLeastSquares._fit_transit(
            time, flux, flux_err, [t0_guess],
            [(t0_guess - duration_window / 4, t0_guess + duration_window / 4)],
            transit_params, ['t0'], exposure_times_seconds=exposure_times_seconds)

    @staticmethod
    def fit_single_t0_rp(time, flux, flux_err, t0_guess, rp_guess, rp_err,
                          duration_window, transit_params, exposure_times_seconds=None):
        """Fit t0 and rp for a single transit. a, inc pre-set on transit_params.

        Returns (best_theta, best_model) where best_theta is [t0, rp].
        """
        return ExoMoonLeastSquares._fit_transit(
            time, flux, flux_err, [t0_guess, rp_guess],
            [(t0_guess - duration_window / 2, t0_guess + duration_window / 2),
             (max(rp_guess - rp_err, 1e-5), rp_guess + rp_err)],
            transit_params, ['t0', 'rp'], exposure_times_seconds=exposure_times_seconds)

    @staticmethod
    def _profile_local_baseline(time, flux, flux_err, center, duration):
        """Profile a conservative local line from protected, two-sided OOT data.

        The planet and any potentially overlapping moon are excluded from the
        baseline fit. A linear nuisance is used only when each side has at
        least three OOT samples; otherwise the unit baseline is retained.
        """
        time, flux, flux_err = ExoMoonLeastSquares._validate_light_curve(time, flux, flux_err)
        if not np.isfinite(duration) or duration <= 0:
            return np.ones_like(flux), {'status': 'disabled_invalid_duration'}
        offset = (time - center) / duration
        protected = np.abs(time - center) < 1.5 * duration
        left = (~protected) & (offset < 0)
        right = (~protected) & (offset > 0)
        if np.sum(left) < 3 or np.sum(right) < 3:
            return np.ones_like(flux), {
                'status': 'disabled_insufficient_two_sided_oot',
                'oot_left': int(np.sum(left)), 'oot_right': int(np.sum(right)),
            }
        oot = left | right
        design = np.column_stack((np.ones(np.sum(oot)), offset[oot]))
        weights = 1.0 / flux_err[oot] ** 2
        normal = design.T @ (weights[:, None] * design)
        if not np.all(np.isfinite(normal)) or np.linalg.cond(normal) > 1e10:
            return np.ones_like(flux), {'status': 'disabled_ill_conditioned'}
        coefficients = np.linalg.solve(normal, design.T @ (weights * flux[oot]))
        baseline = coefficients[0] + coefficients[1] * offset
        if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
            return np.ones_like(flux), {'status': 'disabled_nonpositive_baseline'}
        covariance = np.linalg.inv(normal)
        return baseline, {
            'status': 'profiled_linear_oot', 'intercept': float(coefficients[0]),
            'slope_per_duration': float(coefficients[1]),
            'intercept_uncertainty': float(np.sqrt(covariance[0, 0])),
            'slope_uncertainty': float(np.sqrt(covariance[1, 1])),
            'oot_left': int(np.sum(left)), 'oot_right': int(np.sum(right)),
        }

    @staticmethod
    def signal_transfer_metrics(injected_signal, residual_difference, flux_err):
        """Return weighted paired-residual transfer and orthogonal distortion."""
        injected_signal, residual_difference, flux_err = ExoMoonLeastSquares._validate_light_curve(
            injected_signal, residual_difference, flux_err)
        weights = 1.0 / flux_err ** 2
        denominator = np.sum(weights * injected_signal ** 2)
        if denominator <= 0:
            raise ValueError("Injected signal must have nonzero weighted norm")
        transfer = float(np.sum(weights * residual_difference * injected_signal) / denominator)
        distortion = residual_difference - transfer * injected_signal
        return {
            'weighted_transfer_coefficient': transfer,
            'weighted_orthogonal_distortion': float(np.sqrt(np.sum(weights * distortion ** 2) / denominator)),
        }

    @staticmethod
    def _residual_diagnostics(time, residual, flux_err, center, duration):
        """Describe residuals without treating unexplained structure as a fit failure."""
        time, residual, flux_err = ExoMoonLeastSquares._validate_light_curve(time, residual, flux_err)
        weights = 1.0 / flux_err ** 2
        mean = float(np.average(residual, weights=weights))
        scatter = float(np.sqrt(np.average((residual - mean) ** 2, weights=weights)))
        order = np.argsort(time, kind='stable')
        standardized = (residual[order] - mean) / flux_err[order]
        if len(standardized) > 2 and np.std(standardized[:-1]) > 0 and np.std(standardized[1:]) > 0:
            lag_one_correlation = float(np.corrcoef(standardized[:-1], standardized[1:])[0, 1])
        else:
            lag_one_correlation = np.nan
        half_duration = duration / 2
        in_transit = np.abs(time - center) < half_duration
        return {
            'weighted_residual_mean': mean,
            'weighted_residual_scatter': scatter,
            # Kept for existing diagnostics consumers; this is weighted scatter,
            # not an independently calibrated uncertainty estimate.
            'weighted_residual_rms': scatter,
            'lag_one_residual_correlation': lag_one_correlation,
            'in_transit_cadences': int(np.sum(in_transit)),
            'ingress_covered': bool(np.any(in_transit & (time < center))),
            'egress_covered': bool(np.any(in_transit & (time > center))),
        }

    @staticmethod
    def fit_single_transit(time, flux, flux_err, guess_params, guess_params_bounds,
                           transit_params: batman.TransitParams, expected_duration=None,
                           exposure_times_seconds=None):
        """Fit a single transit to light curve data.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        guess_params : list
            Initial guess parameters [t0, a, inc, rp].
        guess_params_bounds : list
            Bounds for each parameter.
        transit_params : batman.TransitParams
            BATMAN transit parameters object.
        expected_duration : float or None, optional
            Expected transit duration in days. If provided, models with
            shorter duration are penalized to prevent the optimizer from
            shrinking the transit to avoid noisy wings.

        Returns
        -------
        tuple
            (best_theta, best_model) where best_theta is [t0, a, inc, rp]
            and best_model is the model light curve array.
        """
        return ExoMoonLeastSquares._fit_transit(
            time, flux, flux_err, guess_params, guess_params_bounds, transit_params,
            ['t0', 'a', 'inc', 'rp'], expected_duration, exposure_times_seconds)

    def fit_single_transits(self, time, flux, flux_err, plot=False, exposure_times_seconds=None):
        """Fit all individual planet transits to refine T0 values.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        plot : bool, optional
            If True, generate transit fit plots, default False.
        """
        if exposure_times_seconds is None:
            exposure_times_seconds = self.exposure_times_seconds
        exposure_times_seconds = self._validate_exposure_times(exposure_times_seconds, len(time))
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        planet_duration_plot = self.planet_duration * 2
        params = batman.TransitParams()
        params.per = self.planet_period
        params.ecc = self.planet_ecc  # eccentricity
        params.w = self.planet_arg_periastron  # longitude of periastron (deg)
        params.rp = self.planet_radius / star_radius_rearth  # planet radius / stellar radius
        params.a = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                u.au)  # semi-major axis / stellar radius
        params.inc = self.planet_inc  # orbital inclination [deg]
        params.u = self.ab  # limb darkening coefficients
        params.limb_dark = "quadratic"
        fitted_t0s = []
        for i, guess_t0 in enumerate(self.bary_t0s):
            mask = np.abs(time - guess_t0) < self.planet_duration * 2
            if self.is_skip_t0(time[mask]):
                continue
            params.t0 = guess_t0  # this will be updated for each transit
            guess_params = [guess_t0, params.a, params.inc, params.rp]
            guess_params_bounds = [(guess_t0 - self.planet_duration / 2,
                                    guess_t0 + self.planet_duration / 2),
                      (guess_params[1] - guess_params[1] / 100, guess_params[1] + guess_params[1] / 100),
                      (guess_params[2] - self.planet_inc_err, guess_params[2] + self.planet_inc_err),
                                   (guess_params[3] - self.planet_radius_err / star_radius_rearth,
                                    guess_params[3] + self.planet_radius_err / star_radius_rearth)]
            best_theta, best_model = ExoMoonLeastSquares.fit_single_transit(time[mask], flux[mask], flux_err[mask],
                                                                             guess_params, guess_params_bounds, params,
                                                                             expected_duration=self.planet_duration,
                                                                             exposure_times_seconds=(
                                                                                 None if exposure_times_seconds is None
                                                                                 else exposure_times_seconds[mask]))
            best_t0 = best_theta[0]
            fitted_t0s.append(best_t0)
            if plot:
                times_in_t0 = time[(time > best_t0 - planet_duration_plot) & (time < best_t0 + planet_duration_plot)]
                if len(times_in_t0) > 20:
                    fig_transit, axs = plt.subplots(1, 1, figsize=(12, 12))
                    axs.scatter(time[mask],
                                flux[mask],
                                color='gray', alpha=1, rasterized=True, label="Flux")
                    axs.plot(time[mask], best_model,
                             color='red', alpha=1, rasterized=True, label="Flux")
                    axs.set_title("Planet transit at t0=" + str(best_t0))
                    axs.set_xlabel('Time')
                    axs.set_ylabel('Flux')
                    plt.savefig(self.object_dir + "/T0_" + str(round(best_t0, 3)) + ".png")
                    plt.close(fig_transit)

        return np.asarray(fitted_t0s)

    def subtract_planet_transit(self, time, flux, flux_err, mode='subtract', plot=False,
                                exposure_times_seconds=None):
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
        time, flux, flux_err = self._validate_light_curve(time, flux, flux_err)
        if exposure_times_seconds is None:
            exposure_times_seconds = self.exposure_times_seconds
        exposure_times_seconds = self._validate_exposure_times(exposure_times_seconds, len(time))
        if mode not in ('subtract', 'mask'):
            raise ValueError("Planet removal mode must be 'subtract' or 'mask'")
        if not len(self.bary_t0s):
            raise ValueError("No usable planetary epochs")
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        # Keep the observed span as extraction context, separately from the
        # physical T14 that may be refined by the shared fit below.
        duration_window = self.planet_duration * 2
        result_model = flux.copy()
        result_time = time.copy()
        result_indices = np.arange(len(time))
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
                                                                                u.au)
        params.inc = self.planet_inc
        params.u = self.ab
        params.limb_dark = "quadratic"
        # Phase-fold all transits to get a reliable transit shape
        fold_time = []
        fold_flux = []
        fold_flux_err = []
        fold_exposure_times = []
        for t0 in self.bary_t0s:
            mask = np.abs(time - t0) < duration_window
            if len(time[mask]) < self.min_cadences_per_t0:
                continue
            fold_time.append(time[mask] - t0)
            fold_flux.append(flux[mask])
            if flux_err is not None:
                fold_flux_err.append(flux_err[mask])
            if exposure_times_seconds is not None:
                fold_exposure_times.append(exposure_times_seconds[mask])
        if len(fold_time) > 0:
            fold_time = np.concatenate(fold_time)
            fold_flux = np.concatenate(fold_flux)
            fold_flux_err = np.concatenate(fold_flux_err) if flux_err is not None else None
            order = np.argsort(fold_time, kind='stable')
            fold_time, fold_flux, fold_flux_err = (x[order] for x in (fold_time, fold_flux, fold_flux_err))
            fold_exposure_times = (np.concatenate(fold_exposure_times)[order]
                                   if exposure_times_seconds is not None else None)
            fold_t0 = 0.0
            fold_guess = [fold_t0, params.a, self.planet_inc, params.rp]
            fold_bounds = [(fold_time.min(), fold_time.max()),
                           # No semimajor-axis uncertainty is part of the
                           # public EMLS input contract. Keep its catalog
                           # value fixed rather than inventing a +/-2% prior.
                           (fold_guess[1], fold_guess[1]),
                             (max(fold_guess[2] - self.planet_inc_err, 0),
                              min(fold_guess[2] + self.planet_inc_err, 90)),
                            (max(1e-5, fold_guess[3] - self.planet_radius_err / star_radius_rearth),
                             fold_guess[3] + self.planet_radius_err / star_radius_rearth)]
        elif not self.skip_fold_fit:
            raise ValueError("No epochs have enough observations for the shared fit")
        if not self.skip_fold_fit:
            shared_fit = ExoMoonLeastSquares._fit_transit_result(
                fold_time, fold_flux, fold_flux_err, fold_guess, fold_bounds, params,
                ['t0', 'a', 'inc', 'rp'], exposure_times_seconds=fold_exposure_times)
            self.shared_fit_status = shared_fit
            if shared_fit.success:
                fold_theta = shared_fit.theta
                logging.info("Fold fit: a=%.2f->%.2f, inc=%.1f->%.1f, rp=%.4f->%.4f (depth %.1f->%.1f ppm)",
                             fold_guess[1], fold_theta[1],
                             fold_guess[2], fold_theta[2],
                             fold_guess[3], fold_theta[3],
                             fold_guess[3]**2 * 1e6, fold_theta[3]**2 * 1e6)
                params.a = fold_theta[1]
                params.inc = fold_theta[2]
                params.rp = fold_theta[3]
            else:
                fold_theta = np.asarray(fold_guess, dtype=float)
                logging.warning("Shared planet fit rejected [%s]: %s; using catalog geometry for local fit validation.",
                                shared_fit.status, shared_fit.message)
        else:
            self.shared_fit_status = TransitFitResult(None, None, False, 'skipped',
                                                       'Shared fold fit disabled by configuration.', 0,
                                                       np.nan, np.nan, {}, np.array([], dtype=bool))
        if not self.skip_fold_fit and self.shared_fit_status.success:
            # Recompute planet_duration from the best-fit model
            fold_t0_val = fold_theta[0]; fold_per_val = params.per
            fold_rp_val = fold_theta[3]; fold_a_val = fold_theta[1]
            fold_inc_val = fold_theta[2]
            params_best = batman.TransitParams()
            params_best.t0 = fold_t0_val; params_best.per = fold_per_val
            params_best.rp = fold_rp_val; params_best.a = fold_a_val
            params_best.inc = fold_inc_val
            params_best.ecc = params.ecc; params_best.w = params.w
            params_best.u = self.ab; params_best.limb_dark = "quadratic"
            self.planet_duration = self.contact_duration(params_best)
            # Do not discard observed context if the fitted physical duration
            # is shorter, but guarantee two T14s for the local timing fits.
            duration_window = max(duration_window, self.planet_duration * 2)
            if plot:
                # Build initial-guess model
                params_guess = batman.TransitParams()
                params_guess.t0 = fold_guess[0]; params_guess.per = params.per
                params_guess.rp = fold_guess[3]; params_guess.a = fold_guess[1]
                params_guess.inc = fold_guess[2]
                params_guess.ecc = params.ecc; params_guess.w = params.w
                params_guess.u = self.ab; params_guess.limb_dark = "quadratic"
                mg = batman.TransitModel(params_guess, fold_time)
                model_guess = mg.light_curve(params_guess)
                # Build best-fit model
                params_best = batman.TransitParams()
                params_best.t0 = fold_theta[0]; params_best.per = params.per
                params_best.rp = fold_theta[3]; params_best.a = fold_theta[1]
                params_best.inc = fold_theta[2]
                params_best.ecc = params.ecc; params_best.w = params.w
                params_best.u = self.ab; params_best.limb_dark = "quadratic"
                mb = batman.TransitModel(params_best, fold_time)
                model_best = mb.light_curve(params_best)
                fig_fold, ax_fold = plt.subplots(1, 1, figsize=(10, 6))
                ax_fold.scatter(fold_time, fold_flux, color='gray', alpha=0.5, s=5)
                ax_fold.plot(fold_time, model_guess, 'b--', alpha=0.7, label='Initial guess')
                ax_fold.plot(fold_time, model_best, 'r-', alpha=1, label='Fold fit')
                ax_fold.set_title(f"Phase-folded fit: rp={fold_guess[3]:.4f}->{fold_theta[3]:.4f}, depth={fold_guess[3]**2*1e6:.0f}->{fold_theta[3]**2*1e6:.0f} ppm")
                ax_fold.set_xlabel('Time from T0 (days)')
                ax_fold.set_ylabel('Flux')
                ax_fold.legend()
                plt.savefig(os.path.join(self.object_dir, "fold_fit.png"))
                plt.close(fig_fold)
        self.shared_planet_params = copy.deepcopy(params)
        # Per-epoch radius fitting can absorb an overlapping moon.  The shared
        # shape is intentionally anchored after the folded fit; the public
        # flag remains accepted for compatibility but is not used here.
        if self.fit_rp_per_transit:
            logging.info("EMLS uses the shared planet radius for local subtraction to preserve moon signals.")
        self.ttvs = []
        self.local_t0s = []
        self.epoch_diagnostics = []
        self.subtraction_status = {'status': 'running', 'reason': None,
                                   'retained_epochs': 0, 'rejected_epochs': 0}
        accepted_t0s = []
        rejected_t0s = []
        for t0 in self.bary_t0s:
            params = copy.deepcopy(self.shared_planet_params)
            mask = np.abs(time - t0) < duration_window
            intransit_indexes = np.argwhere(np.abs(time - t0) < duration_window).flatten()
            # Always fit t0 and compute TTV
            params.t0 = t0
            params.u = self.ab
            try:
                local_time, local_flux, local_err = time[mask], flux[mask], flux_err[mask]
                local_exposure = None if exposure_times_seconds is None else exposure_times_seconds[mask]
                # First estimate timing from the shared shape.  The baseline is
                # then profiled only on protected OOT points and timing is
                # refined once on the normalized data.
                initial_theta, _ = ExoMoonLeastSquares.fit_single_t0(
                    local_time, local_flux, local_err, t0, duration_window, params, local_exposure)
                baseline, baseline_diagnostic = self._profile_local_baseline(
                    local_time, local_flux, local_err, initial_theta[0], self.planet_duration)
                normalized_flux = local_flux / baseline
                normalized_err = local_err / baseline
                best_theta, best_model = ExoMoonLeastSquares.fit_single_t0(
                    local_time, normalized_flux, normalized_err, initial_theta[0], duration_window,
                    params, local_exposure)
            except (ValueError, RuntimeError) as exc:
                rejected_t0s.append(t0)
                self.epoch_diagnostics.append({
                    'epoch': t0, 'accepted': False, 'rejection_reason': str(exc),
                    'reason': str(exc), 'fit_converged': False,
                    'residual_classification': 'poor_planet_model_fit',
                })
                continue
            best_t0 = best_theta[0]
            residual = normalized_flux - best_model
            residual_diagnostic = self._residual_diagnostics(
                local_time, residual, normalized_err, best_t0, self.planet_duration)
            # A successful optimizer with geometric support is a usable planet
            # subtraction. Residual structure can be a moon, spot crossing, or
            # other astrophysics; it is reported, never rejected by a scatter cut.
            diagnostic = {
                'epoch': t0, 'fitted_t0': best_t0, 'timing_offset_days': best_t0 - t0,
                'accepted': True, 'reason': 'ok', 'rejection_reason': None,
                'fit_converged': True, 'fit_status': 'ok',
                'baseline': baseline_diagnostic,
                'residual_classification': 'unexplained_residual_not_rejected',
                'shape_uncertainty': 'not_propagated_correlated_residuals_possible',
                **residual_diagnostic,
            }
            self.ttvs.append(t0 - best_t0)
            self.local_t0s.append(best_t0)
            accepted_t0s.append(t0)
            if mode == 'subtract':
                result_model[intransit_indexes] = normalized_flux - (best_model - 1)
                result_flux_err[intransit_indexes] = normalized_err
                if plot:
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
                    plt.savefig(os.path.join(view_dir, "T0_" + str(round(best_t0, 3)) + "_sub.png"))
                    plt.close()
            else:
                intransit_mask_local = np.abs(result_time - best_t0) <= self.planet_duration / 2 * self.mask_duration_factor
                if plot:
                    view_dir = os.path.join(self.object_dir, "t0s_view")
                    os.makedirs(view_dir, exist_ok=True)
                    fig_t, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    # Before mask: windowed flux + model
                    ax1.scatter(time[mask], flux[mask], color='gray', alpha=0.7, s=10)
                    ax1.plot(time[mask], best_model, color='orange', alpha=1, label='Planet model')
                    ax1.axvspan(best_t0 - self.planet_duration / 2 * self.mask_duration_factor,
                                best_t0 + self.planet_duration / 2 * self.mask_duration_factor,
                                color='red', alpha=0.15, label='Masked region')
                    ax1.axvline(best_t0, color='red', linestyle='--')
                    ax1.set_title(f"T0={best_t0:.4f} (before mask)")
                    ax1.set_ylabel('Flux'); ax1.legend()
                    # After mask: remaining data at this window
                    keep_mask_window = ~(np.abs(time[mask] - best_t0) <= self.planet_duration / 2 * self.mask_duration_factor)
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
                result_indices = result_indices[~intransit_mask_local]
                result_model = result_model[~intransit_mask_local]
                if result_flux_err is not None:
                    result_flux_err = result_flux_err[~intransit_mask_local]
            self.epoch_diagnostics.append(diagnostic)
        if len(rejected_t0s) > 0:
            logging.info("Rejected %d/%d transits because no valid planet model was available",
                          len(rejected_t0s), len(self.bary_t0s))
            # Failed fits must not leave their unmodeled cadence window in the
            # search input while the corresponding epoch is absent.
            keep = np.ones(len(result_time), dtype=bool)
            for t0 in rejected_t0s:
                keep &= np.abs(result_time - t0) >= duration_window
            result_time, result_model, result_flux_err, result_indices = (
                x[keep] for x in (result_time, result_model, result_flux_err, result_indices))
        self.bary_t0s = accepted_t0s
        self.retained_cadence_indices = result_indices
        self.subtraction_status = {
            'status': 'ok' if accepted_t0s else 'no_usable_epochs',
            'reason': None if accepted_t0s else 'All epochs failed planet-model validation.',
            'retained_epochs': len(accepted_t0s), 'rejected_epochs': len(rejected_t0s),
        }
        if not self.bary_t0s:
            logging.warning("No usable planetary epochs remain; returning synchronized non-search arrays.")
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(self.ttvs)), self.ttvs, marker='o')
            plt.xticks(range(len(self.ttvs)), [f"{t0:.4f}" for t0 in self.bary_t0s], rotation=45)
            plt.xlabel("Barycentric T0 Values")
            plt.ylabel("TTV (days)")
            plt.title("Transit Timing Variations")
            plt.tight_layout()
            if self.object_dir:
                plt.savefig(os.path.join(self.object_dir, "ttvs.png"))
            plt.close()
        if plot and len(self.ttvs) > 5:
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
        min_t1 = planet_t0 - half_moon_orbit_transit_duration
        min_t0 = min_t1 + half_moon_transit_duration
        max_moon_t0_tau = np.abs(planet_t0 - min_t0)
        return max_moon_t0_tau

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
    def assess_period_alias(injected_period_days, recovered_period_days, planet_period_days,
                             maximum_alias_order=5):
        """Compare periods using one relative-to-injected-period convention.

        Alias frequencies use the fixed barycentric ephemeris in days.  The
        phase reference is intentionally not involved: moving its epoch only
        transforms phase, never a period or alias frequency.
        """
        if injected_period_days <= 0 or recovered_period_days <= 0 or planet_period_days <= 0:
            raise ValueError("Alias periods must be positive days")
        candidates = []
        for k in range(-maximum_alias_order, maximum_alias_order + 1):
            frequency = 1 / injected_period_days + k / planet_period_days
            if frequency > 0:
                period = 1 / frequency
                candidates.append((abs(period - recovered_period_days) / injected_period_days, k, period))
        relative_error, alias_k, matched_period = min(candidates)
        return {
            'comparison': 'absolute_period_difference_days / injected_period_days',
            'direct_relative_error': abs(injected_period_days - recovered_period_days) / injected_period_days,
            'relative_error': float(relative_error),
            'absolute_error_days': float(abs(matched_period - recovered_period_days)),
            'alias_k': int(alias_k),
            'matched_period_days': float(matched_period),
        }

    @staticmethod
    def match_recovered_events(truth_events, recovered_by_epoch_id, event_status=None):
        """Match recovered moon times to immutable epoch IDs, never list rows."""
        event_status = {} if event_status is None else event_status
        errors = {}
        statuses = {}
        for epoch_id, event in truth_events.items():
            status = event_status.get(epoch_id, event.coverage)
            statuses[epoch_id] = status
            recovered_t0 = recovered_by_epoch_id.get(epoch_id)
            if status in ('covered', 'partial_coverage') and event.moon_t0 is not None and recovered_t0 is not None:
                errors[epoch_id] = abs(recovered_t0 - event.moon_t0)
        return errors, statuses


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
        list or None
            List of [moon_alpha, time, flux, flux_err] for each scenario,
            or None if no valid scenarios are found.
        """
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = None
        for t0 in bary_t0s:
            max_moon_t0_tau = ExoMoonLeastSquares.compute_moon_ts(t0, half_moon_orbit_transit_duration, half_moon_transit_duration)
            moon_t0, moon_alpha = ExoMoonLeastSquares.compute_moon_t0(planet_t0, t0, moon_period, moon_initial_alpha, max_moon_t0_tau)
            moon_t1 = moon_t0 - half_moon_transit_duration
            # t1 = moon_orbit_range[1]
            # phase_delta = (t0 - planet_t0) % moon_period * 2 * np.pi
            # alpha = (moon_initial_alpha + phase_delta) % (2 * np.pi)
            # time_alpha = np.cos(alpha) * moon_orbit_transit_length / 2
            # moon_t1 = t1 + time_alpha
            # Assign overlapping windows to the nearest immutable epoch, so a
            # cadence contributes once even when two extraction windows overlap.
            distance = np.abs(time - t0)
            belongs = distance <= half_moon_max_orbit_transit_duration
            for other in bary_t0s:
                if other != t0:
                    other_distance = np.abs(time - other)
                    belongs &= (distance < other_distance) | ((distance == other_distance) & (t0 < other))
            time_orbit_range_args = np.flatnonzero(belongs)
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
                if orbit_scenarios is None:
                    orbit_scenarios = []
                orbit_scenarios.append([moon_alpha, time_moon_transit, flux_moon_transit, flux_err_moon_transit])
        return orbit_scenarios

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
            (alternative_chi2, null_chi2, radius, model_snr) if return_lc is
            False, or (alternative_chi2, null_chi2, radius, model_snr,
            it_time, it_flux, oot_time, oot_flux, residual_model) if
            return_lc is True.
        """
        if not len(self.bary_t0s):
            raise ValueError("No usable planetary epochs for moon search")
        transit_scenarios = ExoMoonLeastSquares.compute_moon_transit_scenarios(self.time, self.flux, self.flux_err,
                                                                               self.bary_t0s[0], search_input.moon_alpha,
                                                                search_input.moon_period, self.bary_t0s,
                                                                               search_input.half_moon_orbit_transit_duration,
                                                                search_input.half_moon_max_orbit_transit_duration,
                                                                               search_input.half_moon_transit_duration)
        scenario_time = []
        scenario_flux = []
        scenario_flux_err = []
        for normalized_moon_transit_scenario in transit_scenarios or []:
            scenario_time.append(normalized_moon_transit_scenario[1].flatten())
            scenario_flux.append(normalized_moon_transit_scenario[2].flatten())
            scenario_flux_err.append(normalized_moon_transit_scenario[3].flatten())
        if scenario_time:
            scenario_time = np.concatenate(scenario_time)
            scenario_flux = np.concatenate(scenario_flux)
            scenario_flux_err = np.concatenate(scenario_flux_err)
        else:
            scenario_time = np.array([], dtype=float)
            scenario_flux = np.array([], dtype=float)
            scenario_flux_err = np.array([], dtype=float)
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
        t0_flux_mean_value = np.mean(it_flux) if len(it_flux) else 1.0
        if len(self.models) == 0:
            # No supported alternative is exactly the null hypothesis.
            residual_baseline = (np.sum(((it_flux - 1) / it_flux_err) ** 2)
                                 + np.sum(((oot_flux - 1) / oot_flux_err) ** 2))
            residual_calculation = residual_baseline
            residual_radius = self.min_radius
            residual_model = np.ones_like(it_flux)
        else:
            residual_calculation, residual_baseline, residual_radius, residual_model = (
                self.calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                                         t0_flux_mean_value, len(scenario_time), self.models, self.min_radius,
                                         self.max_radius, self.star_radius, self.radius_grid_size))
        #oot_noise = np.std(oot_flux)
        #oot_noise = np.sqrt(np.std(oot_flux) * (np.sqrt(np.sum(oot_flux_err ** 2) / len(oot_flux_err))))
        oot_noise = np.std(oot_flux) if len(oot_flux) else 0.0
        model_snr = (1 - t0_flux_mean_value) / oot_noise * np.sqrt(len(it_flux)) if oot_noise > 0 else 0.0
        if return_lc:
            return residual_calculation, residual_baseline, residual_radius, model_snr, it_time, it_flux, oot_time, oot_flux, residual_model
        else:
            return residual_calculation, residual_baseline, residual_radius, model_snr

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
        """Compute SDE from finite lower-is-better scores.

        SDE = max(SR_detrended) / std(SR_detrended), where
        SR = min(chi²) / chi² and detrending is via running-median.

        No global normalisation; avoids instability from flat baselines.
        """
        chi2 = np.asarray(chi2, dtype=float)
        if chi2.ndim != 1 or not len(chi2) or np.any(~np.isfinite(chi2)):
            raise ValueError("Spectrum requires nonempty, finite scores")
        if (not np.isfinite(oversampling_factor) or not np.isfinite(kernel_size)
                or int(oversampling_factor) != oversampling_factor or int(kernel_size) != kernel_size
                or oversampling_factor < 1 or kernel_size < 1):
            raise ValueError("Spectrum oversampling_factor and kernel_size must be positive integers")
        # Scores may be -Delta chi-square, so shift them before the legacy
        # lower-is-better SR transformation.
        shifted = chi2 - np.min(chi2) + 1.0
        SR = 1.0 / shifted
        kernel = int(oversampling_factor) * int(kernel_size)
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
    def spectrum_candidates(period_grid, alpha_scores, radii):
        """Keep period-indexed spectrum labels tied to evaluated candidates."""
        period_grid = np.asarray(period_grid, dtype=float)
        alpha_scores = np.asarray(alpha_scores, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if (period_grid.ndim != 1 or alpha_scores.shape != (len(period_grid), 2)
                or radii.shape != (len(period_grid),)
                or not np.all(np.isfinite(period_grid)) or np.any(period_grid <= 0)
                or not np.all(np.isfinite(alpha_scores)) or not np.all(np.isfinite(radii))):
            raise ValueError("Spectrum candidates require aligned finite period, phase, score, and radius values")
        return [
            {'period_days': float(period), 'moon_alpha_radians': float(alpha_score[0]),
             'score': float(alpha_score[1]), 'radius_earth': float(radius)}
            for period, alpha_score, radius in zip(period_grid, alpha_scores, radii)
        ]

    @staticmethod
    def accept_refined_alpha(grid_alpha, grid_score, proposed_alpha, objective):
        """Accept a phase proposal only after its finite objective is better."""
        if not (np.isfinite(grid_alpha) and np.isfinite(grid_score) and np.isfinite(proposed_alpha)):
            raise ValueError("Phase acceptance requires finite grid and proposed values")
        proposed_score = objective(float(proposed_alpha % (2 * np.pi)))
        if np.isfinite(proposed_score) and proposed_score < grid_score:
            return float(proposed_alpha % (2 * np.pi)), float(proposed_score)
        return float(grid_alpha % (2 * np.pi)), float(grid_score)

    @staticmethod
    def refine_alpha(alpha_grid, residuals_matrix):
        """Refine the best alpha per period via parabolic interpolation.

        For each period (column of residuals_matrix), fits a quadratic
        through the three alpha values bracketing the grid minimum, then
        returns the alpha at the parabola's vertex. This gives sub-grid
        alpha proposals. The caller must evaluate proposals before accepting
        them: the returned residual is the evaluated grid score, not a prediction.

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
        alpha_grid = np.asarray(alpha_grid, dtype=float)
        n_alpha = len(alpha_grid)
        residuals_matrix = np.asarray(residuals_matrix, dtype=float)
        if n_alpha == 0 or residuals_matrix.ndim != 2 or residuals_matrix.shape[0] != n_alpha:
            raise ValueError("Phase grid and residual matrix must be nonempty and aligned")
        alpha_grid = np.mod(alpha_grid, 2 * np.pi)
        if (not np.all(np.isfinite(alpha_grid)) or np.any(np.diff(alpha_grid) <= 0)
                or len(np.unique(alpha_grid)) != n_alpha):
            raise ValueError("Phase grid must contain finite, strictly increasing unique phases modulo 2*pi")
        refined = np.empty((residuals_matrix.shape[1], 2))
        for col in range(residuals_matrix.shape[1]):
            r_col = residuals_matrix[:, col]
            if not np.all(np.isfinite(r_col)):
                raise ValueError("Phase refinement requires finite residuals")
            best_j = np.argmin(r_col)
            refined[col] = [alpha_grid[best_j] % (2 * np.pi), r_col[best_j]]
            if n_alpha < 3:
                continue
            j_left = (best_j - 1) % n_alpha
            j_right = (best_j + 1) % n_alpha
            a_center = alpha_grid[best_j]
            a_local = np.array([
                (alpha_grid[j_left] - a_center + np.pi) % (2 * np.pi) - np.pi,
                0.0,
                (alpha_grid[j_right] - a_center + np.pi) % (2 * np.pi) - np.pi,
            ])
            r_local = np.array([r_col[j_left], r_col[best_j], r_col[j_right]])
            # A proposal needs a true local bracket, not merely three finite
            # values.  This also rejects flat and concave neighborhoods.
            if not (a_local[0] < 0 < a_local[2] and r_local[1] <= r_local[0]
                    and r_local[1] <= r_local[2]):
                continue
            coeffs = np.polyfit(a_local, r_local, 2)
            # Parabola r = c2*a^2 + c1*a + c0, minimum at a = -c1 / (2*c2)
            curvature_scale = max(1.0, np.max(np.abs(r_local)) / np.max(a_local ** 2))
            if not np.all(np.isfinite(coeffs)) or coeffs[0] <= np.finfo(float).eps * curvature_scale:
                continue
            delta = -coeffs[1] / (2.0 * coeffs[0])
            if not (a_local[0] <= delta <= a_local[2]):
                continue
            refined[col, 0] = (a_center + delta) % (2.0 * np.pi)
            refined[col, 1] = r_col[best_j]
        return refined

    @staticmethod
    #@njit(fastmath=True, parallel=False)
    def calculate_residuals(it_time, it_flux, it_flux_err, oot_time, oot_flux, oot_flux_err,
                            t0_flux_mean_value, datapoints, models, min_radius, max_radius,
                            star_radius, radius_grid_size, ootr_weight=1.0):
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
            Physical templates [(radius_Rearth, relative_time_days, flux), ...].
        min_radius : float
            Minimum moon radius in Earth radii.
        max_radius : float
            Maximum moon radius in Earth radii.
        star_radius : float
            Stellar radius in solar radii.
        radius_grid_size : int
            Number of grid points for radius search.
        ootr_weight : float, optional
            Weight for out-of-transit residuals, default 1.0. The default is
            the chi-square likelihood for the supplied independent errors.

        Returns
        -------
        tuple
            (best_residual, residuals_baseline, best_radius, best_model).
        """
        if not np.isfinite(ootr_weight) or ootr_weight < 0:
            raise ValueError("OOT weight must be finite and nonnegative")
        if (not (np.isfinite(min_radius) and np.isfinite(max_radius) and 0 < min_radius <= max_radius)
                or isinstance(radius_grid_size, bool) or int(radius_grid_size) != radius_grid_size
                or radius_grid_size < 1 or not np.isfinite(star_radius) or star_radius <= 0):
            raise ValueError("Moon template search requires finite physical radius bounds and stellar radius")
        if len(it_time):
            it_time, it_flux, it_flux_err = ExoMoonLeastSquares._validate_light_curve(it_time, it_flux, it_flux_err)
        if len(oot_time):
            oot_time, oot_flux, oot_flux_err = ExoMoonLeastSquares._validate_light_curve(oot_time, oot_flux, oot_flux_err)
        oot_radius_residuals = np.sum(((oot_flux - 1) / oot_flux_err) ** 2)
        n_it = len(it_flux)
        model_baseline = np.ones(n_it)
        residuals_baseline = np.sum(((it_flux - model_baseline) / it_flux_err) ** 2) + ootr_weight * oot_radius_residuals
        best_radius = min_radius
        best_residual = residuals_baseline
        best_model = model_baseline
        if not n_it or not np.isfinite(residuals_baseline):
            return best_residual, residuals_baseline, best_radius, best_model
        for radius, template_time, template_flux in models:
            if not np.isfinite(radius) or not min_radius <= radius <= max_radius:
                continue
            if (len(template_time) < 2 or len(template_time) != len(template_flux)
                    or not np.all(np.isfinite(template_time)) or not np.all(np.isfinite(template_flux))
                    or np.any(np.diff(template_time) <= 0)):
                raise ValueError("Moon templates require finite, strictly increasing physical times")
            # A no-dip or brightening template is not a physical moon-transit
            # alternative.  Its null-equivalent score must not win by noise.
            if np.min(template_flux) >= 1.0:
                continue
            model_sample = np.interp(it_time, template_time, template_flux, left=1.0, right=1.0)
            # Apply the same observations and error weights to both hypotheses.
            residual = np.sum(((it_flux - model_sample) / it_flux_err) ** 2) + ootr_weight * oot_radius_residuals
            if np.isfinite(residual) and residual < best_residual:
                best_residual, best_radius, best_model = residual, radius, model_sample
        return best_residual, residuals_baseline, best_radius, best_model

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
        if len(data) == 0 or kernel < 1:
            raise ValueError("Running median requires nonempty data and a positive kernel")
        kernel = min(int(kernel), len(data))
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
        injected_flux, _ = self._inject_moon_with_truth(
            time, flux, flux_err, t0s, planet_semimajor_axis, planet_ecc,
            moon_radius, moon_period, moon_mass, initial_alpha, plot,
            inject_planet, inject_moon)
        return injected_flux

    def _inject_moon_with_truth(self, time, flux, flux_err, t0s, planet_semimajor_axis, planet_ecc,
                                moon_radius, moon_period, moon_mass, initial_alpha=0, plot=False,
                                inject_planet=False, inject_moon=True, cadence_ids=None, epoch_ids=None):
        """Return injected copies and immutable truth without changing EMLS state.

        ``epoch_ids`` and ``cadence_ids`` identify the pre-filtered inputs.  A
        caller running the full pipeline may retain the returned truth itself;
        this helper intentionally does not attach it to ``self``.
        """
        # Pandora is needed only for opt-in photodynamical injections, not for
        # ordinary EMLS construction, fitting, or scoring.
        global pandora
        if pandora is None:
            import pandoramoon as pandora

        logging.info("Injecting moon with radius of  %.2fR_e, %.2fdays and %.2frad", moon_radius, moon_period, initial_alpha)
        time = np.asarray(time, dtype=float)
        flux = np.asarray(flux, dtype=float).copy()
        if cadence_ids is None:
            cadence_ids = np.arange(len(time))
        cadence_ids = np.asarray(cadence_ids)
        if len(cadence_ids) != len(time):
            raise ValueError("cadence_ids must align with injected observations")
        if epoch_ids is None:
            epoch_ids = range(len(t0s))
        epoch_ids = list(epoch_ids)
        if len(epoch_ids) != len(t0s):
            raise ValueError("epoch_ids must align with injected epochs")
        star_radius_m = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.m)
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        params = pandora.model_params()
        params.R_star = star_radius_m  # [m]
        params.u1 = self.ab[0]
        params.u2 = self.ab[1]
        # planet parameters
        params.per_bary = self.planet_period  # [days]
        params.a_bary = self.planet_semimajor_axis / LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun,
                                                                                     u.au)  # [R_star]
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
        injected_flux_total, injected_flux_planet, injected_flux_moon = model.light_curve(time)
        events = {}
        nearest_epoch = (np.argmin(np.abs(time[:, None] - np.asarray(t0s)[None, :]), axis=1)
                         if len(t0s) else np.empty(0, dtype=int))
        for index, (epoch_id, epoch_time) in enumerate(zip(epoch_ids, t0s)):
            event_rows = np.flatnonzero(nearest_epoch == index)
            event_flux = injected_flux_moon[event_rows]
            event_cadence_ids = tuple(cadence_ids[event_rows].tolist())
            if not len(event_rows) or not np.any(event_flux < 1.0):
                event = InjectionTruthEvent(int(epoch_id), float(epoch_time), event_cadence_ids,
                                            None, None, 'missing_coverage')
            else:
                minimum_local = int(np.argmin(event_flux))
                minimum_row = event_rows[minimum_local]
                at_edge = minimum_local in (0, len(event_rows) - 1)
                event = InjectionTruthEvent(
                    int(epoch_id), float(epoch_time), event_cadence_ids,
                    float(time[minimum_row]), int(cadence_ids[minimum_row]),
                    'partial_coverage' if at_edge else 'covered')
            events[event.epoch_id] = event
        truth = {
            'phase_reference_time': float(self.planet_t0),
            'phase_reference': 'planet_t0_days',
            'moon_period_days': float(moon_period),
            'moon_alpha_radians': float(initial_alpha),
            'cadence_ids': tuple(cadence_ids.tolist()),
            'events': events,
            'moon_model_flux': injected_flux_moon.copy(),
            'planet_model_flux': injected_flux_planet.copy(),
        }
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
        return flux, truth

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
        max_period = self.au_to_period(max_planet_mass * M_earth_to_M_sun, self.compute_hill_radius(self.star_mass, max_planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis))
        moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, max_period)
        moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                              self.planet_semimajor_axis * AU_TO_RSUN,
                                                              self.planet_period,
                                                              moon_semimajor_axis * AU_TO_RSUN)
        mask_parts = []
        for t0 in t0s:
            mask_parts.append(np.argwhere(
                (time > t0 - moon_orbit_transit_duration) & (time < t0 + moon_orbit_transit_duration)).flatten())
        if not mask_parts:
            raise ValueError("No usable planetary epochs")
        flux_mask = np.unique(np.concatenate(mask_parts).astype(int))
        time = time[flux_mask]
        flux = flux[flux_mask]
        flux_err = flux_err[flux_mask]
        self.retained_cadence_indices = flux_mask
        return time, flux, flux_err

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
    def sorted_diagnostic_copy(rows, score_column):
        """Return a detached, stably ranked copy for presentation only."""
        rows = np.asarray(rows)
        if rows.ndim != 2 or not 0 <= score_column < rows.shape[1]:
            raise ValueError("Diagnostic rows and score column must be aligned")
        diagnostic_rows = rows.copy()
        return diagnostic_rows[np.argsort(diagnostic_rows[:, score_column], kind='stable')].copy()

    def run(self, plot=False, inject_params=None):
        """Run the full exomoon search over the parameter grid.

        Parameters
        ----------
        plot : bool, optional
            If True, generate diagnostic plots, default False.
        inject_params : dict or None, optional
            If a dict with keys 'moon_radius', 'moon_period', 'initial_alpha',
            'moon_mass', injects a moon signal into the flux before the grid
            search to test injection-recovery. Default None (no injection).
        """
        planet_mass_grid = self.planet_mass_grid
        moon_inc_grid = self.moon_inc_grid
        moon_ecc_grid = self.moon_ecc_grid
        moon_arg_periastron_grid = self.moon_arg_periastron_grid
        self.inject_flux_time = self.time.copy()  # save full time before trimming
        self.time, self.flux, self.flux_err = self.remove_non_transit_flux(self.time, self.flux, self.flux_err, self.bary_t0s, np.max(planet_mass_grid))
        if self.exposure_times_seconds is not None:
            self.exposure_times_seconds = self.exposure_times_seconds[self.retained_cadence_indices]
        self.cadence_ids = self.cadence_ids[self.retained_cadence_indices]
        rms = np.std(self.flux)
        if inject_params is not None:
            # Include ephemeris epochs excluded before extraction too: their
            # truth explicitly records absent coverage rather than vanishing.
            epoch_ids = [self.epoch_ids_by_time[float(t0)] for t0 in self.original_bary_t0s]
            self.flux, self.injection_truth = self._inject_moon_with_truth(
                self.time, self.flux, self.flux_err, self.original_bary_t0s, self.planet_semimajor_axis,
                self.planet_ecc, inject_params['moon_radius'], inject_params['moon_period'],
                inject_params['moon_mass'], initial_alpha=inject_params['initial_alpha'],
                inject_planet=False, inject_moon=True, cadence_ids=self.cadence_ids,
                epoch_ids=epoch_ids)
            # Compatibility adapter for the existing optional plot code.
            self.injected_flux_moon_pandora = self.injection_truth['moon_model_flux']
            self.injected_flux_planet_pandora = self.injection_truth['planet_model_flux']
            self.injected_moon_period = self.injection_truth['moon_period_days']
            self.injected_moon_alpha = self.injection_truth['moon_alpha_radians']
            logging.info("Injected moon: P=%.4f d, R=%.2f R_e, alpha=%.4f rad",
                         inject_params['moon_period'], inject_params['moon_radius'], inject_params['initial_alpha'])
            self.inject_flux_time = self.time.copy()
        self.time, self.flux, self.flux_err = self.subtract_planet_transit(
            self.time, self.flux, self.flux_err, mode=self.subtract_mode, plot=plot,
            exposure_times_seconds=self.exposure_times_seconds)
        if self.exposure_times_seconds is not None:
            self.exposure_times_seconds = self.exposure_times_seconds[self.retained_cadence_indices]
        self.cadence_ids = self.cadence_ids[self.retained_cadence_indices]
        ma = copy.deepcopy(self.shared_planet_params)
        ma.t0 = 0.0
        ma.u = self.ab  # limb darkening coefficients
        ma.limb_dark = "quadratic"  # limb darkening model
        time_model = np.linspace(ma.t0 - self.planet_duration * 0.7, ma.t0 + self.planet_duration * 0.7, 1000)
        self.models = []
        for r in np.linspace(self.min_radius, self.max_radius, self.radius_grid_size):
            ma.rp = r * R_earth_to_R_sun / self.star_radius
            m = batman.TransitModel(ma, time_model)
            model_r = m.light_curve(ma)
            if np.all(np.isfinite(model_r)) and np.any(model_r < 1):
                self.models.append((r, time_model - ma.t0, model_r))
        search_inputs = []
        if len(self.models) == 0:
            raise ValueError("No transiting moon templates for the supplied geometry")
        alpha_grid_size = 25
        alpha_grid = np.linspace(0, np.pi * 2 - np.pi * 2 / 25, alpha_grid_size)
        #alpha_grid = [0.0]
        impact_params_grid = [self.planet_impact_param]
        for planet_mass in planet_mass_grid:
            if not np.isfinite(planet_mass) or planet_mass <= 0:
                raise ValueError("Moon search requires a finite positive planet mass")
            if not (np.isfinite(self.planet_ecc) and 0 <= self.planet_ecc < 1):
                raise ValueError("Moon search requires planetary eccentricity in [0, 1)")
            # CITE Édouard Roche (1849, Académie des Sciences de Montpellier) for rigid body
            # planet_volume_m3 = (4/3) * np.pi * ((LcbuilderHelper.convert_from_to(self.planet_radius, u.R_earth, u.m)) ** 3)
            # planet_mass_kg = LcbuilderHelper.convert_from_to(self.planet_mass, u.M_earth, u.kg)
            # planet_density = planet_mass_kg / planet_volume_m3
            # roche_lobe = 2.44 * self.planet_radius * (planet_density / self.max_moon_density) ** (1/3)
            min_period = ExoMoonLeastSquares.compute_roche_period(self.max_moon_density)
            if self.min_orbit_window > 0:
                a_norm = self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius
                p_min_win = ExoMoonLeastSquares.min_period_for_orbit_duration(
                    self.star_radius, a_norm, self.planet_period,
                    planet_mass * M_earth_to_M_sun,
                    self.planet_duration * self.min_orbit_window,
                    self.planet_impact_param)
                min_period = max(min_period, p_min_win)
            logging.info(f"Roche Limit Period: {min_period}")
            max_period = self.au_to_period(
                planet_mass * M_earth_to_M_sun,
                self.compute_hill_radius(self.star_mass, planet_mass * M_earth_to_M_sun,
                                         self.planet_semimajor_axis, self.planet_ecc)) * 0.5
            logging.info(f"Hill Limit Period: {max_period}")
            period_grid = self.compute_moon_period_grid(min_period, max_period, samples=self.period_grid_size, mode="log")
            #period_grid = [2.5, 50]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, max_period)
            for moon_inc in moon_inc_grid:
                if not np.isfinite(moon_inc) or not 0 <= moon_inc <= 180:
                    raise ValueError("Moon inclination must be finite and within [0, 180] degrees")
                for moon_ecc in moon_ecc_grid:
                    if not np.isfinite(moon_ecc) or not 0 <= moon_ecc < 1:
                        raise ValueError("Moon eccentricity must be finite and within [0, 1)")
                    for moon_arg_periastron in moon_arg_periastron_grid:
                        if not np.isfinite(moon_arg_periastron):
                            raise ValueError("Moon argument of periastron must be finite")
                        for impact_param in impact_params_grid:
                            if not np.isfinite(impact_param) or impact_param < 0:
                                raise ValueError("Moon impact parameter must be finite and nonnegative")
                            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                                self.planet_semimajor_axis * AU_TO_RSUN,
                                                                                self.planet_period,
                                                                                self.max_radius * R_earth_to_R_sun,
                                                                                impact_param)
                            half_moon_transit_duration = moon_transit_duration / 2
                            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                                            self.planet_period,
                                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                                            impact_param)
                            half_moon_max_orbit_transit_duration = moon_max_orbit_transit_duration / 2
                            for moon_period in period_grid:
                                moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun,
                                                                                  moon_period)
                                moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                                            self.planet_period,
                                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                                            impact_param)
                                half_moon_orbit_transit_duration = moon_orbit_transit_duration / 2
                                for moon_initial_alpha in alpha_grid:
                                    #TODO moon_orbit_ranges should use moon_radius ?
                                    search_inputs.append(SearchInput(
                                        moon_period, moon_initial_alpha, moon_ecc, moon_inc, moon_arg_periastron,
                                        self.min_radius, impact_param, moon_max_orbit_transit_duration,
                                        moon_transit_duration, moon_orbit_transit_duration, half_moon_transit_duration,
                                        half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration))
        with Pool(processes=self.cpus, maxtasksperchild=500) as pool:
            all_residuals = list(
                tqdm(pool.imap(self.search, search_inputs, chunksize=100),
                     total=len(search_inputs))
            )
        best_residuals_per_scenarios = []
        best_residuals_spectra = []
        best_residuals_spectra_values = []
        best_residual_per_period = np.inf
        best_residual_per_period_values = []
        best_alpha_per_period = []
        best_radius_per_period = []
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
            # This is -Delta chi-square: negative values favor the moon
            # template, zero means that it is indistinguishable from the null.
            corrected_residual = residuals - residuals_baseline
            if not np.isfinite(corrected_residual):
                corrected_residual = 0.0
            if corrected_residual > max_residual:
                max_residual = corrected_residual
            best_residuals_per_scenarios.append(
                [moon_period, moon_initial_alpha, corrected_residual, radius])
            residuals_matrix[i % alpha_grid_size][
                i // alpha_grid_size] = corrected_residual
            snr_matrix[i % alpha_grid_size][
                i // alpha_grid_size] = model_snr
            if i % alpha_grid_size == 0:
                best_alpha_per_period.append([moon_initial_alpha, corrected_residual])
                best_radius_per_period.append(radius)
            elif best_alpha_per_period[i // alpha_grid_size][1] > corrected_residual:
                best_alpha_per_period[i // alpha_grid_size] = [moon_initial_alpha, corrected_residual]
                best_radius_per_period[i // alpha_grid_size] = radius
            if best_residual_per_period > corrected_residual:
                best_residual_per_period = corrected_residual
                best_residual_per_period_values = [residuals_baseline, radius, moon_period, moon_initial_alpha]
            if i % alpha_grid_size == 0:
                best_snr_per_period.append([moon_initial_alpha, model_snr])
            elif best_snr_per_period[i // alpha_grid_size][1] < model_snr:
                best_snr_per_period[i // alpha_grid_size] = [moon_initial_alpha, model_snr]
        best_residuals_spectra = best_residuals_spectra + [best_residual_per_period]
        best_residuals_spectra_values = best_residuals_spectra_values + best_residual_per_period_values
        if not np.all(np.isfinite(residuals_matrix)) or not np.all(np.isfinite(snr_matrix)):
            raise RuntimeError("Moon search produced nonfinite phase-grid scores")
        refined_alpha_per_period = ExoMoonLeastSquares.refine_alpha(alpha_grid, residuals_matrix)
        snr_refined_alpha_per_period = ExoMoonLeastSquares.refine_alpha(alpha_grid, -snr_matrix)
        # Interpolated vertices are proposals, never evaluated detections. Keep
        # the original grid solution unless an actual model improves its score.
        for j in range(len(period_grid)):
            candidate = copy.copy(search_inputs[j * alpha_grid_size])
            refined_radius = [None]

            def refined_objective(alpha):
                candidate.moon_alpha = alpha
                residual, baseline, radius, _ = self.search(candidate)
                refined_radius[0] = radius
                return residual - baseline

            accepted_alpha, accepted_score = ExoMoonLeastSquares.accept_refined_alpha(
                best_alpha_per_period[j][0], best_alpha_per_period[j][1],
                refined_alpha_per_period[j, 0], refined_objective)
            refined_alpha_per_period[j] = [accepted_alpha, accepted_score]
            if accepted_score < best_alpha_per_period[j][1]:
                best_alpha_per_period[j] = [accepted_alpha, accepted_score]
                best_radius_per_period[j] = refined_radius[0]
            else:
                refined_alpha_per_period[j] = best_alpha_per_period[j]
            candidate.moon_alpha = snr_refined_alpha_per_period[j, 0]
            _, _, _, snr = self.search(candidate)
            if np.isfinite(snr) and snr > best_snr_per_period[j][1]:
                best_snr_per_period[j] = [candidate.moon_alpha, snr]
            snr_refined_alpha_per_period[j] = [best_snr_per_period[j][0], -best_snr_per_period[j][1]]
        # Spectrum values must refer to the same evaluated phase candidates used
        # for labels and follow-up searches, never an interpolated prediction.
        best_residuals_spectra = np.asarray([candidate[1] for candidate in best_alpha_per_period])
        self.spectrum_candidates = ExoMoonLeastSquares.spectrum_candidates(
            period_grid, best_alpha_per_period, best_radius_per_period)
        best_residuals_per_scenarios = np.array(best_residuals_per_scenarios)
        transposed_best_residuals_per_scenarios = np.transpose(best_residuals_per_scenarios)
        residuals_per_scenario_baseline_args = np.argwhere(transposed_best_residuals_per_scenarios[2] > ExoMoonLeastSquares.MAX_RESIDUAL / 2 ).flatten()
        transposed_best_residuals_per_scenarios[2][residuals_per_scenario_baseline_args] = max_residual
        best_residuals_per_scenarios = np.transpose(transposed_best_residuals_per_scenarios)
        snr_spectrum = np.transpose(best_snr_per_period)[1]
        kernel = 31
        SR, power_raw, power, SDE_raw, SDE = ExoMoonLeastSquares.spectra(best_residuals_spectra, kernel_size=kernel)
        snr_median = ExoMoonLeastSquares.running_median(snr_spectrum, kernel=kernel)
        snr_norm = snr_spectrum - snr_median
        snr_norm = snr_norm / np.std(snr_norm) if np.std(snr_norm) > 0 else np.zeros_like(snr_norm)
        if plot:
            # Ranking is presentation-only. Advanced indexing creates a
            # detached diagnostic copy, so plotting cannot reorder fit/search
            # arrays used by later numeric recovery calculations.
            best_residuals_per_scenarios_plot = self.sorted_diagnostic_copy(
                best_residuals_per_scenarios, 2)
            fig_transit, axs = plt.subplots(4, 1, figsize=(20, 28))
            for ax, values, label in zip(axs, (best_residuals_spectra, snr_spectrum, power, snr_norm),
                                         ('-Delta chi-square', 'SNR', 'SDE (-Delta chi-square)', 'SDE (SNR)')):
                ax.plot(period_grid, values, color='gray')
                ax.set_xlabel('P (days)')
                ax.set_ylabel(label)
            plt.savefig(self.object_dir + "/stats.png")
            plt.close(fig_transit)
            residuals_score = -residuals_matrix
            plt.imshow(residuals_score, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.savefig(self.object_dir + "/residuals_map.png")
            plt.close()
        for i in range(min(5, len(best_residuals_per_scenarios_plot)) if plot else 0):
            moon_period = best_residuals_per_scenarios_plot[i][0]
            moon_initial_alpha = best_residuals_per_scenarios_plot[i][1]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, moon_period)
            moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                SearchInput(moon_period, moon_initial_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_max_orbit_transit_duration,
                            moon_transit_duration, moon_orbit_transit_duration, moon_transit_duration / 2,
                            moon_orbit_transit_duration / 2, moon_max_orbit_transit_duration / 2),
                return_lc=True)
            if len(model) == 0:
                continue
            moon_depth = 1 - np.min(model)
            moon_radius = radius
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
        for i, top_sde_index in enumerate(top_sde_args[0:5] if plot else []):
            moon_period = period_grid[top_sde_index]
            moon_initial_alpha = refined_alpha_per_period[top_sde_index][0]
            sde = power[top_sde_index]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, moon_period)
            moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                SearchInput(moon_period, moon_initial_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_max_orbit_transit_duration,
                            moon_transit_duration, moon_orbit_transit_duration, moon_transit_duration / 2,
                            moon_orbit_transit_duration / 2, moon_max_orbit_transit_duration / 2),
                return_lc=True)
            if len(model) == 0:
                continue
            scenario_time = np.append(it_time, oot_time)
            scenario_flux = np.append(it_flux, oot_flux)
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            moon_depth = 1 - np.min(model)
            moon_radius = radius
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
        for i in range(min(5, len(best_scenario_per_snr_args)) if plot else 0):
            index = best_scenario_per_snr_args[-1 -i]
            moon_period = period_grid[index]
            moon_initial_alpha = best_snr_per_period[index][0]
            moon_max_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
            moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_period,
                                                                  self.max_radius * R_earth_to_R_sun,
                                                                  self.planet_impact_param)
            moon_max_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_max_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            moon_semimajor_axis = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, moon_period)
            moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                            self.planet_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_period,
                                                                            moon_semimajor_axis * AU_TO_RSUN,
                                                                            self.planet_impact_param)
            residuals, residuals_baseline, radius, model_snr, it_time, it_flux, oot_time, oot_flux, model = self.search(
                SearchInput(moon_period, moon_initial_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_max_orbit_transit_duration,
                            moon_transit_duration, moon_orbit_transit_duration, moon_transit_duration / 2,
                            moon_orbit_transit_duration / 2, moon_max_orbit_transit_duration / 2),
                return_lc=True)
            if len(model) == 0:
                logging.info("Best SNR for period %s, alpha %s: no in-transit data, skipping",
                             moon_period, moon_initial_alpha)
                continue
            fig_transit, axs = plt.subplots(2, 1, figsize=(20, 20))
            moon_depth = 1 - np.min(model)
            moon_radius = radius
            if i == 0:
                self.best_snr_radius = moon_radius
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
        # --- Compute best-candidate T0s (reused by recovery + plotting) ---
        best_snr_idx = best_scenario_per_snr_args[-1]
        best_snr_period = period_grid[best_snr_idx]
        best_snr_alpha = snr_refined_alpha_per_period[best_snr_idx][0]
        moon_max_sma = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, period_grid[-1])
        moon_dur_orbit_max = self.compute_transit_duration(self.star_radius,
            self.planet_semimajor_axis * AU_TO_RSUN, self.planet_period,
            moon_max_sma * AU_TO_RSUN, self.planet_impact_param)
        half_orbit_dur_max = moon_dur_orbit_max / 2
        moon_sma_rec = self.compute_semimajor_axis(self.planet_mass * M_earth_to_M_sun, best_snr_period)
        moon_dur_orbit_rec = self.compute_transit_duration(self.star_radius,
            self.planet_semimajor_axis * AU_TO_RSUN, self.planet_period,
            moon_sma_rec * AU_TO_RSUN, self.planet_impact_param)
        half_orbit_dur_rec = moon_dur_orbit_rec / 2
        moon_dur = self.compute_transit_duration(self.star_radius,
            self.planet_semimajor_axis * AU_TO_RSUN, self.planet_period,
            self.max_radius * R_earth_to_R_sun, self.planet_impact_param)
        half_moon_dur = moon_dur / 2
        rec_t0s, inj_t0s = [], []
        injected_by_epoch = {}
        recovered_by_epoch = {}
        recovered_by_epoch_id = {}
        has_injected = hasattr(self, 'injection_truth')
        event_status = {}
        if has_injected:
            diagnostic_by_epoch = {float(item['epoch']): item for item in self.epoch_diagnostics}
            for epoch_id, event in self.injection_truth['events'].items():
                diagnostic = diagnostic_by_epoch.get(event.epoch_time)
                if diagnostic is not None and not diagnostic['accepted']:
                    event_status[epoch_id] = 'rejected'
                else:
                    event_status[epoch_id] = event.coverage
        for t0_i in self.bary_t0s:
            max_tau = ExoMoonLeastSquares.compute_moon_ts(t0_i, half_orbit_dur_rec, half_moon_dur)
            rec_t0, _ = ExoMoonLeastSquares.compute_moon_t0(
                self.injection_truth['phase_reference_time'] if has_injected else self.planet_t0,
                t0_i, best_snr_period, best_snr_alpha, max_tau)
            rec_t0s.append(rec_t0)
            recovered_by_epoch[t0_i] = rec_t0
            if has_injected:
                epoch_id = self.epoch_ids_by_time[float(t0_i)]
                recovered_by_epoch_id[epoch_id] = rec_t0
                event = self.injection_truth['events'][epoch_id]
                if event.moon_t0 is not None:
                    inj_t0s.append(event.moon_t0)
                    injected_by_epoch[t0_i] = event.moon_t0
        self.recovered_t0s = rec_t0s
        self.injected_t0s = inj_t0s if inj_t0s else None
        self.injected_t0s_by_epoch = injected_by_epoch
        self.recovered_t0s_by_epoch = recovered_by_epoch
        self.recovered_period = float(best_snr_period)
        self.recovered_alpha = float(best_snr_alpha)

        if inject_params is not None:
            alias_match = self.assess_period_alias(
                inject_params['moon_period'], best_snr_period, self.planet_period)
            period_err = alias_match['direct_relative_error']
            alpha_err = abs(np.cos(best_snr_alpha) - np.cos(inject_params['initial_alpha']))
            alias_period = alias_match['matched_period_days']
            alias_k = alias_match['alias_k']
            min_alias_err = alias_match['relative_error']
            logging.info("INJECTION RECOVERY (SNR): injected P=%.4f d, alpha=%.4f rad -> recovered P=%.4f d, alpha=%.4f rad (delta_P=%.2f%%, delta_cos-alpha=%.4f)",
                         inject_params['moon_period'], inject_params['initial_alpha'],
                         best_snr_period, best_snr_alpha, period_err * 100, alpha_err)
            if alias_k is not None and alias_k != 0:
                logging.info("  Best alias match: k=%d, P_alias=%.4f d (delta=%.2f%%)",
                             alias_k, alias_period, min_alias_err * 100)
            self.recovered_period = float(best_snr_period)
            self.recovered_alpha = float(best_snr_alpha)
            timing_matches, event_status = self.match_recovered_events(
                self.injection_truth['events'], recovered_by_epoch_id, event_status)
            timing_values = np.array(list(timing_matches.values()))
            timing_recovered = bool(len(timing_values) and np.median(timing_values) < 0.1)
            self.recovery_result = {
                'detection_score': {'model_snr': float(best_snr_per_period[best_snr_idx][1])},
                'period_alias_match': {**alias_match, 'matched': bool(min_alias_err < 0.05)},
                'timing_recovery': {'phase_reference': 'planet_t0_days',
                                    'errors_days_by_epoch_id': timing_matches,
                                    'median_error_days': (float(np.median(timing_values))
                                                          if len(timing_values) else None),
                                    'matched': timing_recovered,
                                    'event_status_by_epoch_id': event_status},
                'radius_recovery': {'radius_earth': None, 'matched': None},
            }
            # Legacy boolean keeps the former timing-first meaning without
            # conflating it with score, period aliases, or radius recovery.
            self.recovered = timing_recovered
        # Plot best moon model on each single transit
        try:
            has_injected_plot = hasattr(self, 'injected_flux_moon_pandora')
            # Get recovered radius from search at best (P, alpha)
            _, _, recovered_radius, _, _, _, _, _, _ = self.search(
                SearchInput(best_snr_period, best_snr_alpha,
                            0, 90, 0, self.min_radius, self.planet_impact_param,
                            moon_dur_orbit_max, moon_dur, moon_dur_orbit_rec,
                            half_moon_dur, half_orbit_dur_rec, half_orbit_dur_max),
                return_lc=True)
            logging.info("Recovered moon radius: %.4f R_earth", recovered_radius)
            self.best_snr_radius = recovered_radius
            if hasattr(self, 'recovery_result'):
                injected_radius = inject_params['moon_radius']
                self.recovery_result['radius_recovery'] = {
                    'radius_earth': float(recovered_radius),
                    'absolute_error_earth': float(abs(recovered_radius - injected_radius)),
                    'matched': None,
                }
            for i, t0 in enumerate(self.bary_t0s if plot else []):
                mask_orbit = np.abs(self.time - t0) < half_orbit_dur_max
                if len(self.time[mask_orbit]) < self.min_cadences_per_t0:
                    continue
                t_local = self.time[mask_orbit]
                f_local = self.flux[mask_orbit]
                # Use pre-computed T0s
                moon_t0_t = rec_t0s[i]
                moon_t0_t_inj = injected_by_epoch.get(t0)
                ma_moon = batman.TransitParams()
                ma_moon.t0 = moon_t0_t
                ma_moon.per = self.planet_period
                ma_moon.rp = max(recovered_radius, self.min_radius) * R_earth_to_R_sun / self.star_radius
                ma_moon.a = self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius
                ma_moon.inc = self.planet_inc
                ma_moon.ecc = self.planet_ecc
                ma_moon.w = 0
                ma_moon.u = self.ab
                ma_moon.limb_dark = "quadratic"
                m_moon = batman.TransitModel(ma_moon, t_local)
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

        self.max_snr = float(np.nanmax(np.transpose(best_snr_per_period)[1]))
        self.max_sde = float(SDE)

        # Changed templates/objectives invalidate the previous IR calibration.
        self.calibration_status = 'uncalibrated: physical-time templates and consistent OOT weights'
        logging.info("EMLS screening statistics are %s", self.calibration_status)


class SearchInput:
    """Container for moon search input parameters."""
    def __init__(self, moon_period, moon_alpha, moon_ecc, moon_inc, moon_arg_periastron, min_radius, impact_param,
                 moon_max_orbit_transit_duration, moon_transit_duration, moon_orbit_transit_duration,
                 half_moon_transit_duration, half_moon_orbit_transit_duration, half_moon_max_orbit_transit_duration) -> None:
        """Initialize search input parameters for a single moon search scenario.

        Parameters
        ----------
        moon_period : float
            Moon orbital period in days.
        moon_alpha : float
            Moon initial orbital phase in radians.
        moon_ecc : float
            Moon orbital eccentricity.
        moon_inc : float
            Moon orbital inclination in degrees.
        moon_arg_periastron : float
            Moon argument of periastron in degrees.
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
        self.moon_ecc = moon_ecc
        self.moon_inc = moon_inc
        self.moon_arg_periastron = moon_arg_periastron
        self.min_radius = min_radius
        self.impact_param = impact_param
        self.moon_max_orbit_transit_duration = moon_max_orbit_transit_duration
        self.moon_transit_duration = moon_transit_duration
        self.moon_orbit_transit_duration = moon_orbit_transit_duration
        self.half_moon_transit_duration = half_moon_transit_duration
        self.half_moon_orbit_transit_duration = half_moon_orbit_transit_duration
        self.half_moon_max_orbit_transit_duration = half_moon_max_orbit_transit_duration

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
    if catalog not in ('tois', 'kois'):
        raise ValueError("--catalog must be 'tois' or 'kois'")
    csv_path = f"/home/martin/.sherlockpipe/{catalog}.csv"
    tois_df = pd.read_csv(csv_path)
    cadence = [120]
    author = 'SPOC'
    if catalog == "kois":
        author = 'Kepler'
        cadence = [60]
        if "Predicted Mass (M_Earth)" not in tois_df.columns:
            tois_df["Predicted Mass (M_Earth)"] = tois_df["Planet Radius (R_Earth)"] ** 1.5
        if "Stellar Mass (M_Sun)" not in tois_df.columns:
            G = 6.674e-11
            g = 10 ** tois_df["Stellar log(g) (cm/s^2)"] / 100
            r_m = tois_df["koi_srad"] * 6.957e8
            tois_df["Stellar Mass (M_Sun)"] = g * r_m ** 2 / (G * 1.989e30)
        if "Stellar Radius (R_Sun)" not in tois_df.columns:
            tois_df["Stellar Radius (R_Sun)"] = tois_df["koi_srad"]
        tois_df["Planet SNR"] = tois_df["koi_model_snr"]
        tois_df["Planet Radius (R_Earth) err"] = tois_df["koi_prad_err1"].fillna(0.1 * tois_df["Planet Radius (R_Earth)"])
        tois_df = tois_df.rename(columns={"Period (days) err": "Period (days)"})
        tois_df["planet_impact"] = tois_df["koi_impact"].fillna(0)
    confirmed_tois_df = tois_df.loc[(tois_df['Disposition'] == 'CP') | (tois_df['Disposition'] == 'KP') | (tois_df['Disposition'] == 'CONFIRMED') | (tois_df['Disposition'] == 'CANDIDATE')]
    confirmed_tois_df = confirmed_tois_df.loc[confirmed_tois_df['Planet SNR'] > 15]
    confirmed_tois_df = confirmed_tois_df.sort_values(by='Planet SNR', ascending=False)
    # Moon dynamical stability filter
    def hill_roche_stable(period, p_radius, p_mass, star_mass):
        """Check if Hill radius exceeds Roche limit: r_Hill > 2 * r_Roche.

        The Hill sphere (Hill 1878, Am. J. Math. 1, 5) defines the region
        where the planet's gravity dominates over the star:
            r_Hill = a_p × (M_p / 3M_star)^(1/3)

        The Roche limit (Roche 1849, Acad. Sci. Montpellier) for a rigid
        satellite is the closest orbit before tidal disruption:
            r_Roche = 2.44 × R_p × (ρ_p / ρ_m)^(1/3)

        A moon can only exist if r_Hill > r_Roche. We require a factor 2
        margin for long-term stability (Barnes & O'Brien 2002, ApJ 575, 1087)."""
        a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, period)
        r_hill = a * (p_mass * M_earth_to_M_sun / (3 * star_mass)) ** (1 / 3)
        moon_density = 5.5
        p_density = p_mass / p_radius ** 3
        r_roche = 2.44 * p_radius * max(p_density / moon_density, 0.1) ** (1 / 3)
        r_roche_au = r_roche * R_earth_to_R_sun / 215.032
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
        object_dir = target_name + "_EMLS/"
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
            planet_radius = row['Planet Radius (R_Earth)']
            planet_radius_err = row['Planet Radius (R_Earth) err']
            planet_period = row['Period (days)']
            planet_t0 = row['Epoch (BJD)'] - bjd_offset
            planet_t0 = times[0] + (planet_t0 - times[0]) % planet_period
            planet_duration = row['Duration (hours)'] / 24
            planet_ecc = 0.15
            planet_ecc_err = 0.15
            planet_arg_periastron = 0
            catalog_impact = float(row["planet_impact"]) if catalog == "kois" else 0.0
            geometry = ExoMoonLeastSquares.catalog_planet_geometry(
                catalog, star_mass, star_radius, planet_period, planet_radius,
                catalog_impact, planet_ecc, planet_arg_periastron)
            planet_semimajor_axis = geometry['planet_semimajor_axis']
            planet_inc = geometry['planet_inc']
            planet_inc_err = geometry['planet_inc_err']
            planet_impact_param = geometry['planet_impact_param']
            min_radius = geometry['min_radius']
            max_radius = geometry['max_radius']
            planet_mass = row['Predicted Mass (M_Earth)']
            if np.isnan(planet_mass) or planet_mass is None:
                planet_mass = planet_radius ** 1.5
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
                                       min_transit_completeness=0.75, max_subtraction_rms_ratio=1.5,
                                       exposure_times_seconds=cadence[0] if len(cadence) == 1 else None)
            #emls.fit_single_transits(times, flux, flux_err)
            # emls.flux = emls.inject_moon(emls.time, emls.flux, emls.flux_err, emls.bary_t0s, planet_semimajor_axis,
            #                             planet_ecc, moon_radius, moon_period, moon_mass, plot=True, initial_alpha=moon_alpha,
            #                              inject_planet=False, inject_moon=False)
            #emls.fit_single_transits(times, flux, flux_err, plot=True)
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
