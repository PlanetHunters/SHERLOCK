"""Moon orbit dynamics: semi-major axes, Hill, Domingos and Roche limits, periods, moon
transit times and offsets, and the search hypotheses.
"""
import logging
import math
import warnings
import batman
import numpy as np
from sherlockpipe.system_stability.mr_forecast import MrForecast
from sherlockpipe.exomoon.constants import AU_TO_M, AU_TO_RSUN, EARTH_DENSITY, G, M_earth_to_M_sun, Msolar_to_kg, R_earth_to_R_sun


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


class MoonOrbits:
    """Moon orbit dynamics: semi-major axes, Hill, Domingos and Roche limits, periods, moon
    transit times and offsets, and the search hypotheses.
    """

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

    @classmethod
    def compute_min_moon_period(cls, moon_density, planet_mass, planet_radius, max_moon_radius):
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
        contact_period = cls.au_to_period(planet_mass * M_earth_to_M_sun, contact_au)
        return max(cls.compute_roche_period(moon_density), contact_period)

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

    # Domingos, Winter & Yokoyama (2006), circular moon orbits: a_max = 0.4895 (1 - 1.0305 e_p) r_H
    # prograde and 0.9309 (1 - 1.0764 e_p) r_H retrograde, with r_H without the (1 - e_p) factor, as
    # quoted by Rosario-Franco et al. (2020, Table 1) and Heller et al. (2014).
    DOMINGOS_PROGRADE = (0.4895, 1.0305)
    DOMINGOS_RETROGRADE = (0.9309, 1.0764)

    @classmethod
    def max_moon_semimajor_axis(cls, star_mass, planet_mass, planet_semimajor_axis, planet_ecc=0.0,
                                retrograde=False):
        """Return the widest stable moon orbit in AU (audit items G5, G6).

        a_max = 0.4895 (1 - 1.0305 e_p) r_H for a prograde moon and 0.9309 (1 - 1.0764 e_p) r_H for
        a retrograde one (Domingos et al. 2006), with the Hill radius of a circular orbit, so the
        eccentricity is counted once: 0.49 r_H prograde for a circular planet orbit and 0.34 r_H
        at e_p = 0.3.

        Parameters
        ----------
        star_mass, planet_mass : float
            Masses in solar masses.
        planet_semimajor_axis : float
            Planet semi-major axis in AU.
        planet_ecc : float, optional
            Planet eccentricity, default 0.
        retrograde : bool, optional
            Return the retrograde limit instead of the prograde one, default False.

        Returns
        -------
        float
            Semi-major axis in AU.
        """
        coefficient, slope = cls.DOMINGOS_RETROGRADE if retrograde else cls.DOMINGOS_PROGRADE
        factor = coefficient * (1 - slope * planet_ecc)
        if not factor > 0:
            raise ValueError(f"No stable {'retrograde' if retrograde else 'prograde'} moon orbit for a planet "
                             f"eccentricity of {planet_ecc}")
        return factor * cls.compute_hill_radius(star_mass, planet_mass, planet_semimajor_axis)

    def check_model_validity(self):
        """Log where the screening moon model stops holding (audit items G3, G4, G14, G16).

        The moon model is coplanar and circular, with no sense of rotation (the projected offset
        cos(alpha + 2 pi t / P) is the same for a retrograde moon at -alpha), the moon phase frozen at the
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

    SPEED_MISMATCH = 0.1

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
        level = logging.WARNING if abs(ratio - 1) > self.SPEED_MISMATCH else logging.INFO
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
        level = logging.WARNING if abs(ratio - 1) > self.SPEED_MISMATCH else logging.INFO
        logging.log(level, "Pandora injection a/R* %.2f from the transit planet speed; the input a gives %.2f "
                    "(ratio %.2f)%s", a_over_rstar, input_a, ratio,
                    ", so the input a, e or w does not match the transit" if level == logging.WARNING else "")
        return float(a_over_rstar)

    def max_moon_period(self, planet_mass):
        """Return the period in days of the widest stable moon orbit searched for a planet mass in Earth masses.

        It is the retrograde limit when `search_retrograde` is set (the default of the search), the
        prograde one otherwise.
        """
        mass = planet_mass * M_earth_to_M_sun
        return self.au_to_period(mass, self.max_moon_semimajor_axis(
            self.star_mass, mass, self.planet_semimajor_axis, self.planet_ecc,
            retrograde=getattr(self, 'search_retrograde', False)))

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

    @classmethod
    def min_period_for_orbit_duration(cls, star_radius, planet_semimajor_axis,
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
        moon_transit = cls.compute_transit_duration(star_radius, planet_a_rsun, planet_period,
                                                                    moon_radius, impact_parameter)
        a_moon_rsun = (orbit_duration_days - moon_transit) * np.pi * planet_a_rsun / planet_period
        if a_moon_rsun <= 0:
            return 0.0
        return cls.au_to_period(planet_mass, a_moon_rsun / AU_TO_RSUN)

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

    @classmethod
    def compute_moon_orbit_transit_duration(cls, star_radius, planet_semimajor_axis, planet_period, moon_semimajor_axis,
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
        moon_transit_duration = cls.compute_transit_duration(
            star_radius, planet_semimajor_axis if transit_semimajor_axis is None else transit_semimajor_axis,
            planet_period, moon_radius, impact_parameter)
        max_offset = cls.compute_max_moon_offset(moon_semimajor_axis, planet_semimajor_axis,
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
