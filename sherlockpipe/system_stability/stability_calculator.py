import itertools
import json
import logging
import multiprocessing
import random
from typing import List, Dict

from abc import ABC, abstractmethod
from multiprocessing import Pool
import numpy as np
import rebound
from astropy import units as u
from lcbuilder.helper import LcbuilderHelper

from sherlockpipe.system_stability.mr_forecast import MrForecast

"""Includes classes to be used ase base for stability simulations"""


class PlanetInput:
    """
    Defines the planet parameters for system stability calculations
    """
    def __init__(self, period, period_low_err, period_up_err,
                 radius, radius_low_err, radius_up_err, eccentricity, ecc_low_err, ecc_up_err,
                 inclination, inc_low_err, inc_up_err, omega, omega_low_err, omega_up_err,
                 omega_big=None, omega_big_low_err=None, omega_big_up_err=None,
                 mass=None, mass_low_err=None, mass_up_err=None, period_bins=3,
                 mass_bins=3, ecc_bins=3, inc_bins=3, omega_bins=3, omega_big_bins=3):
        """
        Initialize planet parameters for stability calculations.

        Parameters
        ----------
        period : float
            Orbital period in days.
        period_low_err : float
            Lower error on the orbital period in days.
        period_up_err : float
            Upper error on the orbital period in days.
        radius : float
            Planet radius in Earth radii.
        radius_low_err : float
            Lower error on the planet radius in Earth radii.
        radius_up_err : float
            Upper error on the planet radius in Earth radii.
        eccentricity : float
            Orbital eccentricity.
        ecc_low_err : float
            Lower error on the eccentricity.
        ecc_up_err : float
            Upper error on the eccentricity.
        inclination : float
            Orbital inclination in degrees.
        inc_low_err : float
            Lower error on the inclination in degrees.
        inc_up_err : float
            Upper error on the inclination in degrees.
        omega : float
            Argument of periastron in degrees.
        omega_low_err : float
            Lower error on the argument of periastron in degrees.
        omega_up_err : float
            Upper error on the argument of periastron in degrees.
        omega_big : float, optional
            Longitude of ascending node in degrees.
        omega_big_low_err : float, optional
            Lower error on the longitude of ascending node in degrees.
        omega_big_up_err : float, optional
            Upper error on the longitude of ascending node in degrees.
        mass : float, optional
            Planet mass in Earth masses. If not provided, estimated from radius.
        mass_low_err : float, optional
            Lower error on the planet mass in Earth masses.
        mass_up_err : float, optional
            Upper error on the planet mass in Earth masses.
        period_bins : int, optional
            Number of bins for the period grid. Default is 3.
        mass_bins : int, optional
            Number of bins for the mass grid. Default is 3.
        ecc_bins : int, optional
            Number of bins for the eccentricity grid. Default is 3.
        inc_bins : int, optional
            Number of bins for the inclination grid. Default is 3.
        omega_bins : int, optional
            Number of bins for the argument of periastron grid. Default is 3.
        omega_big_bins : int, optional
            Number of bins for the longitude of ascending node grid. Default is 3.
        """
        self.period = period
        self.period_low_err = period_low_err
        self.period_up_err = period_up_err
        self.radius = radius
        self.radius_low_err = radius_low_err
        self.radius_up_err = radius_up_err
        self.eccentricity = eccentricity
        self.ecc_low_err = ecc_low_err
        self.ecc_up_err = ecc_up_err
        self.inclination = inclination
        self.inc_low_err = inc_low_err
        self.inc_up_err = inc_up_err
        self.omega_big = omega_big
        self.omega_big_low_err = omega_big_low_err
        self.omega_big_up_err = omega_big_up_err
        self.omega = omega
        self.omega_low_err = omega_low_err
        self.omega_up_err = omega_up_err
        self.mass = mass
        self.mass_low_err = mass_low_err
        self.mass_up_err = mass_up_err
        self.mass_bins = mass_bins if mass_bins is not None else 3
        self.period_bins = period_bins if period_bins is not None else 3
        self.ecc_bins = ecc_bins if ecc_bins is not None else 3
        self.inc_bins = inc_bins if inc_bins is not None else 3
        self.omega_bins = omega_bins if omega_bins is not None else 3
        self.omega_big_bins = omega_big_bins if omega_big_bins is not None else 3


class SimulationInput:
    """
    Used as input for the simulations done for each scenario
    """

    def __init__(self, star_mass, mass_arr, planet_periods, ecc_arr, inc_arr, omega_arr, omega_big_arr, index,
                 seed=0):
        """
        Initialize a simulation scenario with sampled parameters.

        Parameters
        ----------
        star_mass : float
            Stellar mass in solar masses.
        mass_arr : ndarray
            Array of planet masses in Earth masses.
        planet_periods : ndarray
            Array of planet orbital periods in days.
        ecc_arr : ndarray
            Array of planet eccentricities.
        inc_arr : ndarray
            Array of planet inclinations in degrees.
        omega_arr : ndarray
            Array of planet arguments of periastron in degrees.
        omega_big_arr : ndarray
            Array of planet longitudes of ascending node in degrees.
        index : int
            Scenario identifier.
        seed : int, optional
            Seed of the random angles (the ones set to 'rand'). Each repetition of the scenario derives its own
            stream from it, so the same seed reproduces the same simulations.
        """
        self.star_mass = star_mass
        self.mass_arr = mass_arr
        self.planet_periods = planet_periods
        self.ecc_arr = ecc_arr
        self.inc_arr = inc_arr
        self.omega_arr = omega_arr
        self.omega_big_arr = omega_big_arr
        self.index = index
        self.seed = seed


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types. Got from https://stackoverflow.com/a/49677241/4198726"""

    def default(self, obj):
        """
        Default method invoked to convert the object

        :param obj: the input object
        :return: the json encoded object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class StabilityCalculator(ABC):
    """
    Template class for system stability calculation algorithms
    """

    def __init__(self, dt=0.05):
        """
        Initialize the stability calculator.

        Parameters
        ----------
        dt : float, optional
            Fraction of the minimum orbital period used as the integration timestep.
            Default is 0.05.
        """
        self.dt = dt

    @staticmethod
    def mass_from_radius(radius):
        """
        Computation of mass-radius relationship from
        Bashi D., Helled R., Zucker S., Mordasini C., 2017, A&A, 604, A83. doi:10.1051/0004-6361/201629922

        :param radius: the radius value in earth radius
        :return: the mass in earth masses
        """
        return radius ** (1 / 0.55) if radius <= 12.1 else radius ** (1 / 0.01)

    @staticmethod
    def prepare_star_masses(star_mass_low, star_mass_up, star_mass_bins):
        """
        Creates a star masses grid

        :param star_mass_low: the lowest star mass value
        :param star_mass_up: the highest star mass value
        :param star_mass_bins: the number of star masses to sample. It will be ignored if star_mass_low == star_mass_up.
        :return: the star masses grid
        """
        return np.linspace(star_mass_low, star_mass_up, star_mass_bins) if star_mass_low != star_mass_up \
            else np.linspace(star_mass_low, star_mass_up, 1)

    @staticmethod
    def prepare_planet_params(planet_params: List[PlanetInput]):
        """
        Fills the planet masses if missing

        :param planet_params: the planet inputs
        :return: the planet inputs with the filled masses
        """
        for planet_param in planet_params:
            if planet_param.radius is None and (planet_param.mass_low_err is None or planet_param.mass_up_err is None):
                raise ValueError("There is one body without either radius or mass information: " +
                                 json.dumps(planet_param.__dict__))
            if planet_param.radius is not None and planet_param.mass is None:
                mass, mass_up_err, mass_low_err = MrForecast.Rstat2M(planet_param.radius, np.max([planet_param.radius_up_err, planet_param.radius_low_err]))
                planet_param.mass = mass
                planet_param.mass_low_err = mass_low_err
                planet_param.mass_up_err = mass_up_err
        return planet_params

    def init_rebound_simulation(self, simulation_input, repetition=0):
        """
        Initializes the simulation for rebound-based algorithms

        :param simulation_input: the input data for the simulation scenario
        :param repetition: the repetition of the scenario. The angles set to 'rand' are drawn from a generator seeded
            with the scenario seed and the repetition, so they are reproducible and different between repetitions.
        :return: the rebound initialized simulation scenario
        """
        rng = np.random.default_rng([int(simulation_input.seed), int(repetition)])
        sim = rebound.Simulation()
        sim.units = ('Yr', 'AU', 'Msun')
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.add(m=simulation_input.star_mass)
        min_period = 1000
        for planet_key, mass in enumerate(simulation_input.mass_arr):
            period = simulation_input.planet_periods[planet_key]
            min_period = min_period if period > min_period else period
            ecc = simulation_input.ecc_arr[planet_key]
            inc = np.deg2rad(simulation_input.inc_arr[planet_key])
            omega = simulation_input.omega_arr[planet_key]
            if omega == 'rand':
                omega = rng.uniform(0, 360)
            omega = np.deg2rad(omega)
            omega_big = simulation_input.omega_big_arr[planet_key]
            if omega_big == 'rand':
                omega_big = rng.uniform(0, 360)
            omega_big = np.deg2rad(omega_big)
            sim.add(m=LcbuilderHelper.convert_from_to(mass, u.M_earth, u.M_sun) / simulation_input.star_mass,
                    P=LcbuilderHelper.convert_from_to(period, u.day, u.year), e=ecc, omega=omega, Omega=omega_big, inc=inc)
        sim.dt = LcbuilderHelper.convert_from_to(min_period, u.day, u.year) * self.dt
        # sim.status()
        sim.move_to_com()
        return sim

    @staticmethod
    def checkpoints(years, shortest_period_years, per_decade=20):
        """
        The times (in years) at which the simulation is paused to read its chaos indicator, spaced logarithmically
        from 10 orbits of the innermost planet to the end of the simulation, with ``per_decade`` points per decade.
        The finer the checkpoints, the closer the recorded loss time is to the real one.

        :param years: the total integration time
        :param shortest_period_years: the shortest orbital period of the system
        :param per_decade: the checkpoints per decade of time
        :return: the checkpoint times, the last one being ``years``
        """
        first = min(10 * shortest_period_years, years)
        if first >= years:
            return np.array([float(years)])
        count = max(2, int(np.ceil(per_decade * np.log10(years / first))) + 1)
        return np.logspace(np.log10(first), np.log10(years), count)

    @staticmethod
    def range_grid(value, low_err, up_err, bins, minimum=None, maximum=None):
        """
        The values of one dimension: the nominal value when it is fixed, otherwise ``bins`` equally spaced values
        between ``value - low_err`` and ``value + up_err`` (clamped to ``minimum`` and ``maximum``).
        """
        if bins == 1 or low_err == up_err == 0:
            return [value]
        low, up = value - low_err, value + up_err
        if minimum is not None:
            low = max(low, minimum)
        if maximum is not None:
            up = min(up, maximum)
        return np.linspace(low, up, bins).tolist()

    @staticmethod
    def angle_grid(value, low_err, up_err, bins, free):
        """
        The values of an angle dimension. ``'rand'`` stays a single entry: the angle is drawn at random (and
        reproducibly) in every repetition. A free angle takes ``bins`` equally spaced values around the circle.
        """
        if free:
            return np.linspace(0, 360, bins, endpoint=False).tolist()
        if isinstance(value, str) or bins == 1 or low_err == up_err == 0:
            return [value]
        return np.linspace(value - low_err, value + up_err, bins).tolist()

    @staticmethod
    def build_simulation_inputs(star_masses, planet_params: List[PlanetInput], free_params=None, ecc_max=0.5,
                                seed=0):
        """
        Builds every scenario of the parameter space: the cartesian product of the star masses and, for every
        planet, of its periods, masses, inclinations, eccentricities, arguments of periastron and longitudes of the
        ascending node. A dimension is fixed (one value), a grid between its errors, or free when named in
        ``free_params`` (``eccentricity``: 0 to ``ecc_max``; ``omega`` and ``omega_big``: the full circle).

        :param star_masses: the star masses to simulate
        :param planet_params: the planets, with their masses already filled
        :param free_params: the dimensions to sample entirely
        :param ecc_max: the highest eccentricity of a free eccentricity
        :param seed: base seed of the random angles
        :return: the list of scenarios, each with a unique index and its own seed
        """
        free_params = [] if free_params is None else free_params
        per_planet = {key: [] for key in ("period", "mass", "ecc", "inc", "omega", "omega_big")}
        for planet in planet_params:
            per_planet["period"].append(StabilityCalculator.range_grid(
                planet.period, planet.period_low_err, planet.period_up_err, planet.period_bins))
            per_planet["mass"].append(StabilityCalculator.range_grid(
                planet.mass, planet.mass_low_err, planet.mass_up_err, planet.mass_bins))
            if "eccentricity" in free_params:
                per_planet["ecc"].append(np.linspace(0, ecc_max, planet.ecc_bins).tolist())
            else:
                per_planet["ecc"].append(StabilityCalculator.range_grid(
                    planet.eccentricity, planet.ecc_low_err, planet.ecc_up_err, planet.ecc_bins, 0, 1))
            per_planet["inc"].append(StabilityCalculator.range_grid(
                planet.inclination, planet.inc_low_err, planet.inc_up_err, planet.inc_bins))
            per_planet["omega"].append(StabilityCalculator.angle_grid(
                planet.omega, planet.omega_low_err, planet.omega_up_err, planet.omega_bins, "omega" in free_params))
            per_planet["omega_big"].append(StabilityCalculator.angle_grid(
                planet.omega_big, planet.omega_big_low_err, planet.omega_big_up_err, planet.omega_big_bins,
                "omega_big" in free_params))
        combos = {key: list(itertools.product(*grids)) for key, grids in per_planet.items()}
        logging.info("Preparing system values for all scenarios")
        logging.info("%.0f star mass scenarios.", len(star_masses))
        for key, label in (("mass", "bodies mass"), ("period", "period"), ("inc", "inclination"),
                           ("ecc", "eccentricity"), ("omega", "arg of periastron"),
                           ("omega_big", "long of asc node")):
            logging.info("%.0f %s scenarios.", len(combos[key]), label)
        scenarios_num = len(star_masses) * int(np.prod([len(c) for c in combos.values()]))
        logging.info("%s = %.0f total scenarios.", " x ".join(
            [str(len(star_masses))] + [str(len(c)) for c in combos.values()]), scenarios_num)
        simulation_inputs = []
        for star_mass, periods, masses, incs, eccs, omegas, omega_bigs in itertools.product(
                star_masses, combos["period"], combos["mass"], combos["inc"], combos["ecc"], combos["omega"],
                combos["omega_big"]):
            index = len(simulation_inputs)
            simulation_inputs.append(SimulationInput(
                star_mass, list(masses), list(periods), list(eccs), list(incs), list(omegas), list(omega_bigs),
                index, seed=int((int(seed) * 1_000_003 + index) % 2 ** 31)))
        return simulation_inputs

    def run(self, results_dir, star_mass_low, star_mass_up, star_mass_bins, planet_params: List[PlanetInput],
            cpus=multiprocessing.cpu_count() - 1, free_params=None, ecc_max=0.5, seed=None):
        """
        Creates possible scenarios of stellar masses, planet masses and planet eccentricities. Afterwards a stability
        analysis is run for each of the scenarios and the results are stored in a file.

        :param results_dir: the directory where the results will be written into
        :param star_mass_low: the lowest star mass
        :param star_mass_up: the highest star mass
        :param star_mass_bins: the number of star masses to sample
        :param List[PlanetInput] planet_params: the planet inputs containing the planets parameters
        :param cpus: the number of cpus to be used
        :param free_params: the parameters to be sampled entirely: 'eccentricity', 'omega' and 'omega_big'
        :param ecc_max: the highest eccentricity of a free eccentricity
        :param seed: the base seed of the random angles. A random one is drawn (and logged) when it is None
        """
        if seed is None:
            seed = random.SystemRandom().randrange(2 ** 31)
        logging.info("Seed of the random angles: %s", seed)
        planet_params = StabilityCalculator.prepare_planet_params(planet_params)
        star_masses = StabilityCalculator.prepare_star_masses(star_mass_low, star_mass_up, star_mass_bins)
        simulation_inputs = StabilityCalculator.build_simulation_inputs(star_masses, planet_params, free_params,
                                                                        ecc_max, seed)
        logging.info("Finished preparing scenarios")
        with Pool(processes=cpus) as pool:
            simulation_results = pool.map(self.log_and_run_simulation, simulation_inputs)
        self.store_simulation_results(simulation_results, results_dir)

    def log_and_run_simulation(self, simulation_input: SimulationInput) -> dict:
        """
        Logs the simulation input and launches the simulation for the given input.

        :param SimulationInput simulation_input:
        :return dict: the resulting dictionary from the simulation
        """
        logging.info("Running scenario number %.0f: %s", simulation_input.index, json.dumps(simulation_input.__dict__,
                                                                                            cls=NumpyEncoder))
        return self.run_simulation(simulation_input)

    @abstractmethod
    def run_simulation(self, simulation_input: SimulationInput) -> dict:
        """
        Runs one stability scenario

        :param SimulationInput simulation_input: the simulation scenario parameters
        :return dict: the result of the scenario in a dictionary
        """
        pass

    @abstractmethod
    def store_simulation_results(self, simulation_results: List[Dict], results_dir: str):
        """
        Writes into disk all the final simulation results

        :param List[Dict] simulation_results: the results of the simulation for all the scenarios
        :param str results_dir: the output directory where results will be written into
        """
        pass
