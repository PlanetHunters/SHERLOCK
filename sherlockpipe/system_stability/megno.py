from typing import Dict, List

import rebound
import numpy as np
from astropy import units as u
from lcbuilder.helper import LcbuilderHelper
from sherlockpipe.system_stability.stability_calculator import StabilityCalculator, SimulationInput
import pandas as pd


class MegnoStabilityCalculator(StabilityCalculator):

    """
    Runs the stability computations by calculating the MEGNO score.
    """
    def __init__(self, years, dt, repetitions, checkpoints_per_decade=20):
        """
        Initialize the MEGNO-based stability calculator.

        Parameters
        ----------
        years : float
            Maximum integration time in years.
        dt : float
            Fraction of the minimum orbital period used as the integration timestep.
        repetitions : int
            Number of simulation repetitions per scenario.
        checkpoints_per_decade : int, optional
            How many times per decade of simulated time the MEGNO is read. It sets the resolution of the time at which
            an unstable scenario is lost. Default is 20.
        """
        super().__init__(dt)
        self.years = years
        self.repetitions = repetitions
        self.checkpoints_per_decade = checkpoints_per_decade

    def run_simulation(self, simulation_input: SimulationInput) -> dict:
        """
        Runs one simulation scenario using the megno module of rebound.

        Every repetition integrates the scenario up to a series of logarithmically spaced checkpoints, reading the
        MEGNO at each of them. A repetition ends when the MEGNO reaches 5, or when a planet escapes or has a close
        encounter, and the time reached by the simulation at that moment is recorded as the time of loss.

        :param SimulationInput simulation_input: the scenario parameters
        :return dict: the results of the simulation with the rebound-specific megno metric: the median, the minimum,
            the maximum and the standard deviation of the MEGNO over the repetitions, and the median time of loss
            (``last_year``, equal to the simulated years when no repetition was lost).
        """
        final_megno = []
        final_year = []
        shortest_period_years = LcbuilderHelper.convert_from_to(min(simulation_input.planet_periods), u.day, u.year)
        checkpoints = self.checkpoints(self.years, shortest_period_years, self.checkpoints_per_decade)
        for repetition in range(self.repetitions):
            sim = self.init_rebound_simulation(simulation_input, repetition)
            sim.init_megno(seed=int(np.random.default_rng([int(simulation_input.seed), repetition, 1]).integers(2 ** 31)))
            sim.exit_max_distance = 20.
            megno = 0
            for year in checkpoints:
                try:
                    sim.integrate(year, exact_finish_time=0)
                    megno = sim.megno()
                    megno = megno if megno < 5 else 5
                except (rebound.Escape, rebound.Encounter):
                    megno = 5
                if megno == 5:
                    break
            final_megno = np.append(final_megno, megno)
            final_year = np.append(final_year, sim.t if megno == 5 else self.years)
        return {"star_mass": simulation_input.star_mass,
                "periods": ",".join([str(planet_period) for planet_period in simulation_input.planet_periods]),
                "masses": ",".join([str(mass_value) for mass_value in simulation_input.mass_arr]),
                "inclinations": ",".join([str(inc_value) for inc_value in simulation_input.inc_arr]),
                "eccentricities": ",".join([str(ecc_value) for ecc_value in simulation_input.ecc_arr]),
                "arg_periastron": ",".join([str(om_value) for om_value in simulation_input.omega_arr]),
                "long_asc_node": ",".join([str(om_big_value) for om_big_value in simulation_input.omega_big_arr]),
                "last_year": np.nanmedian(final_year),
                "megno": np.nanmedian(final_megno),
                "megno_min": np.nanmin(final_megno),
                "megno_max": np.nanmax(final_megno),
                "megno_std": np.nanstd(final_megno),
                "seed": simulation_input.seed}

    def store_simulation_results(self, simulation_results: List[Dict], results_dir: str):
        """
        Stores the megno results in the given directory

        :param List[Dict] simulation_results: the list of results dictionaries
        :param str results_dir: directory to store the file
        """
        result_file = results_dir + "/stability_megno.csv"
        results_df = pd.DataFrame(columns=['star_mass', 'periods', 'masses', 'inclinations', 'eccentricities',
                                           'arg_periastron', 'long_asc_node', 'last_year', 'megno', 'megno_min',
                                           'megno_max', 'megno_std', 'seed'])
        results_df = pd.concat([results_df, pd.DataFrame(simulation_results)], ignore_index=True)
        results_df = results_df.sort_values('megno')
        results_df.to_csv(result_file, index=False)


