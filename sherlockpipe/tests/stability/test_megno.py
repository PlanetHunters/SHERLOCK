"""MEGNO simulations on tiny systems (a few seconds each)."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from sherlockpipe.system_stability.megno import MegnoStabilityCalculator
from sherlockpipe.system_stability.stability_calculator import SimulationInput


def scenario(periods, eccs, masses=(10.0, 10.0), omegas=(60, 200), omega_bigs=(0, 0), index=0, **kw):
    return SimulationInput(0.82, list(masses), list(periods), list(eccs), [0, 0], list(omegas), list(omega_bigs), index, **kw)


class MegnoSimulationTest(unittest.TestCase):
    def setUp(self):
        self.calculator = MegnoStabilityCalculator(years=60, dt=0.05, repetitions=1)

    def test_widely_separated_circular_planets_are_stable(self):
        result = self.calculator.run_simulation(scenario((10.0, 30.0), (0.0, 0.0)))
        self.assertLess(result["megno"], 2.5)
        self.assertEqual(result["last_year"], 60)

    def test_crossing_orbits_are_lost_early_and_flagged_with_megno_five(self):
        result = self.calculator.run_simulation(scenario((4.0, 5.0), (0.4, 0.4), masses=(100.0, 100.0)))
        self.assertEqual(result["megno"], 5)
        self.assertLess(result["last_year"], 60)

    def test_the_result_has_the_scenario_parameters_as_comma_separated_strings(self):
        result = self.calculator.run_simulation(scenario((10.0, 30.0), (0.0, 0.1)))
        self.assertEqual(result["periods"], "10.0,30.0")
        self.assertEqual(result["eccentricities"], "0.0,0.1")
        self.assertEqual(result["star_mass"], 0.82)


class MegnoStorageTest(unittest.TestCase):
    def test_results_are_sorted_by_megno_in_stability_megno_csv(self):
        results = [{"star_mass": 0.8, "periods": "1,2", "masses": "1,1", "inclinations": "0,0", "eccentricities": "0,0",
                    "arg_periastron": "0,0", "long_asc_node": "0,0", "last_year": 10, "megno": m} for m in (5, 1.5, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            MegnoStabilityCalculator(10, 0.05, 1).store_simulation_results(results, tmp)
            frame = pd.read_csv(os.path.join(tmp, "stability_megno.csv"))
        self.assertEqual(frame.megno.tolist(), [1.5, 3, 5])


if __name__ == "__main__":
    unittest.main()
