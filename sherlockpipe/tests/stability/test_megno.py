"""MEGNO simulations on tiny systems (a few seconds each)."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from sherlockpipe.system_stability.megno import MegnoStabilityCalculator
from sherlockpipe.system_stability.stability_calculator import SimulationInput


def scenario(periods, eccs, masses=(10.0, 10.0), omegas=(60, 200), omega_bigs=(0, 0), index=0, seed=0):
    return SimulationInput(0.82, list(masses), list(periods), list(eccs), [0, 0], list(omegas), list(omega_bigs), index, seed)


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


class CheckpointsTest(unittest.TestCase):
    def test_they_are_logarithmic_from_ten_orbits_to_the_end(self):
        points = MegnoStabilityCalculator.checkpoints(300, 0.01)
        self.assertAlmostEqual(points[0], 0.1)
        self.assertAlmostEqual(points[-1], 300)
        ratios = points[1:] / points[:-1]
        self.assertTrue(np.allclose(ratios, ratios[0]))

    def test_they_have_the_requested_resolution(self):
        decades = np.log10(300 / 0.1)
        self.assertGreaterEqual(len(MegnoStabilityCalculator.checkpoints(300, 0.01, per_decade=20)), int(20 * decades))
        self.assertGreater(len(MegnoStabilityCalculator.checkpoints(300, 0.01, per_decade=40)),
                           len(MegnoStabilityCalculator.checkpoints(300, 0.01, per_decade=10)))

    def test_a_short_simulation_has_a_single_checkpoint(self):
        points = MegnoStabilityCalculator.checkpoints(0.05, 0.01)
        self.assertEqual(points.tolist(), [0.05])


class MegnoResolutionTest(unittest.TestCase):
    def test_the_time_of_loss_is_the_simulated_time_not_a_checkpoint(self):
        calculator = MegnoStabilityCalculator(years=60, dt=0.05, repetitions=1)
        result = calculator.run_simulation(scenario((4.0, 5.0), (0.4, 0.4), masses=(100.0, 100.0)))
        checkpoints = calculator.checkpoints(60, 4.0 / 365.25, 20)
        self.assertLess(result["last_year"], 60)
        self.assertGreater(result["last_year"], 0)
        # finer than the old 5 logarithmic points: more than one value between 0.1 and the first old checkpoint
        self.assertGreater(len(checkpoints), 20)

    def test_the_loss_time_has_a_resolution_below_a_third_of_a_decade(self):
        calculator = MegnoStabilityCalculator(years=300, dt=0.05, repetitions=1)
        coarse = MegnoStabilityCalculator(years=300, dt=0.05, repetitions=1, checkpoints_per_decade=3)
        a = calculator.run_simulation(scenario((4.0, 5.0), (0.4, 0.4), masses=(100.0, 100.0)))
        b = coarse.run_simulation(scenario((4.0, 5.0), (0.4, 0.4), masses=(100.0, 100.0)))
        self.assertLessEqual(a["last_year"], b["last_year"] * 1.0001)


class MegnoRepetitionsTest(unittest.TestCase):
    def test_the_spread_of_the_repetitions_is_reported(self):
        calculator = MegnoStabilityCalculator(years=30, dt=0.05, repetitions=3)
        result = calculator.run_simulation(scenario((10.0, 30.0), (0.05, 0.05), omegas=("rand", "rand"), seed=3))
        for key in ("megno_min", "megno_max", "megno_std", "seed"):
            self.assertIn(key, result)
        self.assertLessEqual(result["megno_min"], result["megno"])
        self.assertLessEqual(result["megno"], result["megno_max"])
        self.assertEqual(result["seed"], 3)

    def test_the_same_seed_reproduces_the_same_result_and_another_seed_changes_the_angles(self):
        calculator = MegnoStabilityCalculator(years=30, dt=0.05, repetitions=2)
        build = lambda seed: scenario((6.0, 9.0), (0.3, 0.3), masses=(50.0, 50.0), omegas=("rand", "rand"), seed=seed)
        first, again = calculator.run_simulation(build(5)), calculator.run_simulation(build(5))
        self.assertEqual((first["megno"], first["last_year"]), (again["megno"], again["last_year"]))
        sim_a = calculator.init_rebound_simulation(build(5), 0)
        sim_b = calculator.init_rebound_simulation(build(6), 0)
        self.assertNotEqual(sim_a.particles[1].omega, sim_b.particles[1].omega)

    def test_repetitions_of_a_scenario_draw_different_angles(self):
        calculator = MegnoStabilityCalculator(years=30, dt=0.05, repetitions=2)
        item = scenario((6.0, 9.0), (0.3, 0.3), omegas=("rand", "rand"), seed=5)
        self.assertNotEqual(calculator.init_rebound_simulation(item, 0).particles[1].omega,
                            calculator.init_rebound_simulation(item, 1).particles[1].omega)


class MegnoStorageTest(unittest.TestCase):
    def test_results_are_sorted_by_megno_in_stability_megno_csv(self):
        results = [{"star_mass": 0.8, "periods": "1,2", "masses": "1,1", "inclinations": "0,0", "eccentricities": "0,0",
                    "arg_periastron": "0,0", "long_asc_node": "0,0", "last_year": 10, "megno": m, "megno_min": m,
                    "megno_max": m, "megno_std": 0.0, "seed": 1} for m in (5, 1.5, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            MegnoStabilityCalculator(10, 0.05, 1).store_simulation_results(results, tmp)
            frame = pd.read_csv(os.path.join(tmp, "stability_megno.csv"))
        self.assertEqual(frame.megno.tolist(), [1.5, 3, 5])


if __name__ == "__main__":
    unittest.main()
