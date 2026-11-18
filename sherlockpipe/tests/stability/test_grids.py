"""Characterization of how the parameter space is turned into scenarios (before the fixes)."""
import tempfile
import unittest

from sherlockpipe.tests.stability.fixtures import RecordingCalculator, column, distinct, make_planet


def scenarios(planets, free_params=None, star=(0.82, 0.82, 1)):
    calculator = RecordingCalculator()
    with tempfile.TemporaryDirectory() as tmp:
        calculator.run(tmp, star[0], star[1], star[2], planets, cpus=1, free_params=free_params)
    return calculator.collected


class GridSizeTest(unittest.TestCase):
    def test_all_fixed_gives_a_single_scenario(self):
        self.assertEqual(len(scenarios([make_planet(), make_planet(period=6.0)])), 1)

    def test_the_space_is_the_cartesian_product_of_every_dimension(self):
        planets = [make_planet(eccentricity=0.1, ecc_low_err=0.1, ecc_up_err=0.3, ecc_bins=3),
                   make_planet(period=6.0, mass=20, mass_low_err=8, mass_up_err=20, mass_bins=2,
                               eccentricity=0.1, ecc_low_err=0.1, ecc_up_err=0.3, ecc_bins=3)]
        self.assertEqual(len(scenarios(planets)), 3 * 2 * 3)

    def test_star_mass_bins_multiply_the_scenarios(self):
        self.assertEqual(len(scenarios([make_planet()], star=(0.7, 0.9, 3))), 3)


class GridValuesTest(unittest.TestCase):
    def test_eccentricity_grid_is_clamped_at_zero(self):
        planet = make_planet(eccentricity=0.1, ecc_low_err=0.5, ecc_up_err=0.3, ecc_bins=5)
        self.assertEqual([round(e, 3) for e in distinct(scenarios([planet]), "ecc", 0)], [0.0, 0.1, 0.2, 0.3, 0.4])

    def test_period_and_mass_grids_span_value_minus_low_to_value_plus_up(self):
        planet = make_planet(period=6.0, period_low_err=0.5, period_up_err=1.0, period_bins=4,
                             mass=10, mass_low_err=4, mass_up_err=8, mass_bins=3)
        results = scenarios([planet])
        self.assertAlmostEqual(min(column(results, "periods", 0)), 5.5)
        self.assertAlmostEqual(max(column(results, "periods", 0)), 7.0)
        self.assertAlmostEqual(min(column(results, "masses", 0)), 6.0)
        self.assertAlmostEqual(max(column(results, "masses", 0)), 18.0)

    def test_free_eccentricity_spans_zero_to_half(self):
        planet = make_planet(ecc_bins=6)
        values = distinct(scenarios([planet], free_params=["eccentricity"]), "ecc", 0)
        self.assertEqual([round(v, 2) for v in values], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    def test_rand_angles_are_passed_through_to_the_simulation(self):
        planet = make_planet(omega="rand", omega_big="rand")
        results = scenarios([planet])
        self.assertEqual(column(results, "omega", 0), ["rand"])
        self.assertEqual(column(results, "omega_big", 0), ["rand"])

    def test_a_fixed_argument_of_periastron_is_kept(self):
        self.assertEqual(column(scenarios([make_planet(omega=60, omega_big=15)]), "omega", 0), [60])


class FixedBugsTest(unittest.TestCase):
    """Bugs of the grid that were fixed."""

    def test_longitude_of_ascending_node_uses_its_own_values(self):
        results = scenarios([make_planet(omega=60, omega_big=0)])
        self.assertEqual(column(results, "omega_big", 0), [0])

    def test_a_fixed_longitude_of_ascending_node_is_kept(self):
        self.assertEqual(column(scenarios([make_planet(omega=60, omega_big=15)]), "omega_big", 0), [15])

    def test_scenario_indexes_are_unique(self):
        planet = make_planet(omega=60, omega_bins=2, omega_low_err=10, omega_up_err=10,
                             omega_big=0, omega_big_bins=3, omega_big_low_err=10, omega_big_up_err=10)
        results = scenarios([planet])
        self.assertEqual(len(results), 6)
        self.assertEqual(len({r["index"] for r in results}), 6)

    def test_free_omega_gives_the_requested_number_of_values(self):
        values = distinct(scenarios([make_planet(omega_bins=7)], free_params=["omega"]), "omega", 0)
        self.assertEqual(len(values), 7)


class NewBehaviourTest(unittest.TestCase):
    def test_free_omega_big_is_separate_from_free_omega(self):
        planet = make_planet(omega=60, omega_big=15, omega_bins=4, omega_big_bins=3)
        results = scenarios([planet], free_params=["omega_big"])
        self.assertEqual(len(results), 3)
        self.assertEqual(distinct(results, "omega", 0), [60])
        self.assertEqual(sorted(distinct(results, "omega_big", 0)), [0.0, 120.0, 240.0])

    def test_free_omega_does_not_touch_omega_big(self):
        results = scenarios([make_planet(omega=60, omega_big=15, omega_bins=4)], free_params=["omega"])
        self.assertEqual(distinct(results, "omega_big", 0), [15])
        self.assertEqual(sorted(distinct(results, "omega", 0)), [0.0, 90.0, 180.0, 270.0])

    def test_free_eccentricity_uses_the_given_maximum(self):
        calculator = RecordingCalculator()
        inputs = calculator.build_simulation_inputs([0.82], [make_planet(ecc_bins=4)], ["eccentricity"], ecc_max=0.3)
        self.assertEqual([round(i.ecc_arr[0], 2) for i in inputs], [0.0, 0.1, 0.2, 0.3])

    def test_rand_with_many_bins_stays_a_single_random_entry(self):
        results = scenarios([make_planet(omega="rand", omega_low_err=20, omega_up_err=20, omega_bins=5)])
        self.assertEqual(column(results, "omega", 0), ["rand"])

    def test_every_scenario_has_its_own_seed_and_the_same_base_seed_gives_the_same_seeds(self):
        planets = [make_planet(eccentricity=0.1, ecc_low_err=0.1, ecc_up_err=0.2, ecc_bins=4)]
        first = RecordingCalculator.build_simulation_inputs([0.82], planets, seed=7)
        again = RecordingCalculator.build_simulation_inputs([0.82], planets, seed=7)
        other = RecordingCalculator.build_simulation_inputs([0.82], planets, seed=8)
        self.assertEqual([i.seed for i in first], [i.seed for i in again])
        self.assertEqual(len({i.seed for i in first}), 4)
        self.assertNotEqual([i.seed for i in first], [i.seed for i in other])

    def test_an_unspecified_mass_and_radius_is_reported(self):
        planet = make_planet(mass=None, mass_low_err=None, mass_up_err=None)
        planet.radius = None
        with self.assertRaises(ValueError):
            RecordingCalculator.prepare_planet_params([planet])


if __name__ == "__main__":
    unittest.main()
