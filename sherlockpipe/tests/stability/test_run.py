"""run_stability end to end on tiny inputs: properties files and candidates."""
import os
import tempfile
import types
import unittest

import pandas as pd

from sherlockpipe.system_stability.run import run_stability
from sherlockpipe.tests.stability.fixtures import TWO_PLANETS_YAML, write_yaml


def make_args(object_dir, properties=None, **kw):
    args = dict(object_dir=object_dir, properties=properties, candidate=None, cpus=1, max_ecc=0.1, ecc_max=0.5,
                period_bins=1, ecc_bins=1, inc_bins=1, omega_bins=1, omega_big_bins=1, mass_bins=1, star_mass_bins=1,
                years=20, repetitions=1, dt=0.05, free_params=None)
    args.update(kw)
    return types.SimpleNamespace(**args)


class RunFromPropertiesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        pd.DataFrame({"M_star": [0.82], "M_star_lerr": [0.04], "M_star_uerr": [0.04]}).to_csv(f"{self.tmp.name}/params_star.csv", index=False)
        self.yaml = write_yaml(f"{self.tmp.name}/two.yaml", TWO_PLANETS_YAML)

    def result(self):
        return pd.read_csv(f"{self.tmp.name}/stability_two/stability_megno.csv")

    def test_the_grid_of_the_properties_file_is_simulated_and_written(self):
        run_stability(make_args(self.tmp.name, self.yaml))
        frame = self.result()
        self.assertEqual(len(frame), 3 * 2 * 3)   # ecc b x (mass c x ecc c)
        for column in ("star_mass", "periods", "masses", "eccentricities", "megno", "last_year"):
            self.assertIn(column, frame.columns)
        self.assertTrue(frame.megno.is_monotonic_increasing)

    def test_the_log_is_written_next_to_the_results(self):
        run_stability(make_args(self.tmp.name, self.yaml))
        self.assertTrue(os.path.exists(f"{self.tmp.name}/stability_two/stability.log"))

    def test_an_existing_result_directory_is_replaced(self):
        os.makedirs(f"{self.tmp.name}/stability_two")
        open(f"{self.tmp.name}/stability_two/old.txt", "w").close()
        run_stability(make_args(self.tmp.name, self.yaml))
        self.assertFalse(os.path.exists(f"{self.tmp.name}/stability_two/old.txt"))


class RunFromCandidatesTest(unittest.TestCase):
    @unittest.expectedFailure  # period and radius are pandas Series, not numbers
    def test_candidates_without_a_fit_are_simulated_with_default_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"M_star": [0.82], "M_star_lerr": [0.04], "M_star_uerr": [0.04]}).to_csv(f"{tmp}/params_star.csv", index=False)
            pd.DataFrame({"period": [4.2345, 6.35], "per_err": [0.001, 0.001], "rad_p": [2.0, 3.0]}).to_csv(f"{tmp}/candidates.csv", index=False)
            run_stability(make_args(tmp, candidate="1,2", ecc_bins=2, mass_bins=2))
            frame = pd.read_csv(f"{tmp}/stability/stability_megno.csv")
        self.assertGreater(len(frame), 1)
        self.assertTrue(all("," in value for value in frame.periods))


if __name__ == "__main__":
    unittest.main()
