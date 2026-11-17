import inspect
import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from astropy.time import Time

from sherlockpipe.bayesian_fit.allesfitter_data_extractor import AllesfitterDataExtractor
from sherlockpipe.tests.plan.fixtures import EVENT_COLUMNS, offline_astropy, write_observatories

try:
    from sherlockpipe.observation_plan import run as plan_run
    from sherlockpipe.observation_plan.run import run_plan
except ImportError as error:  # alexfitter, which the run imports, needs the compiled ellc
    raise unittest.SkipTest(f"the plan run needs alexfitter: {error}")

try:
    import pypdfium2
except ImportError:  # pragma: no cover
    pypdfium2 = None


def write_fit(object_dir, candidates=("SOI_1",), object_id="TIC 224298134", epoch=1713.45, depth_ppt=0.6486):
    """The files of an allesfitter fit that the plan reads: fit_[1]/main/{params_star.csv, results/ns_*.csv}."""
    main = f"{object_dir}/fit_[1]/main"
    os.makedirs(main + "/results", exist_ok=True)
    rows = ["#name,median,lower_error,upper_error,label,unit", "#Fitted parameters,,,,,"]
    derived = ["#property,value,lower_error,upper_error,source"]
    for c in candidates:
        rows += [f"{c}_rr,0.0231,0.0008,0.0006,$R_b / R_\\star$,", f"{c}_epoch,{epoch},0.002,0.002,$T_0$,BJD",
                 f"{c}_period,1.49233,2e-05,2e-05,$P_b$,d"]
        derived += [f"Total transit duration {c}; $T_\\mathrm{{tot;{c}}}$ (h),0.69,0.02,0.03,derived",
                    f"Transit depth (dil.) {c}; $\\delta_\\mathrm{{tr; dil; {c}; lc}}$ (ppt),{depth_ppt},0.04,0.05,derived",
                    f"Occultation depth (dil.) {c}; $\\delta_\\mathrm{{occ}}$ (ppt),0.03,0.02,0.03,derived"]
    with open(main + "/results/ns_table.csv", "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(main + "/results/ns_derived_table.csv", "w") as f:
        f.write("\n".join(derived) + "\n")
    pd.DataFrame([{"obj_id": object_id, "ra": 185.3394, "dec": 57.7525, "v": 12.8, "j": 9.5, "h": 8.8, "k": 8.6}]) \
        .to_csv(main + "/params_star.csv", index=False)
    return main


def called_with(mock_call):
    """The arguments of a recorded Planner.create_observation_observables call, by parameter name."""
    signature = inspect.signature(plan_run.Planner.create_observation_observables)
    bound = signature.bind(*mock_call.args, **mock_call.kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def alles_for(candidates=("SOI_1",), epoch=1713.45):
    rng = np.random.default_rng(1)
    posteriors = {}
    for c in candidates:
        posteriors[f"{c}_period"] = rng.normal(1.49233, 2e-5, 4000)
        posteriors[f"{c}_epoch"] = rng.normal(epoch, 0.002, 4000)
    return SimpleNamespace(posterior_params=posteriors)


def arguments(directory, observatories, **overrides):
    values = dict(object_dir=directory, candidate=1, observatories=observatories, lat=None, lon=None, alt=None, tz=None,
                  min_altitude=30, moon_min_dist=30, moon_max_dist=70, max_days=12, transit_fraction=0.5, baseline=0.5,
                  since="2024-03-01", error_sigma=2, no_error_alert=False, time_unit=None)
    values.update(overrides)
    return Namespace(**values)


class TestArguments(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_observatories_or_coordinates_are_required(self):
        with self.assertRaisesRegex(ValueError, "observatories.*lat, lon and alt"):
            run_plan(arguments(self.directory, None))
        with self.assertRaisesRegex(ValueError, "observatories.*lat, lon and alt"):
            run_plan(arguments(self.directory, None, lat=28.3, lon=-16.5))

    def test_the_fit_posteriors_are_required(self):
        with self.assertRaisesRegex(ValueError, "posteriors files .*ns_table.csv.*not found"):
            run_plan(arguments(self.directory, write_observatories(self.directory)))

    def test_the_object_directory_defaults_to_the_current_one(self):
        write_fit(self.directory)
        path = write_observatories(self.directory)
        with mock.patch("os.getcwd", return_value=self.directory), \
                mock.patch.object(plan_run, "load_alexfitter_results", return_value=alles_for()), \
                mock.patch.object(plan_run.Planner, "create_observation_observables") as planner, \
                mock.patch.object(plan_run, "ObservationReport"):
            plan = self.directory + "/fit_[1]/main/plan"
            planner.return_value = (pd.DataFrame(), pd.DataFrame(columns=EVENT_COLUMNS), None, plan, plan + "/images")
            run_plan(arguments(None, path))
        self.assertEqual(self.directory + "/fit_[1]/main", called_with(planner.call_args)["object_dir"])


class TestOrchestration(unittest.TestCase):
    """What run_plan hands to the planner and the report, with both replaced by recorders."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.observatories = write_observatories(self.directory)
        self.fit = write_fit(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def _run(self, candidates=("SOI_1",), object_id="TIC 224298134", epoch=1713.45, **overrides):
        write_fit(self.directory, candidates, object_id, epoch)
        planner_result = (pd.DataFrame({"name": ["Calar-Alto"], "tz": [1], "lat": [37.2], "lon": [-2.5], "alt": [2168]}),
                          pd.DataFrame(columns=EVENT_COLUMNS), None, self.fit + "/plan", self.fit + "/plan/images")
        with mock.patch.object(plan_run, "load_alexfitter_results", return_value=alles_for(candidates, epoch)), \
                mock.patch.object(plan_run.Planner, "create_observation_observables", return_value=planner_result) as planner, \
                mock.patch.object(plan_run, "ObservationReport") as report:
            run_plan(arguments(self.directory, self.observatories, **overrides))
        return planner, report

    def test_a_target_from_tess_uses_btjd_unless_the_user_says_otherwise(self):
        planner, _ = self._run()
        self.assertEqual("btjd", called_with(planner.call_args)["time_unit"])
        planner, _ = self._run(time_unit="jd")
        self.assertEqual("jd", called_with(planner.call_args)["time_unit"])

    def test_a_target_from_kepler_uses_bkjd(self):
        planner, _ = self._run(object_id="KIC 8462852")
        self.assertEqual("bkjd", called_with(planner.call_args)["time_unit"])

    @unittest.expectedFailure
    def test_the_time_unit_given_by_the_user_wins_for_k2_too(self):
        """Known bug: `args.time_unit is None and mission == "Kepler" or mission == "K2"` lets K2 override the user."""
        planner, _ = self._run(object_id="EPIC 211945201", time_unit="jd")
        self.assertEqual("jd", called_with(planner.call_args)["time_unit"])

    def test_the_error_sigma_selects_the_percentile_of_the_posteriors(self):
        for sigma, percentile in ((3, 99.7), (2, 95), (1, 68)):
            with mock.patch.object(plan_run.AllesfitterDataExtractor, "extract_period",
                                   wraps=AllesfitterDataExtractor.extract_period) as extract:
                self._run(error_sigma=sigma)
            self.assertEqual(percentile, extract.call_args.args[3])

    def test_the_candidate_numbers_and_the_name_come_from_the_fit(self):
        planner, report = self._run()
        args = called_with(planner.call_args)
        self.assertEqual("TIC 224298134", args["object_id"])
        self.assertEqual(self.fit, args["object_dir"])
        self.assertEqual("TIC 224298134_SOI_1", args["name"])
        self.assertAlmostEqual(1713.45, args["epoch"], places=3)
        self.assertAlmostEqual(1.49233, args["period"], places=4)
        self.assertAlmostEqual(0.69, args["duration"], places=3)
        self.assertAlmostEqual(185.3394, args["ra"], places=3)
        self.assertGreater(args["period_up_err"], 0)
        self.assertGreater(args["epoch_low_err"], 0)

    def test_the_user_constraints_are_passed_to_the_planner(self):
        planner, _ = self._run(max_days=7, min_altitude=40, moon_min_dist=20, moon_max_dist=50, transit_fraction=0.25,
                               baseline=1.0)
        args = called_with(planner.call_args)
        self.assertEqual((7, 40, 20, 50, 0.25, 1.0), (args["max_days"], args["min_altitude"], args["moon_min_dist"],
                                                     args["moon_max_dist"], args["transit_fraction"], args["baseline"]))
        self.assertTrue(args["error_alert"])  # on by default
        planner, _ = self._run(no_error_alert=True)
        self.assertFalse(called_with(planner.call_args)["error_alert"])

    def test_every_candidate_of_the_fit_gets_its_own_plan_and_report(self):
        planner, report = self._run(candidates=("SOI_1", "SOI_2"))
        self.assertEqual(2, planner.call_count)
        self.assertEqual(2, report.call_count)
        self.assertEqual(["TIC 224298134_SOI_1", "TIC 224298134_SOI_2"], [called_with(c)["name"] for c in planner.call_args_list])
        self.assertEqual(2, report.return_value.create_report.call_count)

    def test_the_plan_starts_today_unless_a_date_is_given(self):
        planner, _ = self._run(since=None)
        self.assertLess(abs((called_with(planner.call_args)["since"] - Time.now()).to_value("s")), 120)
        planner, _ = self._run(since="2024-03-01")
        self.assertEqual("2024-03-01 00:00:00.000", called_with(planner.call_args)["since"].iso)

    def test_the_previous_plan_directory_is_replaced(self):
        os.makedirs(self.fit + "/plan")
        open(self.fit + "/plan/old.txt", "w").close()
        self._run()
        self.assertFalse(os.path.exists(self.fit + "/plan/old.txt"))

    def test_the_images_are_removed_after_the_report_is_made(self):
        os.makedirs(self.fit + "/plan/images")
        self._run()
        self.assertFalse(os.path.exists(self.fit + "/plan/images"))

    @unittest.expectedFailure
    def test_a_single_observatory_can_be_given_by_its_coordinates(self):
        """Known bug: the documented --lat --lon --alt path builds the observatory with DataFrame.append("Obs-1", ...),
        which no longer exists in pandas 2 and has the wrong signature, and uses a column named "long" that the rest
        of the code reads as "lon"."""
        write_fit(self.directory)
        with offline_astropy(), mock.patch.object(plan_run, "load_alexfitter_results", return_value=alles_for()):
            run_plan(arguments(self.directory, None, lat="37.2236", lon="-2.5463", alt="2168", tz="1"))
        self.assertTrue(os.path.exists(self.fit + "/plan/TIC 224298134_SOI_1_observation_plan.pdf"))


@unittest.skipIf(pypdfium2 is None, "pypdfium2 is needed to read the PDF")
class TestEndToEnd(unittest.TestCase):
    """The real planner and the real report on a synthetic allesfitter fit, offline."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.observatories = write_observatories(self.directory)
        self.fit = write_fit(self.directory)
        self._offline = offline_astropy()
        self._offline.__enter__()

    def tearDown(self):
        self._offline.__exit__(None, None, None)
        shutil.rmtree(self.directory)

    def _run(self):
        with mock.patch.object(plan_run, "load_alexfitter_results", return_value=alles_for()):
            run_plan(arguments(self.directory, self.observatories))
        return self.fit + "/plan"

    def test_a_pdf_and_a_csv_are_written_and_the_images_are_cleaned(self):
        plan_dir = self._run()
        self.assertTrue(os.path.exists(plan_dir + "/TIC 224298134_SOI_1_observation_plan.pdf"))
        self.assertTrue(os.path.exists(plan_dir + "/observation_plan.csv"))
        self.assertFalse(os.path.exists(plan_dir + "/images"))

    def test_the_pdf_describes_the_target_and_lists_the_observatories_with_events(self):
        plan_dir = self._run()
        document = pypdfium2.PdfDocument(plan_dir + "/TIC 224298134_SOI_1_observation_plan.pdf")
        text = "\n".join(document[i].get_textpage().get_text_range() for i in range(len(document)))
        self.assertIn("TIC 224298134 OBSERVATION PLAN", text)
        self.assertIn("Calar Alto", text)

    @unittest.expectedFailure
    def test_the_depth_of_the_fit_is_written_in_ppt(self):
        """Known bug: the fit tables already give the depth in ppt (0.6486) and AllesfitterDataExtractor.extract_depth
        multiplies it by 1000 again. The extractor is shared with the vetting and single transits runs."""
        depth, _, _ = AllesfitterDataExtractor.extract_depth(0, pd.read_csv(self.fit + "/results/ns_derived_table.csv"))
        self.assertAlmostEqual(0.6486, float(depth), places=3)


if __name__ == '__main__':
    unittest.main()
