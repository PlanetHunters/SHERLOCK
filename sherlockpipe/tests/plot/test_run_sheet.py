import os
import shutil
import sys
import tempfile
import time as timer
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sherlockpipe.plot import run_sheet as rs
from sherlockpipe.search.sherlock import Sherlock
from sherlockpipe.search.transitresult import TransitResult

DURATION = 0.05
DEPTH_PPT = 2.0


def synthetic_curve(period, days, seed=1, cadence=120, gaps=()):
    """A noisy light curve with a box transit of known depth and duration, and optional observing gaps."""
    rng = np.random.default_rng(seed)
    time = np.arange(1.0, 1.0 + days, cadence / 86400)
    for a, b in gaps:
        time = time[(time < a) | (time > b)]
    phase = ((time - 1.5 + period / 2) % period) - period / 2
    flux = 1 + rng.normal(0, 1e-3, len(time))
    flux[np.abs(phase) < DURATION / 2] -= DEPTH_PPT / 1e3
    return time, flux, np.full(len(time), 1e-3)


def transit_result(base_period, time, with_model=True, **overrides):
    """A real TransitResult around a minimal stand-in for the TLS results object."""
    periods = np.linspace(base_period * 0.5, base_period * 2, 400)
    power = np.random.default_rng(2).normal(0, 1, 400)
    power[np.argmin(np.abs(periods - base_period))] = 15
    phase = np.linspace(0, 1, 2001)
    model = np.ones_like(phase)
    model[np.abs((phase - 0.5) * base_period) < DURATION / 2] = 1 - DEPTH_PPT / 1e3
    results = SimpleNamespace(periods=periods, power=power, rp_rs=0.045)
    if with_model:
        results.model_folded_phase, results.model_folded_model = phase, model
    t0s = np.arange(1.5, time.max(), base_period)
    values = dict(period=base_period, per_err=1e-4, duration=DURATION, t0=1.5, depth=DEPTH_PPT, depth_err=0.05, count=len(t0s),
                  snr=12.0, sde=15.0, fap=1e-4, border_score=0.95, harmonic=None, harmonic_spectrum=None,
                  odd_even_mismatch=0.5)
    values.update(overrides)
    return TransitResult({}, results, values["period"], values["per_err"], values["duration"], values["t0"], t0s,
                         np.full(len(t0s), DEPTH_PPT), np.full(len(t0s), 0.5), values["depth"], values["depth_err"],
                         values["odd_even_mismatch"], DEPTH_PPT, DEPTH_PPT, values["count"], values["snr"],
                         values["sde"], values["fap"], values["border_score"], np.zeros(len(time), bool),
                         values["harmonic"], values["harmonic_spectrum"])


class TestEpochs(unittest.TestCase):

    def test_clean_curve_drops_invalid_points_and_sorts(self):
        time, flux, err = rs.clean_curve([3, 1, np.nan, 2, 4], [1, 1.1, 1, np.nan, 1.3], [0.1, 0.1, 0.1, 0.1, 0.1])
        self.assertEqual([1, 3, 4], list(time))
        self.assertEqual([1.1, 1, 1.3], list(flux))

    def test_clean_curve_replaces_missing_errors_by_the_scatter(self):
        time, flux, err = rs.clean_curve([1, 2, 3], [1.0, 1.1, 0.9], None)
        self.assertTrue(np.all(err == np.std([1.0, 1.1, 0.9])))

    def test_epochs_are_found_and_measured(self):
        time, flux, err = synthetic_curve(1.0, 20)
        epochs = rs.compute_epochs(time, flux, err, 1.0, 1.5, DURATION, DEPTH_PPT)
        self.assertGreaterEqual(len(epochs), 18)
        self.assertAlmostEqual(DEPTH_PPT, rs.mean_depth(epochs)[0], delta=0.4)
        self.assertTrue((epochs["err"] > 0).all())
        self.assertTrue(((epochs["hi"] - epochs["lo"]) > 0).all())

    def test_epochs_without_enough_data_are_left_out(self):
        time, flux, err = synthetic_curve(1.0, 20, gaps=[(5.0, 9.0)])
        epochs = rs.compute_epochs(time, flux, err, 1.0, 1.5, DURATION, DEPTH_PPT)
        centres = epochs["tc"].to_numpy()
        self.assertFalse(((centres > 5.2) & (centres < 8.8)).any())

    def test_a_long_gap_starts_a_new_segment(self):
        time, flux, err = synthetic_curve(1.0, 40, gaps=[(10.0, 25.0)])
        epochs = rs.compute_epochs(time, flux, err, 1.0, 1.5, DURATION, DEPTH_PPT)
        self.assertEqual(2, epochs["segment"].nunique())

    def test_invalid_parameters_give_no_epochs(self):
        time, flux, err = synthetic_curve(1.0, 10)
        for period, duration in ((np.nan, DURATION), (1.0, np.nan), (-1.0, DURATION), (1.0, 0.0)):
            self.assertEqual(0, len(rs.compute_epochs(time, flux, err, period, 1.5, duration, DEPTH_PPT)))

    def test_cost_does_not_grow_with_points_times_epochs(self):
        time = np.arange(1.0, 1500.0, 120 / 86400)  # ~1 million points, thousands of epochs
        flux = 1 + np.random.default_rng(3).normal(0, 1e-3, len(time))
        started = timer.time()
        epochs = rs.compute_epochs(time, flux, np.full(len(time), 1e-3), 0.3, 1.5, 0.03, 0.5)
        self.assertGreater(len(epochs), 4000)
        self.assertLess(timer.time() - started, 30)


class TestStackingAndTiers(unittest.TestCase):

    def _epochs(self, snr, segments):
        import pandas as pd
        return pd.DataFrame({"snr": snr, "segment": segments, "k": range(len(snr)), "tc": range(len(snr)),
                             "depth": snr, "err": 1.0, "z": 0.0})

    def test_layout_tiers_follow_the_number_of_epochs_with_data(self):
        self.assertEqual([1, 1, 2, 2, 3], [rs.layout_tier(n) for n in (0, 24, 25, 150, 151)])

    def test_every_stack_reaches_the_target_and_segments_are_never_mixed(self):
        epochs = self._epochs([1.0] * 40 + [1.0] * 30, [0] * 40 + [1] * 30)
        groups, target = rs.stack_epochs(epochs, snr_target=4.0, max_rows=100)
        self.assertEqual(4.0, target)
        for g in groups:
            self.assertEqual(1, epochs["segment"].iloc[g].nunique())
        self.assertEqual(70, sum(len(g) for g in groups))
        for g in groups[:-1]:
            self.assertGreaterEqual(np.sqrt(len(g)), 4.0 - 1e-9)

    def test_a_leftover_stack_joins_the_previous_one_of_its_segment(self):
        epochs = self._epochs([1.0] * 18 + [1.0] * 2, [0] * 18 + [1] * 2)
        groups, _ = rs.stack_epochs(epochs, snr_target=4.0, max_rows=100)
        sizes = [len(g) for g in groups]
        self.assertEqual(20, sum(sizes))
        self.assertTrue(all(s >= 2 for s in sizes))  # nothing is left as a single epoch of its own
        self.assertEqual([0, 1], sorted(set(epochs["segment"].iloc[g[0]] for g in groups)))

    def test_the_target_grows_until_there_are_few_enough_rows(self):
        epochs = self._epochs([1.0] * 400, [0] * 400)
        groups, target = rs.stack_epochs(epochs, snr_target=2.0, max_rows=10)
        self.assertLessEqual(len(groups), 10)
        self.assertGreater(target, 2.0)

    def test_flags(self):
        self.assertEqual(["ok", "warn", "bad"], [rs.flag(v, "odd_even_sigma") for v in (1.0, 2.5, 3.5)])
        self.assertEqual(["ok", "warn", "bad"], [rs.flag(v, "border_score") for v in (0.9, 0.6, 0.2)])
        self.assertIsNone(rs.flag(np.nan, "fap"))
        self.assertIsNone(rs.flag(None, "chi2"))


class TestModelAndParameters(unittest.TestCase):

    def test_model_comes_from_the_folded_model_of_the_search(self):
        time, flux, err = synthetic_curve(1.0, 10)
        model = rs.transit_model(transit_result(1.0, time), 1.0, DURATION, DEPTH_PPT / 1e3)
        self.assertAlmostEqual(1 - DEPTH_PPT / 1e3, float(model(0.0)), places=6)
        self.assertEqual(1.0, float(model(5.0)))
        self.assertEqual(1.0, float(model(-5.0)))

    def test_model_falls_back_to_a_box(self):
        time, flux, err = synthetic_curve(1.0, 10)
        model = rs.transit_model(transit_result(1.0, time, with_model=False), 1.0, DURATION, 0.002)
        self.assertAlmostEqual(0.998, float(model(0.0)))
        self.assertEqual(1.0, float(model(DURATION * 24)))

    def test_parameters_flag_what_is_below_the_search_thresholds(self):
        time, flux, err = synthetic_curve(1.0, 20)
        result = transit_result(1.0, time, sde=4.0, snr=3.0, fap=np.nan, harmonic="P/2")
        epochs = rs.compute_epochs(time, flux, err, 1.0, 1.5, DURATION, DEPTH_PPT)
        groups = dict(rs.parameter_groups(result, epochs, 20, snr_min=5, sde_min=5))
        detection = {name: (value, state) for name, value, state in groups["Detection"]}
        self.assertEqual("bad", detection["SDE"][1])
        self.assertEqual("bad", detection["SNR"][1])
        self.assertEqual("–", detection["FAP"][0])  # a missing FAP is not printed as nan
        self.assertEqual("warn", detection["Harmonic of another signal"][1])

    def test_parameters_have_no_flag_without_thresholds(self):
        time, flux, err = synthetic_curve(1.0, 20)
        epochs = rs.compute_epochs(time, flux, err, 1.0, 1.5, DURATION, DEPTH_PPT)
        groups = dict(rs.parameter_groups(transit_result(1.0, time), epochs, 20))
        detection = {name: state for name, value, state in groups["Detection"]}
        self.assertIsNone(detection["SDE"])
        self.assertIsNone(detection["SNR"])

    def test_the_stacked_map_description_is_filled_with_the_real_target(self):
        text = rs.CAPTIONS[3].format(target=4)
        self.assertIn("S/N ≥ 4", text)
        self.assertIn("stacked epoch map", text)
        self.assertIn("red vertical band", text)


class TestSheetFiles(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp() + "/"

    def tearDown(self):
        shutil.rmtree(self.directory)

    def _write(self, period, days, name, **kwargs):
        time, flux, err = synthetic_curve(period, days)
        result = transit_result(period, time, **kwargs.pop("result", {}))
        path = rs.save_run_sheet("TIC1_all", self.directory, name, time, flux, err, result, 120, 1, "SELECTED (1)",
                                 **kwargs)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 20000)
        return path

    def test_one_panel_per_epoch_up_to_24(self):
        self._write(1.0, 20, "few.png")

    def test_thumbnails_up_to_150(self):
        self._write(0.5, 40, "dozens.png")

    def test_stacked_map_above_150(self):
        self._write(0.2, 60, "hundreds.png")

    def test_harmonics_spectrum_is_added_on_request(self):
        spectrum = np.random.default_rng(5).normal(0, 1, 400)
        self._write(1.0, 20, "harmonics.png", plot_harmonics=True, result={"harmonic_spectrum": spectrum})

    def test_sheet_is_written_when_there_are_no_valid_transit_parameters(self):
        time, flux, err = synthetic_curve(1.0, 10)
        result = transit_result(1.0, time, period=np.nan, duration=np.nan, t0=np.nan, depth=np.nan)
        path = rs.save_run_sheet("TIC1_all", self.directory, "invalid.png", time, flux, err, result, 120, 1, "SELECTED")
        self.assertTrue(os.path.exists(path))

    def test_sheet_is_written_when_no_epoch_has_enough_data(self):
        time, flux, err = synthetic_curve(1.0, 3)
        result = transit_result(1.0, time, t0=1000.0)
        path = rs.save_run_sheet("TIC1_all", self.directory, "empty.png", time, flux, err, result, 120, 1, "SELECTED")
        self.assertTrue(os.path.exists(path))

    def test_curves_with_nan_points_are_accepted(self):
        time, flux, err = synthetic_curve(1.0, 20)
        flux = flux.copy()
        flux[::50] = np.nan
        path = rs.save_run_sheet("TIC1_all", self.directory, "nan.png", time, flux, err, transit_result(1.0, time), 120,
                                 1, "SELECTED")
        self.assertTrue(os.path.exists(path))


class TestFailuresNeverStopTheSearch(unittest.TestCase):
    """The sheet is a convenience: nothing that goes wrong with it may break a SHERLOCK execution."""

    def setUp(self):
        self.directory = tempfile.mkdtemp() + "/"
        self.time, self.flux, self.err = synthetic_curve(1.0, 20)
        self.result = transit_result(1.0, self.time)
        self.target = SimpleNamespace(use_harmonics_spectra=False, snr_min=5, sde_min=5)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def _hook(self, file="Run_1_ws=0.5000_TIC1_all.png", label="win_size 0.5000"):
        return Sherlock(None)._Sherlock__save_run_sheet("TIC1_all", self.directory, file, 1, self.time, self.flux,
                                                      self.err, self.result, 120, label, self.target)

    def test_the_hook_writes_the_sheet_next_to_the_other_run_files(self):
        self._hook()
        self.assertEqual(["Run_1_ws=0.5000_TIC1_all.png"], os.listdir(self.directory))

    def test_sheet_file_names_and_labels_follow_the_old_run_plots(self):
        name = Sherlock._Sherlock__run_sheet_name
        wl = [0, 0.1858, 0.5626]
        self.assertEqual(("Run_1_PDCSAP-FLUX_TIC1_all.png", "PDCSAP flux"), name(1, 0, wl, "biweight", "TIC1_all"))
        self.assertEqual(("Run_2_ws=0.5626_TIC1_all.png", "win_size 0.5626"), name(2, 2, wl, "biweight", "TIC1_all"))
        self.assertEqual(("Run_1_ks=0.1858_TIC1_all.png", "ker_size 0.1858"), name(1, 1, wl, "gp", "TIC1_all"))

    def test_the_names_can_be_read_back_by_the_target_info_report(self):
        from sherlockpipe.search.target_info_report import window_lengths_from_runs
        os.makedirs(self.directory + "1")
        name = Sherlock._Sherlock__run_sheet_name
        for i, w in enumerate([0, 0.2, 0.7]):
            open(self.directory + "1/" + name(1, i, [0, 0.2, 0.7], "biweight", "TIC1_all")[0], "w").close()
        self.assertEqual([0.2, 0.7], window_lengths_from_runs(self.directory[:-1], "TIC1_all"))

    def test_the_hook_survives_an_error_an_exit_and_an_import_failure(self):
        target = "sherlockpipe.plot.run_sheet.save_run_sheet"
        for error in (RuntimeError("boom"), SystemExit(1), KeyError("k"), ValueError("v")):
            with mock.patch(target, side_effect=error):
                self._hook()
        with mock.patch.dict(sys.modules, {"sherlockpipe.plot.run_sheet": None}):
            self._hook()
        self.assertEqual([], os.listdir(self.directory))

    def test_the_hook_survives_a_broken_result(self):
        self.result = SimpleNamespace()  # not even a TransitResult
        self._hook()
        self.assertEqual([], os.listdir(self.directory))

    def test_keyboard_interrupt_still_stops_the_run(self):
        with mock.patch("sherlockpipe.plot.run_sheet.save_run_sheet", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self._hook()


if __name__ == '__main__':
    unittest.main()
