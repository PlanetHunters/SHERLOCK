import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo

from sherlockpipe.search import target_info_report as tir
from sherlockpipe.search.sherlock import Sherlock
from sherlockpipe.search.sherlock_target import SherlockTarget
from sherlockpipe.search.target_info_report import (ReportConfig, TargetInfoReport, availability_rows,
                                                    config_from_sherlock_target, group_availability_rows,
                                                    measure_preprocessing,
                                                    parse_available_data, parse_report_log, read_autodetrend,
                                                    read_pulsations, read_used_products, sector_rows, split_blocks,
                                                    transit_mask, write_target_info_report)

LOG = """SHERLOCK (Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-base seeKers)
USER DEFINITIONS
Detrend method: Bi-Weight
Sigma threshold for upper outliers clipping: 3.0
Retrieving star catalog info
There are TPF data: TESS Sector 15, Year 2019, Author: TESS Sector 15, ExpTime: 120.0
There are TPF data: TESS Sector 15, Year 2019, Author: TESS Sector 15, ExpTime: 1800.0
There are TessCut data: TESS Sector 15, Year 2019, Author: TESScut, ExpTime: 1425.599379
There are LightCurve data: TESS Sector 15, Year 2019, Author: SPOC, ExpTime: 120.0
There are LightCurve data: TESS Sector 15, Year 2019, Author: QLP, ExpTime: 1800.0
There are LightCurve data: TESS Sector 21, Year 2020, Author: QLP, ExpTime: 1800.0
* Initial mask since day 10.00 to day 12.00. *
* Transit mask with P=2.00 d, T0=1.00 d, Dur=60.00 min *
wl/ks_min: 0.19
wl/ks_max: 2.45
"""


def _rows(sectors, signature=None):
    availability = {s: {"SPOC": {120}, "QLP": {1800}} if signature is None else signature(s) for s in sectors}
    return availability_rows(availability, {s: {("SPOC", 120)} for s in sectors})


class TestAvailability(unittest.TestCase):

    def test_parse_available_data(self):
        availability = parse_available_data(LOG)
        self.assertEqual({15, 21}, set(availability))
        self.assertEqual({120, 1800}, availability[15]["SPOC TPF"])
        self.assertEqual({1426}, availability[15]["TESScut (FFI)"])
        self.assertEqual({120}, availability[15]["SPOC"])
        self.assertEqual({1800}, availability[21]["QLP"])

    def test_parse_report_log(self):
        parsed = parse_report_log(LOG)
        self.assertEqual("Bi-Weight", parsed["user_definitions"]["Detrend method"])
        self.assertEqual([[10.0, 12.0]], parsed["initial_mask"])
        self.assertEqual([{"P": 2.0, "T0": 1.0, "D": 60.0}], parsed["transit_mask"])
        self.assertEqual((0.19, 2.45), parsed["window_range"])

    def test_parse_report_log_only_reads_the_last_execution(self):
        text = "SHERLOCK (Searching for old\nUSER DEFINITIONS\nDetrend method: gp\nRetrieving star catalog\n" + LOG
        self.assertEqual("Bi-Weight", parse_report_log(text)["user_definitions"]["Detrend method"])

    def test_used_product_is_bracketed_and_marked(self):
        rows = availability_rows(parse_available_data(LOG), {15: {("SPOC", 120)}})
        by_sector = {r["label"]: r for r in rows}
        self.assertEqual("[120]", by_sector["15"]["cells"]["SPOC"])
        self.assertEqual({"SPOC"}, by_sector["15"]["used_columns"])
        self.assertTrue(by_sector["15"]["in_use"])
        self.assertFalse(by_sector["21"]["in_use"])
        self.assertEqual("–", by_sector["21"]["cells"]["SPOC"])

    def test_short_lists_are_not_collapsed_nor_split(self):
        rows = _rows(range(1, 13))
        self.assertEqual(rows, group_availability_rows(rows))
        self.assertEqual(1, len(split_blocks(rows)))

    def test_medium_lists_are_split_in_blocks_without_losing_sectors(self):
        rows = group_availability_rows(_rows(range(1, 21), signature=lambda s: {"SPOC": {120 + s}}))
        blocks = split_blocks(rows)
        self.assertEqual(2, len(blocks))
        self.assertEqual(20, sum(len(b) for b in blocks))
        self.assertTrue(all(len(b) <= tir.SINGLE_TABLE_MAX_SECTORS for b in blocks))

    def test_long_lists_collapse_consecutive_identical_sectors(self):
        rows = _rows(list(range(1, 31)))
        grouped = group_availability_rows(rows)
        self.assertEqual(1, len(grouped))
        self.assertEqual("1–30", grouped[0]["label"])
        self.assertEqual(list(range(1, 31)), grouped[0]["sectors"])

    def test_long_lists_do_not_merge_across_a_gap_or_a_different_availability(self):
        sectors = list(range(1, 14)) + list(range(20, 33))
        rows = _rows(sectors, signature=lambda s: {"SPOC": {120}} if s < 25 else {"SPOC": {20}})
        grouped = group_availability_rows(rows)
        self.assertEqual(["1–13", "20–24", "25–32"], [r["label"] for r in grouped])

    def test_used_products_are_read_from_the_apertures_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "apertures.yaml"), "w") as f:
                f.write("sectors: {15: {SPOC: {120: [[1, 2]]}}, 41: {SPOC: {120: [[1, 2]], 20: [[1, 2]]}}}\n")
            self.assertEqual({15: {("SPOC", 120)}, 41: {("SPOC", 120), ("SPOC", 20)}}, read_used_products(directory))
            self.assertEqual({}, read_used_products(os.path.join(directory, "missing")))


def _synthetic_execution(directory, outliers=3, mask=(5.0, 6.0), with_sectors=True):
    """A small execution whose preprocessing is known: deterministic bounded noise, spikes and one initial mask."""
    time = np.linspace(1.0, 20.0, 1200)
    flux = 1 + 1e-4 * np.sin(time * 50)
    flux_err = np.full(len(time), 1e-4)
    spikes = np.linspace(100, 1000, outliers).astype(int)
    flux[spikes] += 0.05
    lc = pd.DataFrame({"time": time, "flux": flux, "flux_err": flux_err})
    lc["sector"] = np.where(time < 10, 1, 2) if with_sectors else np.nan
    lc.to_csv(os.path.join(directory, "lc.csv"), index=False)
    keep = np.ones(len(time), bool)
    keep[spikes] = False
    keep &= ~((time >= mask[0]) & (time <= mask[1]))
    pre = lc.loc[keep, ["time", "flux", "flux_err"]]
    pre.to_csv(os.path.join(directory, "lc_predetrends.csv"), index=False)
    pre.to_csv(os.path.join(directory, "lc_0.csv"), index=False)
    detrended = pre.copy()
    detrended["flux"] = detrended["flux"] / (1 + 1e-4 * np.cos(detrended["time"]))
    detrended.to_csv(os.path.join(directory, "lc_1.csv"), index=False)
    pd.DataFrame([{"obj_id": "TIC 1", "ra": 10.0, "dec": -5.0, "R_star": 0.5, "R_star_lerr": 0.01,
                   "R_star_uerr": 0.01, "M_star": 0.5, "M_star_lerr": 0.02, "M_star_uerr": 0.02, "Teff_star": 3500,
                   "Teff_star_lerr": 100, "ld_a": 0.2, "ld_b": 0.4, "logg": 4.7, "lum": 0.03, "v": 12.0, "j": 9.0,
                   "h": 8.5, "k": 8.4}]).to_csv(os.path.join(directory, "params_star.csv"), index=False)
    return lc, pre


class TestPreprocessingMeasurement(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_outliers_and_masks_are_counted_and_explain_the_final_curve(self):
        lc, pre = _synthetic_execution(self.directory)
        in_mask = int(((lc["time"] >= 5) & (lc["time"] <= 6)).sum())
        config = ReportConfig(outliers_sigma_upper=3.0, initial_mask=[[5.0, 6.0]])
        steps = measure_preprocessing(lc, pre, config)
        removed = {(name, setting): count for name, setting, count in steps[1:]}
        self.assertEqual(3, removed[("Outlier clipping", "upper 3 σ")])
        self.assertTrue(in_mask > 0)
        self.assertEqual(in_mask, [c for n, s, c in steps if n == "Initial mask"][0])
        self.assertNotIn("Other steps", [n for n, _, _ in steps])

    def test_what_the_steps_do_not_explain_is_reported(self):
        lc, pre = _synthetic_execution(self.directory)
        pre = pre.iloc[:-7]  # something else removed 7 more points
        steps = measure_preprocessing(lc, pre, ReportConfig(outliers_sigma_upper=3.0, initial_mask=[[5.0, 6.0]]))
        other = [c for n, _, c in steps if n == "Other steps"]
        self.assertEqual([7], other)

    def test_transit_mask_matches_the_tls_definition(self):
        time = np.arange(0, 10, 0.01)
        mask = transit_mask(time, 2.0, 0.1, 1.0)
        self.assertTrue(mask[np.argmin(np.abs(time - 3.0))])
        self.assertFalse(mask[np.argmin(np.abs(time - 3.5))])
        self.assertAlmostEqual(0.1, mask.sum() * 0.01 / 5, delta=0.01)

    def test_sector_rows_measure_the_used_points(self):
        lc, pre = _synthetic_execution(self.directory)
        rows = {r["sector"]: r for r in sector_rows(lc, pre, tess=False)}
        self.assertEqual({1, 2}, set(rows))
        self.assertEqual(len(lc[lc["sector"] == 1]), rows[1]["points"])
        self.assertEqual(len(pre[pre["time"] < 10]), rows[1]["used"])
        self.assertLess(rows[1]["kept"], 100)

    def test_sector_rows_without_sector_information_use_a_single_row(self):
        lc, pre = _synthetic_execution(self.directory, with_sectors=False)
        rows = sector_rows(lc, pre, tess=False)
        self.assertEqual([0], [r["sector"] for r in rows])
        self.assertEqual(len(pre), rows[0]["used"])


class TestConfiguration(unittest.TestCase):

    def test_configuration_is_taken_from_the_sherlock_target(self):
        object_info = MissionObjectInfo('all', "TIC 12345", initial_mask=[[1710, 1712]], outliers_sigma=3,
                                        initial_transit_mask=[{"P": 2.0, "T0": 1.0, "D": 60.0}],
                                        high_rms_enabled=True, reduce_simple_oscillations=True, initial_trim=50)
        target = SherlockTarget(object_info=object_info, detrends_number=6, period_min=1, period_max=100)
        config = config_from_sherlock_target(target, [0, 0.19, 0.56])
        self.assertEqual([[1710, 1712]], config.initial_mask)
        self.assertEqual([{"P": 2.0, "T0": 1.0, "D": 60.0}], config.transit_mask)
        self.assertEqual(3, config.outliers_sigma_upper)
        self.assertIsNone(config.outliers_sigma_lower)
        self.assertEqual(50, config.initial_trim)
        self.assertTrue(config.high_rms_enabled)
        self.assertTrue(config.oscillations_enabled)
        self.assertEqual([0.19, 0.56], config.window_lengths)  # the 0 of the non detrended curve is dropped
        self.assertIn(("Detrend models", "6"), config.setup_rows)

    def test_defaults_have_no_masks_and_no_trim(self):
        config = config_from_sherlock_target(SherlockTarget(object_info=MissionObjectInfo('all', "TIC 12345")))
        self.assertEqual([], config.initial_mask)
        self.assertEqual([], config.transit_mask)
        self.assertIsNone(config.initial_trim)
        self.assertIsNone(config.window_lengths)


class TestOptionalOutputs(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_pulsations_are_read_with_their_plots(self):
        os.makedirs(os.path.join(self.directory, "sa"))
        pd.DataFrame([{"period_s": 1000.0, "frequency_microHz": 1000.0, "amplitude": 1e-4, "phase": 0.5, "snr": 8.0,
                       "number": 0}]).to_csv(os.path.join(self.directory, "sa", "signals.csv"), index=False)
        signals = read_pulsations(self.directory)
        self.assertEqual(1, len(signals))
        self.assertTrue(signals.loc[0, "folded"].endswith(os.path.join("sa", "0", "folded_curve.png")))
        self.assertIsNone(read_pulsations(os.path.join(self.directory, "missing")))

    def test_autodetrend_period_is_read_from_the_file_name(self):
        open(os.path.join(self.directory, "Phase_detrend_period_TIC1_[5]_8.50_days.png"), "w").close()
        period, path = read_autodetrend(self.directory)
        self.assertEqual(8.5, period)
        self.assertTrue(path.endswith("_days.png"))
        self.assertIsNone(read_autodetrend(os.path.join(self.directory, "missing")))


class TestReportDocument(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_report_is_written_with_minimum_inputs(self):
        _synthetic_execution(self.directory)
        path = TargetInfoReport(self.directory, "TIC1_all").create()
        self.assertEqual(os.path.join(self.directory, "TIC1_all_info.pdf"), path)
        with open(path, "rb") as f:
            self.assertEqual(b"%PDF", f.read(4))

    def test_report_with_log_and_optional_pages(self):
        _synthetic_execution(self.directory)
        with open(os.path.join(self.directory, "TIC1_all_report.log"), "w") as f:
            f.write(LOG)
        with open(os.path.join(self.directory, "apertures.yaml"), "w") as f:
            f.write("sectors: {1: {SPOC: {120: [[1, 2]]}}}\n")
        open(os.path.join(self.directory, "Phase_detrend_period_TIC1_all_3.00_days.png"), "w").close()
        path = TargetInfoReport(self.directory, "TIC1_all").create()
        self.assertIsNotNone(path)
        self.assertGreater(os.path.getsize(path), 1000)

    def test_report_is_not_written_without_the_light_curves(self):
        self.assertIsNone(TargetInfoReport(self.directory, "TIC1_all").create())

    def test_writer_never_raises(self):
        self.assertIsNone(write_target_info_report(os.path.join(self.directory, "missing"), "TIC1_all"))
        with open(os.path.join(self.directory, "lc.csv"), "w") as f:
            f.write("not,a,light,curve\n")
        with open(os.path.join(self.directory, "lc_predetrends.csv"), "w") as f:
            f.write("nope\n")
        self.assertIsNone(write_target_info_report(self.directory, "TIC1_all"))


class TestFailuresNeverStopTheSearch(unittest.TestCase):
    """The report is a convenience: nothing that goes wrong with it may break a SHERLOCK execution."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        _synthetic_execution(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def _files(self):
        return sorted(f for f in os.listdir(self.directory) if "info" in f)

    def test_writer_swallows_errors_and_exits_raised_by_a_library(self):
        for error in (RuntimeError("boom"), SystemExit(1), KeyError("k"), ImportError("no module")):
            with mock.patch.object(TargetInfoReport, "create", side_effect=error):
                self.assertIsNone(write_target_info_report(self.directory, "TIC1_all"), error)

    def test_keyboard_interrupt_still_stops_the_run(self):
        with mock.patch.object(TargetInfoReport, "create", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                write_target_info_report(self.directory, "TIC1_all")

    def test_one_failing_section_does_not_remove_the_others(self):
        with mock.patch.object(TargetInfoReport, "_periodicity_page", side_effect=RuntimeError("boom")):
            path = TargetInfoReport(self.directory, "TIC1_all").create()
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_failure_while_building_leaves_no_partial_file(self):
        class BrokenDocument:
            def __init__(self, filename, **kwargs):
                self.filename = filename

            def build(self, story):
                with open(self.filename, "wb") as f:
                    f.write(b"%PDF-truncated")
                raise OSError("disk full")

        with mock.patch.object(tir, "BaseDocTemplate", BrokenDocument):
            self.assertIsNone(write_target_info_report(self.directory, "TIC1_all"))
        self.assertEqual([], self._files())

    def test_failed_regeneration_keeps_the_previous_good_report(self):
        good = write_target_info_report(self.directory, "TIC1_all")
        with open(good, "rb") as f:
            before = f.read()

        class BrokenDocument:
            def __init__(self, filename, **kwargs):
                self.filename = filename

            def build(self, story):
                with open(self.filename, "wb") as f:
                    f.write(b"half")
                raise OSError("disk full")

        with mock.patch.object(tir, "BaseDocTemplate", BrokenDocument):
            self.assertIsNone(write_target_info_report(self.directory, "TIC1_all"))
        with open(good, "rb") as f:
            self.assertEqual(before, f.read())
        self.assertEqual(["TIC1_all_info.pdf"], self._files())

    def test_missing_optional_dependency_only_leaves_out_the_field_of_view(self):
        os.makedirs(os.path.join(self.directory, "fov"))
        open(os.path.join(self.directory, "fov", "fov_sector_1.pdf"), "wb").close()
        with mock.patch.dict(sys.modules, {"pymupdf": None}):
            path = TargetInfoReport(self.directory, "TIC1_all").create()
        self.assertIsNotNone(path)

    def test_sherlock_hook_survives_an_error_an_exit_and_an_import_failure(self):
        sherlock = Sherlock(None)
        hook = sherlock._Sherlock__write_target_info_report
        target = "sherlockpipe.search.target_info_report.write_target_info_report"
        for error in (RuntimeError("boom"), SystemExit(1)):
            with mock.patch(target, side_effect=error):
                hook(self.directory, "TIC1_all", None, None)
        # the module cannot even be imported (e.g. a missing dependency): SHERLOCK must still run
        with mock.patch.dict(sys.modules, {"sherlockpipe.search.target_info_report": None}):
            hook(self.directory, "TIC1_all", None, None)


if __name__ == '__main__':
    unittest.main()
