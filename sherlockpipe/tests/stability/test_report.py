"""The stability PDF report, on synthetic result tables (no simulation)."""
import logging
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from sherlockpipe.system_stability import stability_report as sr
from sherlockpipe.system_stability.run import run_stability
from sherlockpipe.tests.stability.fixtures import TWO_PLANETS_YAML, write_yaml
from sherlockpipe.tests.stability.test_run import make_args


def write_results(path, rows, planets=2, extra=None):
    """A stability_megno.csv with comma separated parameters of every body, like MegnoStabilityCalculator writes."""
    records = []
    for row in rows:
        records.append({"star_mass": row.get("star_mass", 0.82),
                        "periods": ",".join(str(row.get(f"P{i}", 4.0 * 1.5 ** (i - 1))) for i in range(1, planets + 1)),
                        "masses": ",".join(str(row.get(f"m{i}", 10.0)) for i in range(1, planets + 1)),
                        "inclinations": ",".join("0" for _ in range(planets)),
                        "eccentricities": ",".join(str(row.get(f"e{i}", 0.0)) for i in range(1, planets + 1)),
                        "arg_periastron": ",".join(str(row.get("omega", 60)) for _ in range(planets)),
                        "long_asc_node": ",".join("0" for _ in range(planets)),
                        "last_year": row.get("last_year", 300.0), "megno": row["megno"],
                        "megno_min": row["megno"], "megno_max": row["megno"], "megno_std": 0.0, "seed": len(records)})
    frame = pd.DataFrame(records)
    if extra is not None:
        frame = extra(frame)
    frame.to_csv(path, index=False)
    return path


def ecc_grid_rows(n=5, planets=2, megno=lambda e1, e2: 2.0 if e1 + e2 < 0.4 else 5.0):
    rows = []
    for e1 in np.linspace(0, 0.4, n):
        for e2 in np.linspace(0, 0.4, n):
            m = megno(e1, e2)
            rows.append({"e1": round(e1, 3), "e2": round(e2, 3), "megno": m, "last_year": 300.0 if m < 5 else 2.0})
    return rows


class TableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv = f"{self.tmp.name}/stability_megno.csv"

    def load(self, rows, **kw):
        write_results(self.csv, rows, **kw)
        return sr.load_scenarios(self.csv)

    def test_one_column_per_parameter_of_every_body(self):
        table, bodies = self.load(ecc_grid_rows(3))
        self.assertEqual(bodies, 2)
        for column in ("p1_period", "p2_period", "p1_mass", "p2_eccentricity", "p1_omega", "p2_omega_big", "star_mass", "megno"):
            self.assertIn(column, table.columns)
        self.assertEqual(len(table), 9)

    def test_three_bodies(self):
        table, bodies = self.load([{"megno": 2.0}, {"megno": 5.0}], planets=3)
        self.assertEqual(bodies, 3)
        self.assertIn("p3_period", table.columns)

    def test_states_follow_the_thresholds(self):
        table, _ = self.load([{"megno": m} for m in (1.9, 2.49, 2.5, 4.99, 5.0)])
        self.assertEqual(table.state.tolist(), ["regular", "regular", "chaotic", "chaotic", "unstable"])
        self.assertEqual(table.lost.tolist(), [False, False, False, False, True])

    def test_crossing_orbits_are_flagged_from_the_periods_and_eccentricities(self):
        table, _ = self.load([{"P1": 4.2, "P2": 6.4, "e1": 0.4, "e2": 0.4, "megno": 5.0},
                              {"P1": 4.2, "P2": 6.4, "e1": 0.0, "e2": 0.0, "megno": 2.0}])
        self.assertEqual(table.crossing.tolist(), [True, False])

    def test_random_angles_stay_as_text_and_are_not_swept(self):
        table, bodies = self.load([{"megno": 2.0, "omega": "rand"}, {"megno": 5.0, "omega": "rand"}])
        dims = {d.key: d for d in sr.dimensions(table, bodies)}
        self.assertFalse(dims["p1_omega"].swept)
        self.assertEqual(dims["p1_omega"].values, ["rand"])

    def test_swept_dimensions_are_the_ones_with_more_than_one_value(self):
        table, bodies = self.load(ecc_grid_rows(3))
        swept = {d.key for d in sr.dimensions(table, bodies) if d.swept}
        self.assertEqual(swept, {"p1_eccentricity", "p2_eccentricity"})


class AxisAndSensitivityTest(unittest.TestCase):
    def test_few_values_get_a_cell_each(self):
        index, centres, labels = sr.axis_bins(pd.Series([0.0, 0.1, 0.1, 0.2]))
        self.assertEqual(index.tolist(), [0, 1, 1, 2])
        self.assertEqual(labels, ["0", "0.1", "0.2"])

    def test_many_values_are_cut_in_cells_of_equal_count(self):
        series = pd.Series(np.linspace(0, 1, 1000))
        index, centres, _ = sr.axis_bins(series)
        self.assertLessEqual(len(centres), sr.MAX_CELLS)
        counts = np.bincount(index)
        self.assertLess(counts.max() - counts.min(), 5)

    def test_the_decisive_dimension_has_the_highest_sensitivity(self):
        rows = [{"e1": e1, "e2": e2, "P2": p2, "megno": 5.0 if p2 > 6.5 else 2.0} for e1 in (0.0, 0.1) for e2 in (0.0, 0.1) for p2 in (6.0, 7.0)]
        with tempfile.TemporaryDirectory() as tmp:
            table, bodies = sr.load_scenarios(write_results(f"{tmp}/r.csv", rows))
        dims = {d.key: d for d in sr.dimensions(table, bodies)}
        self.assertEqual(sr.sensitivity(table, dims["p2_period"]), 1.0)
        self.assertEqual(sr.sensitivity(table, dims["p1_eccentricity"]), 0.0)
        self.assertEqual(sr.sensitivity(table, dims["p2_mass"]), 0.0)


class HeroPairTest(unittest.TestCase):
    def table(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        table, bodies = sr.load_scenarios(write_results(f"{tmp.name}/r.csv", rows))
        return table, [d for d in sr.dimensions(table, bodies) if d.swept]

    def test_the_eccentricities_of_two_bodies_are_preferred_even_if_another_dimension_is_more_decisive(self):
        rows = [{"e1": e1, "e2": e2, "P2": p2, "megno": 5.0 if p2 > 6.5 else 2.0} for e1 in (0.0, 0.1) for e2 in (0.0, 0.1) for p2 in (6.0, 7.0)]
        table, swept = self.table(rows)
        pair = sr.hero_pair(table, swept)
        self.assertEqual((pair[0].key, pair[1].key), ("p1_eccentricity", "p2_eccentricity"))

    def test_without_two_eccentricities_the_two_most_decisive_dimensions_are_used(self):
        rows = [{"P2": p2, "m2": m2, "e1": e1, "megno": 5.0 if p2 > 6.5 else (3.0 if m2 > 15 else 2.0)}
                for p2 in (6.0, 7.0) for m2 in (10.0, 20.0) for e1 in (0.0, 0.1)]
        table, swept = self.table(rows)
        pair = sr.hero_pair(table, swept)
        self.assertEqual({pair[0].key, pair[1].key}, {"p2_period", "p2_mass"})

    def test_with_a_single_swept_dimension_there_is_no_pair(self):
        table, swept = self.table([{"e1": 0.0, "megno": 2.0}, {"e1": 0.1, "megno": 5.0}])
        self.assertIsNone(sr.hero_pair(table, swept))


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv = f"{self.tmp.name}/stability_megno.csv"
        self.pdf = f"{self.tmp.name}/stability_report.pdf"
        self.info = sr.RunInfo(object_id="TIC 1", years=300, dt=0.05, repetitions=3, cpus=4, seed=7, properties_file="x.yaml")

    def pages(self, rows, **kw):
        write_results(self.csv, rows, **kw)
        table, bodies = sr.load_scenarios(self.csv)
        return sr.build_pages(table, bodies, self.info)

    @staticmethod
    def pdf_pages(path):
        with open(path, "rb") as handle:
            return len(re.findall(rb"/Type\s*/Page[^s]", handle.read()))

    def test_four_landscape_pages_when_three_or_more_dimensions_are_swept(self):
        rows = [dict(r, P2=p2) for r in ecc_grid_rows(3) for p2 in (6.0, 6.4)]
        pages = self.pages(rows)
        self.assertEqual(len(pages), 4)
        for fig in pages:
            w, h = fig.get_size_inches()
            self.assertGreater(w, h)

    def test_three_pages_when_only_two_dimensions_are_swept(self):
        self.assertEqual(len(self.pages(ecc_grid_rows(4))), 3)

    def test_a_single_swept_dimension_still_builds_every_page(self):
        pages = self.pages([{"e1": e, "megno": 2.0 if e < 0.2 else 5.0} for e in np.linspace(0, 0.4, 5)])
        self.assertEqual(len(pages), 3)

    def test_a_single_scenario_builds_a_report(self):
        pages = self.pages([{"megno": 2.0}])
        self.assertEqual(len(pages), 3)

    def test_the_pdf_is_written_atomically_with_the_pages(self):
        write_results(self.csv, ecc_grid_rows(4))
        pages = sr.write_stability_report(self.csv, self.pdf, self.info)
        self.assertEqual(self.pdf_pages(self.pdf), pages)
        self.assertFalse(os.path.exists(self.pdf + ".partial"))

    def test_the_text_of_the_report(self):
        try:
            import pypdfium2 as pdfium
        except ImportError:
            self.skipTest("pypdfium2 is not installed")
        write_results(self.csv, ecc_grid_rows(5))
        sr.write_stability_report(self.csv, self.pdf, self.info)
        doc = pdfium.PdfDocument(self.pdf)
        text = "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
        for expected in ("map and summary", "e (b) vs e (c)", "What drives the stability", "Representative scenarios", "Seed of the random angles", "7"):
            self.assertIn(expected, text)

    def test_a_failing_page_becomes_a_stub_and_the_rest_survives(self):
        write_results(self.csv, ecc_grid_rows(4))
        table, bodies = sr.load_scenarios(self.csv)
        with mock.patch.object(sr, "page_drivers", side_effect=RuntimeError("boom")), self.assertLogs(level=logging.ERROR):
            pages = sr.build_pages(table, bodies, self.info)
        self.assertEqual(len(pages), 3)
        self.assertTrue(any("Not available: RuntimeError: boom" in t.get_text() for t in pages[1].texts))

    def test_pages_are_drawn_with_the_report_style_whatever_the_global_style(self):
        write_results(self.csv, ecc_grid_rows(4))
        seen, original = [], sr.PdfPages.savefig

        def spy(pdf, fig, *args, **kwargs):
            p = sr.matplotlib.rcParams
            seen.append((p["font.size"], p["axes.labelsize"], p["xtick.labelsize"]))
            return original(pdf, fig, *args, **kwargs)

        hostile = {"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 16}
        with sr.matplotlib.rc_context(hostile), mock.patch.object(sr.PdfPages, "savefig", spy):
            sr.write_stability_report(self.csv, self.pdf, self.info)
        self.assertEqual(set(seen), {(sr.RC["font.size"], "medium", "medium")})


class GuardTest(unittest.TestCase):
    def test_a_missing_results_file_does_not_raise_and_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertLogs(level=logging.ERROR):
            self.assertIsNone(sr.write_stability_report_guarded(f"{tmp}/none.csv", f"{tmp}/r.pdf"))
            self.assertFalse(os.path.exists(f"{tmp}/r.pdf"))

    def test_an_empty_results_file_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(columns=["star_mass", "periods", "megno"]).to_csv(f"{tmp}/e.csv", index=False)
            with self.assertLogs(level=logging.ERROR):
                self.assertIsNone(sr.write_stability_report_guarded(f"{tmp}/e.csv", f"{tmp}/r.pdf"))

    def test_system_exit_is_contained_and_the_partial_file_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_results(f"{tmp}/r.csv", ecc_grid_rows(3))
            with mock.patch.object(sr, "build_pages", side_effect=SystemExit(1)), self.assertLogs(level=logging.ERROR):
                self.assertIsNone(sr.write_stability_report_guarded(f"{tmp}/r.csv", f"{tmp}/r.pdf"))
            self.assertEqual([f for f in os.listdir(tmp) if f.endswith(".partial")], [])


class RunIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        pd.DataFrame({"M_star": [0.82], "M_star_lerr": [0.04], "M_star_uerr": [0.04]}).to_csv(f"{self.tmp.name}/params_star.csv", index=False)
        self.yaml = write_yaml(f"{self.tmp.name}/two.yaml", TWO_PLANETS_YAML)

    def test_run_stability_writes_the_report_next_to_the_results(self):
        run_stability(make_args(self.tmp.name, self.yaml, seed=3))
        out = f"{self.tmp.name}/stability_two"
        self.assertTrue(os.path.exists(f"{out}/stability_report.pdf"))
        self.assertTrue(os.path.exists(f"{out}/stability_megno.csv"))

    def test_a_failing_report_does_not_break_the_run(self):
        with mock.patch.object(sr, "write_stability_report", side_effect=RuntimeError("boom")):
            run_stability(make_args(self.tmp.name, self.yaml))
        out = f"{self.tmp.name}/stability_two"
        self.assertTrue(os.path.exists(f"{out}/stability_megno.csv"))
        self.assertFalse(os.path.exists(f"{out}/stability_report.pdf"))

    def test_an_import_failure_of_the_report_module_does_not_break_the_run(self):
        import sys
        with mock.patch.dict(sys.modules, {"sherlockpipe.system_stability.stability_report": None}):
            run_stability(make_args(self.tmp.name, self.yaml))
        self.assertTrue(os.path.exists(f"{self.tmp.name}/stability_two/stability_megno.csv"))


if __name__ == "__main__":
    unittest.main()
