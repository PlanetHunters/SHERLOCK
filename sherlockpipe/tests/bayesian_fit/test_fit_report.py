"""The landscape fit report, on synthetic fit results (no allesfitter, no sampling)."""
import logging
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from sherlockpipe.bayesian_fit import fit_landscape_report as flr
from sherlockpipe.bayesian_fit.fit_results import Prior, load_fit_results, parse_priors, star_density_prior
from sherlockpipe.bayesian_fit.fitter import ODD_EVEN_ORBIT_SCALE
from sherlockpipe.tests.bayesian_fit.fixtures import FakeAlles, make_fit_results, write_fit_directory


class PriorsTest(unittest.TestCase):
    def test_uniform_normal_and_truncated_normal_priors_are_parsed_and_fixed_lines_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/params.csv"
            with open(path, "w") as handle:
                handle.write("#name,value,fit,bounds,label,unit\n#companion b,,,,,\nSOI_1_rr,0.02,1,uniform 0.01 0.03,x,\n"
                             "ln_err,-6,1,normal -6.3 0.01,x,\nq1,0.5,1,trunc_normal 0 1 0.5 0.2,x,\nfixed,0,0,,x,\n")
            priors = parse_priors(path)
        self.assertEqual(priors["SOI_1_rr"].bounds, (0.01, 0.03))
        self.assertIsNone(priors["ln_err"].bounds)
        self.assertEqual(priors["q1"].bounds, (0.0, 1.0))
        self.assertNotIn("fixed", priors)

    def test_a_missing_file_gives_no_priors(self):
        self.assertEqual(parse_priors("/does/not/exist.csv"), {})

    def test_the_stellar_density_comes_from_the_mass_and_the_radius(self):
        density, error = star_density_prior({"M_star": 0.47851, "R_star": 0.480553, "M_star_lerr": 0.02, "M_star_uerr": 0.02,
                                             "R_star_lerr": 0.014, "R_star_uerr": 0.014})
        self.assertAlmostEqual(density, 6.08, places=1)
        self.assertGreater(error, 0.3)

    def test_without_a_star_there_is_no_density_prior(self):
        self.assertIsNone(star_density_prior({}))
        self.assertIsNone(star_density_prior({"M_star": np.nan, "R_star": 0.5}))


class EdgeFlagTest(unittest.TestCase):
    def setUp(self):
        self.r = make_fit_results()

    def test_a_centered_posterior_is_not_flagged(self):
        self.assertIsNone(flr.edge_flag(self.r, "SOI_1_rsuma"))

    def test_a_posterior_piled_against_the_upper_edge_is_flagged_high(self):
        self.r.samples["SOI_1_rr"] = np.random.default_rng(1).uniform(0.0235, 0.0245, len(self.r.samples))
        self.assertEqual(flr.edge_flag(self.r, "SOI_1_rr"), "high")

    def test_a_flat_posterior_is_not_flagged(self):
        self.r.samples["SOI_1_rr"] = np.random.default_rng(1).uniform(0.015, 0.0245, len(self.r.samples))
        self.assertIsNone(flr.edge_flag(self.r, "SOI_1_rr"))

    def test_physical_limits_are_not_flagged_but_the_other_edge_is(self):
        self.r.samples["SOI_1_cosi"] = np.random.default_rng(1).uniform(0.0, 0.003, len(self.r.samples))
        self.assertIsNone(flr.edge_flag(self.r, "SOI_1_cosi"))   # cos i = 0 is a central transit
        self.r.samples["SOI_1_cosi"] = np.random.default_rng(1).uniform(0.058, 0.06, len(self.r.samples))
        self.assertEqual(flr.edge_flag(self.r, "SOI_1_cosi"), "high")
        self.r.samples["SOI_1_sbratio_lc"] = np.random.default_rng(1).uniform(0.0, 0.01, len(self.r.samples))
        self.assertIsNone(flr.edge_flag(self.r, "SOI_1_sbratio_lc"))

    def test_the_epoch_is_never_flagged_because_allesfitter_moves_its_origin(self):
        self.assertIsNone(flr.edge_flag(self.r, "SOI_1_epoch"))

    def test_edge_flags_only_list_the_planet_and_the_shared_parameters(self):
        r = make_fit_results(planets=("SOI_1", "SOI_2"))
        r.samples["SOI_2_rr"] = np.random.default_rng(1).uniform(0.0235, 0.0245, len(r.samples))
        self.assertEqual(flr.edge_flags(r, "SOI_1"), {})
        self.assertIn("SOI_2_rr", flr.edge_flags(r, "SOI_2"))


class NoiseAndConvergenceTest(unittest.TestCase):
    def test_white_noise_has_a_beta_close_to_one(self):
        beta = flr.red_noise_beta(np.random.default_rng(1).normal(0, 1, 20000))
        self.assertTrue(0.9 < beta < 1.15)

    def test_correlated_noise_has_a_large_beta(self):
        rng = np.random.default_rng(1)
        red = np.convolve(rng.normal(0, 1, 20000), np.ones(60) / 60, mode="same")
        self.assertGreater(flr.red_noise_beta(red), 2)

    def test_too_few_points_give_no_beta(self):
        self.assertTrue(np.isnan(flr.red_noise_beta(np.ones(50))))

    def test_the_effective_samples_set_the_convergence_status(self):
        for ess, status in ((5000, "ok"), (500, "warn"), (100, "fail")):
            r = make_fit_results(n_samples=1000)
            r.ess = ess
            self.assertEqual(flr.convergence_checks(r)[0].status, status)

    def test_a_large_evidence_error_is_a_warning(self):
        r = make_fit_results()
        r.logzerr = 2.0
        self.assertEqual(flr.convergence_checks(r)[1].status, "warn")

    def test_red_noise_residuals_raise_a_warning(self):
        r = make_fit_results()
        r.residuals = np.convolve(np.random.default_rng(1).normal(0, 1, len(r.residuals)), np.ones(40) / 40, mode="same")
        self.assertEqual(flr.noise_checks(r)[0].status, "warn")


class DensityTest(unittest.TestCase):
    def test_a_density_that_agrees_with_the_star_is_ok(self):
        self.assertEqual(flr.density_check(make_fit_results(), "SOI_1").status, "ok")

    def test_a_density_far_from_the_star_is_a_warning(self):
        r = make_fit_results()
        r.derived["SOI_1_host_density"] = np.random.default_rng(1).normal(12.0, 0.6, len(r.samples))
        self.assertEqual(flr.density_check(r, "SOI_1").status, "warn")

    def test_without_a_density_prior_there_is_no_check(self):
        r = make_fit_results()
        r.star_density = None
        self.assertIsNone(flr.density_check(r, "SOI_1"))


class OddEvenTest(unittest.TestCase):
    def test_the_scale_of_the_odd_even_priors_is_the_inverse_of_the_one_the_fitter_applies(self):
        self.assertAlmostEqual(flr.VARIANT_SCALE["rsuma"] * ODD_EVEN_ORBIT_SCALE, 1.0)
        self.assertAlmostEqual(flr.VARIANT_SCALE["cosi"] * ODD_EVEN_ORBIT_SCALE, 1.0)

    def test_consistent_fits_are_ok(self):
        self.assertEqual(flr.odd_even_check(make_fit_results(), "SOI_1").status, "ok")

    def test_fits_piled_against_their_priors_cannot_be_judged(self):
        check = flr.odd_even_check(make_fit_results(pegged=True), "SOI_1")
        self.assertEqual(check.status, "warn")
        self.assertIn("cannot be judged", check.detail)

    def test_a_different_radius_ratio_is_a_warning(self):
        r = make_fit_results()
        table = r.variants["even"].table
        table.loc[table["#name"] == "SOI_1_rr", "median"] += 0.01
        self.assertEqual(flr.odd_even_check(r, "SOI_1").status, "warn")

    def test_without_the_odd_and_even_fits_it_is_information(self):
        self.assertEqual(flr.odd_even_check(make_fit_results(variants=False), "SOI_1").status, "info")


class ComparisonTableTest(unittest.TestCase):
    def rows(self, **kw):
        return {r.label: r for r in flr.comparison_rows(make_fit_results(**kw), "SOI_1")}

    def test_every_parameter_has_the_main_odd_and_even_values(self):
        rows = self.rows()
        for label in ("Rp / R★", "Depth [ppt]", "Rp [R⊕]", "Impact parameter b", "Total duration [h]", "a / R★", "Stellar density [g/cm³]"):
            self.assertIn(label, rows)
            row = rows[label]
            self.assertTrue(all(v is not None for v in (row.main, row.odd, row.even)), label)

    def test_a_over_r_star_of_the_odd_and_even_fits_is_scaled_back_to_the_main_fit(self):
        row = self.rows()["a / R★"]
        self.assertAlmostEqual(row.odd[0], 9.03, delta=.01)   # 9.03 * 2^(2/3) from the fit, times 2^(-2/3)

    def test_the_difference_is_in_sigmas_between_the_odd_and_the_even_values(self):
        row = self.rows()["Depth [ppt]"]
        self.assertAlmostEqual(row.sigma, abs(row.odd[0] - row.even[0]) / np.hypot(row.odd[1], row.even[1]), places=6)

    def test_the_status_follows_two_and_three_sigmas(self):
        base = dict(label="x", fmt="{}", main=(1, 1), odd=(1, 1), even=(1, 1))
        for sigma, status in ((.5, "ok"), (2.5, "warn"), (3.5, "fail"), (None, "info")):
            self.assertEqual(flr.ComparisonRow(sigma=sigma, **base).status, status)

    def test_without_the_odd_and_even_fits_only_the_main_column_is_filled(self):
        rows = self.rows(variants=False)
        self.assertIsNotNone(rows["Depth [ppt]"].main)
        self.assertIsNone(rows["Depth [ppt]"].odd)
        self.assertIsNone(rows["Depth [ppt]"].sigma)

    def test_the_odd_even_check_uses_the_depth_when_it_is_there(self):
        r = make_fit_results()
        check = flr.odd_even_check(r, "SOI_1")
        self.assertIn("depth differs by", check.detail)
        self.assertEqual(check.status, "ok")

    def test_a_very_different_depth_is_a_warning(self):
        r = make_fit_results()
        table = r.variants["even"].derived
        table.loc[table["#property"].str.startswith("Transit depth (undil.)"), "value"] += 1.0
        self.assertEqual(flr.odd_even_check(r, "SOI_1").status, "warn")

    def test_without_derived_tables_the_check_falls_back_to_the_radius_ratio(self):
        r = make_fit_results()
        for variant in r.variants.values():
            variant.derived = None
        self.assertIn("Rp/R★ differs by", flr.odd_even_check(r, "SOI_1").detail)

    def test_the_consistency_page_has_the_table_with_every_row(self):
        fig = flr.page_consistency(make_fit_results(), "SOI_1", 2)
        text = " ".join(t.get_text() for t in fig.texts)
        for label in ("ODD AND EVEN FITS AGAINST THE MAIN FIT", "Depth [ppt]", "Stellar density [g/cm³]", "odd − even"):
            self.assertIn(label, text)

    def test_the_loader_keeps_the_derived_table_of_the_odd_and_even_fits(self):
        from sherlockpipe.bayesian_fit.fit_results import read_variants
        source = make_fit_results(n_samples=1200, n_time=2000)
        with tempfile.TemporaryDirectory() as tmp:
            fit_dir = f"{tmp}/fit_[1]/main"
            write_fit_directory(fit_dir, source)
            for kind in ("odd", "even"):
                source.variants[kind].derived.to_csv(f"{tmp}/fit_[1]/{kind}/results/ns_derived_table.csv", index=False)
            variants = read_variants(fit_dir)
        self.assertEqual(sorted(variants), ["even", "odd"])
        self.assertIsNotNone(variants["odd"].derived)
        self.assertGreater(len(variants["odd"].derived), 5)


class TransitsTest(unittest.TestCase):
    def test_only_the_transits_with_data_are_listed(self):
        r = make_fit_results()
        times, t14, total = flr.transit_times(r, "SOI_1")
        self.assertGreater(len(times), 0)
        self.assertGreaterEqual(total, len(times))
        self.assertAlmostEqual(t14 * 24, 1.21, delta=.1)

    def test_a_gap_removes_the_transits_inside_it(self):
        r = make_fit_results()
        full = len(flr.transit_times(r, "SOI_1")[0])
        keep = (r.time < 2540.0) | (r.time > 2541.5)
        for key in ("time", "flux", "baseline", "model", "residuals"):
            setattr(r, key, getattr(r, key)[keep])
        self.assertLess(len(flr.transit_times(r, "SOI_1")[0]), full)

    def test_gaps_in_the_residuals_do_not_turn_the_chi2_into_nan(self):
        r = make_fit_results()
        r.residuals = r.residuals.copy()
        r.residuals[::50] = np.nan
        times, t14, _ = flr.transit_times(r, "SOI_1")
        chi2 = flr.transit_chi2(r, "SOI_1", times, t14)
        self.assertTrue(np.isfinite(chi2).all())
        self.assertTrue(0.5 < np.median(chi2) < 1.6)

    def test_chi2_is_close_to_one_for_white_residuals(self):
        r = make_fit_results()
        times, t14, _ = flr.transit_times(r, "SOI_1")
        chi2 = flr.transit_chi2(r, "SOI_1", times, t14)
        self.assertTrue(0.5 < np.nanmedian(chi2) < 1.6)


class VerdictTest(unittest.TestCase):
    def test_clean_warnings_and_failures(self):
        ok, warn, fail = flr.Check("ok", "a", ""), flr.Check("warn", "b", ""), flr.Check("fail", "c", "")
        self.assertIn("clean", flr.verdict([ok])[0])
        self.assertIn("1 thing to check", flr.verdict([ok, warn])[0])
        self.assertIn("2 things", flr.verdict([warn, warn])[0])
        self.assertIn("problems", flr.verdict([warn, fail])[0])

    def test_pegged_odd_even_fits_and_an_edge_posterior_make_the_planet_checks_warn(self):
        r = make_fit_results(pegged=True)
        r.samples["SOI_1_rr"] = np.random.default_rng(1).uniform(0.0235, 0.0245, len(r.samples))
        warnings = [c.label for c in flr.planet_checks(r, "SOI_1") if c.status == "warn"]
        self.assertIn("Prior edges", warnings)
        self.assertIn("Odd / even", warnings)


class ReportTest(unittest.TestCase):
    @staticmethod
    def pdf_pages(path):
        with open(path, "rb") as handle:
            return len(re.findall(rb"/Type\s*/Page[^s]", handle.read()))

    def test_one_planet_gives_four_landscape_pages_in_the_agreed_order(self):
        pages = flr.build_pages(make_fit_results())
        self.assertEqual(len(pages), 4)
        for fig in pages:
            w, h = fig.get_size_inches()
            self.assertGreater(w, h)
        titles = [fig.texts[0].get_text() for fig in pages]
        self.assertTrue(titles[0].startswith("Bayesian fit"))
        self.assertTrue(titles[1].startswith("Consistency"))     # the second page is the odd / even one
        self.assertTrue(titles[2].startswith("Posteriors"))
        self.assertTrue(titles[3].startswith("Did the sampling"))

    def test_every_planet_gets_its_own_pages_and_the_convergence_page_is_shared(self):
        pages = flr.build_pages(make_fit_results(planets=("SOI_1", "SOI_2")))
        self.assertEqual(len(pages), 7)
        titles = [fig.texts[0].get_text() for fig in pages]
        self.assertEqual([t.split("·")[-1].strip() for t in titles[:6]], ["SOI_1", "SOI_1", "SOI_1", "SOI_2", "SOI_2", "SOI_2"])
        self.assertTrue(titles[6].startswith("Did the sampling"))

    def test_masked_candidates_are_named_on_the_summary(self):
        fig = flr.page_summary(make_fit_results(masked=("SOI_2",)), "SOI_1", 1)
        self.assertTrue(any("masked in the fit: SOI_2" in t.get_text() for t in fig.axes[0].texts))

    def test_the_corner_plot_includes_the_stellar_density(self):
        labels = [c[0] for c in flr.corner_columns(make_fit_results(), "SOI_1")]
        self.assertTrue(any(label.startswith("ρ★") for label in labels))
        self.assertEqual(len(labels), 6)

    def test_the_corner_period_is_centered_so_it_has_no_offset_text(self):
        period = [c for c in flr.corner_columns(make_fit_results(), "SOI_1") if c[0].startswith("P")][0][1]
        self.assertAlmostEqual(float(np.median(period)), 0.0, delta=1)

    def test_reports_without_odd_and_even_fits_still_build(self):
        self.assertEqual(len(flr.build_pages(make_fit_results(variants=False))), 4)

    def test_the_pdf_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            pages = flr.write_fit_report(make_fit_results(), f"{tmp}/r.pdf")
            self.assertEqual(self.pdf_pages(f"{tmp}/r.pdf"), pages)
            self.assertFalse(os.path.exists(f"{tmp}/r.pdf.partial"))

    def test_a_failing_page_becomes_a_stub_and_the_rest_survives(self):
        with mock.patch.object(flr, "page_corner", side_effect=RuntimeError("boom")), self.assertLogs(level=logging.ERROR):
            pages = flr.build_pages(make_fit_results())
        self.assertEqual(len(pages), 4)
        self.assertTrue(any("Not available: RuntimeError: boom" in t.get_text() for t in pages[2].texts))

    def test_pages_are_drawn_with_the_report_style_whatever_the_global_style(self):
        seen, original = [], flr.PdfPages.savefig

        def spy(pdf, fig, *args, **kwargs):
            p = flr.matplotlib.rcParams
            seen.append((p["font.size"], p["axes.labelsize"], p["xtick.labelsize"]))
            return original(pdf, fig, *args, **kwargs)

        hostile = {"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 16}
        with tempfile.TemporaryDirectory() as tmp, flr.matplotlib.rc_context(hostile), mock.patch.object(flr.PdfPages, "savefig", spy):
            flr.write_fit_report(make_fit_results(), f"{tmp}/r.pdf")
        self.assertEqual(set(seen), {(flr.RC["font.size"], "medium", "medium")})

    def test_a_fit_without_planets_is_an_error(self):
        r = make_fit_results()
        r.planets = []
        with self.assertRaises(ValueError):
            flr.write_fit_report(r, "/tmp/never.pdf")


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = make_fit_results(variants=True, n_samples=2000, n_time=3000)
        self.fit_dir = f"{self.tmp.name}/fit_[1]/main"
        write_fit_directory(self.fit_dir, self.source)
        self.star = pd.Series({"ra": 185.3, "dec": 57.7, "v": 12.8, "j": 9.4, "h": 8.8, "k": 8.6, "M_star": 0.48, "R_star": 0.48,
                               "M_star_lerr": 0.02, "M_star_uerr": 0.02, "R_star_lerr": 0.014, "R_star_uerr": 0.014})

    def load(self, **kw):
        candidates = pd.DataFrame({"name": ["SOI_1", "SOI_2"]})
        return load_fit_results(self.fit_dir, "TIC 1", self.star, candidates, alles=FakeAlles(self.source), **kw)

    def test_the_samples_tables_and_priors_are_read(self):
        r = self.load()
        self.assertEqual(r.planets, ["SOI_1"])
        self.assertEqual(r.masked, ["SOI_2"])
        self.assertIn("SOI_1_rr", r.samples.columns)
        self.assertEqual(r.priors["SOI_1_rr"].bounds, (0.015, 0.0245))
        self.assertAlmostEqual(r.ess, 2000, delta=1)
        self.assertAlmostEqual(r.logz, 96896.45)
        self.assertAlmostEqual(r.star_density[0], 6.08 + 0.1, delta=.3)
        self.assertEqual(r.magnitudes["v"], 12.8)

    def test_the_residuals_are_the_data_minus_the_model_and_the_baseline(self):
        r = self.load()
        np.testing.assert_allclose(r.residuals, r.flux - r.model - r.baseline)
        np.testing.assert_allclose(r.detrended, r.flux - r.baseline)

    def test_the_odd_and_even_fits_are_found_next_to_the_main_one(self):
        self.assertEqual(sorted(self.load().variants), ["even", "odd"])

    def test_the_fit_can_be_loaded_without_odd_and_even_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_dir = f"{tmp}/fit_[1]/main"
            write_fit_directory(fit_dir, self.source, variants=False)
            r = load_fit_results(fit_dir, "TIC 1", self.star, None, alles=FakeAlles(self.source))
        self.assertEqual(r.variants, {})

    def test_the_resampling_is_reproducible(self):
        a, b = self.load(seed=5), self.load(seed=5)
        np.testing.assert_array_equal(a.samples["SOI_1_rr"], b.samples["SOI_1_rr"])

    def test_a_mismatch_between_the_samples_and_the_table_is_an_error(self):
        table = pd.read_csv(f"{self.fit_dir}/results/ns_table.csv")
        table.iloc[:-2].to_csv(f"{self.fit_dir}/results/ns_table.csv", index=False)
        with self.assertRaises(ValueError):
            self.load()

    def test_a_baseline_that_cannot_be_computed_is_taken_as_zero(self):
        alles = FakeAlles(self.source)
        alles.get_posterior_median_baseline = mock.Mock(side_effect=RuntimeError("no gp"))
        with self.assertLogs(level=logging.ERROR):
            r = load_fit_results(self.fit_dir, "TIC 1", self.star, None, alles=alles)
        self.assertTrue(np.all(r.baseline == 0))

    def test_the_whole_report_can_be_written_from_the_files(self):
        r = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(flr.write_fit_report(r, f"{tmp}/r.pdf"), 4)


class GuardTest(unittest.TestCase):
    def test_a_missing_directory_does_not_raise_and_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertLogs(level=logging.ERROR):
            self.assertIsNone(flr.write_fit_report_guarded(f"{tmp}/none", "TIC 1", pd.Series(dtype=float), None))

    def test_system_exit_is_contained_and_no_partial_file_is_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("sherlockpipe.bayesian_fit.fit_results.load_fit_results", side_effect=SystemExit(1)), self.assertLogs(level=logging.ERROR):
                self.assertIsNone(flr.write_fit_report_guarded(tmp, "TIC 1", pd.Series(dtype=float), None))
            self.assertEqual([f for f in os.listdir(tmp) if f.endswith(".partial")], [])

    def test_the_report_goes_next_to_the_fit_with_the_target_name(self):
        source = make_fit_results(n_samples=1500, n_time=3000)
        with tempfile.TemporaryDirectory() as tmp:
            fit_dir = f"{tmp}/fit_[1]/main"
            write_fit_directory(fit_dir, source)
            with mock.patch("sherlockpipe.bayesian_fit.fit_results.alexfitter", create=True):
                import sherlockpipe.bayesian_fit.fit_results as fr
                real = fr.load_fit_results
                with mock.patch.object(fr, "load_fit_results", lambda d, i, s, c: real(d, i, s, c, alles=FakeAlles(source))):
                    path = flr.write_fit_report_guarded(fit_dir, "TIC 1", pd.Series({"M_star": 0.48, "R_star": 0.48}), None)
            self.assertEqual(path, f"{fit_dir}/TIC 1_fit.pdf")
            self.assertTrue(os.path.exists(path))


class RunIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.star = pd.DataFrame([{"obj_id": "TIC 1", "ra": 1.0, "dec": 2.0}])
        self.candidates = pd.DataFrame({"name": ["SOI_1"]})

    def test_the_run_asks_for_the_report_of_the_fit_directory(self):
        from sherlockpipe.bayesian_fit import run
        with mock.patch.object(flr, "write_fit_report_guarded") as writer:
            run.save_fit_report("/fit/main", self.star, self.candidates)
        args = writer.call_args[0]
        self.assertEqual((args[0], args[1]), ("/fit/main", "TIC 1"))
        self.assertEqual(args[2]["ra"], 1.0)

    def test_a_failing_report_does_not_break_the_fit(self):
        from sherlockpipe.bayesian_fit import run
        with mock.patch.object(flr, "write_fit_report_guarded", side_effect=RuntimeError("boom")), self.assertLogs(level=logging.ERROR):
            run.save_fit_report("/fit/main", self.star, self.candidates)

    def test_an_import_failure_of_the_report_module_does_not_break_the_fit(self):
        import sys
        from sherlockpipe.bayesian_fit import run
        with mock.patch.dict(sys.modules, {"sherlockpipe.bayesian_fit.fit_landscape_report": None}), self.assertLogs(level=logging.ERROR):
            run.save_fit_report("/fit/main", self.star, self.candidates)

    def test_the_old_portrait_report_is_gone(self):
        import importlib
        with self.assertRaises(ImportError):
            importlib.import_module("sherlockpipe.bayesian_fit.fit_report")


if __name__ == "__main__":
    unittest.main()
