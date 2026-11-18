"""The light curves of the odd and the even transit fits."""
import unittest

import numpy as np
import pandas as pd

from sherlockpipe.bayesian_fit.fitter import Fitter
from sherlockpipe.tests.bayesian_fit.fixtures import candidates_df, main_params_text

PERIOD, T0, WIDTH = 1.5, 100.0, 0.2


def curve(n_transits=8, per_transit=40, width=0.15):
    """The points around ``n_transits`` transits of a 1.5 d planet (epoch 100), and nothing in between."""
    rng = np.random.default_rng(1)
    time = np.concatenate([T0 + n * PERIOD + rng.uniform(-width, width, per_transit) for n in range(n_transits)])
    return pd.DataFrame({"time": np.sort(time), "flux": 1.0, "flux_err": 1e-3})


def transit_numbers(lc, t0=T0, period=PERIOD):
    return np.round((lc["time"].to_numpy() - t0) / period).astype(int)


class OddEvenCurveTest(unittest.TestCase):
    def setUp(self):
        self.candidates = candidates_df((1, PERIOD, T0, 0.5))
        self.lc = curve()

    def variant(self, name, candidates=None):
        return Fitter.odd_even_curve(self.lc, self.candidates if candidates is None else candidates, name, WIDTH)

    def test_the_odd_fit_keeps_the_first_third_and_fifth_transits(self):
        self.assertEqual(sorted(set(transit_numbers(self.variant("odd")))), [0, 2, 4, 6])

    def test_the_even_fit_keeps_the_second_fourth_and_sixth_transits(self):
        self.assertEqual(sorted(set(transit_numbers(self.variant("even")))), [1, 3, 5, 7])

    def test_between_them_the_two_fits_keep_every_point_exactly_once(self):
        odd, even = self.variant("odd"), self.variant("even")
        self.assertEqual(len(odd) + len(even), len(self.lc))
        self.assertEqual(set(odd["time"]).isdisjoint(set(even["time"])), True)

    def test_the_points_far_from_every_transit_are_kept_by_both_fits(self):
        lc = pd.concat([self.lc, pd.DataFrame({"time": [T0 + 0.75 + n * PERIOD for n in range(8)], "flux": 1.0, "flux_err": 1e-3})]).sort_values("time")
        odd = Fitter.odd_even_curve(lc, self.candidates, "odd", 0.2)
        self.assertEqual(sum(abs((odd["time"] - T0) % PERIOD - 0.75) < 1e-9), 8)

    def test_the_other_columns_and_the_order_are_kept(self):
        odd = self.variant("odd")
        self.assertEqual(list(odd.columns), ["time", "flux", "flux_err"])
        self.assertTrue(odd["time"].is_monotonic_increasing)
        self.assertEqual(list(odd.index), list(range(len(odd))))

    def test_every_candidate_is_split_by_its_own_ephemeris(self):
        candidates = candidates_df((1, PERIOD, T0, 0.5), (2, 2 * PERIOD, T0, 0.4))   # its 2nd transit is the 3rd of the first one
        both = Fitter.odd_even_curve(self.lc, candidates, "odd", WIDTH)
        alone = Fitter.odd_even_curve(self.lc, self.candidates, "odd", WIDTH)
        self.assertLess(len(both), len(alone))

    def test_the_original_curve_is_not_modified(self):
        before = self.lc.copy()
        self.variant("odd")
        pd.testing.assert_frame_equal(self.lc, before)

    def test_a_curve_without_transits_of_the_excluded_parity_is_unchanged(self):
        lc = curve(n_transits=1)
        self.assertEqual(len(Fitter.odd_even_curve(lc, self.candidates, "odd", WIDTH)), len(lc))


class OddEvenParamsTest(unittest.TestCase):
    def test_the_variants_use_the_params_of_the_main_fit(self):
        """The fit of the odd or the even transits keeps the period, the epoch and every prior of the main fit."""
        self.assertFalse(hasattr(Fitter, "odd_even_params_text"))
        self.assertFalse(hasattr(Fitter, "scale_prior"))
        self.assertIn("SOI_1_period", main_params_text(candidates_df((1, 1.4923, 1713.45, 0.49))))


if __name__ == "__main__":
    unittest.main()
