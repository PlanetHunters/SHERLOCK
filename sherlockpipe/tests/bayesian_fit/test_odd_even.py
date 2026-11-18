"""The parameters of the odd and the even transit fits."""
import unittest

from sherlockpipe.bayesian_fit.fitter import Fitter
from sherlockpipe.tests.bayesian_fit.fixtures import candidates_df, main_params_text, prior

SCALE = 0.5 ** (2 / 3)  # the same star with twice the period: a/R* grows by 2^(2/3)


class OddEvenParamsTest(unittest.TestCase):
    def setUp(self):
        self.candidates = candidates_df((1, 1.4923, 1713.45, 0.49))
        self.main = main_params_text(self.candidates)

    def variant(self, name, candidates=None):
        return Fitter.odd_even_params_text(self.main, self.candidates if candidates is None else candidates, name)

    def test_the_period_is_doubled_and_its_prior_centered_on_it(self):
        value, low, high = prior(self.variant("odd"), "SOI_1_period")
        self.assertAlmostEqual(value, 2 * 1.4923)
        self.assertAlmostEqual(low, 2 * 1.4923 - 2 * 4.8e-5)
        self.assertAlmostEqual(high, 2 * 1.4923 + 2 * 4.8e-5)

    def test_the_odd_epoch_is_kept_and_the_even_one_is_shifted_a_period(self):
        self.assertAlmostEqual(prior(self.variant("odd"), "SOI_1_epoch")[0], 1713.45)
        self.assertAlmostEqual(prior(self.variant("even"), "SOI_1_epoch")[0], 1713.45 + 1.4923)

    def test_the_epoch_prior_keeps_its_width(self):
        _, low, high = prior(self.variant("even"), "SOI_1_epoch")
        self.assertAlmostEqual(high - low, 0.04)

    def test_the_radius_ratio_prior_is_untouched(self):
        self.assertEqual(prior(self.variant("odd"), "SOI_1_rr"), prior(self.main, "SOI_1_rr"))

    def test_every_candidate_is_rewritten(self):
        candidates = candidates_df((1, 1.4923, 1713.45, 0.49), (2, 2.8662, 1712.95, 0.58))
        text = Fitter.odd_even_params_text(main_params_text(candidates), candidates, "odd")
        self.assertAlmostEqual(prior(text, "SOI_2_period")[0], 2 * 2.8662)
        self.assertAlmostEqual(prior(text, "SOI_1_period")[0], 2 * 1.4923)

    def test_the_rest_of_the_text_is_unchanged(self):
        before, after = self.main.splitlines(), self.variant("odd").splitlines()
        changed = [b for b, a in zip(before, after) if a != b]
        self.assertEqual(len(before), len(after))
        self.assertTrue(all(line.split(",")[0].endswith(("_epoch", "_period", "_rsuma", "_cosi")) for line in changed))


class OddEvenOrbitScalingTest(unittest.TestCase):
    """With the period doubled (same star) (R*+Rp)/a and cos i fall by 2^(-2/3); the old priors kept the main ones, so
    both odd and even fits piled up against the edge of their priors."""

    def setUp(self):
        self.candidates = candidates_df((1, 1.4923, 1713.45, 0.49))
        self.main = main_params_text(self.candidates)
        self.odd = Fitter.odd_even_params_text(self.main, self.candidates, "odd")

    def test_the_sum_of_radii_over_a_prior_is_scaled_with_the_period(self):
        for old, new in zip(prior(self.main, "SOI_1_rsuma"), prior(self.odd, "SOI_1_rsuma")):
            self.assertAlmostEqual(new, old * SCALE, places=9)

    def test_the_cosine_of_the_inclination_prior_is_scaled_with_the_period(self):
        for old, new in zip(prior(self.main, "SOI_1_cosi"), prior(self.odd, "SOI_1_cosi")):
            self.assertAlmostEqual(new, old * SCALE, places=9)

    def test_the_even_fit_gets_the_same_orbit_priors(self):
        even = Fitter.odd_even_params_text(self.main, self.candidates, "even")
        self.assertEqual(prior(even, "SOI_1_rsuma"), prior(self.odd, "SOI_1_rsuma"))

    def test_the_real_posterior_of_the_main_fit_fits_inside_the_scaled_prior(self):
        """Main fit of TIC 224298134: (R*+Rp)/a = 0.1133 and cos i = 0.041 -> 0.0714 and 0.0258 with 2P."""
        _, low, high = prior(self.odd, "SOI_1_rsuma")
        self.assertTrue(low < 0.1133 * SCALE < high)
        self.assertTrue(prior(self.odd, "SOI_1_cosi")[1] < 0.041 * SCALE < prior(self.odd, "SOI_1_cosi")[2])


if __name__ == "__main__":
    unittest.main()
