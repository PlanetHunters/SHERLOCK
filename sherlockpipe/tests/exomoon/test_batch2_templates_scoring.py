import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares, SearchInput


def _score(it_time, it_flux, it_err, oot_time, oot_flux, oot_err, template):
    return ExoMoonLeastSquares.calculate_residuals(
        np.asarray(it_time), np.asarray(it_flux), np.asarray(it_err),
        np.asarray(oot_time), np.asarray(oot_flux), np.asarray(oot_err),
        1.0, len(it_time) + len(oot_time), [(0.3, *template)], 0.3, 0.3, 1.0, 1,
    )


def test_asymmetric_template_uses_physical_times_and_unit_baseline_outside_support():
    template_time = np.array([-0.20, -0.05, 0.0, 0.10, 0.30])
    template_flux = np.array([1.0, 0.97, 0.94, 0.98, 1.0])
    observation_time = np.array([0.30, -0.40, 0.0, -0.05, 0.45, 0.10, -0.05, -0.20])
    expected = np.array([1.0, 1.0, 0.94, 0.97, 1.0, 0.98, 0.97, 1.0])
    errors = np.full(len(observation_time), 0.01)

    _, _, _, model = _score(observation_time, expected, errors, [], [], [],
                             (template_time, template_flux))

    np.testing.assert_allclose(model, expected)
    # Removing a cadence and reordering the observations cannot alter values at
    # any surviving physical observation time.
    keep = np.array([True, False, True, True, True, True, False, True])
    order = np.array([4, 0, 5, 2, 1, 3])
    _, _, _, subset_model = _score(observation_time[keep][order], expected[keep][order],
                                   errors[keep][order], [], [], [],
                                   (template_time, template_flux))
    np.testing.assert_allclose(subset_model, expected[keep][order])


def test_template_predictions_and_scores_are_invariant_to_large_time_origin_and_gaps():
    template_time = np.array([-0.2, -0.04, 0.02, 0.15])
    template_flux = np.array([1.0, 0.96, 0.975, 1.0])
    it_time = np.array([-0.19, -0.04, 0.02, 0.14])
    flux = np.interp(it_time, template_time, template_flux, left=1.0, right=1.0)
    errors = np.array([0.01, 0.02, 0.015, 0.03])
    oot_time = np.array([-0.8, 0.6])
    oot_flux = np.array([1.01, 0.98])
    oot_errors = np.array([0.04, 0.05])

    reference = _score(it_time, flux, errors, oot_time, oot_flux, oot_errors,
                       (template_time, template_flux))
    shifted = _score(it_time + 2457000.0, flux, errors, oot_time + 2457000.0,
                     oot_flux, oot_errors,
                     (template_time + 2457000.0, template_flux))
    np.testing.assert_allclose(reference[:3], shifted[:3])
    np.testing.assert_allclose(reference[3], shifted[3])


def test_batman_template_is_interpolated_at_actual_observation_times():
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = 10.0
    params.rp = 0.04
    params.a = 12.0
    params.inc = 89.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.2, 0.1]
    params.limb_dark = "quadratic"
    template_time = np.linspace(-0.2, 0.2, 4001)
    template_flux = batman.TransitModel(params, template_time).light_curve(params)
    observation_time = np.array([0.0, -0.17, 0.073, -0.031, 0.2, 0.27])
    direct = batman.TransitModel(params, observation_time).light_curve(params)

    _, _, _, sampled = _score(observation_time, direct, np.full(6, 0.001), [], [], [],
                               (template_time, template_flux))

    np.testing.assert_allclose(sampled[:-1], direct[:-1], atol=2e-8)
    assert sampled[-1] == 1.0


def test_weighted_null_and_alternative_match_hand_calculation_and_reordering():
    template = (np.array([-0.1, 0.0, 0.1]), np.array([1.0, 0.95, 1.0]))
    it_time = np.array([-0.1, 0.0, 0.1])
    it_flux = np.array([0.99, 0.96, 1.02])
    it_errors = np.array([0.01, 0.02, 0.04])
    oot_time = np.array([-0.3, 0.3])
    oot_flux = np.array([1.01, 0.98])
    oot_errors = np.array([0.05, 0.10])

    alternative, null, _, _ = _score(it_time, it_flux, it_errors, oot_time, oot_flux,
                                      oot_errors, template)
    oot_chi2 = np.sum(((oot_flux - 1.0) / oot_errors) ** 2)
    expected_null = np.sum(((it_flux - 1.0) / it_errors) ** 2) + oot_chi2
    expected_alternative = np.sum(((it_flux - template[1]) / it_errors) ** 2) + oot_chi2
    assert null == pytest.approx(expected_null)
    assert alternative == pytest.approx(expected_alternative)

    permuted = _score(it_time[[2, 0, 1]], it_flux[[2, 0, 1]], it_errors[[2, 0, 1]],
                      oot_time[::-1], oot_flux[::-1], oot_errors[::-1], template)
    np.testing.assert_allclose(permuted[:2], (alternative, null))


def test_null_equivalent_and_exact_signal_have_expected_chi_square_improvements():
    time = np.array([-0.1, 0.0, 0.1])
    errors = np.array([0.01, 0.02, 0.04])
    flat = (time, np.ones(3))
    alternative, null, _, _ = _score(time, np.ones(3), errors, [], [], [], flat)
    assert alternative - null == pytest.approx(0.0)

    signal = (time, np.array([1.0, 0.95, 1.0]))
    alternative, null, _, _ = _score(time, signal[1], errors, [], [], [], signal)
    assert alternative == pytest.approx(0.0)
    assert alternative - null < 0.0


def test_outside_support_and_oot_changes_are_shared_by_both_hypotheses():
    template = (np.array([-0.1, 0.0, 0.1]), np.array([1.0, 0.95, 1.0]))
    it_time = np.array([-0.4, 0.0, 0.4])
    it_flux = np.array([0.97, 0.95, 1.03])
    it_errors = np.array([0.02, 0.01, 0.03])
    oot_time = np.array([-0.5, 0.5])
    oot_errors = np.array([0.02, 0.04])
    first = _score(it_time, it_flux, it_errors, oot_time, np.array([1.0, 1.0]), oot_errors,
                   template)
    second = _score(it_time, it_flux, it_errors, oot_time, np.array([1.02, 0.96]), oot_errors,
                    template)

    assert first[3][0] == 1.0
    assert first[3][-1] == 1.0
    assert (second[0] - first[0]) == pytest.approx(second[1] - first[1])
    assert (second[0] - second[1]) == pytest.approx(first[0] - first[1])


def test_search_uses_gapped_physical_context_for_null_and_alternative():
    template_time = np.array([-0.1, 0.0, 0.1])
    template_flux = np.array([1.0, 0.96, 1.0])
    emls = object.__new__(ExoMoonLeastSquares)
    emls.time = np.array([-0.28, -0.1, -0.03, 0.0, 0.08, 0.27])
    emls.flux = np.interp(emls.time, template_time, template_flux, left=1.0, right=1.0)
    emls.flux_err = np.array([0.03, 0.02, 0.01, 0.01, 0.02, 0.04])
    emls.bary_t0s = [0.0]
    emls.models = [(0.3, template_time, template_flux)]
    emls.min_radius = emls.max_radius = 0.3
    emls.star_radius = 1.0
    emls.radius_grid_size = 1
    search_input = SearchInput(1.0, np.pi / 2, 0.0, 90.0, 0.0, 0.3, 0.0,
                               0.6, 0.2, 0.4, 0.1, 0.3, 0.3)

    alternative, null, radius, _ = emls.search(search_input)

    assert radius == 0.3
    assert alternative == pytest.approx(0.0)
    assert alternative - null < 0.0
