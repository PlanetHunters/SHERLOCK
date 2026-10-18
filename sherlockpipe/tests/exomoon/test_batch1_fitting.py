import copy

import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares


def _params(t0=0.0, rp=0.08):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = 20.0
    params.rp = rp
    params.a = 12.0
    params.inc = 89.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.2, 0.1]
    params.limb_dark = "quadratic"
    return params


def _model(time, t0, rp=0.08):
    params = _params(t0, rp)
    return batman.TransitModel(params, time).light_curve(params)


def _emls(time, flux, flux_err, t0s=(2.0, 6.0, 10.0), fit_rp=True):
    return ExoMoonLeastSquares(
        "", 1, 1.0, 1.0, [0.2, 0.1], 8.72, 0.8, 100.0, 20.0,
        2.0, 0.25, 12.0 / 215.032, 89.0, 0.05, 0.0, 0.5, 90.0, 0.0, 0.3, 0.6,
        t0s, time, flux, flux_err, 5.5, min_cadences_per_t0=3,
        skip_fold_fit=True, fit_rp_per_transit=fit_rp,
    )


def test_light_curve_validation_is_explicit_and_does_not_mutate_inputs():
    time = np.array([0.0, 1.0, 2.0])
    flux = np.array([1.0, 0.99, 1.0])
    errors = np.array([0.001, 0.002, 0.003])
    originals = tuple(array.copy() for array in (time, flux, errors))

    returned = ExoMoonLeastSquares._validate_light_curve(time, flux, errors)

    for actual, expected in zip(returned, originals):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip((time, flux, errors), originals):
        np.testing.assert_array_equal(actual, expected)

    with pytest.raises(ValueError, match="flux_err is required"):
        ExoMoonLeastSquares._validate_light_curve(time, flux, None)
    for invalid_errors in (np.array([0.001, 0.0, 0.003]), np.array([0.001, np.nan, 0.003])):
        with pytest.raises(ValueError, match="strictly positive"):
            ExoMoonLeastSquares._validate_light_curve(time, flux, invalid_errors)
    with pytest.raises(ValueError, match="aligned"):
        ExoMoonLeastSquares._validate_light_curve(time, flux[:-1], errors)
    with pytest.raises(ValueError, match="No observations"):
        ExoMoonLeastSquares._validate_light_curve([], [], [])


def test_mask_mode_keeps_exact_cadence_error_alignment():
    time = np.arange(0.0, 12.0, 0.02)
    flux = np.ones_like(time)
    for t0 in (2.0, 6.0, 10.0):
        flux *= _model(time, t0)
    errors = np.linspace(0.0001, 0.0007, len(time))
    emls = _emls(time, flux, errors)
    original_time, original_flux, original_errors = time.copy(), flux.copy(), errors.copy()

    result_time, result_flux, result_errors = emls.subtract_planet_transit(
        time, flux, errors, mode="mask")

    expected_ids = emls.retained_cadence_indices
    np.testing.assert_array_equal(result_time, original_time[expected_ids])
    np.testing.assert_array_equal(result_flux, original_flux[expected_ids])
    np.testing.assert_array_equal(result_errors, original_errors[expected_ids])
    np.testing.assert_array_equal(errors, original_errors)
    assert not np.any(np.abs(result_time - 6.0) < 0.1)


def test_epoch_scenarios_are_unique_and_score_is_permutation_invariant():
    time = np.array([0.0, 10.0, 10.1, 20.0, 20.1, 20.2])
    flux = np.array([1.0, 0.99, 1.0, 1.0, 0.98, 1.0])
    errors = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    epochs = [-10.0, 0.0, 10.0, 20.0]

    scenarios = ExoMoonLeastSquares.compute_moon_transit_scenarios(
        time, flux, errors, 0.0, 0.0, 1.0, epochs, 0.1, 0.2, 0.05)

    assert [len(scenario[1]) for scenario in scenarios] == [1, 2, 3]
    concatenated_times = np.concatenate([scenario[1] for scenario in scenarios])
    assert len(concatenated_times) == len(time)
    assert sum(len(scenario[1]) for scenario in scenarios) == len(time)

    template_time = np.array([-0.05, 0.0, 0.05])
    template_flux = np.array([1.0, 0.98, 1.0])
    score = ExoMoonLeastSquares.calculate_residuals(
        np.array([-0.05, 0.0, 0.05]), template_flux, np.array([0.002, 0.003, 0.004]),
        np.array([-0.1, 0.1]), np.ones(2), np.array([0.005, 0.006]), 0.99, 5,
        [(0.3, template_time, template_flux)], 0.3, 0.3, 1.0, 1,
    )[0]
    permuted_score = ExoMoonLeastSquares.calculate_residuals(
        np.array([0.05, -0.05, 0.0]), np.array([1.0, 1.0, 0.98]), np.array([0.004, 0.002, 0.003]),
        np.array([0.1, -0.1]), np.ones(2), np.array([0.006, 0.005]), 0.99, 5,
        [(0.3, template_time, template_flux)], 0.3, 0.3, 1.0, 1,
    )[0]
    assert score == pytest.approx(permuted_score)


@pytest.mark.parametrize("fit_rp", [False, True])
def test_per_epoch_fits_are_order_independent_and_preserve_shared_params(fit_rp):
    time = np.arange(-0.4, 10.5, 0.005)
    errors = np.full_like(time, 0.0002)
    flux = _model(time, 2.0, 0.06) * _model(time, 6.0, 0.10) * _model(time, 10.0, 0.07)
    shared = _params(0.0, 0.08)
    shared_before = copy.deepcopy(shared)

    results = {}
    for order in ((2.0, 6.0, 10.0), (10.0, 6.0, 2.0), (6.0, 2.0, 10.0)):
        order_results = {}
        for epoch in order:
            # B2 requires OOT context on both sides of a usable local fit.
            mask = np.abs(time - epoch) < 0.45
            if fit_rp:
                theta, _ = ExoMoonLeastSquares.fit_single_t0_rp(
                    time[mask], flux[mask], errors[mask], epoch, shared.rp, 0.04, 0.9, shared)
            else:
                theta, _ = ExoMoonLeastSquares.fit_single_t0(
                    time[mask], flux[mask], errors[mask], epoch, 0.9, shared)
            order_results[epoch] = theta
        results[order] = order_results

    reference = results[(2.0, 6.0, 10.0)]
    for order_results in results.values():
        for epoch, theta in reference.items():
            np.testing.assert_allclose(order_results[epoch], theta, rtol=0, atol=1e-8)
    assert shared.t0 == shared_before.t0
    assert shared.rp == shared_before.rp
    assert shared.a == shared_before.a
    assert shared.inc == shared_before.inc
