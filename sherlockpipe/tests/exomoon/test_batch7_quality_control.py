import batman
import numpy as np

from sherlockpipe.exomoon import ExoMoonLeastSquares


def _params(t0=0.0):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = 20.0
    params.rp = 0.08
    params.a = 12.0
    params.inc = 89.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.2, 0.1]
    params.limb_dark = 'quadratic'
    return params


def _emls(time, flux, errors, epochs=(0.0,)):
    return ExoMoonLeastSquares(
        '', 1, 1.0, 1.0, [0.2, 0.1], 8.72, 0.8, 100.0, 20.0,
        0.0, 0.30, 12.0 / 215.032, 89.0, 0.5, 0.0, 0.3, 90.0, 0.0,
        0.3, 0.6, epochs, time, flux, errors, 5.5, min_cadences_per_t0=3,
        skip_fold_fit=True, fit_rp_per_transit=False,
    )


def _planet_fixture():
    time = np.arange(-0.65, 0.651, 0.01)
    errors = np.full_like(time, 2e-4)
    flux = batman.TransitModel(_params(), time).light_curve(_params())
    return time, flux, errors


def _subtract(time, flux, errors):
    emls = _emls(time, flux, errors)
    result = emls.subtract_planet_transit(time, flux, errors)
    return emls, result


def test_residual_diagnostics_report_offset_scatter_and_correlation_without_rejection():
    time, planet, errors = _planet_fixture()
    flux = planet.copy()
    in_transit = np.abs(time) < 0.12
    flux[in_transit] += 3e-4
    originals = tuple(item.copy() for item in (time, flux, errors))

    emls, (result_time, result_flux, result_errors) = _subtract(time, flux, errors)

    diagnostic = emls.epoch_diagnostics[0]
    assert diagnostic['accepted']
    assert diagnostic['fit_converged']
    assert diagnostic['weighted_residual_mean'] > 1e-5
    assert diagnostic['weighted_residual_scatter'] >= 0
    assert np.isfinite(diagnostic['lag_one_residual_correlation'])
    assert diagnostic['ingress_covered'] and diagnostic['egress_covered']
    assert diagnostic['residual_classification'] == 'unexplained_residual_not_rejected'
    assert diagnostic['shape_uncertainty'] == 'not_propagated_correlated_residuals_possible'
    assert len(result_time) == len(result_flux) == len(result_errors)
    for actual, expected in zip((time, flux, errors), originals):
        np.testing.assert_array_equal(actual, expected)


def test_bad_exposure_spot_crossing_and_overlapping_moon_remain_astrophysical_residuals():
    time, planet, errors = _planet_fixture()
    cases = {
        'bad_exposure': np.eye(1, len(time), np.argmin(np.abs(time - 0.03))).ravel() * 0.01,
        'spot_crossing': 9e-4 * np.exp(-0.5 * (time / 0.025) ** 2),
        'overlapping_moon': -8e-4 * np.exp(-0.5 * ((time - 0.035) / 0.025) ** 2),
    }
    for perturbation in cases.values():
        emls, _ = _subtract(time, planet + perturbation, errors)
        diagnostic = emls.epoch_diagnostics[0]
        assert diagnostic['accepted']
        assert diagnostic['residual_classification'] == 'unexplained_residual_not_rejected'
        assert diagnostic['weighted_residual_scatter'] > 0


def test_missing_ingress_and_all_rejected_return_synchronized_safe_status():
    time, planet, errors = _planet_fixture()
    missing_ingress = time >= 0
    emls, result = _subtract(time[missing_ingress], planet[missing_ingress], errors[missing_ingress])
    assert emls.subtraction_status['status'] == 'no_usable_epochs'
    assert emls.epoch_diagnostics[0]['residual_classification'] == 'poor_planet_model_fit'
    assert 'insufficient_coverage' in emls.epoch_diagnostics[0]['rejection_reason']
    assert len(result[0]) == len(result[1]) == len(result[2]) == len(emls.retained_cadence_indices)

    flat = np.ones_like(time)
    flat_before = flat.copy()
    all_rejected, result = _subtract(time, flat, errors)
    assert all_rejected.subtraction_status['status'] == 'no_usable_epochs'
    assert not all_rejected.bary_t0s
    assert len(result[0]) == len(result[1]) == len(result[2]) == len(all_rejected.retained_cadence_indices)
    np.testing.assert_array_equal(flat, flat_before)


def test_paired_planet_and_injected_selection_match_and_preserve_inputs():
    time, planet, errors = _planet_fixture()
    moon = 8e-4 * np.exp(-0.5 * ((time - 0.035) / 0.025) ** 2)
    originals = tuple(item.copy() for item in (time, planet, errors))

    planet_emls, (_, planet_residual, planet_errors) = _subtract(time, planet, errors)
    injected_emls, (_, injected_residual, injected_errors) = _subtract(time, planet - moon, errors)

    assert planet_emls.bary_t0s == injected_emls.bary_t0s == [0.0]
    np.testing.assert_array_equal(planet_emls.retained_cadence_indices, injected_emls.retained_cadence_indices)
    assert len(planet_errors) == len(injected_errors)
    assert np.all(planet_errors > 0) and np.all(injected_errors > 0)
    metrics = ExoMoonLeastSquares.signal_transfer_metrics(
        -moon, injected_residual - planet_residual, injected_errors)
    assert np.isfinite(metrics['weighted_transfer_coefficient'])
    for actual, expected in zip((time, planet, errors), originals):
        np.testing.assert_array_equal(actual, expected)
