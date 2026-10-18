import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares


def _params():
    params = batman.TransitParams()
    params.t0 = 0.013
    params.per = 10.0
    params.rp = 0.09
    params.a = 14.0
    params.inc = 89.25
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.25, 0.15]
    params.limb_dark = "quadratic"
    return params


def _fine_batman(params, time, exposure_seconds, samples_per_exposure=900):
    """Independent high-resolution reference, deliberately finer than EMLS."""
    exposure_seconds = np.broadcast_to(exposure_seconds, len(time))
    result = np.empty(len(time))
    for exposure in np.unique(exposure_seconds):
        mask = exposure_seconds == exposure
        if exposure == 0:
            result[mask] = batman.TransitModel(params, time[mask]).light_curve(params)
        else:
            result[mask] = batman.TransitModel(
                params, time[mask], supersample_factor=samples_per_exposure,
                exp_time=exposure / 86400.0).light_curve(params)
    return result


@pytest.mark.parametrize("exposure_seconds", [20.0, 120.0, 1800.0])
def test_finite_exposure_model_matches_fine_batman_and_beats_instantaneous(exposure_seconds):
    time = np.linspace(-0.2, 0.2, 301)
    params = _params()
    truth = _fine_batman(params, time, exposure_seconds)
    integrated = ExoMoonLeastSquares._transit_model(params, time, exposure_seconds)
    instantaneous = ExoMoonLeastSquares._transit_model(params, time)

    integrated_error = np.sqrt(np.mean((integrated - truth) ** 2))
    instantaneous_error = np.sqrt(np.mean((instantaneous - truth) ** 2))
    assert integrated_error < instantaneous_error
    assert integrated_error < 2e-6


def test_exposure_supersampling_converges_and_improves_long_cadence_subtraction():
    time = np.linspace(-0.25, 0.25, 121)
    params = _params()
    truth = _fine_batman(params, time, 1800.0, samples_per_exposure=1800)
    errors = []
    for samples in (15, 45, 90):
        approximation = batman.TransitModel(
            params, time, supersample_factor=samples, exp_time=1800.0 / 86400.0).light_curve(params)
        errors.append(np.sqrt(np.mean((approximation - truth) ** 2)))
    assert errors[2] < errors[1] < errors[0]

    integrated = ExoMoonLeastSquares._transit_model(params, time, 1800.0)
    instantaneous = ExoMoonLeastSquares._transit_model(params, time)
    integrated_residual = truth - (integrated - 1.0)
    instantaneous_residual = truth - (instantaneous - 1.0)
    assert np.sqrt(np.mean((integrated_residual - 1.0) ** 2)) < 0.1 * np.sqrt(
        np.mean((instantaneous_residual - 1.0) ** 2))

    fitted_integrated, _ = ExoMoonLeastSquares.fit_single_t0_rp(
        time, truth, np.full(len(time), 2e-5), 0.0, 0.075, 0.03, 0.5, params,
        exposure_times_seconds=1800.0)
    fitted_instantaneous, _ = ExoMoonLeastSquares.fit_single_t0_rp(
        time, truth, np.full(len(time), 2e-5), 0.0, 0.075, 0.03, 0.5, params)
    assert abs(fitted_integrated[1] - params.rp) < abs(fitted_instantaneous[1] - params.rp)


def test_mixed_exposures_are_integrated_at_their_own_timestamps():
    time = np.linspace(-0.2, 0.2, 180)
    exposures = np.resize(np.array([20.0, 120.0, 1800.0, 0.0]), len(time))
    params = _params()
    truth = _fine_batman(params, time, exposures)
    integrated = ExoMoonLeastSquares._transit_model(params, time, exposures)
    instantaneous = ExoMoonLeastSquares._transit_model(params, time)

    assert np.sqrt(np.mean((integrated - truth) ** 2)) < 2e-6
    assert np.sqrt(np.mean((integrated - truth) ** 2)) < np.sqrt(np.mean((instantaneous - truth) ** 2))


def test_exposure_input_is_scalar_or_aligned_array_and_none_is_legacy_instantaneous():
    time = np.linspace(-0.1, 0.1, 31)
    params = _params()
    direct = batman.TransitModel(params, time).light_curve(params)
    np.testing.assert_array_equal(ExoMoonLeastSquares._transit_model(params, time), direct)
    np.testing.assert_allclose(
        ExoMoonLeastSquares._transit_model(params, time, 120.0),
        ExoMoonLeastSquares._transit_model(params, time, np.full(len(time), 120.0)))
    with pytest.raises(ValueError, match="align"):
        ExoMoonLeastSquares._transit_model(params, time, [20.0, 120.0])
