import copy

import batman
import numpy as np
import pytest

import sherlockpipe.exomoon as exomoon_module
from sherlockpipe.exomoon import ExoMoonLeastSquares


def _params(t0=0.0, a=12.0, rp=0.08):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = 20.0
    params.rp = rp
    params.a = a
    params.inc = 89.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.2, 0.1]
    params.limb_dark = "quadratic"
    return params


def _flux(time, params):
    return batman.TransitModel(params, time).light_curve(params)


def _result(time, flux, params, guess, bounds, fields):
    return ExoMoonLeastSquares._fit_transit_result(
        time, flux, np.full(len(time), 2e-4), guess, bounds, params, fields)


def test_good_fit_offset_epoch_and_large_absolute_time_shift_are_invariant():
    relative_time = np.linspace(-0.7, 0.7, 401)
    truth = _params(t0=0.037, rp=0.075)
    flux = _flux(relative_time, truth)
    guess = [0.0, 0.08]
    bounds = [(-0.2, 0.2), (0.04, 0.12)]

    result = _result(relative_time, flux, _params(rp=0.08), guess, bounds, ['t0', 'rp'])
    shifted = _result(relative_time + 2457000.0, flux, _params(rp=0.08),
                      [2457000.0, 0.08], [(2456999.8, 2457000.2), (0.04, 0.12)],
                      ['t0', 'rp'])

    assert result.success and shifted.success
    np.testing.assert_allclose(result.theta, [truth.t0, truth.rp], atol=2e-7)
    np.testing.assert_allclose(shifted.theta - [2457000.0, 0], result.theta, atol=2e-7)
    # Exact instantaneous synthetic data leave only optimizer/model roundoff.
    np.testing.assert_allclose(flux - (result.model - 1.0), 1.0, atol=2e-10)


def test_difficult_wrong_density_shape_fit_recovers_without_bound_solution():
    rng = np.random.default_rng(117)
    time = np.linspace(-0.8, 0.8, 501)
    truth = _params(t0=0.041, a=10.5, rp=0.072)
    flux = _flux(time, truth) + rng.normal(0, 4e-5, len(time))
    result = _result(time, flux, _params(a=14.0, rp=0.10), [0.0, 14.0, 89.0, 0.10],
                     [(-0.15, 0.15), (8.0, 16.0), (86.0, 90.0), (0.04, 0.14)],
                     ['t0', 'a', 'inc', 'rp'])

    assert result.success
    assert result.theta[0] == pytest.approx(truth.t0, abs=3e-4)
    assert result.theta[1] == pytest.approx(truth.a, abs=0.15)
    assert result.theta[3] == pytest.approx(truth.rp, abs=2e-3)
    assert not np.any(result.active_bounds)


def test_no_dip_partial_coverage_flat_objective_and_invalid_bounds_are_rejected():
    time = np.linspace(-0.7, 0.7, 301)
    params = _params()
    no_dip = _result(time, np.ones_like(time), params, [0.0], [(-0.2, 0.2)], ['t0'])
    partial_time = time[time >= 0.0]
    partial = _result(partial_time, _flux(partial_time, params), params, [0.0], [(-0.2, 0.2)], ['t0'])
    # Every datum is far from the transit, so timing has a flat model/objective.
    flat = _result(np.linspace(2, 3, 50), np.ones(50), params, [0.0], [(-0.2, 0.2)], ['t0'])
    invalid = _result(time, _flux(time, params), params, [0.0], [(0.2, -0.2)], ['t0'])

    assert no_dip.status == 'no_improvement'
    assert partial.status == 'insufficient_coverage'
    assert flat.status == 'insufficient_coverage'
    assert invalid.status == 'invalid_bounds'


def test_active_bound_and_optimizer_failure_are_not_accepted(monkeypatch):
    time = np.linspace(-0.7, 0.7, 401)
    truth = _params(t0=0.1)
    active = _result(time, _flux(time, truth), _params(), [0.0], [(-0.2, 0.1)], ['t0'])
    assert active.status == 'active_bound'

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic optimizer failure')

    monkeypatch.setattr(exomoon_module, 'least_squares', fail)
    failed = _result(time, _flux(time, truth), _params(), [0.0], [(-0.2, 0.2)], ['t0'])
    assert not failed.success
    assert failed.status == 'optimizer_failure'


def test_structured_result_does_not_mutate_caller_params():
    time = np.linspace(-0.7, 0.7, 301)
    params = _params()
    before = copy.deepcopy(params)
    result = _result(time, _flux(time, _params(t0=0.03)), params, [0.0], [(-0.2, 0.2)], ['t0'])

    assert result.success
    assert params.t0 == before.t0
    assert params.a == before.a
    assert params.rp == before.rp


def test_failed_local_fit_is_not_used_for_planet_subtraction():
    time = np.linspace(-0.7, 0.7, 301)
    flux = np.ones_like(time)
    errors = np.full_like(time, 2e-4)
    emls = ExoMoonLeastSquares(
        "", 1, 1.0, 1.0, [0.2, 0.1], 8.72, 0.8, 100.0, 20.0,
        0.0, 0.6, 12.0 / 215.032, 89.0, 0.5, 0.0, 0.3, 90.0, 0.0, 0.3, 0.6,
        [0.0], time, flux, errors, 5.5, min_cadences_per_t0=3,
        skip_fold_fit=True, fit_rp_per_transit=False,
    )

    result_time, result_flux, result_errors = emls.subtract_planet_transit(time, flux, errors)
    assert emls.subtraction_status['status'] == 'no_usable_epochs'
    assert len(result_time) == len(result_flux) == len(result_errors)
    assert emls.epoch_diagnostics[0]['accepted'] is False
    assert 'no_improvement' in emls.epoch_diagnostics[0]['reason']
    np.testing.assert_array_equal(flux, np.ones_like(time))
