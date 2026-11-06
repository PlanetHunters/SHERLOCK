import logging

import batman
import numpy as np
import pytest
from scipy.optimize import least_squares

from sherlockpipe.exomoon import (AU_TO_RSUN, ExoMoonLeastSquares, PlanetModelError, R_earth_to_R_sun,
                                  TransitFitError)

PERIOD = 5.0
A_TRUE = 12.0
RP = 0.08
LIMB_DARKENING = [0.3, 0.2]
EPOCHS = np.arange(10) * PERIOD
CADENCE = 120.0 / 86400.0


def _inclination(impact_param, a=A_TRUE):
    return np.rad2deg(np.arccos(impact_param / a))


def _duration(a=A_TRUE, impact_param=0.3, rp=RP, period=PERIOD):
    inc = np.deg2rad(_inclination(impact_param, a))
    return period / np.pi * np.arcsin(np.sqrt((1 + rp) ** 2 - impact_param ** 2) / (a * np.sin(inc)))


def _params(t0, inc, a=A_TRUE, rp=RP, period=PERIOD):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = t0, period, rp, a, inc
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, LIMB_DARKENING, "quadratic"
    return params


def _light_curve(sigma, seed=11, rp=RP, offsets=None, epochs=EPOCHS):
    """Planet transits at each epoch plus the optional per-epoch timing offsets."""
    offsets = np.zeros(len(epochs)) if offsets is None else offsets
    duration = _duration(rp=rp)
    local = np.arange(-3 * duration, 3 * duration, CADENCE)
    time = np.concatenate([epoch + local for epoch in epochs])
    flux = np.ones_like(time)
    for epoch, offset in zip(epochs, offsets):
        near = np.abs(time - epoch) < 3 * duration
        params = _params(epoch + offset, _inclination(0.3), rp=rp)
        flux[near] = batman.TransitModel(params, time[near]).light_curve(params)
    flux = flux + np.random.default_rng(seed).normal(0, sigma, len(time))
    return time, flux, np.full(len(time), sigma), duration


def _emls(time, flux, errors, duration, a=A_TRUE, inc=None, inc_err=3.0, rp=RP, **kwargs):
    radius = rp / R_earth_to_R_sun
    inc = _inclination(0.3) if inc is None else inc
    return ExoMoonLeastSquares(
        "", 1, 1.0, 1.0, LIMB_DARKENING, radius, 0.15 * radius, 100.0, PERIOD, 0.0, duration,
        a / AU_TO_RSUN, inc, inc_err, 0.0, 0.0, 90.0, 0.0, 0.3, 0.6, EPOCHS, time, flux, errors, 5.5,
        min_cadences_per_t0=20, fit_rp_per_transit=False, min_transit_completeness=0.1,
        max_subtraction_rms_ratio=100.0, **kwargs)


def _worst_folded_residual(emls, result):
    time, flux, _ = result
    nearest = EPOCHS[np.argmin(np.abs(time[:, None] - EPOCHS), axis=1)]
    offset = time - nearest
    window = np.abs(offset) < 0.6 * emls.planet_duration
    bins = np.digitize(offset[window], np.linspace(-0.6 * emls.planet_duration, 0.6 * emls.planet_duration, 13))
    return max(abs(np.mean(flux[window][bins == index] - 1)) for index in range(1, 13))


def _prior_excludes_truth():
    time, flux, errors, duration = _light_curve(3e-4)
    return _emls(time, flux, errors, duration, inc=90.0, inc_err=0.2), time, flux, errors


def test_failed_shared_fit_stops_with_a_clear_status():
    emls, time, flux, errors = _prior_excludes_truth()

    with pytest.raises(PlanetModelError, match="active_bound"):
        emls.subtract_planet_transit(time, flux, errors)

    assert emls.subtraction_status['status'] == 'shared_fit_failed'
    assert 'active_bound' in emls.subtraction_status['reason']
    assert emls.subtraction_status['retained_epochs'] == 0
    assert not hasattr(emls, 'shared_planet_params')


def test_mask_mode_also_stops_on_a_failed_shared_fit():
    emls, time, flux, errors = _prior_excludes_truth()

    with pytest.raises(PlanetModelError):
        emls.subtract_planet_transit(time, flux, errors, mode='mask')


def test_skipping_the_shared_fit_remains_an_explicit_catalog_choice():
    time, flux, errors, duration = _light_curve(3e-4)
    emls = _emls(time, flux, errors, duration, skip_fold_fit=True)

    emls.subtract_planet_transit(time, flux, errors)

    assert emls.shared_fit_status.status == 'skipped'
    assert emls.subtraction_status['status'] == 'ok'


@pytest.mark.parametrize("bias", [0.8, 0.9, 1.1, 1.25])
def test_a_over_rstar_uncertainty_removes_catalog_bias(bias):
    time, flux, errors, duration = _light_curve(3e-4)
    emls = _emls(time, flux, errors, duration, a=A_TRUE * bias, planet_a_over_rstar_err=0.15 * A_TRUE)

    result = emls.subtract_planet_transit(time, flux, errors)

    assert emls.shared_fit_status.status == 'ok'
    assert emls.planet_duration == pytest.approx(duration, rel=0.02)
    assert _worst_folded_residual(emls, result) < 1.5e-4
    assert emls.shared_fit_residual['status'] == 'ok'


def test_a_over_rstar_stays_fixed_without_an_uncertainty():
    time, flux, errors, duration = _light_curve(3e-4)
    emls = _emls(time, flux, errors, duration, a=A_TRUE * 1.1)

    emls.subtract_planet_transit(time, flux, errors)

    assert emls.shared_planet_params.a == pytest.approx(A_TRUE * 1.1)


def test_biased_fixed_a_over_rstar_is_flagged_in_the_folded_residual(caplog):
    time, flux, errors, duration = _light_curve(3e-4)
    emls = _emls(time, flux, errors, duration, a=A_TRUE * 1.1)

    with caplog.at_level(logging.WARNING):
        emls.subtract_planet_transit(time, flux, errors)

    assert emls.shared_fit_status.status == 'ok'
    assert emls.shared_fit_residual['status'] == 'structured'
    assert emls.shared_fit_residual['max_abs_sigma'] > 10
    assert "Folded planet residual" in caplog.text


def test_a_over_rstar_pinned_to_its_uncertainty_limit_stops_the_target():
    time, flux, errors, duration = _light_curve(3e-4)
    emls = _emls(time, flux, errors, duration, a=A_TRUE * 1.25, planet_a_over_rstar_err=0.1)

    with pytest.raises(PlanetModelError, match="active_bound"):
        emls.subtract_planet_transit(time, flux, errors)


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_invalid_a_over_rstar_uncertainty_is_rejected(value):
    time, flux, errors, duration = _light_curve(3e-4)

    with pytest.raises(ValueError, match="planet_a_over_rstar_err"):
        _emls(time, flux, errors, duration, planet_a_over_rstar_err=value)


def test_catalog_a_over_rstar_uncertainty_propagates_stellar_errors():
    plain = ExoMoonLeastSquares.catalog_planet_geometry('tois', 1.0, 1.0, 10.0, 4.0, 0.0)
    geometry = ExoMoonLeastSquares.catalog_planet_geometry('tois', 1.0, 1.0, 10.0, 4.0, 0.0,
                                                           star_mass_err=0.09, star_radius_err=0.04)

    a_over_rstar = geometry['planet_semimajor_axis'] * AU_TO_RSUN
    assert plain['planet_a_over_rstar_err'] == 0.0
    assert geometry['planet_a_over_rstar_err'] == pytest.approx(a_over_rstar * 0.05)
    with pytest.raises(ValueError, match="uncertainties"):
        ExoMoonLeastSquares.catalog_planet_geometry('tois', 1.0, 1.0, 10.0, 4.0, 0.0, star_mass_err=-1.0)


def _evaluation_limited(*args, **kwargs):
    result = least_squares(*args, **kwargs)
    result.success, result.status = False, 0
    return result


def test_evaluation_limit_does_not_discard_the_best_candidate(monkeypatch):
    time = np.linspace(-0.7, 0.7, 801)
    truth = _params(0.0, _inclination(0.3))
    flux = batman.TransitModel(truth, time).light_curve(truth)
    monkeypatch.setattr('sherlockpipe.exomoon.least_squares', _evaluation_limited)
    result = ExoMoonLeastSquares._fit_transit_result(
        time, flux, np.full(len(time), 2e-4), [0.02, 0.07], [(-0.2, 0.2), (0.05, 0.11)],
        _params(0.0, _inclination(0.3)), ['t0', 'rp'])

    assert result.success
    assert result.theta[0] == pytest.approx(0.0, abs=1e-3)


def test_evaluation_limit_result_must_still_pass_validity_checks(monkeypatch):
    time = np.linspace(-0.7, 0.7, 801)
    monkeypatch.setattr('sherlockpipe.exomoon.least_squares', _evaluation_limited)
    result = ExoMoonLeastSquares._fit_transit_result(
        time, np.ones_like(time), np.full(len(time), 2e-4), [0.02, 0.07], [(-0.2, 0.2), (0.05, 0.11)],
        _params(0.0, _inclination(0.3)), ['t0', 'rp'])

    assert not result.success


def test_genuine_optimizer_failure_is_not_treated_as_an_evaluation_limit(monkeypatch):
    def failed(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        result.success, result.status = False, -1
        return result

    time = np.linspace(-0.7, 0.7, 801)
    truth = _params(0.0, _inclination(0.3))
    monkeypatch.setattr('sherlockpipe.exomoon.least_squares', failed)

    result = ExoMoonLeastSquares._fit_transit_result(
        time, batman.TransitModel(truth, time).light_curve(truth), np.full(len(time), 2e-4), [0.02, 0.07],
        [(-0.2, 0.2), (0.05, 0.11)], _params(0.0, _inclination(0.3)), ['t0', 'rp'])

    assert result.status == 'optimizer_failure'


def test_strong_ttvs_do_not_smear_the_shared_fit():
    offsets = 0.04 * np.sin(np.arange(len(EPOCHS)))
    time, flux, errors, duration = _light_curve(3e-4, offsets=offsets)
    emls = _emls(time, flux, errors, duration, inc_err=0.3)

    emls.subtract_planet_transit(time, flux, errors)

    assert emls.shared_fit_status.status == 'ok'
    assert emls.shared_planet_params.rp == pytest.approx(RP, rel=0.02)
    np.testing.assert_allclose(emls.local_t0s, EPOCHS + offsets, atol=2e-3)


def test_weak_epochs_are_not_aligned_on_their_noise():
    sigma, rp = 2e-3, 0.03
    time, flux, errors, duration = _light_curve(sigma, rp=rp)
    emls = _emls(time, flux, errors, duration, rp=rp)
    near = np.abs(time - EPOCHS[0]) < 2 * duration
    params = _params(EPOCHS[0], _inclination(0.3), rp=rp)

    offset = emls._fold_timing_offset(time[near], flux[near], errors[near], EPOCHS[0], 2 * duration, params, None)

    assert offset == 0.0


def _shallow_light_curve(target_snr, seed):
    rp = 0.03
    duration = _duration(rp=rp)
    sigma = rp ** 2 * np.sqrt(duration / CADENCE) / target_snr
    return (*_light_curve(sigma, seed=seed, rp=rp), rp)


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_shallow_epochs_are_kept_with_ephemeris_timing(seed):
    time, flux, errors, duration, rp = _shallow_light_curve(2, seed)
    emls = _emls(time, flux, errors, duration, rp=rp, skip_fold_fit=True)

    emls.subtract_planet_transit(time, flux, errors)

    assert len(emls.bary_t0s) == len(EPOCHS)
    sources = [item['timing_source'] for item in emls.epoch_diagnostics]
    assert emls.subtraction_status['ephemeris_timing_epochs'] == sources.count('ephemeris')
    for item in emls.epoch_diagnostics:
        if item['timing_source'] == 'ephemeris':
            assert item['fitted_t0'] == item['epoch']
            assert item['fit_converged'] is False


def test_ephemeris_timing_is_used_only_for_no_improvement(monkeypatch):
    time, flux, errors, duration, rp = _shallow_light_curve(2, 100)
    emls = _emls(time, flux, errors, duration, rp=rp)
    params = _params(0.0, _inclination(0.3), rp=rp)

    def reject(*args):
        raise TransitFitError('insufficient_coverage', 'no coverage')

    monkeypatch.setattr(ExoMoonLeastSquares, 'fit_single_t0', staticmethod(reject))

    with pytest.raises(TransitFitError, match="insufficient_coverage"):
        emls._fit_epoch_t0(time, flux, errors, 0.0, 0.0, 2 * duration, params, None)


def test_detectable_epoch_without_its_transit_is_still_rejected():
    time, flux, errors, duration = _light_curve(3e-4)
    missing = np.abs(time - EPOCHS[4]) < 3 * duration
    flux[missing] = 1 + np.random.default_rng(5).normal(0, 3e-4, np.sum(missing))
    emls = _emls(time, flux, errors, duration)

    emls.subtract_planet_transit(time, flux, errors)

    assert EPOCHS[4] not in emls.bary_t0s
    assert emls.subtraction_status['rejected_epochs'] == 1


# A one-year planet: its Hill sphere reaches far beyond 1.5 transit durations.
WIDE_PERIOD = 365.25
WIDE_A = 215.0
WIDE_EPOCHS = np.arange(6) * WIDE_PERIOD
WIDE_DURATION = WIDE_PERIOD / np.pi * np.arcsin(1.1 / WIDE_A)


def _wide_flux(time, slope=0.0):
    flux = np.ones_like(time)
    for epoch in WIDE_EPOCHS:
        near = np.abs(time - epoch) < 4
        params = _params(epoch, 89.95, a=WIDE_A, rp=0.1, period=WIDE_PERIOD)
        flux[near] = batman.TransitModel(params, time[near]).light_curve(params) * (1 + slope * (time[near] - epoch))
    return flux


def _wide_emls(time, flux, errors):
    return ExoMoonLeastSquares(
        "", 1, 1.0, 1.0, LIMB_DARKENING, 10.9, 0.5, 300.0, WIDE_PERIOD, 0.0, WIDE_DURATION,
        WIDE_A / AU_TO_RSUN, 89.95, 0.05, 0.0, 0.0, 90.0, 0.0, 0.5, 3.0, WIDE_EPOCHS, time, flux, errors, 5.5,
        min_cadences_per_t0=20, fit_rp_per_transit=False, min_transit_completeness=0.1,
        max_subtraction_rms_ratio=100.0)


def _wide_time():
    return np.concatenate([epoch + np.arange(-4, 4, 10 / 1440) for epoch in WIDE_EPOCHS])


def test_moon_orbit_window_matches_the_hill_limit():
    time = _wide_time()
    emls = _wide_emls(time, _wide_flux(time), np.full(len(time), 2e-4))

    planet_mass_sun = 300.0 * 3.003e-6
    hill = WIDE_A * (planet_mass_sun / 3) ** (1 / 3) * 0.5 ** (2 / 3)
    expected = WIDE_PERIOD / np.pi * np.arcsin((1 + hill) / WIDE_A) / 2
    assert emls.moon_orbit_half_window() == pytest.approx(expected, rel=0.02)
    assert emls.moon_orbit_half_window() > 1.5 * WIDE_DURATION


@pytest.mark.parametrize("offset", [1.0, 1.25])
def test_moon_beyond_the_planet_survives_baseline_profiling(offset):
    time = _wide_time()
    errors = np.full(len(time), 2e-4)
    planet = _wide_flux(time) + np.random.default_rng(3).normal(0, 2e-4, len(time))
    moon = sum(5e-4 * np.exp(-0.5 * ((time - (epoch + offset)) / 0.1) ** 2) for epoch in WIDE_EPOCHS)
    reference = _wide_emls(time, planet, errors)
    _, reference_flux, reference_errors = reference.subtract_planet_transit(time, planet, errors)
    _, moon_flux, _ = _wide_emls(time, planet - moon, errors).subtract_planet_transit(time, planet - moon, errors)

    metrics = ExoMoonLeastSquares.signal_transfer_metrics(
        -moon[reference.retained_cadence_indices], moon_flux - reference_flux, reference_errors)

    assert metrics['weighted_transfer_coefficient'] == pytest.approx(1.0, abs=0.01)
    assert metrics['weighted_orthogonal_distortion'] < 0.01


def test_slope_is_still_recovered_beyond_the_moon_window():
    time = _wide_time()
    errors = np.full(len(time), 2e-4)
    flux = _wide_flux(time, slope=4e-4) + np.random.default_rng(3).normal(0, 2e-4, len(time))
    emls = _wide_emls(time, flux, errors)

    emls.subtract_planet_transit(time, flux, errors)

    assert all(item['baseline']['status'] == 'profiled_linear_oot' for item in emls.epoch_diagnostics)
    slopes = [item['baseline']['slope_per_duration'] / WIDE_DURATION for item in emls.epoch_diagnostics]
    assert np.median(slopes) == pytest.approx(4e-4, abs=3e-5)


def test_baseline_falls_back_to_unity_when_the_moon_window_leaves_no_data():
    time = np.linspace(-1.0, 1.0, 201)
    evaluate_time = np.linspace(-0.2, 0.2, 5)

    baseline, status = ExoMoonLeastSquares._profile_local_baseline(
        time, 1.01 + 0 * time, np.full(len(time), 1e-4), 0.0, 0.1, protected_half_window=2.0,
        evaluate_time=evaluate_time)

    np.testing.assert_array_equal(baseline, np.ones(5))
    assert status['status'] == 'disabled_insufficient_two_sided_oot'


def test_baseline_is_evaluated_at_the_requested_times():
    time = np.linspace(-3.0, 3.0, 301)
    flux = 1 + 0.01 * time
    evaluate_time = np.array([-0.2, 0.0, 0.2])

    baseline, status = ExoMoonLeastSquares._profile_local_baseline(
        time, flux, np.full(len(time), 1e-4), 0.0, 0.1, protected_half_window=1.0, evaluate_time=evaluate_time)

    assert status['status'] == 'profiled_linear_oot'
    np.testing.assert_allclose(baseline, 1 + 0.01 * evaluate_time)
