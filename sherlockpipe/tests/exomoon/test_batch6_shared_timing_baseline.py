import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares


EPOCHS = np.array([0.0, 5.0, 10.0])


def _params(t0, a=12.0, rp=0.08):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = 5.0
    params.rp = rp
    params.a = a
    params.inc = 89.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.2, 0.1]
    params.limb_dark = "quadratic"
    return params


def _planet_flux(time, centers):
    flux = np.ones_like(time)
    for center, reference_epoch in zip(centers, EPOCHS):
        params = _params(center)
        local = np.abs(time - reference_epoch) < 1.0
        flux[local] = batman.TransitModel(params, time[local]).light_curve(params)
    return flux


def _fixture(centers=EPOCHS, slope=0.0, errors=(2e-4, 2e-4, 2e-4), moon_offset=None):
    local_time = np.arange(-0.65, 0.651, 0.005)
    time = np.concatenate([center + local_time for center in EPOCHS])
    nearest = EPOCHS[np.argmin(np.abs(time[:, None] - EPOCHS), axis=1)]
    flux = _planet_flux(time, centers) * (1 + slope * (time - nearest))
    error = np.concatenate([np.full(len(local_time), value) for value in errors])
    if moon_offset is not None:
        # An independent, smooth injected deficit avoids using the planet
        # template as its own preservation oracle.
        moon = np.zeros_like(time)
        for center in centers:
            moon += 8e-4 * np.exp(-0.5 * ((time - (center + moon_offset)) / 0.025) ** 2)
        flux -= moon
    return time, flux, error


def _emls(time, flux, errors):
    # The catalog radius is deliberately modestly low but has a supplied
    # uncertainty, so the shared fit may correct it while local epochs never
    # acquire their own depth freedom.
    return ExoMoonLeastSquares(
        "", 1, 1.0, 1.0, [0.2, 0.1], 8.5, 0.8, 100.0, 5.0,
        0.0, 0.30, 12.0 / 215.032, 89.0, 0.1, 0.0, 0.1, 90.0, 0.0,
        0.3, 0.6, EPOCHS, time, flux, errors, 5.5, min_cadences_per_t0=20,
        fit_rp_per_transit=True, min_transit_completeness=0.1, max_subtraction_rms_ratio=100.0,
    )


def _subtract(time, flux, errors):
    emls = _emls(time, flux, errors)
    result = emls.subtract_planet_transit(time, flux, errors)
    return emls, result


def test_shared_shape_then_local_timing_recovers_no_ttv_and_known_ttv():
    for offsets in (np.zeros(3), np.array([-0.035, 0.0, 0.028])):
        time, flux, errors = _fixture(EPOCHS + offsets)
        emls, _ = _subtract(time, flux, errors)

        # The shared fit is attempted first; B2 may conservatively retain the
        # catalog shape when a grazing geometry reaches a supplied bound.
        assert emls.shared_fit_status.status in ('ok', 'active_bound')
        assert hasattr(emls, 'shared_planet_params')
        np.testing.assert_allclose(emls.local_t0s, EPOCHS + offsets, atol=8e-4)
        np.testing.assert_allclose(emls.ttvs, -offsets, atol=8e-4)
        assert all(item['timing_offset_days'] == pytest.approx(offset, abs=8e-4)
                   for item, offset in zip(emls.epoch_diagnostics, offsets))


def test_profiled_baseline_requires_two_sided_oot_and_records_slope():
    time, flux, errors = _fixture(slope=0.004)
    emls, _ = _subtract(time, flux, errors)

    diagnostics = emls.epoch_diagnostics
    assert all(item['baseline']['status'] == 'profiled_linear_oot' for item in diagnostics)
    # Slope is expressed per fitted physical transit duration, not per day.
    assert np.median([item['baseline']['slope_per_duration'] for item in diagnostics]) / emls.planet_duration == pytest.approx(
        0.004, abs=1e-5)
    baseline, status = ExoMoonLeastSquares._profile_local_baseline(
        np.linspace(-0.4, 0.4, 41), np.ones(41), np.full(41, 1e-4), 0.0, 0.30)
    np.testing.assert_array_equal(baseline, np.ones(41))
    assert status['status'] == 'disabled_insufficient_two_sided_oot'


def test_heterogeneous_epoch_snr_and_paired_moon_transfer_are_recordable():
    rng = np.random.default_rng(117)
    time, planet, errors = _fixture(errors=(1e-4, 8e-4, 3e-4))
    noisy_planet = planet + rng.normal(0, errors)
    emls, (_, residual_planet, residual_errors) = _subtract(time, noisy_planet, errors)

    assert len(emls.local_t0s) == len(EPOCHS)
    assert all(np.isfinite(item['weighted_residual_rms']) for item in emls.epoch_diagnostics)
    assert all(item['baseline']['status'] == 'profiled_linear_oot' for item in emls.epoch_diagnostics)

    for moon_offset in (0.03, 0.42):
        _, injected, injected_errors = _fixture(errors=(1e-4, 8e-4, 3e-4), moon_offset=moon_offset)
        noisy_injected = injected + (noisy_planet - planet)
        _, (_, residual_injected, output_errors) = _subtract(time, noisy_injected, injected_errors)
        metrics = ExoMoonLeastSquares.signal_transfer_metrics(
            injected - planet, residual_injected - residual_planet, output_errors)
        # Timing is intentionally permitted, so a strongly overlapping moon
        # can project onto a timing derivative. The metric makes that loss
        # explicit; the separate moon is protected from baseline/depth fitting.
        assert metrics['weighted_transfer_coefficient'] > (0.30 if moon_offset < 0.1 else 0.70)
        assert metrics['weighted_orthogonal_distortion'] < 0.55
        assert len(output_errors) == len(residual_errors)
        assert np.all(np.isfinite(output_errors)) and np.all(output_errors > 0)
