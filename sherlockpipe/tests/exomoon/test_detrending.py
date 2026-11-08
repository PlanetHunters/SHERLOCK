"""Masked detrending, injection before detrending and the per-epoch baseline: audit items A1, A2 and A3.

A1: EMLS detrends the undetrended curve with every moon window masked, so neither the planet nor
a moon shapes the trend, and joins the trend linearly across each mask. A2: an injected moon goes
into the curve before that detrending. A3: each epoch is divided by a straight line fitted outside
the moon window, so a residual slope does not move the planet mid-time.
"""
import tempfile
from unittest import mock

import batman
import numpy as np
import pytest
import wotan

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon import test_planet_fit as fit
from sherlockpipe.tests.exomoon import test_search_integrity as guard

PERIOD, PLANET_RADIUS, MOON_RADIUS, CADENCE, EPOCHS = 30.0, 5.0, 2.5, 2 / 1440, 12
A_AU = ExoMoonLeastSquares.compute_semimajor_axis(1.0, PERIOD)


def transit(radius, t0):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = t0, PERIOD, radius * R_earth_to_R_sun, A_AU * AU_TO_RSUN
    params.inc, params.ecc, params.w, params.u, params.limb_dark = 90.0, 0.0, 90.0, [0.4, 0.2], 'quadratic'
    return params


def audit_fixture(moon_offset, seed, noise=1e-3, trend=None):
    """The audit's A1 fixture: P = 30 d, 5 R_earth planet, 2.5 R_earth moon, 2-min cadence, 12 epochs."""
    t0s = np.arange(EPOCHS) * PERIOD
    time = np.arange(-PERIOD / 2, t0s[-1] + PERIOD / 2, CADENCE)
    planet_params = transit(PLANET_RADIUS, 0.0)
    planet = batman.TransitModel(planet_params, time).light_curve(planet_params)
    t14 = PERIOD / np.pi * np.arcsin((1 + planet_params.rp) / planet_params.a)
    moon_params = transit(MOON_RADIUS, moon_offset * t14)
    moon = batman.TransitModel(moon_params, time).light_curve(moon_params)
    flux = planet + moon - 1 + np.random.default_rng(seed).normal(0, noise, len(time))
    if trend is not None:
        flux = flux * trend(time)
    return t0s, time, flux, planet, moon, t14


def emls_for(t0s, time, flux, t14, planet_mass=3000.0):
    # A heavy planet, so the widest moon window (+-0.55 d) holds a moon at 1.5 T14.
    return ExoMoonLeastSquares(tempfile.mkdtemp(), 1, 1.0, 1.0, [0.4, 0.2], PLANET_RADIUS, 0.5, planet_mass, PERIOD,
                               0.0, t14, A_AU, 90.0, 0.5, 0.0, 0.0, 90.0, 0.0, 0.25, 2.5, list(t0s), time, flux,
                               np.full(len(time), 1e-3), 5.5, period_grid_size=40, radius_grid_size=4,
                               detrend_window=3 * t14)


def amplitude(flat, model):
    deficit = 1 - model
    used = deficit > 0
    return np.sum((1 - flat[used]) * deficit[used]) / np.sum(deficit[used] ** 2)


# --- A1 ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('moon_offset', [0.25, 1.5])
def test_masked_detrending_keeps_the_planet_and_the_moon(moon_offset):
    """Acceptance: at 1000 ppm the moon keeps at least 0.9 and the planet 0.97 of their depth. The
    unmasked biweight of the catalog script (5 durations) keeps 0.24-0.46 of the moon."""
    planets, moons, unmasked = [], [], []
    for seed in range(3):
        t0s, time, flux, planet, moon, t14 = audit_fixture(moon_offset, seed)
        _, flat, _ = emls_for(t0s, time, flux, t14).detrend(time, flux, None)
        planets.append(amplitude(flat / moon, planet))
        moons.append(amplitude(flat / planet, moon))
        coded = wotan.flatten(time, flux, method='biweight', window_length=5 * t14)
        ok = np.isfinite(coded)
        unmasked.append(amplitude(coded[ok] / planet[ok], moon[ok]))

    assert np.mean(planets) >= 0.97
    assert np.mean(moons) >= 0.9
    assert np.mean(unmasked) < 0.6


def test_masked_detrending_removes_stellar_variability():
    variability = lambda time: 1 + 2e-3 * np.sin(2 * np.pi * time / 5.0)
    t0s, time, flux, planet, moon, t14 = audit_fixture(0.25, 0, noise=3e-4, trend=variability)

    _, flat, _ = emls_for(t0s, time, flux, t14).detrend(time, flux, None)

    residual = flat / (planet * moon)
    assert np.std(residual) < 1.2 * 3e-4
    assert amplitude(flat / moon, planet) == pytest.approx(1.0, abs=0.03)


def test_masks_cover_every_moon_window():
    t0s, time, flux, _, _, t14 = audit_fixture(0.25, 0)
    emls = emls_for(t0s, time, flux, t14)
    half = emls.moon_window_half_width()
    captured = {}
    real_flatten = wotan.flatten

    def spy(*args, **kwargs):
        captured['mask'] = kwargs['mask']
        return real_flatten(*args, **kwargs)

    with mock.patch.object(wotan, 'flatten', spy):
        sorted_time = emls.detrend(time, flux, None)[0]
    for t0 in t0s:
        assert np.all(captured['mask'][np.abs(sorted_time - t0) < half])
    assert not np.any(captured['mask'][np.min(np.abs(sorted_time[:, None] - t0s[None, :]), axis=1) > half])


# --- A2 ----------------------------------------------------------------------------------------------------------

def test_injected_moon_is_added_before_detrending(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    emls.detrend_window = 3 * duration
    seen = {}
    real_detrend = ExoMoonLeastSquares.detrend

    def spy(self, time_in, flux_in, flux_err_in):
        if 'flux' not in seen:
            seen['time'], seen['flux'] = np.array(time_in), np.array(flux_in)
        return real_detrend(self, time_in, flux_in, flux_err_in)

    with mock.patch.object(ExoMoonLeastSquares, 'detrend', spy):
        emls.run(inject_params=dict(moon_radius=3.0, moon_period=13.27, moon_mass=5.0, initial_alpha=0.52))

    # The first detrended curve already holds the injected moon.
    assert len(seen['time']) == len(time)
    assert np.min(seen['flux'] - flux) < -5e-4


# --- A3 ----------------------------------------------------------------------------------------------------------

def slope_ttv_rms(slope, seeds=2):
    truth = fit.planet_params()
    rms = []
    for seed in range(seeds):
        signs = np.random.default_rng(seed + 50).choice([-1, 1], 24)
        extra = lambda time, t0: 1 + signs[int(round(t0 / fit.PERIOD))] * slope * (time - t0)
        data = fit.curve(truth, epochs=24, noise=3e-4, seed=seed, extra=extra)
        emls = fit.emls_for(tempfile.mkdtemp(), data, fit.t14(truth), skip_fold_fit=True)
        emls.subtract_planet_transit(*data[1:])
        rms.append(np.sqrt(np.mean(np.square(emls.ttvs))))
    return np.mean(rms)


def test_epoch_slope_does_not_move_the_planet_mid_time():
    """Acceptance: with +-3000 ppm/d per epoch the t0 rms is within 20 % of the no-slope value.
    With the flat median baseline it was 1.27 min against 0.24 min."""
    assert slope_ttv_rms(3e-3) == pytest.approx(slope_ttv_rms(0.0), rel=0.2)


def test_epoch_baseline_does_not_absorb_a_moon(tmp_path):
    truth = fit.planet_params()
    moon_dip = lambda time, t0: 1 - 5e-4 * (np.abs(time - (t0 + 0.4)) < 0.1)
    depths = []
    for slope in (0.0, 3e-3):
        extra = lambda time, t0, slope=slope: moon_dip(time, t0) * (1 + slope * (time - t0))
        t0s, time, flux, flux_err = fit.curve(truth, noise=1e-5, seed=1, extra=extra)
        emls = fit.emls_for(tmp_path, (t0s, time, flux, flux_err), fit.t14(truth), skip_fold_fit=True)
        normalised, _ = emls.normalise_epochs(time, flux, flux_err)
        inside = np.zeros(len(time), dtype=bool)
        around = np.zeros(len(time), dtype=bool)
        for t0 in t0s:
            inside |= np.abs(time - (t0 + 0.4)) < 0.08
            around |= (time - (t0 + 0.4) > 0.12) & (time - (t0 + 0.4) < 0.3)  # after the dip, clear of the planet
        depths.append(np.mean(normalised[around]) - np.mean(normalised[inside]))

    assert depths[1] == pytest.approx(depths[0], rel=0.02)
    assert depths[0] == pytest.approx(5e-4, rel=0.05)
