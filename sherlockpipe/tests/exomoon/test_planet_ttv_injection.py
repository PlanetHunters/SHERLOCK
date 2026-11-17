"""Injections that move the real planet by the moon's TTV (planet_ttv, 2026-11-17).

The planet orbits the planet-moon barycentre, so a moon makes its transits early or late. The
injection used to add only the moon flux, with the real planet left on its own times; with
planet_ttv=True each epoch of the light curve is shifted by the planet's offset that the Pandora
model gives, and the moon is injected at the shifted times.
"""
import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon.test_moon_orbit import JUPITER, system

MINUTE = 1 / 1440
EARTH_MASS_KG = 5.9722e24


def analytic_amplitude(emls, moon_period, moon_mass, planet_mass=JUPITER):
    """a_m M_m / (M_p + M_m) / v_p for a circular, coplanar orbit."""
    g, msun_earth = 6.674e-11, 332946.0
    a_m = (g * (planet_mass + moon_mass) * EARTH_MASS_KG * (moon_period * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    v_p = 2 * np.pi * emls.planet_semimajor_axis * 1.495978707e11 / (emls.planet_period * 86400)
    return a_m * moon_mass / (planet_mass + moon_mass) / v_p / 86400


def wobble_params(emls, moon_period, moon_mass, alpha):
    """Run the injection once to build the Pandora parameters, and return the shifts it applied."""
    time = np.concatenate([t0 + np.linspace(-0.6, 0.6, 2001) for t0 in emls.bary_t0s])
    emls.inject_moon(time, np.ones_like(time), np.full(len(time), 1e-4), emls.bary_t0s, emls.planet_semimajor_axis,
                     0.0, 2.0, moon_period, moon_mass, initial_alpha=alpha, planet_ttv=True)
    return np.array(emls.injected_planet_ttvs)


def test_the_wobble_follows_the_barycentre_formula(tmp_path):
    """A heavy moon (100 M_earth, a 40-min wobble) on a 3.7 d orbit of a Jupiter at 30 d: the shifts
    follow A sin(2 pi t / P_m + phase), with A the analytic amplitude."""
    emls = system(tmp_path, JUPITER, 11.2, 30.0)
    emls.bary_t0s = list(np.arange(40) * 30.0)
    shifts = wobble_params(emls, 3.7, 100.0, 1.0)
    amplitude = analytic_amplitude(emls, 3.7, 100.0)

    t = np.asarray(emls.bary_t0s)
    design = np.column_stack([np.sin(2 * np.pi * t / 3.7), np.cos(2 * np.pi * t / 3.7)])
    coefficients = np.linalg.lstsq(design, shifts, rcond=None)[0]
    assert np.hypot(*coefficients) == pytest.approx(amplitude, rel=0.05)
    assert np.std(shifts - design @ coefficients) < 0.03 * amplitude


def test_an_earth_mass_moon_moves_a_jupiter_by_half_a_minute(tmp_path):
    emls = system(tmp_path, JUPITER, 11.2, 30.0)
    emls.bary_t0s = list(np.arange(40) * 30.0)
    shifts = wobble_params(emls, 3.7, 1.0, 1.0)
    amplitude = analytic_amplitude(emls, 3.7, 1.0)
    assert amplitude == pytest.approx(0.53 * MINUTE, rel=0.05)
    assert 0.9 * amplitude < np.max(np.abs(shifts)) < 1.02 * amplitude


def test_each_epoch_of_the_curve_is_moved_by_its_shift(tmp_path):
    """The real transit in the data moves with the planet: its fitted mid-time changes by the shift."""
    emls = system(tmp_path, JUPITER, 11.2, 30.0)
    params = batman.TransitParams()
    params.per, params.rp, params.a = 30.0, 11.2 * R_earth_to_R_sun, emls.planet_semimajor_axis * AU_TO_RSUN
    params.inc, params.ecc, params.w, params.u, params.limb_dark = 90.0, 0.0, 90.0, [0.4, 0.2], "quadratic"
    time = np.concatenate([t0 + np.arange(-1.0, 1.0, 1 / 1440) for t0 in emls.bary_t0s])
    flux = np.ones_like(time)
    for t0 in emls.bary_t0s:
        params.t0 = t0
        near = np.abs(time - t0) < 1
        flux[near] = batman.TransitModel(params, time[near]).light_curve(params)

    injected = emls.inject_moon(time, flux, np.full(len(time), 1e-4), emls.bary_t0s, emls.planet_semimajor_axis, 0.0,
                                0.01, 3.7, 100.0, initial_alpha=1.0, planet_ttv=True)
    moved = emls.injected_time
    assert np.all(np.diff(moved) >= 0)
    for t0, shift in zip(emls.bary_t0s, emls.injected_planet_ttvs):
        near = np.abs(moved - t0) < 1
        weights = 1 - injected[near]
        centre = np.sum(moved[near] * weights) / np.sum(weights)
        assert centre - t0 == pytest.approx(shift, abs=0.05 * MINUTE)
    assert np.max(np.abs(emls.injected_planet_ttvs)) > 5 * MINUTE


def test_without_planet_ttv_the_curve_keeps_its_times(tmp_path):
    emls = system(tmp_path, JUPITER, 11.2, 30.0)
    time = np.concatenate([t0 + np.linspace(-0.6, 0.6, 2001) for t0 in emls.bary_t0s])
    emls.inject_moon(time, np.ones_like(time), np.full(len(time), 1e-4), emls.bary_t0s, emls.planet_semimajor_axis,
                     0.0, 2.0, 3.7, 100.0, initial_alpha=1.0)
    assert emls.injected_time is None and emls.injected_planet_ttvs is None


def test_a_barycentre_ttv_moves_planet_and_moon_together(tmp_path):
    """A companion's TTV moves the whole planet-moon system: the real transit and the injected moon dip
    are both shifted by the barycentre offset of their epoch."""
    emls = system(tmp_path, JUPITER, 11.2, 30.0)
    params = batman.TransitParams()
    params.per, params.rp, params.a = 30.0, 11.2 * R_earth_to_R_sun, emls.planet_semimajor_axis * AU_TO_RSUN
    params.inc, params.ecc, params.w, params.u, params.limb_dark = 90.0, 0.0, 90.0, [0.4, 0.2], "quadratic"
    time = np.concatenate([t0 + np.arange(-1.0, 1.0, 1 / 1440) for t0 in emls.bary_t0s])
    flux = np.ones_like(time)
    for t0 in emls.bary_t0s:
        params.t0 = t0
        near = np.abs(time - t0) < 1
        flux[near] = batman.TransitModel(params, time[near]).light_curve(params)
    kwargs = dict(initial_alpha=1.0)

    emls.inject_moon(time, flux.copy(), np.full(len(time), 1e-4), emls.bary_t0s, emls.planet_semimajor_axis, 0.0,
                     2.0, 3.7, 1e-6, **kwargs)
    still = np.asarray(emls.injected_flux_moon_pandora)
    bary = dict(amplitude=40 * MINUTE, period=100.0, phase=0.3)
    emls.inject_moon(time, flux.copy(), np.full(len(time), 1e-4), emls.bary_t0s, emls.planet_semimajor_axis, 0.0,
                     2.0, 3.7, 1e-6, barycentre_ttv=bary, **kwargs)
    moved_time, moved_moon = emls.injected_time, np.asarray(emls.injected_flux_moon_pandora)

    for t0, shift in zip(emls.bary_t0s, emls.injected_barycentre_ttvs):
        assert shift == pytest.approx(40 * MINUTE * np.sin(2 * np.pi * t0 / 100.0 + 0.3))
        near = np.abs(time - t0) < 1
        near_moved = np.abs(moved_time - t0) < 1
        centre = lambda t, f: np.sum(t * (1 - f)) / np.sum(1 - f)
        assert centre(moved_time[near_moved], moved_moon[near_moved]) - centre(time[near], still[near]) == \
            pytest.approx(shift, abs=0.2 * MINUTE)
