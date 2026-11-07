"""Regression tests for the small EMLS audit items G2, H4, I2, G15, E4 and B7.

Each test fails on the code before its fix. IDs refer to the local audit list.
"""
from unittest import mock

import batman
import numpy as np
import pandoramoon as pandora
import pytest
from astropy import constants as ac

import sherlockpipe.exomoon as exomoon
from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, M_earth_to_M_sun, R_earth_to_R_sun
from sherlockpipe.tests.exomoon import test_search_integrity as guard
from sherlockpipe.tests.exomoon.test_audit_fixes import LD, PERIOD, SMA, angular_error, run_to_search, system

ALPHA_STEP = 2 * np.pi / 25


class SerialPool:
    """Runs the search in this process, so patched methods apply."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap(self, function, inputs, chunksize=1):
        return map(function, inputs)


# --- G2: maximum moon offset -------------------------------------------------------------------------------------

def pandora_crossing(b, moon_period, planet_period=300.0, planet_mass=1.0):
    """Time after the planet mid-transit at which a moon at greatest elongation crosses x = 0."""
    a_rstar = ExoMoonLeastSquares.compute_semimajor_axis(1.0, planet_period) * AU_TO_RSUN
    params = pandora.model_params()
    params.R_star = ac.R_sun.value
    params.u1, params.u2 = LD
    params.per_bary, params.a_bary, params.r_planet, params.b_bary = planet_period, a_rstar, 0.01, b
    params.t0_bary, params.t0_bary_offset = 0.0, 0.0
    params.M_planet, params.w_bary, params.ecc_bary = planet_mass * ac.M_earth.value, 0, 0
    params.r_moon, params.per_moon = 0.003, moon_period
    params.tau_moon = (0.0 / moon_period - np.pi / (2 * np.pi)) % 1
    params.Omega_moon, params.i_moon, params.ecc_moon, params.w_moon = 0, 90, 0, 0
    params.M_moon = 1e-6 * ac.M_earth.value
    params.epoch_distance, params.supersampling_factor = planet_period, 1
    params.occult_small_threshold, params.hill_sphere_threshold = 0.1, 1
    a_moon = ExoMoonLeastSquares.compute_semimajor_axis(planet_mass * M_earth_to_M_sun, moon_period) * AU_TO_RSUN
    span = 1.5 * a_moon * planet_period / (2 * np.pi * a_rstar)
    time = np.linspace(0, span, 400001)
    moon_x = pandora.moon_model(params).coordinates(time)[2]
    crossing = np.where(np.sign(moon_x[:-1]) != np.sign(moon_x[1:]))[0][0]
    return time[crossing], a_moon, a_rstar


@pytest.mark.parametrize('b', [0.0, 0.5, 0.8, 0.9])
@pytest.mark.parametrize('moon_period', [30.0, 80.0])
def test_moon_offset_matches_pandora(b, moon_period):
    crossing, a_moon, a_rstar = pandora_crossing(b, moon_period)

    offset = ExoMoonLeastSquares.compute_max_moon_offset(a_moon, a_rstar, 300.0)

    assert offset == pytest.approx(crossing, rel=0.01)


@pytest.mark.parametrize('b', [0.0, 0.5, 0.8])
def test_search_hypotheses_use_the_moon_offset(tmp_path, b):
    emls = system(tmp_path)
    emls.planet_impact_param = b

    inputs = run_to_search(emls)

    a_planet = SMA * AU_TO_RSUN
    for item in inputs[::97]:
        tau = ExoMoonLeastSquares.compute_moon_ts(0.0, item.half_moon_orbit_transit_duration,
                                                  item.half_moon_transit_duration)
        a_moon = ExoMoonLeastSquares.compute_semimajor_axis(300.0 * M_earth_to_M_sun, item.moon_period) * AU_TO_RSUN
        assert tau == pytest.approx(a_moon * PERIOD / (2 * np.pi * a_planet), rel=1e-9)


def test_data_window_holds_the_widest_moon_transit(tmp_path):
    emls = system(tmp_path)
    emls.planet_impact_param = 0.5

    inputs = run_to_search(emls)

    widest = max(inputs, key=lambda item: item.moon_period)
    tau = ExoMoonLeastSquares.compute_moon_ts(0.0, widest.half_moon_orbit_transit_duration,
                                              widest.half_moon_transit_duration)
    assert widest.half_moon_max_orbit_transit_duration >= tau + widest.half_moon_transit_duration - 1e-12


# --- H4: depth against the out-of-window level -------------------------------------------------------------------

def test_baseline_offset_does_not_create_signal(tmp_path):
    """A constant offset contains no moon. The planet stage is bypassed, as a planet fit with a
    unit baseline turns an offset into structure (audit item A3), which is not what H4 covers."""
    snrs = {}
    for offset in (0.0, -2e-4):
        values = []
        for seed in range(6):
            t0s = np.arange(12) * PERIOD
            time = np.concatenate([t0 + np.arange(-2.0, 2.0, 2 / 1440) for t0 in t0s])
            flux = 1 + np.random.default_rng(5000 + seed).normal(0, 1e-3, len(time)) + offset
            curve = (t0s, time, flux, np.full(len(time), 1e-3), guard_duration())
            emls = system(tmp_path, curve=curve, grid=200)
            with mock.patch.object(ExoMoonLeastSquares, 'subtract_planet_transit', keep_light_curve):
                emls.run()
            values.append(emls.max_snr)
        snrs[offset] = np.median(values)

    assert abs(snrs[-2e-4] - snrs[0.0]) < 0.5, snrs


def keep_light_curve(self, time, flux, flux_err, mode='subtract', plot=False):
    return time, flux, flux_err


def guard_duration():
    rp = 10.0 * R_earth_to_R_sun
    return PERIOD / np.pi * np.arcsin((1 + rp) / (SMA * AU_TO_RSUN))


def test_snr_on_noise_has_no_clipping_bias(tmp_path):
    emls = system(tmp_path)
    inputs = run_to_search(emls)
    hypothesis = inputs[len(inputs) // 2]
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, 5 / 86400) for t0 in emls.bary_t0s])
    emls.time = time
    emls.flux_err = np.full(len(time), 1e-3)
    snrs = []
    for seed in range(150):
        emls.flux = 1 + np.random.default_rng(seed).normal(0, 1e-3, len(time))
        snrs.append(emls.search(hypothesis)[3])

    in_window = emls.search(hypothesis, return_lc=True)[5]
    assert len(in_window) > 2e4
    assert abs(np.mean(snrs)) < 2 * np.std(snrs) / np.sqrt(len(snrs))


# --- G15: alpha reference epoch ----------------------------------------------------------------------------------

def test_alpha_keeps_its_reference_when_the_first_epoch_is_missing(tmp_path):
    t0s, time, flux, flux_err, duration, params = guard.planet_curve(3)
    flux = guard.with_moon(flux, time, t0s, params, 3.0)
    keep = np.abs(time - t0s[0]) > 3.5
    emls = guard.build(flux[keep], time[keep], flux_err[keep], tmp_path, t0s, duration)
    assert emls.bary_t0s[0] == t0s[1]

    emls.run()

    assert abs(emls.recovered_period - guard.moon_period()) / guard.moon_period() < 0.01
    assert angular_error(emls.recovered_alpha, guard.MOON_ALPHA) < ALPHA_STEP
    true_times = guard_moon_times(t0s)[1:]
    assert np.median(np.abs(np.array(emls.recovered_t0s) - true_times)) < 0.05


def guard_moon_times(t0s):
    """Moon mid-times the guard injection uses, referenced to the first catalog epoch."""
    a_rsun = guard.PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN
    b = a_rsun * np.cos(np.deg2rad(guard.PLANET_INC))
    period = guard.moon_period()
    a_moon = ExoMoonLeastSquares.compute_semimajor_axis(guard.PLANET_MASS * M_earth_to_M_sun, period) * AU_TO_RSUN
    orbit = ExoMoonLeastSquares.compute_moon_orbit_transit_duration(1.0, a_rsun, guard.PLANET_PERIOD, a_moon,
                                                                    4.0 * R_earth_to_R_sun, b)
    transit = ExoMoonLeastSquares.compute_transit_duration(1.0, a_rsun, guard.PLANET_PERIOD, 4.0 * R_earth_to_R_sun, b)
    tau = ExoMoonLeastSquares.compute_moon_ts(0.0, orbit / 2, transit / 2)
    return np.array([ExoMoonLeastSquares.compute_moon_t0(t0s[0], t0, period, guard.MOON_ALPHA, tau)[0] for t0 in t0s])


# --- I2: injected and recovered moon times paired by epoch -------------------------------------------------------

def test_epoch_without_injected_dip_keeps_the_pairing(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(301)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    real_inject = ExoMoonLeastSquares.inject_moon

    def inject_without_second_dip(self, *args, **kwargs):
        result = real_inject(self, *args, **kwargs)
        # The data keep the dip, but the stored truth shows none in the second epoch.
        window = np.abs(self.time - self.bary_t0s[1]) < 3.5
        self.injected_flux_moon_pandora[window] = 1.0
        return result

    with mock.patch.object(ExoMoonLeastSquares, 'inject_moon', inject_without_second_dip):
        emls.run(inject_params=dict(moon_radius=3.0, moon_period=13.27, moon_mass=5.0, initial_alpha=0.52))

    assert emls.injected_t0s[1] is None
    assert len(emls.injected_t0s) == len(emls.recovered_t0s)
    diffs = [abs(rec - inj) for rec, inj in zip(emls.recovered_t0s, emls.injected_t0s) if inj is not None]
    assert np.median(diffs) < 0.1
    assert emls.recovered


# --- E4: NaN SNR in one alpha bin --------------------------------------------------------------------------------

def test_refine_alpha_ignores_a_nan_cell():
    alpha = np.linspace(0, 2 * np.pi - ALPHA_STEP, 25)
    column = (1 - np.cos(alpha - alpha[10]))[:, None]
    column[0, 0] = np.nan

    refined = ExoMoonLeastSquares.refine_alpha(alpha, column)

    assert refined[0, 0] == pytest.approx(alpha[10], abs=1e-6)


def test_refine_alpha_does_not_fit_through_a_nan_neighbour():
    alpha = np.linspace(0, 2 * np.pi - ALPHA_STEP, 25)
    column = (1 - np.cos(alpha - alpha[10]))[:, None]
    column[11, 0] = np.nan

    refined = ExoMoonLeastSquares.refine_alpha(alpha, column)

    assert np.isfinite(refined[0, 0])


def test_nan_snr_in_the_first_alpha_bin_keeps_the_period(tmp_path):
    t0s, time, flux, flux_err, duration, params = guard.planet_curve(3)
    flux = guard.with_moon(flux, time, t0s, params, 3.0)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    real_search = ExoMoonLeastSquares.search

    def search_with_empty_first_bin(self, search_input, return_lc=False):
        result = real_search(self, search_input, return_lc)
        if search_input.moon_alpha == 0.0 and not return_lc:
            return result[:3] + (np.nan,) + result[4:]
        return result

    with mock.patch.object(exomoon, 'Pool', SerialPool), \
            mock.patch.object(ExoMoonLeastSquares, 'search', search_with_empty_first_bin):
        emls.run()

    assert np.isfinite(emls.recovered_alpha)
    assert abs(emls.recovered_period - guard.moon_period()) / guard.moon_period() < 0.01
    assert angular_error(emls.recovered_alpha, guard.MOON_ALPHA) < ALPHA_STEP


# --- B7: T14 from model contact times ----------------------------------------------------------------------------

def analytic_t14(params):
    b = params.a * np.cos(np.deg2rad(params.inc))
    return params.per / np.pi * np.arcsin(np.sqrt((1 + params.rp) ** 2 - b ** 2) / (params.a * np.sin(np.deg2rad(params.inc))))


@pytest.mark.parametrize('inc', [90.0, 89.5])
def test_model_duration_matches_the_contact_times(inc):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = 0.0, PERIOD, 0.09, SMA * AU_TO_RSUN, inc
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, LD, 'quadratic'

    duration = ExoMoonLeastSquares.model_transit_duration(params, 0.5)

    assert duration == pytest.approx(analytic_t14(params), rel=1e-3)


def test_fitted_duration_does_not_depend_on_cadence(tmp_path):
    cadence = 30 / 1440
    t0s = np.arange(6) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, cadence) for t0 in t0s])
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = 0.0, PERIOD, 10.0 * R_earth_to_R_sun, SMA * AU_TO_RSUN, 89.9
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, LD, 'quadratic'
    flux = batman.TransitModel(params, time).light_curve(params) + np.random.default_rng(4).normal(0, 2e-5, len(time))
    curve = (t0s, time, flux, np.full(len(time), 2e-5), analytic_t14(params))
    emls = system(tmp_path, curve=curve, min_cadences_per_t0=5)

    run_to_search(emls)

    assert emls.planet_duration == pytest.approx(analytic_t14(params), rel=0.01)
