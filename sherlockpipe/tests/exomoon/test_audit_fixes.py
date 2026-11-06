"""Regression tests for the small defects found in the EMLS audit.

Each test fails on the code before its fix. IDs refer to the local audit list.
"""
import logging
import os
from unittest import mock

import batman
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
from astropy import constants as ac

import sherlockpipe.exomoon as exomoon
from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, M_earth_to_M_sun, R_earth_to_R_sun

PERIOD = 20.0
EPOCHS = 6
SMA = 0.1436
LD = [0.3, 0.2]
PLANET_RADIUS = 10.0
ALPHA_GRID = np.linspace(0, 2 * np.pi - 2 * np.pi / 25, 25)


def light_curve(seed=5, noise=1e-4, shifts=None, epochs=EPOCHS, inc=89.9, ecc=0.0, w=90.0):
    t0s = np.arange(epochs) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, 4 / 1440) for t0 in t0s])
    params = batman.TransitParams()
    params.per = PERIOD
    params.rp = PLANET_RADIUS * R_earth_to_R_sun
    params.a = SMA * AU_TO_RSUN
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.u = LD
    params.limb_dark = 'quadratic'
    flux = np.ones_like(time)
    for i, t0 in enumerate(t0s):
        window = np.abs(time - t0) < 3
        params.t0 = t0 + (shifts[i] if shifts is not None else 0.0)
        flux[window] = batman.TransitModel(params, time[window]).light_curve(params)
    flux = flux + np.random.default_rng(seed).normal(0, noise, len(time))
    duration = PERIOD / np.pi * np.arcsin((1 + params.rp) / params.a)
    return t0s, time, flux, np.full(len(time), noise), duration


def system(tmp_path, curve=None, radius_err=0.5, inc=89.9, ecc=0.0, w=90.0, grid=40, **kwargs):
    t0s, time, flux, flux_err, duration = curve if curve is not None else light_curve()
    return ExoMoonLeastSquares(
        str(tmp_path), 1, 1.0, 1.0, LD, PLANET_RADIUS, radius_err, 300.0, PERIOD, 0.0, duration, SMA, inc, 0.5, ecc,
        0.0, w, 0.0, 1.0, 4.0, t0s, time, flux, flux_err, 5.5, period_grid_size=grid, radius_grid_size=4, **kwargs)


class StopAtSearch(Exception):
    pass


class CapturePool:
    """Stand-in for multiprocessing.Pool that keeps the search inputs and stops the run."""
    inputs = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap(self, function, inputs, chunksize=1):
        CapturePool.inputs = list(inputs)
        raise StopAtSearch


def run_to_search(emls):
    """Run all the setup of EMLS and return the search inputs it builds."""
    with mock.patch.object(exomoon, 'Pool', CapturePool), pytest.raises(StopAtSearch):
        emls.run()
    return CapturePool.inputs


def well(alpha0):
    """A smooth minimum at alpha0 on the alpha grid, as one residual column."""
    return (1 - np.cos(ALPHA_GRID - alpha0))[:, None]


def angular_error(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


# --- G1: first epoch stacked twice -------------------------------------------------------------------------------

def scenario_arguments(emls, search_input):
    return (emls.time, emls.flux, emls.flux_err, emls.bary_t0s[0], search_input.moon_alpha, search_input.moon_period,
            emls.bary_t0s, search_input.half_moon_orbit_transit_duration,
            search_input.half_moon_max_orbit_transit_duration, search_input.half_moon_transit_duration)


def test_each_epoch_enters_a_scenario_once(tmp_path):
    emls = system(tmp_path)
    inputs = run_to_search(emls)

    scenarios = ExoMoonLeastSquares.compute_moon_transit_scenarios(*scenario_arguments(emls, inputs[0]))

    assert len(scenarios) == EPOCHS
    pooled = np.concatenate([scenario[2] for scenario in scenarios])
    assert len(np.unique(pooled)) == len(pooled)


def test_no_epoch_with_data_gives_an_empty_scenario_list():
    empty = np.array([])

    scenarios = ExoMoonLeastSquares.compute_moon_transit_scenarios(
        empty, empty, empty, 0.0, 0.0, 5.0, [0.0], 0.1, 0.2, 0.05)

    assert scenarios == []


def test_search_scores_a_hypothesis_without_data_as_no_detection(tmp_path):
    emls = system(tmp_path)
    inputs = run_to_search(emls)
    emls.time, emls.flux, emls.flux_err = np.array([]), np.array([]), np.array([])

    residual, baseline, _, snr, unusable = emls.search(inputs[0])

    assert residual == baseline == ExoMoonLeastSquares.MAX_RESIDUAL
    assert np.isnan(snr)
    assert not unusable


def test_same_dip_counts_the_same_in_the_first_and_last_epoch(tmp_path):
    def snr(epoch):
        t0s, time, flux, flux_err, duration = light_curve(noise=2e-4)
        flux = flux.copy()
        flux[np.abs(time - t0s[epoch]) < 0.15] -= 3e-3
        emls = system(tmp_path, (t0s, time, flux, flux_err, duration), skip_fold_fit=True)
        inputs = run_to_search(emls)
        return emls.search(inputs[0])[3]

    assert snr(0) == pytest.approx(snr(EPOCHS - 1), rel=0.25)


# --- H16, H17: alpha refinement ----------------------------------------------------------------------------------

@pytest.mark.parametrize('alpha0', [1.1, 3.0, 3.05, 5.0])
def test_refine_alpha_finds_an_interior_minimum(alpha0):
    refined = ExoMoonLeastSquares.refine_alpha(ALPHA_GRID, well(alpha0))[0, 0]

    assert angular_error(refined, alpha0) < 0.01


def test_refine_alpha_does_not_depend_on_the_constant_level():
    low = ExoMoonLeastSquares.refine_alpha(ALPHA_GRID, well(3.05))[0, 0]
    high = ExoMoonLeastSquares.refine_alpha(ALPHA_GRID, well(3.05) - 20.0)[0, 0]

    assert high == pytest.approx(low, abs=1e-9)


@pytest.mark.parametrize('alpha0', [0.1, -0.1, 2 * np.pi - 0.2])
def test_refine_alpha_works_across_the_seam(alpha0):
    refined = ExoMoonLeastSquares.refine_alpha(ALPHA_GRID, well(alpha0))[0, 0]

    assert angular_error(refined, alpha0) < 0.01


def test_refine_alpha_stays_on_the_grid_without_a_minimum():
    flat = np.ones((len(ALPHA_GRID), 1))

    assert ExoMoonLeastSquares.refine_alpha(ALPHA_GRID, flat)[0, 0] == ALPHA_GRID[0]


def test_refine_alpha_moves_at_most_one_grid_step():
    column = np.ones(len(ALPHA_GRID))
    column[10:13] = [0.5, 0.2, 0.2000001]

    refined = ExoMoonLeastSquares.refine_alpha(ALPHA_GRID, column[:, None])[0, 0]

    assert abs(refined - ALPHA_GRID[11]) <= ALPHA_GRID[1]


# --- H2: null and moon objectives --------------------------------------------------------------------------------

def residuals_on_template(depth):
    template_time = np.linspace(-0.2, 0.2, 401)
    template = 1 - 0.01 * (np.abs(template_time) < 0.1)
    it_time = np.linspace(-0.09, 0.09, 50)
    it_flux = np.full(50, 1 - depth)
    oot_flux = np.full(500, 1.0) + np.random.default_rng(1).normal(0, 1e-3, 500)
    errors = np.full(50, 1e-3), np.full(500, 1e-3)
    return ExoMoonLeastSquares.calculate_residuals(
        it_time, it_flux, errors[0], np.linspace(1, 2, 500), oot_flux, errors[1], float(np.mean(it_flux)), 50,
        [(1.0, template_time, template)], 0.1, 100.0, 1.0, 1), it_flux, oot_flux


def test_null_and_moon_models_use_the_same_data_weights():
    (_, baseline, *_), it_flux, oot_flux = residuals_on_template(1e-3)

    expected = np.sum(((it_flux - 1) / 1e-3) ** 2) + np.sum(((oot_flux - 1) / 1e-3) ** 2)

    assert baseline == pytest.approx(expected)


@pytest.mark.parametrize('depth', [3e-4, 1e-3, 1e-2])
def test_a_perfect_template_beats_the_null_at_any_depth(depth):
    (residual, baseline, *_), _, _ = residuals_on_template(depth)

    assert residual / baseline < 1


# --- H5: spectrum kernel -----------------------------------------------------------------------------------------

def test_spectrum_uses_the_stated_kernel():
    chi2 = 1000 + np.linspace(0, 200, 2000)

    power = ExoMoonLeastSquares.spectra(chi2, 1, 31)[2]

    assert abs(np.corrcoef(power, np.arange(2000))[0, 1]) < 0.2


def test_run_passes_the_kernel_as_kernel_size(tmp_path):
    calls = []
    original = ExoMoonLeastSquares.spectra

    def spy(chi2, oversampling_factor=1, kernel_size=30):
        calls.append(oversampling_factor * kernel_size)
        return original(chi2, oversampling_factor, kernel_size)

    with mock.patch.object(ExoMoonLeastSquares, 'spectra', staticmethod(spy)):
        system(tmp_path).run()

    assert calls == [31]


# --- E1: mask mode errors ----------------------------------------------------------------------------------------

def test_mask_mode_keeps_each_cadence_with_its_own_error(tmp_path):
    t0s, time, flux, _, duration = light_curve()
    flux_err = 1e-4 * (1 + np.arange(len(time)) % 7)
    sigma = dict(zip(time, flux_err))
    emls = system(tmp_path, (t0s, time, flux, flux_err, duration), subtract_mode='mask')

    run_to_search(emls)

    assert len(emls.flux_err) == len(emls.time) == len(emls.flux)
    assert np.array_equal(emls.flux_err, [sigma[t] for t in emls.time])


# --- I5: injection side effects ----------------------------------------------------------------------------------

MOON = dict(moon_radius=2.5, moon_period=2.3, moon_mass=5.0, initial_alpha=1.0)


def test_injection_does_not_fit_the_planet(tmp_path):
    emls = system(tmp_path)
    original = ExoMoonLeastSquares.subtract_planet_transit
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    with mock.patch.object(ExoMoonLeastSquares, 'subtract_planet_transit', spy), \
            mock.patch.object(exomoon, 'Pool', CapturePool), pytest.raises(StopAtSearch):
        emls.run(inject_params=MOON)

    assert len(calls) == 1


def test_injected_and_null_runs_start_the_fit_from_the_same_state(tmp_path):
    emls = system(tmp_path)
    duration, epochs = emls.planet_duration, list(emls.bary_t0s)

    emls.inject_moon(emls.time, emls.flux, emls.flux_err, emls.bary_t0s, SMA, 0.0, MOON['moon_radius'],
                     MOON['moon_period'], MOON['moon_mass'], initial_alpha=1.0, inject_planet=False, inject_moon=True)

    assert emls.planet_duration == duration
    assert emls.bary_t0s == epochs


# --- G8, G7: moon period limits ----------------------------------------------------------------------------------

def test_inner_limit_keeps_the_moon_outside_an_inflated_planet():
    radius, mass, max_moon = 14.56, 0.3 / 5.51 * 14.56 ** 3, 4.0

    period = ExoMoonLeastSquares.compute_min_moon_period(5.5, mass, radius, max_moon)

    a_rsun = ExoMoonLeastSquares.compute_semimajor_axis(mass * M_earth_to_M_sun, period) * AU_TO_RSUN
    assert a_rsun >= (radius + max_moon) * R_earth_to_R_sun * 0.999


def test_inner_limit_stays_at_roche_for_a_dense_planet():
    period = ExoMoonLeastSquares.compute_min_moon_period(5.5, 300.0, 10.0, 4.0)

    assert period == ExoMoonLeastSquares.compute_roche_period(5.5)


def test_reversed_period_grid_is_rejected():
    with pytest.raises(ValueError, match='moon period'):
        ExoMoonLeastSquares.compute_moon_period_grid(0.2235, 0.1443, mode='log', samples=10)


# --- A4: planet radius errors ------------------------------------------------------------------------------------

def test_nan_radius_error_gets_a_logged_default(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        emls = system(tmp_path, radius_err=float('nan'))

    assert emls.planet_radius_err == pytest.approx(0.1 * PLANET_RADIUS)
    assert 'radius error' in caplog.text


def test_nan_radius_error_does_not_freeze_rp(tmp_path):
    curve = light_curve()
    fits = []
    original = ExoMoonLeastSquares.fit_single_transit

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        fits.append((args[3][3], result[0][3]))
        return result

    emls = system(tmp_path, curve, radius_err=float('nan'))
    emls.planet_radius = PLANET_RADIUS * 1.05
    with mock.patch.object(ExoMoonLeastSquares, 'fit_single_transit', staticmethod(spy)):
        emls.subtract_planet_transit(curve[1], curve[2], curve[3])

    assert fits[0][1] != fits[0][0]


def test_radius_error_above_the_radius_keeps_rp_positive(tmp_path):
    curve = light_curve()
    bounds = []
    original = ExoMoonLeastSquares.fit_single_transit

    def spy(*args, **kwargs):
        bounds.append(args[4][3])
        return original(*args, **kwargs)

    emls = system(tmp_path, curve, radius_err=2 * PLANET_RADIUS)
    with mock.patch.object(ExoMoonLeastSquares, 'fit_single_transit', staticmethod(spy)):
        emls.subtract_planet_transit(curve[1], curve[2], curve[3])

    assert bounds[0][0] > 0


# --- D3, I9: unusable input --------------------------------------------------------------------------------------

def test_no_epochs_gives_an_explicit_error(tmp_path):
    t0s, time, flux, flux_err, duration = light_curve()
    emls = system(tmp_path, (np.array([]), time, flux, flux_err, duration))

    with pytest.raises(exomoon.NoUsableEpochsError):
        emls.run()


def test_all_epochs_rejected_gives_the_same_error(tmp_path):
    emls = system(tmp_path, max_subtraction_rms_ratio=1e-6)

    with pytest.raises(exomoon.NoUsableEpochsError):
        emls.run()


@pytest.mark.parametrize('grid', [20, 30])
def test_too_small_period_grid_is_rejected_at_construction(tmp_path, grid):
    with pytest.raises(ValueError, match='period_grid_size'):
        system(tmp_path, grid=grid)


# --- I3, I4: alias search ----------------------------------------------------------------------------------------

def test_alias_search_compares_relative_errors():
    alias, k, error = ExoMoonLeastSquares.find_period_alias(10.0, 100.0, 11.0, 0.1)

    assert k == -1
    assert alias == pytest.approx(1 / (1 / 10.0 - 1 / 100.0))
    assert error == pytest.approx(abs(alias - 11.0) / 10.0)


def test_alias_search_survives_a_period_commensurate_with_the_planet():
    alias, k, error = ExoMoonLeastSquares.find_period_alias(4.0, 20.0, 4.0, 1.0)

    assert (alias, k, error) == (4.0, 0, 0.0)


def test_no_alias_keeps_the_starting_error():
    assert ExoMoonLeastSquares.find_period_alias(10.0, 100.0, 50.0, 0.01) == (None, None, 0.01)


# --- G10: moon template geometry ---------------------------------------------------------------------------------

def template_duration(params):
    time = np.linspace(-1.5, 1.5, 60001) + params.t0
    params.rp = 0.02
    flux = batman.TransitModel(params, time).light_curve(params)
    return np.ptp(time[flux < 1])


@pytest.mark.parametrize('w', [90.0, 270.0])
def test_moon_template_uses_the_planet_periastron(tmp_path, w):
    emls = system(tmp_path, ecc=0.3, w=w)
    truth = batman.TransitParams()
    truth.t0, truth.per, truth.a, truth.inc = 10, PERIOD, SMA * AU_TO_RSUN, 89.9
    truth.ecc, truth.w, truth.u, truth.limb_dark = 0.3, w, LD, 'quadratic'

    assert template_duration(emls.moon_template_params()) == pytest.approx(template_duration(truth), rel=0.01)


# --- G13: planet mass grid ---------------------------------------------------------------------------------------

def test_moon_orbit_follows_the_mass_in_the_grid(tmp_path):
    emls = system(tmp_path)
    emls.planet_mass_grid = [300.0, 37.5]

    inputs = run_to_search(emls)

    light = inputs[len(inputs) // 2]
    a_moon = ExoMoonLeastSquares.compute_semimajor_axis(37.5 * M_earth_to_M_sun, light.moon_period) * AU_TO_RSUN
    expected = ExoMoonLeastSquares.compute_transit_duration(1.0, SMA * AU_TO_RSUN, PERIOD, a_moon, light.impact_param)
    assert light.half_moon_orbit_transit_duration == pytest.approx(expected / 2)


# --- B10, B9: grazing geometry -----------------------------------------------------------------------------------

def test_impact_parameter_includes_eccentricity():
    b = ExoMoonLeastSquares.impact_parameter(20.0, 88.0, 0.3, 90.0)

    assert b == pytest.approx(20.0 * np.cos(np.deg2rad(88.0)) * (1 - 0.09) / 1.3)


def grazing_fit(b):
    a, k = 20.0, 0.092
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, 10.0, k, a
    params.inc, params.ecc, params.w, params.u = np.rad2deg(np.arccos(b / a)), 0.0, 90.0, LD
    params.limb_dark = 'quadratic'
    time = np.arange(-0.4, 0.4, 2 / 1440)
    truth = batman.TransitModel(params, time).light_curve(params)
    guess = [0.0, a, np.rad2deg(np.arccos(0.9 / a)), k]
    bounds = [(-0.1, 0.1), (a * 0.98, a * 1.02), (85.0, 90.0), (k * 0.8, k * 1.2)]
    _, model = ExoMoonLeastSquares.fit_single_transit(time, truth, np.full(len(time), 1e-4), guess, bounds, params)
    return np.max(np.abs(model - truth)) / (1 - truth.min())


def test_grazing_transit_can_be_fitted():
    assert grazing_fit(1.06) < 0.02


def test_ordinary_transit_is_still_fitted():
    assert grazing_fit(0.5) < 0.02


def test_lower_inclination_is_the_physical_limit():
    limit = ExoMoonLeastSquares.min_transit_inclination(40.0, 0.1, 0.0, 90.0)

    assert limit == pytest.approx(np.rad2deg(np.arccos(1.1 / 40.0)))


def test_fold_inclination_bound_contains_the_guess_and_is_physical(tmp_path):
    curve = light_curve()
    seen = []
    original = ExoMoonLeastSquares.fit_single_transit

    def spy(*args, **kwargs):
        seen.append((args[3], args[4]))
        return original(*args, **kwargs)

    emls = system(tmp_path, curve, inc=87.0)
    emls.planet_inc_err = 80.0
    with mock.patch.object(ExoMoonLeastSquares, 'fit_single_transit', staticmethod(spy)):
        emls.subtract_planet_transit(curve[1], curve[2], curve[3])

    guess, bounds = seen[0]
    limit = ExoMoonLeastSquares.min_transit_inclination(guess[1], guess[3], 0.0, 90.0)
    assert bounds[2][0] <= guess[2] <= bounds[2][1]
    assert bounds[2][0] == pytest.approx(min(guess[2], limit))


# --- H6, J1, J2: diagnostics -------------------------------------------------------------------------------------

def test_sde_plot_title_shows_the_sde(tmp_path):
    titles = []
    original = matplotlib.figure.Figure.suptitle

    def spy(self, text, *args, **kwargs):
        titles.append(text)
        return original(self, text, *args, **kwargs)

    emls = system(tmp_path)
    with mock.patch.object(matplotlib.figure.Figure, 'suptitle', spy):
        emls.run()

    first = next(t for t in titles if 'SDE->' in t)
    assert float(first.split('SDE->')[1].split(',')[0]) == pytest.approx(emls.max_sde)


def test_late_transit_has_positive_timing_variation(tmp_path):
    shift = 10 / 1440
    curve = light_curve(shifts=np.full(EPOCHS, shift))
    emls = system(tmp_path, curve, skip_fold_fit=True)

    emls.subtract_planet_transit(curve[1], curve[2], curve[3])

    assert np.median(emls.ttvs) == pytest.approx(shift, abs=2 / 1440)


def test_fold_plot_draws_lines_in_time_order(tmp_path):
    drawn = []
    original = matplotlib.axes.Axes.plot

    def spy(self, *args, **kwargs):
        drawn.append(np.asarray(args[0]))
        return original(self, *args, **kwargs)

    curve = light_curve()
    emls = system(tmp_path, curve)
    with mock.patch.object(matplotlib.axes.Axes, 'plot', spy):
        emls.subtract_planet_transit(curve[1], curve[2], curve[3], plot=True)

    assert np.all(np.diff(drawn[0]) >= 0)


# --- C5, C3: fitting state ---------------------------------------------------------------------------------------

def test_unused_single_transit_helper_is_gone():
    assert not hasattr(ExoMoonLeastSquares, 'fit_single_transits')


def per_epoch_guesses(tmp_path, curve):
    guesses = []
    original = ExoMoonLeastSquares.fit_single_t0_rp

    def spy(*args, **kwargs):
        guesses.append(args[4])
        return original(*args, **kwargs)

    emls = system(tmp_path, curve, fit_rp_per_transit=True, skip_fold_fit=True)
    with mock.patch.object(ExoMoonLeastSquares, 'fit_single_t0_rp', staticmethod(spy)):
        result = emls.subtract_planet_transit(curve[1], curve[2], curve[3])
    return guesses, result


def test_every_epoch_starts_from_the_shared_rp(tmp_path):
    guesses, _ = per_epoch_guesses(tmp_path, light_curve())

    assert len(set(guesses)) == 1


def test_epoch_order_does_not_change_the_residuals(tmp_path):
    curve = light_curve()
    reversed_curve = (curve[0][::-1],) + curve[1:]

    _, forward = per_epoch_guesses(tmp_path, curve)
    _, backward = per_epoch_guesses(tmp_path, reversed_curve)

    np.testing.assert_allclose(forward[1], backward[1], atol=1e-9)


# --- A5: constants -----------------------------------------------------------------------------------------------

def test_earth_year_gives_one_astronomical_unit():
    assert ExoMoonLeastSquares.compute_semimajor_axis(1.0, 365.25) == pytest.approx(1.0, abs=1e-4)


def test_period_and_semimajor_axis_conversions_invert_each_other():
    assert ExoMoonLeastSquares.au_to_period(1.0, ExoMoonLeastSquares.compute_semimajor_axis(1.0, 30.0)) == \
        pytest.approx(30.0, rel=1e-9)


def test_conversion_factors_match_astropy():
    assert R_earth_to_R_sun == pytest.approx((ac.R_earth / ac.R_sun).value, rel=1e-6)
    assert M_earth_to_M_sun == pytest.approx((ac.M_earth / ac.M_sun).value, rel=1e-6)
    assert AU_TO_RSUN == pytest.approx((ac.au / ac.R_sun).value, rel=1e-6)
    assert exomoon.AU_TO_M == pytest.approx(ac.au.value, rel=1e-9)


# --- F1, F2, F4, F5, F9: catalog driver --------------------------------------------------------------------------

def test_koi_table_keeps_its_period_column():
    table = pd.DataFrame({'Period (days)': [9.48803557], 'Period (days) err': [1e-5], 'Planet Radius (R_Earth)': [2.0],
                          'koi_srad': [1.0], 'Stellar log(g) (cm/s^2)': [4.4], 'koi_model_snr': [30.0],
                          'koi_prad_err1': [np.nan], 'koi_impact': [np.nan]})

    prepared = exomoon.prepare_koi_table(table)

    assert list(prepared.columns).count('Period (days)') == 1
    assert prepared['Period (days)'].iloc[0] == 9.48803557
    assert prepared['Planet Radius (R_Earth) err'].iloc[0] == pytest.approx(0.2)


def test_koi_inclination_is_computed_from_the_impact_parameter():
    inc = exomoon.koi_planet_inclination(0.5, 1.0, 1.0, 10.0)

    a_rsun = ExoMoonLeastSquares.compute_semimajor_axis(1.0, 10.0) * AU_TO_RSUN
    assert inc == pytest.approx(np.rad2deg(np.arccos(0.5 / a_rsun)))


def test_catalog_path_follows_the_user_home(monkeypatch):
    monkeypatch.setenv('HOME', '/some/home')

    assert exomoon.catalog_csv_path('tois') == os.path.join('/some/home', '.sherlockpipe', 'tois.csv')


def test_planets_of_one_star_get_separate_directories():
    assert exomoon.emls_output_dir('TIC 1', '100.01') != exomoon.emls_output_dir('TIC 1', '100.02')


def test_roche_limit_of_an_earth_analogue():
    assert ExoMoonLeastSquares.compute_roche_radius(1.0, 1.0, 5.51) == pytest.approx(2.44, rel=0.01)
