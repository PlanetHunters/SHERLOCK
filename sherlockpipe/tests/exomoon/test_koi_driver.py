"""Tests that the EMLS catalog driver sets up KOI targets completely (audit item F3).

Before the fix, the moon radius range was only set for TOI-like catalogs, so every KOI
target stopped with a NameError before reaching the search.
"""
import runpy
import sys
from types import SimpleNamespace
from unittest import mock

import batman
import numpy as np
import pandas as pd
import pytest

import sherlockpipe.exomoon as exomoon
from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon.test_audit_fixes import (CapturePool, LD, PERIOD, PLANET_RADIUS, StopAtSearch,
                                                         run_to_search, system)

KEPLER_OFFSET = 2454833


def koi_table(**overrides):
    values = {'Object Id': 'KIC 1', 'OI': 'K00001.01', 'Disposition': 'CONFIRMED', 'Period (days)': PERIOD,
              'Epoch (BJD)': KEPLER_OFFSET + 0.0, 'Duration (hours)': 6.0, 'Planet Radius (R_Earth)': PLANET_RADIUS,
              'koi_srad': 1.0, 'Stellar log(g) (cm/s^2)': 4.438, 'koi_model_snr': 300.0, 'koi_prad_err1': 0.5,
              'koi_impact': 0.2}
    values.update(overrides)
    return pd.DataFrame({key: [value] for key, value in values.items()})


def koi_row(**overrides):
    return exomoon.prepare_koi_table(koi_table(**overrides)).iloc[0]


DRIVER_PERIOD = 100.0


def long_period_curve(noise=1e-4):
    """Four transits of a 10 Re planet at 100 d, long enough to pass the driver's tidal filter."""
    a_rsun = ExoMoonLeastSquares.compute_semimajor_axis(1.0, DRIVER_PERIOD) * AU_TO_RSUN
    t0s = np.arange(4) * DRIVER_PERIOD
    time = np.concatenate([t0 + np.arange(-3.0, 3.0, 2 / 1440) for t0 in t0s])
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, DRIVER_PERIOD, PLANET_RADIUS * R_earth_to_R_sun, a_rsun
    params.inc, params.ecc, params.w, params.u, params.limb_dark = 90.0, 0.0, 90.0, LD, 'quadratic'
    flux = batman.TransitModel(params, time).light_curve(params)
    flux = flux + np.random.default_rng(3).normal(0, noise, len(time))
    duration = DRIVER_PERIOD / np.pi * np.arcsin((1 + params.rp) / a_rsun)
    return time, flux, np.full(len(time), noise), duration


def test_koi_row_gets_the_moon_radius_range():
    params = exomoon.catalog_planet_params(koi_row(), 'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)

    assert params['min_radius'] == pytest.approx(PLANET_RADIUS / 20)
    assert params['max_radius'] == pytest.approx(PLANET_RADIUS / 2)


def test_koi_row_uses_its_impact_parameter():
    params = exomoon.catalog_planet_params(koi_row(koi_impact=0.4), 'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)

    assert params['planet_impact_param'] == 0.4
    assert params['planet_inc'] == pytest.approx(exomoon.koi_planet_inclination(0.4, 1.0, 1.0, PERIOD))
    assert params['planet_inc_err'] == 0.5


def test_toi_row_keeps_its_previous_setup():
    row = pd.Series({'Planet Radius (R_Earth)': 3.0, 'Planet Radius (R_Earth) err': 0.3, 'Period (days)': 10.0,
                     'Epoch (BJD)': 2457000 + 1.5, 'Duration (hours)': 3.0, 'Predicted Mass (M_Earth)': np.nan})

    params = exomoon.catalog_planet_params(row, 'tois', 0.9, 0.8, 100.0, 2457000)

    a = exomoon.ExoMoonLeastSquares.compute_semimajor_axis(0.9, 10.0)
    assert params['min_radius'] == pytest.approx(0.15)
    assert params['max_radius'] == pytest.approx(1.5)
    assert params['planet_inc'] == 90 and params['planet_inc_err'] == 3
    assert params['planet_impact_param'] == pytest.approx(a * exomoon.AU_TO_RSUN / 0.8 * np.cos(np.pi / 2), abs=1e-12)
    assert params['planet_mass'] == pytest.approx(exomoon.catalog_planet_mass(3.0, 0.3))
    assert params['planet_t0'] == pytest.approx(101.5)
    assert params['planet_duration'] == pytest.approx(0.125)
    assert params['planet_ecc'] == 0.0 and params['planet_arg_periastron'] == 0.0


def test_catalog_mass_is_the_emls_command_forecast_not_r_to_the_1_5():
    from sherlockpipe.exomoon_run import GIANT_RADIUS, JUPITER_MASS, guess_planet_mass

    # A giant: R^1.5 gave 37 M_earth for 11.2 R_earth, about 8 times too light for its Hill sphere.
    assert exomoon.catalog_planet_mass(11.2) == pytest.approx(JUPITER_MASS)
    assert exomoon.catalog_planet_mass(GIANT_RADIUS - 1, 0.3) == pytest.approx(guess_planet_mass(GIANT_RADIUS - 1, 0.3))
    assert np.isnan(exomoon.catalog_planet_mass(np.nan))
    # A radius error as large as the radius, common in the KOI table, still gives a mass.
    assert np.isfinite(exomoon.catalog_planet_mass(2.0, 10.0))
    assert np.isfinite(exomoon.catalog_planet_mass(3.0, 3.0))


def test_koi_table_mass_comes_from_the_forecast():
    row = koi_row(**{'Planet Radius (R_Earth)': 11.2})

    from sherlockpipe.exomoon_run import JUPITER_MASS
    assert row['Predicted Mass (M_Earth)'] == pytest.approx(JUPITER_MASS)
    assert exomoon.catalog_planet_params(row, 'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)['planet_mass'] == \
        pytest.approx(JUPITER_MASS)


def test_koi_eccentricity_comes_from_the_catalog_else_circular():
    circular = exomoon.catalog_planet_params(koi_row(), 'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)
    eccentric = exomoon.catalog_planet_params(koi_row(koi_eccen=0.3, koi_longp=120.0), 'kois', 1.0, 1.0, 0.0,
                                              KEPLER_OFFSET)

    assert circular['planet_ecc'] == 0.0 and circular['planet_arg_periastron'] == 0.0
    assert eccentric['planet_ecc'] == 0.3 and eccentric['planet_arg_periastron'] == 120.0


def test_koi_parameters_reach_the_search(tmp_path):
    params = exomoon.catalog_planet_params(koi_row(), 'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)
    emls = system(tmp_path)
    emls.min_radius = params['min_radius']
    emls.max_radius = params['max_radius']

    inputs = run_to_search(emls)

    assert len(inputs) > 0
    assert {item.min_radius for item in inputs} == {params['min_radius']}


def test_koi_driver_reaches_the_search(tmp_path, monkeypatch):
    """Run the module as the KOI driver on one synthetic target and stop at the search."""
    home = tmp_path / 'home'
    (home / '.sherlockpipe').mkdir(parents=True)
    time, flux, flux_err, duration = long_period_curve()
    koi_table(**{'Period (days)': DRIVER_PERIOD, 'Duration (hours)': duration * 24, 'koi_impact': 0.0}).to_csv(home / '.sherlockpipe' / 'kois.csv', index=False)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['exomoon.py', '--catalog', 'kois'])
    build = SimpleNamespace(
        star_info=SimpleNamespace(mass=1.0, radius=1.0, ld_coefficients=LD),
        lc=SimpleNamespace(time=SimpleNamespace(value=time), flux=SimpleNamespace(value=flux),
                           flux_err=SimpleNamespace(value=flux_err)))
    builder = mock.MagicMock()
    builder.return_value.build.return_value = build
    CapturePool.inputs = None
    caught = []
    # The driver replaces the root log handlers, so record what it reports through logging.exception.
    record = lambda *args, **kwargs: caught.append(sys.exc_info()[1])

    with mock.patch('lcbuilder.lcbuilder_class.LcBuilder', builder), \
            mock.patch('multiprocessing.Pool', CapturePool), \
            mock.patch('logging.exception', side_effect=record):
        runpy.run_module('sherlockpipe.exomoon', run_name='__main__')

    assert len(caught) == 1 and isinstance(caught[0], StopAtSearch), repr(caught)
    assert CapturePool.inputs
    assert {item.min_radius for item in CapturePool.inputs} == {PLANET_RADIUS / 20}

