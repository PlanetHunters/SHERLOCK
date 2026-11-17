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
              'koi_impact': 0.2, 'Stellar Eff Temp (K)': 5800.0}
    values.update(overrides)
    return pd.DataFrame({key: [value] for key, value in values.items()})


def confirmed_table(**overrides):
    values = {'koi_name': 'K00001.01', 'pl_name': 'Kepler-1 b', 'pl_bmasse': np.nan, 'pl_bmassprov': 'M-R relationship',
              'pl_orbeccen': 0.0, 'pl_orblper': np.nan}
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
    assert np.isfinite(params['max_surviving_moon_radius'])
    wide = exomoon.catalog_planet_params(koi_row(**{'Period (days)': 400.0}), 'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)
    assert wide['max_radius'] == pytest.approx(PLANET_RADIUS / 2)


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
    # A 10 d sub-Neptune keeps no moon above the smallest one searched even with the generous tidal
    # margin (Barnes & O'Brien 2002), which is reported but does not cut the Rp/2 range.
    assert params['max_surviving_moon_radius'] < 0.15
    assert params['max_radius'] == pytest.approx(1.5)
    assert params['planet_inc'] == 90 and params['planet_inc_err'] == 3
    assert params['planet_impact_param'] == pytest.approx(a * exomoon.AU_TO_RSUN / 0.8 * np.cos(np.pi / 2), abs=1e-12)
    assert params['planet_mass'] == pytest.approx(exomoon.catalog_planet_mass(3.0, 0.3))
    assert params['planet_t0'] == pytest.approx(101.5)
    assert params['planet_duration'] == pytest.approx(0.125)
    assert params['planet_ecc'] == 0.0 and params['planet_arg_periastron'] == 0.0


def test_catalog_mass_is_the_emls_command_forecast_not_r_to_the_1_5():
    from sherlockpipe.exomoon.run import GIANT_RADIUS, JUPITER_MASS, guess_planet_mass

    # A giant: R^1.5 gave 37 M_earth for 11.2 R_earth, about 8 times too light for its Hill sphere.
    assert exomoon.catalog_planet_mass(11.2) == pytest.approx(JUPITER_MASS)
    assert exomoon.catalog_planet_mass(GIANT_RADIUS - 1, 0.3) == pytest.approx(guess_planet_mass(GIANT_RADIUS - 1, 0.3))
    assert np.isnan(exomoon.catalog_planet_mass(np.nan))
    # A radius error as large as the radius, common in the KOI table, still gives a mass.
    assert np.isfinite(exomoon.catalog_planet_mass(2.0, 10.0))
    assert np.isfinite(exomoon.catalog_planet_mass(3.0, 3.0))


def test_koi_table_mass_comes_from_the_forecast():
    row = koi_row(**{'Planet Radius (R_Earth)': 11.2})

    from sherlockpipe.exomoon.run import JUPITER_MASS
    # The table leaves the mass empty; the driver forecasts it only for the rows it keeps.
    assert np.isnan(row['Predicted Mass (M_Earth)'])
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


class CaptureSearchPool(CapturePool):
    """CapturePool that also keeps the search object the driver built (the pool initializer's argument)."""
    emls = None

    def __init__(self, *args, initargs=(), **kwargs):
        CaptureSearchPool.emls = initargs[0] if initargs else None


def test_koi_driver_reaches_the_search(tmp_path, monkeypatch):
    """Run the module as the KOI driver on one synthetic target and stop at the search."""
    home = tmp_path / 'home'
    (home / '.sherlockpipe').mkdir(parents=True)
    time, flux, flux_err, duration = long_period_curve()
    koi_table(**{'Period (days)': DRIVER_PERIOD, 'Duration (hours)': duration * 24, 'koi_impact': 0.0}).to_csv(home / '.sherlockpipe' / 'kois.csv', index=False)
    # The archive table of planets confirmed today, so the driver does not download it.
    confirmed_table().to_csv(home / '.sherlockpipe' / 'kepler_confirmed.csv', index=False)
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
    CaptureSearchPool.emls = None
    caught = []
    # The driver replaces the root log handlers, so record what it reports through logging.exception.
    record = lambda *args, **kwargs: caught.append(sys.exc_info()[1])

    with mock.patch('lcbuilder.lcbuilder_class.LcBuilder', builder), \
            mock.patch('sherlockpipe.exomoon.search.Pool', CaptureSearchPool), \
            mock.patch('logging.exception', side_effect=record):
        runpy.run_module('sherlockpipe.exomoon', run_name='__main__')

    assert len(caught) == 1 and isinstance(caught[0], StopAtSearch), repr(caught)
    assert CapturePool.inputs
    assert {item.min_radius for item in CapturePool.inputs} == {PLANET_RADIUS / 20}
    # The validated set-up: EMLS detrends the raw curve (3 T14 biweight, 'auto' mask) and runs 100 nulls.
    emls = CaptureSearchPool.emls
    assert emls.detrend_window == pytest.approx(3 * duration)
    assert emls.detrend_mask == 'auto'
    assert emls.null_runs == 100



def test_max_surviving_moon_mass_is_barnes_and_obrien_eq_8(monkeypatch):
    """Independent check against the audit's integration of da/dt (f = 0.36, k2 = 0.51, Q = 1e5)."""
    import sherlockpipe.exomoon.tides as tides
    monkeypatch.setattr(tides, 'planet_k2_over_q', lambda radius: 0.51 / 1e5)
    monkeypatch.setattr(tides.MoonOrbits, 'max_moon_semimajor_axis', staticmethod(
        lambda star_mass, planet_mass, a, ecc=0.0, **kwargs: 0.36 * ExoMoonLeastSquares.compute_hill_radius(star_mass,
                                                                                                    planet_mass, a)))
    nominal = {'age_gyr': 5.0, 'k2_q_factor': 1.0}
    assert exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 30, **nominal) == pytest.approx(0.091, rel=0.05)
    assert exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 60, **nominal) == pytest.approx(1.84, rel=0.05)


def test_default_tidal_bound_is_the_generous_margin():
    """k2/Q ten times lower and 1 Gyr instead of 5: a moon 50 times more massive survives."""
    nominal = exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 30, age_gyr=5.0, k2_q_factor=1.0)
    assert exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 30) == pytest.approx(50 * nominal)
    assert exomoon.max_surviving_moon_radius(1.0, 318, 11.2, 30) == pytest.approx(
        exomoon.moon_radius_from_mass(50 * nominal))
    assert np.isnan(exomoon.max_surviving_moon_radius(1.0, np.nan, 11.2, 30))


def test_retrograde_tidal_bound_uses_the_retrograde_stability_limit():
    """Barnes & O'Brien Eq. 12: a captured retrograde moon crosses the region up to the retrograde limit."""
    prograde = exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 30)
    retrograde = exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 30, retrograde=True)
    assert retrograde / prograde == pytest.approx((0.9309 / 0.4895) ** 6.5, rel=1e-6)
    assert exomoon.max_surviving_moon_radius(1.0, 318, 11.2, 30, retrograde=True) > \
        exomoon.max_surviving_moon_radius(1.0, 318, 11.2, 30)


def test_moons_survive_around_wide_giants_not_around_close_small_planets():
    def surviving(mass, radius, period):
        return exomoon.moon_radius_from_mass(exomoon.max_surviving_moon_mass(1.0, mass, radius, period))

    # A hot Jupiter keeps no moon; a 200 d Jupiter keeps a large one.
    assert surviving(318, 11.2, 3) < 11.2 / 20
    assert surviving(318, 11.2, 200) > 11.2 / 2
    # The bound grows with the planet mass and the period (unlike the old P > 10 d M^(1/6) rule).
    assert exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 50) > exomoon.max_surviving_moon_mass(1.0, 30, 11.2, 50)
    assert exomoon.max_surviving_moon_mass(1.0, 17, 3.9, 100) > exomoon.max_surviving_moon_mass(1.0, 17, 3.9, 20)
    # An eccentric orbit narrows the stable region, so the bound drops.
    assert exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 50, 0.3) < exomoon.max_surviving_moon_mass(1.0, 318, 11.2, 50)
    assert np.isnan(exomoon.max_surviving_moon_mass(1.0, np.nan, 11.2, 50))


def test_k2_over_q_follows_hek_vi_and_stops_at_jupiter():
    assert exomoon.planet_k2_over_q(1.0) == pytest.approx(10 ** (-2.90 - 0.187))
    assert exomoon.planet_k2_over_q(20.0) == exomoon.planet_k2_over_q(11.2)


def test_moon_radius_from_mass_is_the_chen_kipping_mean_relation():
    assert exomoon.moon_radius_from_mass(1.0) == pytest.approx(1.008)
    below, above = exomoon.moon_radius_from_mass(2.04 * 0.999), exomoon.moon_radius_from_mass(2.04 * 1.001)
    assert above == pytest.approx(below, rel=0.002)
    assert exomoon.moon_radius_from_mass(17.0) == pytest.approx(1.008 * 2.04 ** 0.279 * (17 / 2.04) ** 0.589)
    assert exomoon.moon_radius_from_mass(0.0) == 0.0


def test_catalog_radius_range_is_not_cut_by_tidal_survival():
    close = exomoon.catalog_planet_params(koi_row(**{'Period (days)': 25.0, 'Planet Radius (R_Earth)': 11.2}), 'kois',
                                          1.0, 1.0, 0.0, KEPLER_OFFSET)
    wide = exomoon.catalog_planet_params(koi_row(**{'Period (days)': 300.0, 'Planet Radius (R_Earth)': 11.2}),
                                         'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)

    assert close['min_radius'] == pytest.approx(11.2 / 20)
    assert close['max_surviving_moon_radius'] < 11.2 / 2
    assert close['max_radius'] == pytest.approx(11.2 / 2)
    assert wide['max_surviving_moon_radius'] > close['max_surviving_moon_radius']
    assert wide['max_radius'] == pytest.approx(11.2 / 2)


def test_mean_planet_mass_is_close_to_the_forecast_median():
    for radius in (1.0, 2.0, 4.0, 6.0):
        assert exomoon.mean_planet_mass(radius) == pytest.approx(exomoon.catalog_planet_mass(radius), rel=0.15)
    from sherlockpipe.exomoon.run import JUPITER_MASS
    assert exomoon.mean_planet_mass(11.2) == JUPITER_MASS
    assert exomoon.moon_radius_from_mass(exomoon.mean_planet_mass(3.0)) == pytest.approx(3.0)
    assert np.isnan(exomoon.mean_planet_mass(np.nan))


def run_driver_on_table(tmp_path, monkeypatch, table, extra_args=(), order=None):
    """Run the KOI driver on a catalog table and return the number of targets whose light curve it started."""
    home = tmp_path / 'home'
    (home / '.sherlockpipe').mkdir(parents=True)
    table.to_csv(home / '.sherlockpipe' / 'kois.csv', index=False)
    # Every KOI of the table is a planet confirmed today, so the driver does not download the archive table.
    pd.concat([confirmed_table(koi_name=oi, pl_name=f'Planet {oi}') for oi in table['OI']], ignore_index=True) \
        .to_csv(home / '.sherlockpipe' / 'kepler_confirmed.csv', index=False)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['exomoon.py', '--catalog', 'kois', *extra_args])
    builder = mock.MagicMock()
    builder.return_value.build.side_effect = StopAtSearch()
    with mock.patch('lcbuilder.lcbuilder_class.LcBuilder', builder), mock.patch('logging.exception'):
        runpy.run_module('sherlockpipe.exomoon', run_name='__main__')
    names = [call.kwargs['target_name'] for call in builder.return_value.build_object_info.call_args_list]
    if order is not None:
        order.extend(dict.fromkeys(names))
    return len(set(names))


def test_driver_drops_a_target_with_a_close_faint_companion(tmp_path, monkeypatch):
    """A companion below the SNR cut still destabilises the moons of its neighbour."""
    target = koi_table(**{'Period (days)': DRIVER_PERIOD})
    companion = koi_table(**{'OI': 'K00001.02', 'Period (days)': 1.1 * DRIVER_PERIOD, 'koi_model_snr': 5.0})
    assert run_driver_on_table(tmp_path / 'alone', monkeypatch, target) == 1
    assert run_driver_on_table(tmp_path / 'pair', monkeypatch, pd.concat([target, companion])) == 0


def test_driver_hill_roche_check_uses_the_domingos_limit_with_eccentricity(tmp_path, monkeypatch, capsys):
    """No stable prograde orbit once 0.4895 (1 - 1.0305 e) r_H falls inside the Roche limit.

    The tidal filter would drop this target too, so the check is read from the filter counts."""
    circular = koi_table(**{'Period (days)': DRIVER_PERIOD, 'koi_eccen': 0.0})
    eccentric = koi_table(**{'Period (days)': DRIVER_PERIOD, 'koi_eccen': 0.97})
    assert run_driver_on_table(tmp_path / 'circular', monkeypatch, circular) == 1
    assert 'Hill-Roche=0,' in capsys.readouterr().out
    assert run_driver_on_table(tmp_path / 'eccentric', monkeypatch, eccentric) == 0
    assert 'Hill-Roche=1,' in capsys.readouterr().out


# --- Confirmed planets: the archive, not the KOI table's disposition ---------------------------------------------

def three_kois():
    """A planet confirmed today, a refuted "CONFIRMED" KOI (KOI-189.01-like) and a validated planet that the
    Kepler data call a stellar eclipse (KOI-125.01-like)."""
    table = pd.concat([koi_table(OI='K00001.01'), koi_table(OI='K00189.01'),
                       koi_table(OI='K00125.01', koi_pdisposition='FALSE POSITIVE')], ignore_index=True)
    return exomoon.prepare_koi_table(table)


def archive(**overrides):
    rows = [confirmed_table(**overrides), confirmed_table(koi_name='K00125.01', pl_name='Kepler-468 b')]
    return pd.concat(rows, ignore_index=True)


def test_only_planets_confirmed_today_are_kept():
    table = exomoon.apply_confirmed_planets(three_kois(), archive())

    assert table.set_index('OI')['confirmed_planet'].to_dict() == {
        'K00001.01': True, 'K00189.01': False, 'K00125.01': False}
    assert table.set_index('OI').loc['K00001.01', 'archive_planet_name'] == 'Kepler-1 b'


def test_a_measured_mass_and_eccentricity_replace_the_guesses():
    table = exomoon.apply_confirmed_planets(three_kois(), archive(pl_bmasse=1678.0, pl_bmassprov='Mass',
                                                                 pl_orbeccen=0.401, pl_orblper=-75.3))
    row = table.set_index('OI').loc['K00001.01']

    assert row['Predicted Mass (M_Earth)'] == pytest.approx(1678.0)
    assert exomoon.catalog_planet_orbit(row, 'kois') == pytest.approx((0.401, -75.3))
    assert exomoon.catalog_planet_params(row, 'kois', 1.0, 1.0, 0.0, KEPLER_OFFSET)['planet_mass'] == \
        pytest.approx(1678.0)


def test_a_measured_brown_dwarf_mass_is_not_a_planet():
    brown_dwarf = exomoon.apply_confirmed_planets(three_kois(), archive(pl_bmasse=6388.1, pl_bmassprov='Mass'))
    giant = exomoon.apply_confirmed_planets(three_kois(), archive(pl_bmasse=12 * 317.83, pl_bmassprov='Mass'))
    upper_limit = exomoon.apply_confirmed_planets(three_kois(), archive(pl_bmasse=6388.1, pl_bmassprov='Mass',
                                                                       pl_bmasselim=1))

    assert not brown_dwarf.set_index('OI').loc['K00001.01', 'confirmed_planet']
    assert giant.set_index('OI').loc['K00001.01', 'confirmed_planet']
    assert upper_limit.set_index('OI').loc['K00001.01', 'confirmed_planet']  # a limit says nothing


def test_relation_masses_and_assumed_circular_orbits_are_not_used():
    table = exomoon.apply_confirmed_planets(three_kois(), archive(pl_bmasse=318.0, pl_bmassprov='M-R relationship',
                                                                 pl_orbeccen=0.0))
    row = table.set_index('OI').loc['K00001.01']

    assert np.isnan(row['Predicted Mass (M_Earth)'])
    assert exomoon.catalog_planet_orbit(row, 'kois') == (0.0, 0.0)


def test_mass_and_eccentricity_limits_are_not_used():
    table = exomoon.apply_confirmed_planets(three_kois(), archive(pl_bmasse=12651.5, pl_bmasselim=1, pl_bmassprov='Mass',
                                                                 pl_orbeccen=0.3, pl_orbeccenlim=1))
    row = table.set_index('OI').loc['K00001.01']

    assert np.isnan(row['Predicted Mass (M_Earth)'])
    assert exomoon.catalog_planet_orbit(row, 'kois') == (0.0, 0.0)


def test_the_confirmed_table_is_downloaded_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exomoon.catalog, 'download_confirmed_kepler_planets',
                        lambda: calls.append(1) or archive())
    path = str(tmp_path / 'kepler_confirmed.csv')

    first = exomoon.load_confirmed_kepler_planets(path)
    second = exomoon.load_confirmed_kepler_planets(path)
    exomoon.load_confirmed_kepler_planets(path, refresh=True)

    assert len(calls) == 2
    assert list(second['koi_name']) == list(first['koi_name'])


# --- Target selection and ranking (2026-11-17) -----------------------------------------------------------------------

def test_targets_are_sun_like_stars_with_wide_planets():
    wide = {'Period (days)': DRIVER_PERIOD}
    table = exomoon.prepare_koi_table(pd.concat([
        koi_table(OI='K00001.01', **wide), koi_table(OI='K00002.01', **wide, **{'Stellar Eff Temp (K)': 4800.0}),
        koi_table(OI='K00003.01', **wide, **{'Stellar Eff Temp (K)': 6600.0}),
        koi_table(OI='K00004.01', **{'Period (days)': 60.0}),
        koi_table(OI='K00005.01', **wide, **{'Stellar Eff Temp (K)': np.nan})], ignore_index=True))

    assert list(exomoon.select_catalog_targets(table)['OI']) == ['K00001.01']
    assert set(exomoon.select_catalog_targets(table, (4000, 7000), 50)['OI']) == \
        {'K00001.01', 'K00002.01', 'K00003.01', 'K00004.01'}


def test_reference_moon_is_one_earth_radius_or_the_largest_the_target_holds():
    # A giant at 100 d holds an Earth: the reference is 1 R_earth.
    assert exomoon.reference_moon_radius(1.0, 300.0, 10.0, 100.0) == 1.0
    # A 1.6 R_earth planet: half its radius, the top of the searched range.
    assert exomoon.reference_moon_radius(1.0, 3.0, 1.6, 100.0) == pytest.approx(0.8)
    # A planet whose moons tides remove: the generous survival radius.
    surviving = exomoon.max_surviving_moon_radius(1.0, 10.0, 3.0, 15.0)
    assert surviving < 1.0
    assert exomoon.reference_moon_radius(1.0, 10.0, 3.0, 15.0) == pytest.approx(surviving)


def test_reference_moon_snr_scales_the_planet_snr_by_the_depth_ratio():
    assert exomoon.reference_moon_snr(400.0, 10.0, 1.0) == pytest.approx(4.0)
    assert exomoon.reference_moon_snr(90.0, 3.0, 1.0) == pytest.approx(10.0)
    assert np.isnan(exomoon.reference_moon_snr(np.nan, 3.0, 1.0))


def test_driver_searches_the_best_moon_snr_first(tmp_path, monkeypatch):
    """A Neptune with SNR 100 (a 1 R_earth moon at SNR 100 / 3^2 = 11) goes before a giant with SNR 300
    (300 / 10^2 = 3), which the planet SNR used to put first."""
    giant = koi_table(**{'Period (days)': DRIVER_PERIOD})
    neptune = koi_table(**{'Object Id': 'KIC 2', 'OI': 'K00002.01', 'Period (days)': DRIVER_PERIOD,
                           'Planet Radius (R_Earth)': 3.0, 'koi_prad_err1': 0.3, 'koi_model_snr': 100.0})
    order = []
    assert run_driver_on_table(tmp_path, monkeypatch, pd.concat([giant, neptune]), order=order) == 2
    assert order == ['KIC 2', 'KIC 1']


def test_driver_selection_and_count_come_from_the_command_line(tmp_path, monkeypatch):
    cool = koi_table(**{'Period (days)': DRIVER_PERIOD, 'Stellar Eff Temp (K)': 4500.0})
    other = koi_table(**{'Object Id': 'KIC 2', 'OI': 'K00002.01', 'Period (days)': DRIVER_PERIOD})
    table = pd.concat([cool, other])
    assert run_driver_on_table(tmp_path / 'default', monkeypatch, table) == 1
    assert run_driver_on_table(tmp_path / 'wide', monkeypatch, table, ['--min_teff', '4000']) == 2
    assert run_driver_on_table(tmp_path / 'one', monkeypatch, table, ['--min_teff', '4000', '--max_targets', '1']) == 1
