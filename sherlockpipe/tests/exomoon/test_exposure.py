"""Regression tests for audit items C2 and G18: finite exposure in the planet fits and moon templates.

Long-cadence data average the flux over each exposure. The reference here averages instantaneous
batman models at 401 sub-exposure midpoints. batman's own supersample_factor is not used as the
reference, because it is still about 9 ppm off at 101 sub-exposures.
"""
import inspect
import logging

import batman
import numpy as np
import pytest
import pandas as pd

from sherlockpipe import exomoon_run
from sherlockpipe.exomoon import AU_TO_RSUN, R_earth_to_R_sun, ExoMoonLeastSquares
from sherlockpipe.tests.exomoon.test_audit_fixes import LD, PERIOD, SMA, run_to_search, system
from sherlockpipe.tests.exomoon.test_emls_run import guard_directory, write_lc

THIRTY_MINUTES = 30 / 1440


def planet_params(radius=10.0, inc=89.9):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = 0.0, PERIOD, radius * R_earth_to_R_sun, SMA * AU_TO_RSUN, inc
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, LD, 'quadratic'
    return params


def integrated(params, time, exposure, sub_exposures=401):
    offsets = ((np.arange(sub_exposures) + 0.5) / sub_exposures - 0.5) * exposure
    return np.mean([batman.TransitModel(params, time + offset).light_curve(params) for offset in offsets], axis=0)


def test_planet_model_matches_the_integrated_transit_at_30_minutes():
    params = planet_params()
    time = np.linspace(-0.3, 0.3, 4001)
    truth = integrated(params, time, THIRTY_MINUTES)

    averaged = ExoMoonLeastSquares.transit_model(params, time, THIRTY_MINUTES).light_curve(params)
    instantaneous = ExoMoonLeastSquares.transit_model(params, time).light_curve(params)

    assert np.max(np.abs(averaged - truth)) < 15e-6
    assert np.max(np.abs(instantaneous - truth)) > 60e-6


def test_two_minute_data_stay_instantaneous():
    params = planet_params()
    time = np.linspace(-0.3, 0.3, 1001)

    model = ExoMoonLeastSquares.transit_model(params, time, 2 / 1440).light_curve(params)

    np.testing.assert_array_equal(model, batman.TransitModel(params, time).light_curve(params))


def test_moon_templates_match_the_integrated_moon_at_30_minutes():
    params = planet_params()
    radii = np.array([1.0, 2.0, 4.0])

    templates = ExoMoonLeastSquares.build_moon_templates(params, radii, 1.0, 0.3, THIRTY_MINUTES)

    for radius, template_time, flux in templates:
        params.rp = radius * R_earth_to_R_sun
        truth = integrated(params, params.t0 + template_time, THIRTY_MINUTES)
        depth = 1 - np.min(truth)
        assert np.max(np.abs(flux - truth)) < 0.01 * depth, f"radius {radius}"


def subtract_30_minute_transits(tmp_path, exposure):
    """Run the planet fit on 30-minute data; return the in-transit residual RMS, the noise and the fit."""
    noise = 5e-5
    params = planet_params()
    t0s = np.arange(6) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, THIRTY_MINUTES) for t0 in t0s])
    flux = integrated(params, time, THIRTY_MINUTES) + np.random.default_rng(6).normal(0, noise, len(time))
    duration = PERIOD / np.pi * np.arcsin((1 + params.rp) / params.a)
    emls = system(tmp_path, curve=(t0s, time, flux, np.full(len(time), noise), duration), exposure_time=exposure,
                  min_cadences_per_t0=5)

    run_to_search(emls)

    in_transit = np.any([np.abs(emls.time - t0) < duration / 2 for t0 in emls.bary_t0s], axis=0)
    return np.std(emls.flux[in_transit]), noise, emls


def test_planet_fit_on_30_minute_data_needs_the_exposure(tmp_path):
    """The free-shape fold fit (B1) hides most of the smearing in its residuals, but not in its geometry."""
    (tmp_path / 'averaged').mkdir()
    (tmp_path / 'instantaneous').mkdir()
    true_t14 = ExoMoonLeastSquares.model_transit_duration(planet_params(), 1.0)

    averaged, noise, with_exposure = subtract_30_minute_transits(tmp_path / 'averaged', THIRTY_MINUTES)
    _, _, without_exposure = subtract_30_minute_transits(tmp_path / 'instantaneous', None)

    assert averaged < 1.15 * noise, f"averaged model leaves {averaged * 1e6:.1f} ppm"
    averaged_error = abs(with_exposure.planet_duration / true_t14 - 1)
    instantaneous_error = abs(without_exposure.planet_duration / true_t14 - 1)
    assert averaged_error < 0.02
    assert instantaneous_error > 2 * averaged_error


def curve_with_spacing(directory, guard_time, spacings_s):
    """Write a curve over the guard data span whose points follow the given spacings, in turn, per segment."""
    segments = np.array_split(np.arange(int(guard_time.min()), int(np.ceil(guard_time.max()))), len(spacings_s))
    time = np.concatenate([np.arange(days[0], days[-1] + 1, spacing / 86400) for days, spacing in
                           zip(segments, spacings_s)])
    lc_file = str(directory / 'lc_custom.csv')
    write_lc(lc_file, time, np.ones_like(time), np.full(len(time), 1e-4))
    return lc_file


@pytest.mark.parametrize('spacing', [120.0, 1765.5, 8640.0])
def test_exposure_is_the_median_point_spacing(tmp_path, spacing):
    _, time, _ = guard_directory(tmp_path)
    lc_file = curve_with_spacing(tmp_path, time, [spacing])

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, lc_file=lc_file)

    assert inputs['exposure_time'] == pytest.approx(spacing, rel=1e-6)
    assert inputs['sources']['exposure_time'] == 'median point spacing of lc_custom.csv'


def test_sherlock_curve_exposure_comes_from_its_spacing(tmp_path):
    _, time, _ = guard_directory(tmp_path)
    (tmp_path / 'properties.yaml').write_text('EXPTIME: [20, 1800]\n')  # a download filter, not read

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    assert inputs['exposure_time'] == pytest.approx(np.median(np.diff(time)) * 86400, rel=1e-6)
    assert inputs['sources']['exposure_time'] == 'median point spacing of lc_0.csv'


def test_gaps_do_not_change_the_exposure(tmp_path):
    _, time, _ = guard_directory(tmp_path)
    lc_file = curve_with_spacing(tmp_path, time, [600.0])
    curve = pd.read_csv(lc_file)
    curve[(curve['time'] % 1) < 0.3].to_csv(lc_file, index=False)  # 0.7-day gap every day

    assert exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, lc_file=lc_file)['exposure_time'] == \
        pytest.approx(600.0, rel=1e-6)


def sector(start, days, cadence_s):
    return start + np.arange(0, days, cadence_s / 86400)


@pytest.mark.parametrize('segments, median_s, other_s', [
    # Kepler long cadence for four quarters and short cadence for one: the median follows the short one.
    ([(0, 360, 1765.5), (400, 90, 58.85)], 58.85, 1765.5),
    # TESS FFI at 600 s and 1800 s, as in a SHERLOCK run with EXPTIME [600, 1800].
    ([(0, 36, 600.0), (40, 62, 1800.0)], 600.0, 1800.0),
    # TESS 2 min for one sector and 30-minute FFIs for three.
    ([(0, 27, 120.0), (30, 81, 1800.0)], 120.0, 1800.0),
])
def test_mixed_cadences_are_warned_and_recorded(segments, median_s, other_s, caplog):
    time = np.concatenate([sector(*segment) for segment in segments])

    with caplog.at_level(logging.INFO):
        seconds, _, cadences = exomoon_run.estimate_exposure_time(time, 'lc_1.csv')

    assert seconds == pytest.approx(median_s, rel=1e-3)
    assert sorted(cadence['seconds'] for cadence in cadences) == pytest.approx(sorted([median_s, other_s]), rel=1e-3)
    assert len(exomoon_run.significant_cadences(cadences)) == 2
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any('mixes cadences' in message and f'{other_s:.1f} s' in message for message in warnings)


def test_removed_points_and_gaps_are_not_cadences(caplog):
    time = sector(0, 27, 120.0)
    time = np.sort(np.random.default_rng(0).choice(time, int(0.7 * len(time)), replace=False))
    time = time[(time % 13.7) > 1.0]  # a one-day gap every 13.7 days

    with caplog.at_level(logging.WARNING):
        seconds, _, cadences = exomoon_run.estimate_exposure_time(time, 'lc_1.csv')

    assert seconds == pytest.approx(120.0, rel=1e-3)
    assert [round(cadence['seconds']) for cadence in exomoon_run.significant_cadences(cadences)] == [120]
    assert not caplog.records


def test_single_cadence_is_not_warned(caplog):
    with caplog.at_level(logging.WARNING):
        seconds, _, cadences = exomoon_run.estimate_exposure_time(sector(0, 90, 1765.5), 'lc_1.csv')

    assert seconds == pytest.approx(1765.5, rel=1e-6)
    assert len(cadences) == 1 and cadences[0]['points_share'] == pytest.approx(1.0)
    assert not caplog.records


def test_results_record_the_cadences(tmp_path):
    _, time, _ = guard_directory(tmp_path)
    lc_file = curve_with_spacing(tmp_path, time, [120.0, 1800.0])

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, lc_file=lc_file)

    assert inputs['exposure_time'] == pytest.approx(120.0, rel=1e-6)
    assert [round(cadence['seconds']) for cadence in inputs['cadences']] == [120, 1800]


def test_irregular_spacing_is_warned(tmp_path, caplog):
    _, time, _ = guard_directory(tmp_path)
    irregular = np.sort(np.random.default_rng(0).uniform(time.min(), time.max(), 5000))
    lc_file = str(tmp_path / 'lc_irregular.csv')
    write_lc(lc_file, irregular, np.ones_like(irregular), np.full(len(irregular), 1e-4))

    with caplog.at_level(logging.WARNING):
        exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, lc_file=lc_file)

    assert any('no regular cadence' in record.getMessage() for record in caplog.records)


def test_user_exposure_off_the_data_is_warned(caplog):
    with caplog.at_level(logging.WARNING):
        seconds, source, _ = exomoon_run.estimate_exposure_time(sector(0, 27, 1800.0), 'lc_1.csv', 30)

    assert (seconds, source) == (30.0, 'user')
    assert any('matches none of the cadences' in record.getMessage() for record in caplog.records)


def test_user_exposure_takes_precedence(tmp_path):
    guard_directory(tmp_path)

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, exposure_time=120)

    assert inputs['exposure_time'] == 120.0 and inputs['sources']['exposure_time'] == 'user'


def test_spacing_longer_than_a_day_is_rejected(tmp_path):
    _, time, _ = guard_directory(tmp_path)
    sparse = np.arange(time.min(), time.max(), 1.5)
    lc_file = str(tmp_path / 'lc_sparse.csv')
    write_lc(lc_file, sparse, np.ones_like(sparse), np.full(len(sparse), 1e-4))

    with pytest.raises(ValueError, match='longer than a day.*--exposure_time'):
        exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, lc_file=lc_file)


@pytest.mark.parametrize('days', [float('inf'), 2.0])
def test_unbounded_exposure_is_rejected(tmp_path, days):
    t0s, time, flux, flux_err, duration = (np.arange(3) * PERIOD, np.linspace(0, 3 * PERIOD, 3000), np.ones(3000),
                                           np.full(3000, 1e-4), 0.2)

    with pytest.raises(ValueError, match='positive number of days up to 1'):
        system(tmp_path, curve=(t0s, time, flux, flux_err, duration), exposure_time=days)


def test_invalid_exposure_is_rejected(tmp_path):
    guard_directory(tmp_path)

    for value in (0, -1, float('inf'), float('nan'), 86401):
        with pytest.raises(ValueError, match='positive number of seconds up to 86400'):
            exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, exposure_time=value)


# --- injected moons follow the same exposure -----------------------------------------------------------------------

INJECTED_MOON = dict(moon_radius=3.0, moon_period=2.3, moon_mass=5.0)


def injected_moon_flux(tmp_path, exposure):
    t0s = np.arange(6) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, THIRTY_MINUTES) for t0 in t0s])
    flux = np.ones_like(time)
    duration = PERIOD / np.pi * np.arcsin((1 + 10.0 * R_earth_to_R_sun) / (SMA * AU_TO_RSUN))
    emls = system(tmp_path, curve=(t0s, time, flux, np.full(len(time), 1e-4), duration), exposure_time=exposure,
                  min_cadences_per_t0=5)
    emls.inject_moon(emls.time, emls.flux.copy(), emls.flux_err, emls.bary_t0s, emls.planet_semimajor_axis,
                     emls.planet_ecc, initial_alpha=1.0, **INJECTED_MOON)
    return emls.injected_flux_moon_pandora


def test_injected_moon_is_averaged_over_the_exposure(tmp_path, monkeypatch):
    averaged = injected_moon_flux(tmp_path, THIRTY_MINUTES)
    instantaneous = injected_moon_flux(tmp_path, None)
    monkeypatch.setattr(ExoMoonLeastSquares, 'SUBEXPOSURE_DAYS', 5 / 86400)  # 360 sub-exposures as reference
    reference = injected_moon_flux(tmp_path, THIRTY_MINUTES)

    depth = 1 - np.min(reference)
    assert depth > 0
    assert np.max(np.abs(averaged - reference)) < 0.01 * depth
    assert np.max(np.abs(instantaneous - reference)) > 0.1 * depth


# --- the class measures the exposure by default --------------------------------------------------------------------

def long_cadence_curve():
    t0s = np.arange(3) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, THIRTY_MINUTES) for t0 in t0s])
    return t0s, time, np.ones_like(time), np.full(len(time), 1e-4), 0.2


def test_default_exposure_is_the_median_point_spacing(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        emls = system(tmp_path, curve=long_cadence_curve(), exposure_time='auto', min_cadences_per_t0=5)

    assert inspect.signature(ExoMoonLeastSquares.__init__).parameters['exposure_time'].default == 'auto'
    assert emls.exposure_time == pytest.approx(THIRTY_MINUTES, rel=1e-9)
    assert emls.exposure_source == 'median point spacing'
    assert any('Exposure time 30.00 min from the median point spacing' in r.getMessage() for r in caplog.records)


def test_instantaneous_model_on_long_cadence_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        emls = system(tmp_path, curve=long_cadence_curve(), exposure_time=None, min_cadences_per_t0=5)

    assert emls.exposure_time is None
    assert any('exposure_time=None' in r.getMessage() and '30.0 min' in r.getMessage() for r in caplog.records)


def test_instantaneous_model_on_short_cadence_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        exposure, _ = ExoMoonLeastSquares.resolve_exposure_time(None, np.arange(0, 1, 1 / 1440))

    assert exposure is None and not caplog.records


def test_unmeasurable_spacing_gives_an_instantaneous_model(caplog):
    with caplog.at_level(logging.WARNING):
        exposure, source = ExoMoonLeastSquares.resolve_exposure_time('auto', np.arange(0, 30, 1.5))

    assert exposure is None and source == 'none (instantaneous model)'
    assert any('cannot be measured' in r.getMessage() for r in caplog.records)
    with pytest.raises(ValueError, match="'auto' or None"):
        ExoMoonLeastSquares.resolve_exposure_time('median', np.arange(10.0))
