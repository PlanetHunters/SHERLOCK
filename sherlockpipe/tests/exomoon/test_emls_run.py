"""Tests for the EMLS entrypoint on SHERLOCK results directories (sherlockpipe.emls)."""
import json
import os
import runpy
import sys

from unittest import mock

import batman
import numpy as np
import pandas as pd
import pytest

from sherlockpipe import exomoon, exomoon_run
from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon.test_audit_fixes import CapturePool, StopAtSearch
from sherlockpipe.tests.exomoon import test_search_integrity as guard

CANDIDATE_COLUMNS = ['curve', 'period', 'per_err', 'duration', 't0', 'depth', 'snr', 'sde', 'border_score', 'oi',
                     'rad_p', 'rp_rs', 'a', 'hz', 'Object Id', 'run', 'score', 'fap', 'harmonic', 'depth_err',
                     'depth_sig', 'transits', 'sectors']


def candidate_row(curve, period, duration_days, t0, radius, a_au):
    row = dict.fromkeys(CANDIDATE_COLUMNS, np.nan)
    row.update(curve=curve, period=period, duration=duration_days * 24 * 60, t0=t0, depth=1.0, rad_p=radius,
               a=a_au, oi='', **{'Object Id': 'TIC 1'})
    return row


def write_star(directory, mass=1.0, radius=1.0, ld=(0.3, 0.2)):
    pd.DataFrame([{'obj_id': 'TIC 1', 'radius': radius, 'mass': mass, 'ld_a': ld[0], 'ld_b': ld[1],
                   'R_star': radius, 'M_star': mass}]).to_csv(os.path.join(directory, 'params_star.csv'), index=False)


def write_lc(path, time, flux, flux_err):
    pd.DataFrame({'time': time, 'flux': flux, 'flux_err': flux_err}).to_csv(path, index=False)


def guard_directory(directory, moon_radius=None, extra_candidate=True):
    """A SHERLOCK directory around the search-guard planet, optionally with its model moon."""
    t0s, time, flux, flux_err, duration, params = guard.planet_curve(3)
    if moon_radius is not None:
        flux = guard.with_moon(flux, time, t0s, params, moon_radius)
    rows = [candidate_row(1, guard.PLANET_PERIOD, duration, 0.0, guard.PLANET_RADIUS, guard.PLANET_SEMIMAJOR_AXIS)]
    if extra_candidate:
        # A second planet whose transits fall in the data windows, to be masked.
        rows.append(candidate_row(1, 7.0, 0.1, 1.3, 2.0, 0.07))
        second = np.abs(((time - 1.3 + 3.5) % 7.0) - 3.5) < 0.05
        flux = flux - 5e-3 * second
    pd.DataFrame(rows, columns=CANDIDATE_COLUMNS).to_csv(os.path.join(directory, 'candidates.csv'), index=False)
    write_star(directory, ld=guard.LIMB_DARKENING)
    write_lc(os.path.join(directory, 'lc_1.csv'), time, flux, flux_err)
    write_lc(os.path.join(directory, 'lc_0.csv'), time, flux + 0.01, flux_err)
    return t0s, time, duration


def write_fit(directory, candidate, period, epoch, radius, a_au, inc, impact, duration_h):
    results = os.path.join(directory, f'fit_[{candidate}]', 'main', 'results')
    os.makedirs(results)
    name = f'SOI_{candidate}'
    pd.DataFrame({'#name': ['#Fitted parameters', f'{name}_rr', f'{name}_epoch', f'{name}_period', f'{name}_f_c',
                            f'{name}_f_s'],
                  'median': [np.nan, 0.09, epoch, period, 0.3, 0.4],
                  'lower_error': 0.0, 'upper_error': 0.0}).to_csv(os.path.join(results, 'ns_table.csv'), index=False)
    rows = [
        (f'Companion radius {name}; $R_\\mathrm{{{name}}}$ ($\\mathrm{{R_{{\\oplus}}}}$)', radius, 0.3),
        (f'Companion radius {name}; $R_\\mathrm{{{name}}}$ ($\\mathrm{{R_{{jup}}}}$)', radius / 11.2, 0.03),
        (f'Semi-major axis {name}; $a_\\mathrm{{{name}}}$ ($\\mathrm{{R_{{\\odot}}}}$)', a_au * AU_TO_RSUN, 0.1),
        (f'Semi-major axis {name}; $a_\\mathrm{{{name}}}$ (AU)', a_au, 0.001),
        (f'Inclination {name}; $i_\\mathrm{{{name}}}$ (deg)', inc, 0.2),
        (f'Impact parameter {name}; $b_\\mathrm{{tra;{name}}}$', impact, 0.05),
        (f'Total transit duration {name}; $T_\\mathrm{{tot;{name}}}$ (h)', duration_h, 0.1),
        (f'Impact parameter occultation {name}; $b_\\mathrm{{occ;{name}}}$', 0.99, 0.05),
    ]
    pd.DataFrame(rows, columns=['#property', 'value', 'lower_error']).assign(
        upper_error=lambda d: d['lower_error'], source='derived').to_csv(
        os.path.join(results, 'ns_derived_table.csv'), index=False)


# --- unit: input selection ---------------------------------------------------------------------------------------

def test_fit_results_replace_the_search_parameters(tmp_path):
    guard_directory(tmp_path)
    write_fit(tmp_path, 1, 100.001, 0.002, 9.5, 0.42, 89.9, 0.12, 13.5)

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    planet = inputs['planet']
    assert planet['period'] == 100.001 and planet['t0'] == 0.002
    assert planet['radius'] == 9.5 and planet['radius_err'] == 0.3
    assert planet['a'] == 0.42 and planet['inc'] == 89.9 and planet['impact'] == 0.12
    assert planet['duration'] == pytest.approx(13.5 / 24)
    assert planet['ecc'] == pytest.approx(0.25)
    assert planet['arg_periastron'] == pytest.approx(np.degrees(np.arctan2(0.4, 0.3)))
    assert inputs['sources']['period'].endswith(os.path.join('fit_[1]', 'main'))


def test_no_fit_keeps_the_candidate_parameters(tmp_path):
    guard_directory(tmp_path)
    write_fit(tmp_path, 1, 100.001, 0.002, 9.5, 0.42, 89.9, 0.12, 13.5)

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, use_fit=False)

    assert inputs['planet']['period'] == guard.PLANET_PERIOD
    assert inputs['planet']['radius'] == guard.PLANET_RADIUS
    assert inputs['sources']['period'] == 'candidates.csv'


def test_light_curve_is_the_one_the_candidate_was_found_in(tmp_path):
    guard_directory(tmp_path)

    found = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)
    chosen = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, curve=0)

    assert found['lc_file'].endswith('lc_1.csv')
    assert chosen['lc_file'].endswith('lc_0.csv')
    assert np.median(chosen['flux']) - np.median(found['flux']) == pytest.approx(0.01, abs=1e-6)


def test_masked_candidate_transits_are_removed(tmp_path):
    guard_directory(tmp_path)

    kept = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)
    masked = exomoon_run.build_inputs(str(tmp_path), 1, mask=[2], planet_mass=300.0)

    second = np.abs(((masked['time'] - 1.3 + 3.5) % 7.0) - 3.5) < 0.1
    assert not np.any(second)
    assert len(kept['time']) - len(masked['time']) == pytest.approx(np.sum(
        np.abs(((kept['time'] - 1.3 + 3.5) % 7.0) - 3.5) < 0.1), abs=2)
    assert np.min(masked['flux']) > 1 - 0.02


@pytest.mark.parametrize('mask, message', [([1], 'cannot mask itself'), ([3], 'not found')])
def test_invalid_masks_are_rejected(tmp_path, mask, message):
    guard_directory(tmp_path)

    with pytest.raises(ValueError, match=message):
        exomoon_run.build_inputs(str(tmp_path), 1, mask=mask, planet_mass=300.0)


def test_missing_candidate_is_rejected(tmp_path):
    guard_directory(tmp_path)

    with pytest.raises(ValueError, match='Candidate 5 not found'):
        exomoon_run.build_inputs(str(tmp_path), 5)


def test_moon_radius_range_defaults_and_overrides(tmp_path):
    guard_directory(tmp_path)

    default = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)['moon']
    custom = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, min_moon_radius=1.0,
                                      max_moon_radius=4.0)['moon']

    assert default == {'min_radius': guard.PLANET_RADIUS / 20, 'max_radius': guard.PLANET_RADIUS / 2}
    assert custom == {'min_radius': 1.0, 'max_radius': 4.0}
    with pytest.raises(ValueError, match='Moon radius range'):
        exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, min_moon_radius=4.0, max_moon_radius=1.0)


def test_mass_is_guessed_when_not_given(tmp_path):
    guard_directory(tmp_path)

    given = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)
    guessed = exomoon_run.build_inputs(str(tmp_path), 1)

    assert given['planet']['mass'] == 300.0 and given['sources']['mass'] == 'user'
    assert guessed['planet']['mass'] == exomoon_run.JUPITER_MASS
    assert guessed['sources']['mass'] == 'Jupiter mass (giant radius)'


def test_small_planet_mass_comes_from_the_forecaster(tmp_path):
    guard_directory(tmp_path)
    candidates = pd.read_csv(tmp_path / 'candidates.csv')
    candidates.loc[0, 'rad_p'] = 3.9
    candidates.to_csv(tmp_path / 'candidates.csv', index=False)

    guessed = exomoon_run.build_inputs(str(tmp_path), 1)

    assert guessed['sources']['mass'].startswith('Chen & Kipping')
    assert guessed['planet']['mass'] == exomoon_run.guess_planet_mass(3.9)


def test_mass_forecast_is_reproducible_and_plausible():
    np.random.seed(1)
    neptune = exomoon_run.guess_planet_mass(3.9)
    np.random.seed(2)
    again = exomoon_run.guess_planet_mass(3.9)
    state = np.random.get_state()

    earth = exomoon_run.guess_planet_mass(1.0)

    assert again == neptune
    assert 0.5 < earth < 2.0
    assert 8 < neptune < 30
    assert np.all(np.random.get_state()[1] == state[1])


def test_giant_mass_falls_back_to_jupiter(caplog):
    with caplog.at_level('WARNING'):
        mass = exomoon_run.guess_planet_mass(11.0)

    assert mass == exomoon_run.JUPITER_MASS
    assert any('--planet_mass' in record.getMessage() for record in caplog.records)


def test_epochs_include_transits_before_the_catalog_epoch():
    time = np.linspace(0, 100, 1000)

    epochs = exomoon_run.epochs_in_span(55.0, 20.0, time)

    assert epochs == [-5.0, 15.0, 35.0, 55.0, 75.0, 95.0]


def test_star_without_limb_darkening_is_rejected(tmp_path):
    guard_directory(tmp_path)
    write_star(tmp_path, ld=(np.nan, 0.2))

    with pytest.raises(ValueError, match='limb darkening'):
        exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)


# --- end to end: the command on a SHERLOCK directory -------------------------------------------------------------

def run_command(directory, monkeypatch, *arguments):
    monkeypatch.setattr(sys, 'argv', ['emls.py', '--object_dir', str(directory), *arguments])
    runpy.run_module('sherlockpipe.emls', run_name='__main__')
    with open(os.path.join(directory, 'emls_1', 'emls_results.json')) as file:
        return json.load(file)


def test_command_finds_a_moon_in_a_sherlock_directory(tmp_path, monkeypatch):
    t0s, _, _ = guard_directory(tmp_path, moon_radius=3.0)
    (tmp_path / 'emls_1').mkdir()
    (tmp_path / 'emls_1' / 'stale.txt').write_text('old run')
    monkeypatch.chdir(tmp_path)

    results = run_command(tmp_path, monkeypatch, '--candidate', '1', '--mask_candidates', '2', '--planet_mass', '300',
                          '--min_moon_radius', '1', '--max_moon_radius', '4',
                          '--period_grid_size', str(guard.PERIOD_GRID), '--radius_grid_size', '4', '--cpus', '2')

    assert results['masked_candidates'] == [2]
    assert results['epochs_used'] == results['epochs_with_data'] == len(t0s)
    assert results['epochs_without_data'] == []
    assert abs(results['best_period'] - guard.moon_period()) / guard.moon_period() < 0.01
    assert results['max_snr'] > 10
    assert not (tmp_path / 'emls_1' / 'stale.txt').exists()
    assert os.path.exists(tmp_path / 'emls_1' / 'emls.log')
    assert os.path.exists(tmp_path / 'emls_1' / 'stats.png')


def test_command_without_a_moon_stays_quiet(tmp_path, monkeypatch):
    guard_directory(tmp_path)
    monkeypatch.chdir(tmp_path)

    results = run_command(tmp_path, monkeypatch, '--candidate', '1', '--mask_candidates', '2', '--planet_mass', '300',
                          '--min_moon_radius', '1', '--max_moon_radius', '4',
                          '--period_grid_size', str(guard.PERIOD_GRID), '--radius_grid_size', '4', '--cpus', '2')

    assert results['max_snr'] < 5
    assert results['sources']['inc'].startswith('no fit')
    text = (tmp_path / 'emls_1' / 'emls_results.json').read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    assert not (tmp_path / 'emls_1.partial').exists()


# --- validator round 1 -------------------------------------------------------------------------------------------

def test_without_a_fit_the_inclination_starts_central_and_is_free(tmp_path, caplog):
    guard_directory(tmp_path)
    candidates = pd.read_csv(tmp_path / 'candidates.csv')
    candidates.loc[0, 'duration'] *= 0.75
    candidates.to_csv(tmp_path / 'candidates.csv', index=False)

    with caplog.at_level('WARNING'):
        planet = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)['planet']

    assert any('no ' in record.getMessage() and 'Run sherlockpipe.fit first' in record.getMessage()
               for record in caplog.records)
    a_over_rstar = guard.PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN
    min_inc = ExoMoonLeastSquares.min_transit_inclination(a_over_rstar, guard.PLANET_RADIUS * R_earth_to_R_sun,
                                                          0.0, 0.0)
    assert planet['inc'] == 90.0 and planet['impact'] == 0.0
    assert planet['inc'] - planet['inc_err'] == pytest.approx(min_inc)


def test_fold_fit_recovers_the_impact_parameter_from_a_short_search_duration(tmp_path):
    """A search duration 25 % shorter than T14, as from TLS, must not bias the planet geometry."""
    period, a_au, b, radius = 100.0, guard.PLANET_SEMIMAJOR_AXIS, 0.6, 10.0
    a_over_rstar = a_au * AU_TO_RSUN
    inc = np.degrees(np.arccos(b / a_over_rstar))
    k = radius * R_earth_to_R_sun
    t14 = period / np.pi * np.arcsin(np.sqrt((1 + k) ** 2 - b ** 2) / (a_over_rstar * np.sin(np.radians(inc))))
    t0s = np.arange(8) * period
    time = np.concatenate([t0 + np.arange(-3.0, 3.0, 5 / 1440) for t0 in t0s])
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = 0.0, period, k, a_over_rstar, inc
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, guard.LIMB_DARKENING, 'quadratic'
    flux = batman.TransitModel(params, time).light_curve(params) + np.random.default_rng(2).normal(0, 1e-4, len(time))
    pd.DataFrame([candidate_row(1, period, 0.75 * t14, 0.0, radius, a_au)], columns=CANDIDATE_COLUMNS).to_csv(
        tmp_path / 'candidates.csv', index=False)
    write_star(tmp_path, ld=guard.LIMB_DARKENING)
    write_lc(tmp_path / 'lc_1.csv', time, flux, np.full(len(time), 1e-4))

    with mock.patch.object(exomoon, 'Pool', CapturePool), pytest.raises(StopAtSearch):
        exomoon_run.run_emls(str(tmp_path), 1, planet_mass=300.0, period_grid_size=40, radius_grid_size=4, cpus=1)

    log = (tmp_path / 'emls_1.partial' / 'emls.log').read_text().splitlines()
    fold = [line.split('INFO', 1)[1].strip() for line in log if 'Fold fit' in line]
    fitted_inc = float(fold[0].split('inc=')[1].split('->')[1].split(',')[0])
    assert fitted_inc == pytest.approx(inc, abs=0.1)


def test_failed_rerun_keeps_previous_results_and_logs_the_error(tmp_path):
    guard_directory(tmp_path)
    (tmp_path / 'emls_1').mkdir()
    (tmp_path / 'emls_1' / 'emls_results.json').write_text('{"previous": true}')

    with pytest.raises(ValueError, match='cannot mask itself'):
        exomoon_run.run_emls(str(tmp_path), 1, mask=[1], planet_mass=300.0)

    assert (tmp_path / 'emls_1' / 'emls_results.json').read_text() == '{"previous": true}'
    log = (tmp_path / 'emls_1.partial' / 'emls.log').read_text()
    assert 'EMLS failed for candidate 1' in log and 'cannot mask itself' in log


def test_symlinked_output_directory_is_refused(tmp_path):
    guard_directory(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep.txt').write_text('keep')
    os.symlink(outside, tmp_path / 'emls_1')

    with pytest.raises(ValueError, match='symbolic link'):
        exomoon_run.run_emls(str(tmp_path), 1, planet_mass=300.0)

    assert (outside / 'keep.txt').read_text() == 'keep'


def test_no_fit_warning_names_its_cause(tmp_path, caplog):
    guard_directory(tmp_path)
    write_fit(tmp_path, 1, 100.001, 0.002, 9.5, 0.42, 89.9, 0.12, 13.5)

    with caplog.at_level('WARNING'):
        exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, use_fit=False)

    messages = [record.getMessage() for record in caplog.records if 'No fitted inclination' in record.getMessage()]
    assert len(messages) == 1
    assert '--no_fit' in messages[0] and 'Run sherlockpipe.fit first' not in messages[0]


def drop_derived_rows(directory, prefixes):
    derived = directory / 'fit_[1]' / 'main' / 'results' / 'ns_derived_table.csv'
    table = pd.read_csv(derived)
    table[~table['#property'].str.startswith(prefixes)].to_csv(derived, index=False)


def test_fit_impact_parameter_gives_the_inclination_without_an_inclination_row(tmp_path, caplog):
    guard_directory(tmp_path)
    write_fit(tmp_path, 1, 100.001, 0.002, 9.5, 0.42, 89.9, 0.12, 13.5)
    drop_derived_rows(tmp_path, ('Inclination',))

    with caplog.at_level('WARNING'):
        inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    planet = inputs['planet']
    a_over_rstar = 0.42 * AU_TO_RSUN
    e, w = 0.25, np.radians(np.degrees(np.arctan2(0.4, 0.3)))
    expected = np.degrees(np.arccos(0.12 / a_over_rstar * (1 + e * np.sin(w)) / (1 - e ** 2)))
    assert planet['impact'] == 0.12
    assert planet['inc'] == pytest.approx(expected)
    assert inputs['sources']['inc'] == 'from the fitted impact parameter'
    assert not any('No fitted inclination' in record.getMessage() for record in caplog.records)


def test_fit_without_inclination_or_impact_is_reported(tmp_path, caplog):
    guard_directory(tmp_path)
    write_fit(tmp_path, 1, 100.001, 0.002, 9.5, 0.42, 89.9, 0.12, 13.5)
    drop_derived_rows(tmp_path, ('Inclination', 'Impact parameter'))

    with caplog.at_level('WARNING'):
        inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    messages = [record.getMessage() for record in caplog.records if 'No fitted inclination' in record.getMessage()]
    assert len(messages) == 1 and 'no inclination or impact parameter' in messages[0]
    assert os.path.join('fit_[1]', 'main') in messages[0].split('a/R* from')[1]
    assert inputs['planet']['period'] == 100.001


def test_default_errors_are_recorded_with_their_source(tmp_path):
    guard_directory(tmp_path)

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    assert inputs['planet']['radius_err'] == pytest.approx(0.1 * guard.PLANET_RADIUS)
    assert inputs['sources']['radius_err'] == 'default: 10% of the radius'
    assert np.isfinite(inputs['planet']['inc_err'])
