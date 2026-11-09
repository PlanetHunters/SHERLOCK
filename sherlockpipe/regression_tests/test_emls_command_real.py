"""Realistic end-to-end test of ``python -m sherlockpipe.emls`` on real Kepler data.

The light curve is the real PDCSAP flux of Kepler-706 b (KOI K00855.01, KIC 6522242), cut to
+-3 d around each transit (``emls_data/kepler706_pdcsap_windows.csv.gz``, written by
``emls_data/make_kepler706_fixture.py``). Unlike the synthetic tests, it has real noise, gaps and
systematics, and the moon comes from an independent model:

1. A Pandora moon is added to the raw flux, averaged over the Kepler long-cadence exposure.
2. The raw curve is written as ``lc_0.csv``, which the command detrends itself with a biweight that
   masks every moon window (audit item A1). The SHERLOCK-style unmasked biweight is written as
   ``lc_1.csv``, the curve the candidate was found in.
3. The candidate duration is 80 % of T14, like a TLS search duration, and there is no fit.
4. The command runs with its default period grid and the guessed planet mass.

The same steps without a moon give the control.

Scope: one wide-orbit moon near the outer period limit (0.9 of the Domingos et al. 2006 prograde period limit), 2 R_earth,
about SNR 16. That is the easiest regime for EMLS: the largest timing offset, well outside the
planet transit. It shows that the command works end to end on real data, not that EMLS recovers
moons in general. Close moons that transit during the planet are still absorbed by the planet fit
(audit items B3 and C1). The 1 % period check needs the default 10000-period grid, because the top
of the grid has a competing peak near 9.94 d on this curve.

The command runs with ``--null_runs 0``: the fixture keeps only +-3 d around each transit, so the
pseudo-epochs of the false-alarm probability would all come from the same thin strip of data just
outside the moon window, and 100 of them take about 45 minutes per run.
"""
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd
import pandoramoon as pandora
import pytest
import wotan
from astropy import constants as ac

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, M_earth_to_M_sun

DATA = os.path.join(os.path.dirname(__file__), 'emls_data', 'kepler706_pdcsap_windows.csv.gz')
PERIOD = 41.408316686
EPOCH = 2455028.78745 - 2454833
T14 = 5.0264 / 24
STAR_RADIUS = 0.747
STAR_LOGG = 4.586
IMPACT = 0.377
PLANET_RADIUS = 11.38
LD = (0.45, 0.20)
JUPITER_MASS = 317.8
KEPLER_LONG_CADENCE = 1765.5  # s: 270 frames of 6.54 s, the window each long-cadence point averages
MOON_RADIUS = 2.0
MOON_PERIOD_FRACTION = 0.9
TAU_MOON = 0.37
DETREND_WINDOW = 6 * T14
CANDIDATE_COLUMNS = ['curve', 'period', 'per_err', 'duration', 't0', 'depth', 'snr', 'sde', 'border_score', 'oi',
                     'rad_p', 'rp_rs', 'a', 'hz', 'Object Id', 'run', 'score', 'fap', 'harmonic', 'depth_err',
                     'depth_sig', 'transits', 'sectors']


def star_mass():
    gravity = 10 ** STAR_LOGG / 100
    return gravity * (STAR_RADIUS * ac.R_sun.value) ** 2 / ac.G.value / ac.M_sun.value


def moon_period():
    mass = JUPITER_MASS * M_earth_to_M_sun
    a_au = ExoMoonLeastSquares.compute_semimajor_axis(star_mass(), PERIOD)
    widest = ExoMoonLeastSquares.max_moon_semimajor_axis(star_mass(), mass, a_au)
    return MOON_PERIOD_FRACTION * ExoMoonLeastSquares.au_to_period(mass, widest)


def pandora_moon(time):
    """Pandora moon flux for Kepler-706 b, averaged over the Kepler long-cadence exposure."""
    a_au = ExoMoonLeastSquares.compute_semimajor_axis(star_mass(), PERIOD)
    params = pandora.model_params()
    params.R_star = STAR_RADIUS * ac.R_sun.value
    params.u1, params.u2 = LD
    params.per_bary, params.a_bary = PERIOD, a_au * AU_TO_RSUN / STAR_RADIUS
    params.r_planet, params.b_bary = 1e-6, IMPACT  # the planet is already in the data
    params.t0_bary, params.t0_bary_offset = EPOCH, 0
    params.M_planet, params.w_bary, params.ecc_bary = JUPITER_MASS * ac.M_earth.value, 0, 0
    params.r_moon = MOON_RADIUS * ac.R_earth.value / params.R_star
    params.per_moon, params.tau_moon = moon_period(), TAU_MOON
    params.Omega_moon, params.i_moon, params.ecc_moon, params.w_moon = 0, 90, 0, 0
    params.M_moon = 1e-3 * ac.M_earth.value
    params.epoch_distance, params.supersampling_factor = PERIOD, 1
    params.occult_small_threshold, params.hill_sphere_threshold = 0.1, 1
    offsets = ((np.arange(15) + 0.5) / 15 - 0.5) * (KEPLER_LONG_CADENCE / 86400)
    fine = (time[:, None] + offsets[None, :]).ravel()
    moon_flux = np.asarray(pandora.moon_model(params).light_curve(fine)[2], float)
    return moon_flux.reshape(len(time), len(offsets)).mean(axis=1)


def sherlock_directory(directory, with_moon):
    data = pd.read_csv(DATA)
    time, flux, flux_err = (data[column].to_numpy(float) for column in ('time', 'flux', 'flux_err'))
    if with_moon:
        flux = flux * pandora_moon(time)
    pd.DataFrame({'time': time, 'flux': flux, 'flux_err': flux_err}).to_csv(os.path.join(directory, 'lc_0.csv'),
                                                                          index=False)
    flat = wotan.flatten(time, flux, method='biweight', window_length=DETREND_WINDOW)
    keep = np.isfinite(flat)
    pd.DataFrame({'time': time[keep], 'flux': flat[keep], 'flux_err': flux_err[keep]}).to_csv(
        os.path.join(directory, 'lc_1.csv'), index=False)
    row = dict.fromkeys(CANDIDATE_COLUMNS, np.nan)
    row.update(curve=1, period=PERIOD, duration=0.8 * T14 * 24 * 60, t0=EPOCH, depth=19.5, rad_p=PLANET_RADIUS,
               a=ExoMoonLeastSquares.compute_semimajor_axis(star_mass(), PERIOD), oi='K00855.01',
               **{'Object Id': 'KIC 6522242'})
    pd.DataFrame([row], columns=CANDIDATE_COLUMNS).to_csv(os.path.join(directory, 'candidates.csv'), index=False)
    pd.DataFrame([{'obj_id': 'KIC 6522242', 'radius': STAR_RADIUS, 'mass': star_mass(), 'ld_a': LD[0],
                   'ld_b': LD[1]}]).to_csv(os.path.join(directory, 'params_star.csv'), index=False)


def run_command(directory, monkeypatch):
    monkeypatch.chdir(directory)
    monkeypatch.setattr(sys, 'argv', ['emls.py', '--object_dir', str(directory), '--candidate', '1',
                                      '--cpus', str(min(8, os.cpu_count() or 1)), '--null_runs', '0'])
    runpy.run_module('sherlockpipe.emls', run_name='__main__')
    with open(os.path.join(directory, 'emls_1', 'emls_results.json')) as file:
        return json.load(file)


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    results = {}
    try:
        for name, with_moon in (('moon', True), ('control', False)):
            directory = tmp_path_factory.mktemp(name)
            sherlock_directory(directory, with_moon)
            results[name] = run_command(directory, monkeypatch)
    finally:
        monkeypatch.undo()
    return results


def test_the_injected_moon_period_is_found(runs):
    found = runs['moon']['best_period']

    assert abs(found - moon_period()) / moon_period() < 0.01, f"found {found:.4f} d, injected {moon_period():.4f} d"


def test_the_moon_stands_out_and_the_control_stays_quiet(runs):
    assert runs['moon']['max_snr'] > 10
    assert runs['control']['max_snr'] < 5
    assert runs['moon']['max_snr'] > 2.5 * runs['control']['max_snr']


@pytest.mark.parametrize('name', ['moon', 'control'])
def test_most_epochs_survive_the_planet_subtraction(runs, name):
    # Two predicted epochs fall in Kepler data gaps and are reported, not silently dropped.
    assert runs[name]['epochs_in_span'] == 35
    assert runs[name]['epochs_kept'] == 33
    assert [epoch['points'] for epoch in runs[name]['epochs_skipped']] == [0, 0]
    assert runs[name]['epochs_used'] >= 30


def test_the_moon_phase_is_found(runs):
    """The recovered alpha must match the injected Pandora phase, which checks the timing model."""
    injected_alpha = (2 * np.pi * (EPOCH / moon_period() - TAU_MOON) - np.pi) % (2 * np.pi)
    error = abs((runs['moon']['best_alpha'] - injected_alpha + np.pi) % (2 * np.pi) - np.pi)

    assert error < 0.3, f"found alpha {runs['moon']['best_alpha']:.3f}, injected {injected_alpha:.3f}"


def test_the_moon_radius_is_of_the_right_size(runs):
    # EMLS scales the template to the mean in-window depth, which sits below the peak (audit item H13).
    assert 1.0 < runs['moon']['best_radius'] < 2.5


def test_inputs_follow_the_no_fit_path(runs):
    sources = runs['moon']['sources']

    assert sources['mass'] == 'Jupiter mass (giant radius)'
    assert sources['inc'].startswith('no fit')
    assert runs['moon']['exposure_time_s'] == pytest.approx(KEPLER_LONG_CADENCE, abs=1)
    assert sources['exposure_time'] == 'median point spacing of lc_0.csv'
    assert runs['moon']['detrending']['by'] == 'emls'
    assert runs['moon']['period_grid_size'] == 10000
