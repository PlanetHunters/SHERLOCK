"""Scrambled-phase nulls (referee comment M1): searches on the real epochs with the moon phases
permuted between epochs keep every planet residual and remove only the coherence of a moon orbit.
"""
import numpy as np
import pytest

import sherlockpipe.exomoon as exomoon
from sherlockpipe.tests.exomoon import test_search_integrity as guard


def scrambled_run(tmp_path, moon_radius=None, residual=None, **kwargs):
    t0s, time, flux, flux_err, duration, params = guard.planet_curve(3)
    if moon_radius is not None:
        flux = guard.with_moon(flux, time, t0s, params, moon_radius)
    if residual is not None:
        # The same dip at the centre of every transit: a planet-model error, not a moon.
        depth, width = residual
        for t0 in t0s:
            flux[np.abs(time - t0) < width / 2] -= depth
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration, **kwargs)
    emls.run()
    return emls


def best_snr(emls):
    periods, alphas, _, residuals = emls.grid_search(progress=False)
    return emls.select_best(periods, alphas, emls.snr_matrix(residuals, len(alphas), len(periods)))[3]


@pytest.fixture(scope='module')
def moon_scrambled(tmp_path_factory):
    return scrambled_run(tmp_path_factory.mktemp('moon_scrambled'), moon_radius=3.0, scramble_runs=20,
                         scramble_trigger_fap=None)


@pytest.fixture(scope='module')
def residual_scrambled(tmp_path_factory):
    return scrambled_run(tmp_path_factory.mktemp('residual_scrambled'), residual=(1.5e-3, 0.05), scramble_runs=20,
                         scramble_trigger_fap=None)


def test_an_identity_phase_map_is_the_real_search(moon_scrambled):
    emls = moon_scrambled
    assert emls.moon_phase_map is None  # restored after the scrambled searches
    emls.moon_phase_map = {float(t0): float(t0) for t0 in emls.bary_t0s}
    try:
        assert best_snr(emls) == pytest.approx(emls.max_snr, rel=1e-9)
    finally:
        emls.moon_phase_map = None
        emls._search_stack_cache = None
    assert best_snr(emls) == pytest.approx(emls.max_snr, rel=1e-9)


def test_a_coherent_moon_passes_the_scrambled_nulls(moon_scrambled):
    emls = moon_scrambled
    assert len(emls.scramble_snrs) == 20 and emls.scramble_stopped_early is False
    assert max(emls.scramble_snrs) < emls.max_snr
    assert emls.scramble_passed is True
    assert emls.detected is None  # no pseudo-epoch nulls requested: the scrambled ones do not detect


def test_a_constant_planet_residual_fails_the_scrambled_nulls(residual_scrambled):
    emls = residual_scrambled
    # The scrambled searches find the residual as well as the real one does.
    assert np.median(emls.scramble_snrs) > 0.8 * emls.max_snr
    assert emls.scramble_fap > 0.1
    assert emls.scramble_passed is False


def test_scrambling_moves_the_moon_but_not_the_data(moon_scrambled):
    emls = moon_scrambled
    search_input = emls.search_input(emls.recovered_period, emls.recovered_alpha, emls.planet_mass,
                                      emls.recovered_period, emls.planet_impact_param)
    epochs = [float(t0) for t0 in emls.bary_t0s]
    real = emls.split_scenarios(search_input, emls.time, emls.flux, emls.flux_err, emls.bary_t0s)
    emls.moon_phase_map = dict(zip(epochs, epochs[1:] + epochs[:1]))
    try:
        scrambled = emls.split_scenarios(search_input, emls.time, emls.flux, emls.flux_err, emls.bary_t0s)
    finally:
        emls.moon_phase_map = None
    # Same points (up to the upper clipping, which depends on the window), but the moon window no
    # longer follows the moon: the window depth drops.
    assert len(scrambled[0]) == pytest.approx(len(real[0]), rel=0.02)
    real_depth = np.mean(real[5]) - np.mean(real[2])
    scrambled_depth = np.mean(scrambled[5]) - np.mean(scrambled[2])
    assert real_depth > 2 * scrambled_depth


def test_scrambled_nulls_need_the_trigger(tmp_path):
    emls = scrambled_run(tmp_path, scramble_runs=20)  # trigger 0.1 but no pseudo-epoch FAP
    assert emls.scramble_snrs is None and emls.scramble_passed is None


def test_too_few_epochs_give_no_scrambled_nulls(moon_scrambled):
    emls = moon_scrambled
    saved = emls.bary_t0s
    emls.bary_t0s = saved[:exomoon.MIN_SCRAMBLE_EPOCHS - 1]
    try:
        assert emls.run_scrambled_nulls(emls.max_snr) == []
    finally:
        emls.bary_t0s = saved


def test_negative_scramble_runs_are_rejected(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    with pytest.raises(ValueError, match='scramble_runs'):
        guard.build(flux, time, flux_err, tmp_path, t0s, duration, scramble_runs=-1)
