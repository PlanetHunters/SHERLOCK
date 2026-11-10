"""The search fits one planet depth correction shared by all epochs with each moon hypothesis.

A moon that sits inside the planet transit in most epochs deepens the fold fit by its mean depth; the
subtraction then leaves that depth back where the moon is absent and the moon window loses it. The
search stacks the epochs in the frame of the moon, as always, and fits the moon window together with
the derivative of the planet model with respect to its depth, placed at each epoch's mid-time.
"""
import copy

import numpy as np

from sherlockpipe.exomoon import ExoMoonLeastSquares
from sherlockpipe.tests.exomoon.test_audit_fixes import EPOCHS, run_to_search, system


def stacked_points(rng, moon_depth, planet_error, noise=1e-4):
    """Stacked moon-frame points of 12 epochs: the moon window is the same template in all of them,
    while the planet transit (column -1 inside it) falls at a different place in each epoch."""
    it_columns, oot_columns = [], []
    for offset in np.linspace(-0.6, 0.6, 12):
        phase = np.linspace(-1, 1, 200)
        inside = np.abs(phase) <= 0.25
        column = np.where(np.abs(phase - offset) < 0.4, -1.0, 0.0)
        it_columns.append(column[inside])
        oot_columns.append(column[~inside])
    it_column, oot_column = np.concatenate(it_columns), np.concatenate(oot_columns)
    it_flux = 1 - moon_depth + planet_error * it_column + rng.normal(0, noise, len(it_column))
    oot_flux = 1 + planet_error * oot_column + rng.normal(0, noise, len(oot_column))
    return it_flux, oot_flux, it_column, oot_column


def test_depth_fit_returns_the_moon_depth_that_a_planet_depth_error_hides():
    rng = np.random.default_rng(3)
    # The fold fit took 300 ppm too much planet depth: the residual is +300 ppm inside the transit.
    it_flux, oot_flux, it_column, oot_column = stacked_points(rng, moon_depth=300e-6, planet_error=-300e-6)
    plain = np.mean(oot_flux) - np.mean(it_flux)

    it_fixed, oot_fixed, correction, inflation = ExoMoonLeastSquares.planet_depth_fit(it_flux, oot_flux,
                                                                                    it_column, oot_column)

    assert plain < 250e-6
    assert abs(np.mean(oot_fixed) - np.mean(it_fixed) - 300e-6) < 15e-6
    assert abs(correction + 300e-6) < 15e-6
    assert inflation > 1


def test_depth_fit_without_a_planet_error_keeps_the_moon_depth():
    rng = np.random.default_rng(4)
    it_flux, oot_flux, it_column, oot_column = stacked_points(rng, moon_depth=300e-6, planet_error=0.0)

    it_fixed, oot_fixed, correction, _ = ExoMoonLeastSquares.planet_depth_fit(it_flux, oot_flux, it_column,
                                                                            oot_column)

    assert abs(correction) < 15e-6
    assert abs(np.mean(oot_fixed) - np.mean(it_fixed) - 300e-6) < 15e-6


def test_depth_fit_without_planet_overlap_is_the_plain_difference_of_means():
    rng = np.random.default_rng(5)
    it_flux, oot_flux = 1 - 3e-4 + rng.normal(0, 1e-4, 50), 1 + rng.normal(0, 1e-4, 200)

    it_fixed, oot_fixed, correction, inflation = ExoMoonLeastSquares.planet_depth_fit(
        it_flux, oot_flux, np.zeros(50), np.zeros(200))

    assert correction == 0.0 and inflation == 1.0
    assert np.array_equal(it_fixed, it_flux) and np.array_equal(oot_fixed, oot_flux)


def test_planet_depth_column_is_the_planet_shape_at_each_epoch(tmp_path):
    emls = system(tmp_path)
    run_to_search(emls)
    column = emls.planet_depth_column
    mid_times = [t0 + ttv for t0, ttv in zip(emls.bary_t0s, emls.ttvs)]
    distance = np.min(np.abs(emls.time[:, None] - np.array(mid_times)[None, :]), axis=1)

    assert column is not None and len(column) == len(emls.time) == len(emls.flux)
    assert len(mid_times) == EPOCHS
    assert np.all(column[distance > emls.planet_duration] == 0)
    assert np.all(column <= 0)
    # Full depth: the flux drops by about one unit per unit of depth (limb darkening adds a little).
    centre = column[distance < 0.1 * emls.planet_duration]
    assert np.all((centre < -0.9) & (centre > -1.5))


def test_mask_mode_has_no_planet_depth_column(tmp_path):
    emls = system(tmp_path, subtract_mode='mask')
    run_to_search(emls)

    assert emls.planet_depth_column is None


def test_search_measures_the_same_moon_window_whatever_the_subtracted_planet_depth(tmp_path):
    emls = system(tmp_path)
    inputs = run_to_search(emls)
    emls.build_search_templates()
    search_input = inputs[len(inputs) // 2]
    deeper = copy.copy(emls)
    # A fold fit 500 ppm too deep leaves +500 ppm where the planet transits.
    deeper.flux = emls.flux - 500e-6 * emls.planet_depth_column
    plain = copy.copy(deeper)
    plain.planet_depth_column = None

    reference, shifted, uncorrected = (e.search(search_input, return_lc=True) for e in (emls, deeper, plain))

    def window_depth(result):
        return np.mean(result[7]) - np.mean(result[5])

    assert abs(window_depth(shifted) - window_depth(reference)) < 1e-6
    assert abs(shifted[3] - reference[3]) < 1e-6
    assert abs(window_depth(uncorrected) - window_depth(reference)) > 20e-6


def test_search_measures_the_same_moon_window_whatever_the_epoch_mid_time_errors(tmp_path):
    emls = system(tmp_path)
    inputs = run_to_search(emls)
    emls.build_search_templates()
    search_input = inputs[len(inputs) // 2]
    shifted = copy.copy(emls)
    # Each epoch's mid-time fit pulled by a different amount, as a moon at a different place does.
    shifts = np.array([3.0, -2.0, 4.0, -1.0, 2.5, -3.5]) / 1440
    error = np.zeros(len(emls.time))
    for index, shift in enumerate(shifts):
        error += np.where(emls.planet_epoch_index == index, shift * emls.planet_t0_column, 0.0)
    shifted.flux = emls.flux - error
    plain = copy.copy(shifted)
    plain.fit_planet_t0 = False

    reference, moved, depth_only = (e.search(search_input, return_lc=True) for e in (emls, shifted, plain))

    def window_depth(result):
        return np.mean(result[7]) - np.mean(result[5])

    assert abs(window_depth(moved) - window_depth(reference)) < 1e-6
    assert abs(moved[3] - reference[3]) < 1e-6
    assert abs(window_depth(depth_only) - window_depth(reference)) > 5e-6


def test_planet_t0_column_is_antisymmetric_about_each_mid_time(tmp_path):
    emls = system(tmp_path)
    run_to_search(emls)
    mid_times = [t0 + ttv for t0, ttv in zip(emls.bary_t0s, emls.ttvs)]

    assert emls.planet_t0_column is not None and set(np.unique(emls.planet_epoch_index)) == {-1, *range(EPOCHS)}
    for index, mid_time in enumerate(mid_times):
        mine = emls.planet_epoch_index == index
        before = mine & (emls.time < mid_time)
        after = mine & (emls.time > mid_time)
        # A later planet leaves more flux at ingress and less at egress.
        assert np.sum(emls.planet_t0_column[before]) > 0 > np.sum(emls.planet_t0_column[after])
        assert abs(np.sum(emls.planet_t0_column[before]) + np.sum(emls.planet_t0_column[after])) < \
            0.1 * np.sum(np.abs(emls.planet_t0_column[mine]))
