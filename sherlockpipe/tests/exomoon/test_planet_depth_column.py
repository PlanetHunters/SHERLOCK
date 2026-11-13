"""The search fits one planet depth correction shared by all epochs with each moon hypothesis.

A moon that sits inside the planet transit in most epochs deepens the fold fit by its mean depth; the
subtraction then leaves that depth back where the moon is absent and the moon window loses it. The
search stacks the epochs in the frame of the moon, as always, and fits the moon window together with
the derivative of the planet model with respect to its depth, placed at each epoch's mid-time.
"""
import copy

import numpy as np
import pytest

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


def test_search_measures_the_same_moon_window_whatever_the_limb_darkening_error(tmp_path):
    emls = system(tmp_path)
    inputs = run_to_search(emls)
    emls.build_search_templates()
    search_input = inputs[len(inputs) // 2]
    wrong = copy.copy(emls)
    # The subtracted planet had limb darkening off by +0.15 in u1 and -0.15 in u2: the subtraction
    # leaves that shape error at every epoch.
    u1_column, u2_column = emls.planet_shape_columns[:, 1], emls.planet_shape_columns[:, 2]
    wrong.flux = emls.flux - (0.15 * u1_column - 0.15 * u2_column)
    plain = copy.copy(wrong)
    plain.fit_planet_shape = False

    reference, fitted, unfitted = (e.search(search_input, return_lc=True) for e in (emls, wrong, plain))

    def window_depth(result):
        return np.mean(result[7]) - np.mean(result[5])

    assert abs(window_depth(fitted) - window_depth(reference)) < 1e-6
    assert abs(fitted[3] - reference[3]) < 1e-6
    assert abs(window_depth(unfitted) - window_depth(reference)) > 5e-6


def test_planet_shape_columns_are_symmetric_and_zero_away_from_the_transits(tmp_path):
    emls = system(tmp_path)
    run_to_search(emls)
    columns = emls.planet_shape_columns
    mid_times = [t0 + ttv for t0, ttv in zip(emls.bary_t0s, emls.ttvs)]
    distance = np.min(np.abs(emls.time[:, None] - np.array(mid_times)[None, :]), axis=1)

    assert columns is not None and columns.shape == (len(emls.time), len(ExoMoonLeastSquares.PLANET_SHAPE_TERMS))
    assert np.all(columns[distance > emls.planet_duration] == 0)
    assert np.all(np.any(columns != 0, axis=0))
    # A shape change is the same before and after each mid-time.
    for mid_time in mid_times:
        before = (emls.time < mid_time) & (emls.time > mid_time - emls.planet_duration)
        after = (emls.time > mid_time) & (emls.time < mid_time + emls.planet_duration)
        for term in range(columns.shape[1]):
            total = np.sum(np.abs(columns[before | after, term]))
            assert abs(np.sum(columns[before, term]) - np.sum(columns[after, term])) < 0.1 * total


def test_fixed_impact_parameter_is_kept_by_the_fold_fit(tmp_path):
    emls = system(tmp_path, inc=89.0, fix_impact_param=True)
    emls.planet_impact_param = 0.3
    run_to_search(emls)

    fitted = ExoMoonLeastSquares.impact_parameter(emls.planet_a_over_rstar, emls.planet_inc, emls.planet_ecc,
                                                  emls.planet_arg_periastron)
    assert abs(fitted - 0.3) < 1e-4


def test_impact_column_does_not_vanish_for_a_central_transit(tmp_path):
    """The flux is even in b, so a b derivative vanishes at b = 0; the column is taken in b^2."""
    emls = system(tmp_path, inc=90.0, fix_impact_param=True)
    run_to_search(emls)
    term = ExoMoonLeastSquares.PLANET_SHAPE_TERMS.index("impact_squared")
    column = emls.planet_shape_columns[:, term]
    assert emls.planet_shape_impact < 0.01

    params = emls.null_planet_params()
    mid_time = emls.bary_t0s[0] + emls.ttvs[0]
    near = np.abs(emls.time - mid_time) < emls.planet_duration
    flux = []
    for b in (0.0, 0.1):
        changed = copy.copy(params)
        changed.t0 = mid_time
        changed.inc = float(np.rad2deg(np.arccos(b / changed.a)))
        flux.append(ExoMoonLeastSquares.transit_model(changed, emls.time[near], emls.exposure_time)
                    .light_curve(changed))
    secant = (flux[1] - flux[0]) / 0.1 ** 2
    assert np.max(np.abs(column[near])) > 0.5 * np.max(np.abs(secant))
    assert np.corrcoef(column[near], secant)[0, 1] > 0.95


def stacked_epochs(rng, moon_depth, epoch_errors, noise=1e-4):
    """Stacked moon-frame points of one epoch per entry of epoch_errors: the moon window is the same
    template in all of them, the planet transit (depth column -1) falls at a different place in each,
    and each epoch's planet depth is wrong by its own amount."""
    flux, inside, columns = [], [], []
    for index, (offset, error) in enumerate(zip(np.linspace(-0.6, 0.6, len(epoch_errors)), epoch_errors)):
        phase = np.linspace(-1, 1, 200)
        column = np.where(np.abs(phase - offset) < 0.4, -1.0, 0.0)
        window = np.abs(phase) <= 0.25
        flux.append(1 - moon_depth * window + error * column + rng.normal(0, noise, len(phase)))
        inside.append(window)
        columns.append(np.column_stack((column, np.zeros(len(phase)), np.full(len(phase), index))))
    return np.concatenate(flux), np.concatenate(inside), np.vstack(columns)


def test_epoch_depth_fit_returns_the_moon_when_each_transit_depth_differs():
    rng = np.random.default_rng(21)
    errors = rng.normal(0, 300e-6, 12)
    flux, inside, columns = stacked_epochs(rng, 300e-6, errors)

    shared = ExoMoonLeastSquares.planet_fit(flux, inside, columns)
    per_epoch = ExoMoonLeastSquares.planet_fit(flux, inside, columns, epoch_depth=True)

    def moon_depth(fit):
        corrected = flux - fit["model"]
        return np.mean(corrected[~inside]) - np.mean(corrected[inside])

    assert abs(moon_depth(per_epoch) - 300e-6) < 15e-6
    assert abs(moon_depth(shared) - 300e-6) > abs(moon_depth(per_epoch) - 300e-6)
    fitted = np.array([per_epoch["epoch_depths"][index] for index in range(12)])
    np.testing.assert_allclose(fitted, errors, atol=30e-6)
    assert per_epoch["depth"] == pytest.approx(np.mean(fitted))
    assert shared["epoch_depths"] == {}


def test_planet_correction_rebuilds_the_fitted_model():
    rng = np.random.default_rng(22)
    flux, inside, columns = stacked_epochs(rng, 300e-6, rng.normal(0, 300e-6, 8))
    for epoch_depth in (False, True):
        fit = ExoMoonLeastSquares.planet_fit(flux, inside, columns, epoch_depth=epoch_depth)
        np.testing.assert_allclose(ExoMoonLeastSquares.planet_correction(fit, columns), fit["model"], atol=1e-12)


def test_epoch_depths_keep_their_epoch_without_the_mid_time_fit(tmp_path):
    emls = system(tmp_path)
    run_to_search(emls)
    emls.fit_planet_t0 = False
    columns = emls.search_planet_depth_column()
    assert np.all(columns[:, 1] == 0)
    assert set(np.unique(columns[columns[:, 0] != 0, 2])) == set(range(len(emls.bary_t0s)))
