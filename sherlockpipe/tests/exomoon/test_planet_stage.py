"""Planet-stage fixes: audit items B2, B6, B8, D2, D4, E2 and E3.

B8: the fold fit ends with a bounded least-squares run, so it converges on large folds. B6: the
fold is refitted on the fitted epoch times when that helps more than noise would. B2: the fold rp
starts from the observed depth. D2: the stacked residual is tested for a planet shape left behind.
D4: mask mode centres an unusable epoch on the ephemeris. E2: the default mask has a margin.
E3: mask mode reports how much of a moon's transit time survives the mask.
"""
import logging

import batman
import numpy as np
import pytest
from scipy.optimize import least_squares

from sherlockpipe.exomoon import ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon.test_audit_fixes import PLANET_RADIUS
from sherlockpipe.tests.exomoon.test_planet_fit import (CADENCE, curve, emls_for, planet_params, stacked_residual_peak,
                                                        t14)


def epoch_count(emls):
    return len(emls.bary_t0s)


# --- B8 ----------------------------------------------------------------------------------------------------------

def test_large_fold_converges_to_the_least_squares_optimum():
    """Acceptance: on a large fold the chi2 is within 0.1 of a least-squares fit from the truth."""
    truth = planet_params(impact=0.5)
    rng = np.random.default_rng(3)
    time = np.sort(rng.uniform(-0.3, 0.3, 62100))
    centred = planet_params(impact=0.5)
    flux = batman.TransitModel(centred, time).light_curve(centred) + rng.normal(0, 5e-4, len(time))
    err = np.full(len(time), 5e-4)
    duration = t14(truth)
    bounds = [(-duration / 2, duration / 2), (duration / 2, 2 * duration), (0.0, 1.2), (0.05, 0.15)]
    guess = [0.0, duration * 1.05, 0.3, truth.rp * 1.03]

    theta, model, _, status = ExoMoonLeastSquares.fit_fold_shape(time, flux, err, guess, bounds, planet_params(),
                                                                 None)
    fitted = np.sum(((flux - model) / err) ** 2)

    params = planet_params()

    def residuals(x):
        geometry = ExoMoonLeastSquares.geometry_from_shape(x[1], x[2], x[3], params.per, params.ecc, params.w)
        if geometry is None:
            return np.full(len(time), 1e6)
        params.t0, params.rp = x[0], x[3]
        params.a, params.inc = geometry
        return (flux - batman.TransitModel(params, time).light_curve(params)) / err

    reference = least_squares(residuals, [0.0, duration, 0.5, truth.rp], bounds=tuple(zip(*bounds)), x_scale='jac')
    assert status['converged']
    assert fitted - 2 * reference.cost < 0.1


# --- B6 ----------------------------------------------------------------------------------------------------------

def test_fold_is_realigned_on_large_ttvs(tmp_path):
    """Acceptance: with a 30-minute sinusoidal TTV the fold rp is within 0.3 % (100 epochs, 500 ppm).
    The audit's 30 ppm residual target is below this binning's noise (about 32 ppm per 10-minute bin),
    so the stacked residual is compared with the same curve without TTVs."""
    truth = planet_params()
    true_rp = truth.rp
    shifts = 30 / 1440 * np.sin(2 * np.pi * np.arange(100) / 17)
    data = curve(truth, epochs=100, noise=5e-4, shifts=shifts, seed=5)
    emls = emls_for(tmp_path, data, t14(truth), ttv_prior_width=40 / 1440)
    reference_data = curve(planet_params(), epochs=100, noise=5e-4, seed=5)
    reference = emls_for(tmp_path, reference_data, t14(truth), ttv_prior_width=40 / 1440)

    result = emls.subtract_planet_transit(*data[1:])
    reference_result = reference.subtract_planet_transit(*reference_data[1:])

    assert emls.fold_realigned is True
    assert emls.planet_rp_over_rstar == pytest.approx(true_rp, rel=0.003)
    peak = stacked_residual_peak(emls, result, np.array(emls.bary_t0s) + np.array(emls.ttvs))
    reference_peak = stacked_residual_peak(reference, reference_result,
                                           np.array(reference.bary_t0s) + np.array(reference.ttvs))
    assert peak < 1.3 * reference_peak


def test_fold_is_not_realigned_without_ttvs(tmp_path):
    truth = planet_params()
    data = curve(truth, epochs=12, noise=5e-4, seed=6)
    emls = emls_for(tmp_path, data, t14(truth))

    emls.subtract_planet_transit(*data[1:])

    assert emls.fold_realigned is False


# --- B2 ----------------------------------------------------------------------------------------------------------

def test_catalog_radius_20_percent_high_is_corrected_from_the_depth(tmp_path):
    """Acceptance: with rp nominal 20 % high and a 5 % error, at least 11/12 epochs are kept and the
    stacked residual is below 5 % of the depth. Before, 0/12 were kept."""
    truth = planet_params(rp=PLANET_RADIUS / 1.2 * R_earth_to_R_sun)
    data = curve(truth, noise=1e-4, seed=7)
    emls = emls_for(tmp_path, data, t14(truth))  # catalog radius PLANET_RADIUS, error 0.5 (5 %)

    result = emls.subtract_planet_transit(*data[1:])

    assert epoch_count(emls) >= 11
    assert stacked_residual_peak(emls, result, np.array(emls.bary_t0s) + np.array(emls.ttvs)) < 0.05 * truth.rp ** 2


# --- D2 ----------------------------------------------------------------------------------------------------------

def test_wrong_planet_shape_fails_the_shape_check(tmp_path):
    truth = planet_params()
    data = curve(truth, noise=1e-4, seed=8)
    # The fold fit is skipped and the input semi-major axis is 15 % short, so T14 is wrong.
    wrong = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, sma=0.85 * 0.1436)
    wrong.subtract_planet_transit(*data[1:])
    right = emls_for(tmp_path, data, t14(truth))
    right.subtract_planet_transit(*data[1:])

    assert wrong.check_planet_shape(*wrong.planet_shape_check_inputs)['ok'] is False
    assert right.check_planet_shape(*right.planet_shape_check_inputs)['ok'] is True


def test_shape_check_bin_errors_follow_the_red_noise_factor(tmp_path):
    truth = planet_params()
    data = curve(truth, noise=1e-4, seed=8)
    wrong = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, sma=0.85 * 0.1436)
    wrong.subtract_planet_transit(*data[1:])

    white = wrong.check_planet_shape(*wrong.planet_shape_check_inputs)
    red = wrong.check_planet_shape(*wrong.planet_shape_check_inputs, noise_factor=3.0)

    assert red['chi2'] == pytest.approx(white['chi2'] / 9, rel=1e-9)
    assert red['noise_factor'] == 3.0


# --- D4 / E2 / E3 ------------------------------------------------------------------------------------------------

def test_mask_mode_centres_an_epoch_without_transit_data_on_the_ephemeris(tmp_path, caplog):
    truth = planet_params()
    t0s, time, flux, flux_err = curve(truth, noise=1e-4, seed=9)
    gap = np.abs(time - t0s[4]) < 0.6 * t14(truth)  # no in-transit point in epoch 4
    data = (t0s, time[~gap], flux[~gap], flux_err[~gap])
    emls = emls_for(tmp_path, data, t14(truth), subtract_mode='mask', min_transit_completeness=0)

    with caplog.at_level(logging.INFO):
        kept_time = emls.subtract_planet_transit(*data[1:], mode='mask')[0]

    assert any(f'Masking transit T0={t0s[4]:.4f} around the ephemeris time: no in-transit data' in record.getMessage()
               for record in caplog.records)
    assert not np.any(np.abs(kept_time - t0s[4]) < emls.mask_half_width(kept_time))


@pytest.mark.parametrize('noise, cadence', [(2e-3, CADENCE), (1e-4, 30 / 1440)])
def test_default_mask_leaves_no_planet_point(tmp_path, noise, cadence):
    """Acceptance: no retained point has a true planet flux below 1 - 1e-7."""
    truth = planet_params()
    t0s = np.arange(12) * truth.per
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, cadence) for t0 in t0s])
    flux = batman.TransitModel(truth, time).light_curve(truth) + np.random.default_rng(10).normal(0, noise, len(time))
    data = (t0s, time, flux, np.full(len(time), noise))
    emls = emls_for(tmp_path, data, t14(truth), subtract_mode='mask', min_transit_completeness=0)

    kept_time = emls.subtract_planet_transit(*data[1:], mode='mask')[0]

    assert np.all(batman.TransitModel(truth, kept_time).light_curve(truth) >= 1 - 1e-7)


@pytest.mark.parametrize('offset, survival', [(0.25, 0.08), (0.5, 0.16), (1.0, 0.32), (2.0, 0.64), (4.0, 0.84)])
def test_mask_survival_matches_the_geometry(offset, survival):
    """The audit's geometric fractions, with the offset in units of the planet half-duration."""
    assert ExoMoonLeastSquares.mask_survival(offset, 1.0, 1.0) == pytest.approx(survival, abs=0.015)


def test_mask_mode_reports_the_moon_survival(tmp_path, caplog):
    truth = planet_params()
    data = curve(truth, noise=1e-4, seed=11)
    emls = emls_for(tmp_path, data, t14(truth), subtract_mode='mask')

    with caplog.at_level(logging.INFO):
        emls.subtract_planet_transit(*data[1:], mode='mask')

    assert 0 < emls.mask_mode_survival < 1
    assert any('Mask mode keeps on average' in record.getMessage() for record in caplog.records)
