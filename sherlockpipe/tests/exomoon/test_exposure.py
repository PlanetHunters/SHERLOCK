"""Regression tests for audit items C2 and G18: finite exposure in the planet fits and moon templates.

Long-cadence data average the flux over each exposure. The reference here averages instantaneous
batman models at 401 sub-exposure midpoints. batman's own supersample_factor is not used as the
reference, because it is still about 9 ppm off at 101 sub-exposures.
"""
import logging

import batman
import numpy as np
import pytest
import yaml

from sherlockpipe import exomoon_run
from sherlockpipe.exomoon import AU_TO_RSUN, R_earth_to_R_sun, ExoMoonLeastSquares
from sherlockpipe.tests.exomoon.test_audit_fixes import LD, PERIOD, SMA, run_to_search, system
from sherlockpipe.tests.exomoon.test_emls_run import guard_directory

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


def in_transit_residual_rms(tmp_path, exposure):
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
    return np.std(emls.flux[in_transit]), noise


def test_planet_subtraction_leaves_noise_on_30_minute_data(tmp_path):
    """Without the exposure, part of a 30-minute transit is left in the data after subtraction."""
    (tmp_path / 'averaged').mkdir()
    (tmp_path / 'instantaneous').mkdir()

    averaged, noise = in_transit_residual_rms(tmp_path / 'averaged', THIRTY_MINUTES)
    instantaneous, _ = in_transit_residual_rms(tmp_path / 'instantaneous', None)

    assert averaged < 1.15 * noise, f"averaged model leaves {averaged * 1e6:.1f} ppm"
    assert instantaneous > 1.2 * averaged, f"instantaneous {instantaneous * 1e6:.1f} ppm"


def write_properties(directory, exptime):
    with open(directory / 'properties.yaml', 'w') as file:
        yaml.safe_dump({'EXPTIME': exptime}, file)


def test_command_reads_the_exposure_from_the_sherlock_properties(tmp_path):
    guard_directory(tmp_path)
    write_properties(tmp_path, [1800])

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    assert inputs['exposure_time'] == 1800.0
    assert inputs['sources']['exposure_time'] == 'properties.yaml'


def test_user_exposure_takes_precedence(tmp_path):
    guard_directory(tmp_path)
    write_properties(tmp_path, [1800])

    inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, exposure_time=120)

    assert inputs['exposure_time'] == 120.0 and inputs['sources']['exposure_time'] == 'user'


def test_mixed_exposures_use_the_longest_with_a_warning(tmp_path, caplog):
    guard_directory(tmp_path)
    write_properties(tmp_path, [120, 600])

    with caplog.at_level(logging.WARNING):
        inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    assert inputs['exposure_time'] == 600.0
    assert any('several exposure times' in record.getMessage() for record in caplog.records)


def test_missing_exposure_falls_back_to_an_instantaneous_model(tmp_path, caplog):
    guard_directory(tmp_path)

    with caplog.at_level(logging.WARNING):
        inputs = exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0)

    assert inputs['exposure_time'] is None
    assert any('instantaneous' in record.getMessage() for record in caplog.records)


def test_invalid_exposure_is_rejected(tmp_path):
    guard_directory(tmp_path)

    with pytest.raises(ValueError, match='must be positive'):
        exomoon_run.build_inputs(str(tmp_path), 1, planet_mass=300.0, exposure_time=0)
