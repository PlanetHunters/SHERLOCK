"""End-to-end EMLS executions launched from ExoMoonLeastSquares.run on synthetic data.

These check that a full execution finishes, writes its products and behaves
consistently with and without a moon. Template numerics are covered by
sherlockpipe/tests/exomoon. The assessment step needs sherlockpipe.exomoon_ir,
which is not in the repository, so it logs an error and writes no
emls_assessment.json here. That is a known state, not something asserted.
"""
import os

import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, PlanetModelError, R_earth_to_R_sun

STAR_MASS = 1.0
STAR_RADIUS = 1.0
LIMB_DARKENING = [0.3, 0.2]
PLANET_PERIOD = 20.0
PLANET_RADIUS = 10.0
PLANET_MASS = 300.0
PLANET_SEMIMAJOR_AXIS = 0.1436
PLANET_INC = 89.9
NOISE = 5e-5
EPOCHS = 12
MOON = {'moon_radius': 2.5, 'moon_period': 2.3, 'moon_mass': 5.0, 'initial_alpha': 1.0}


def planet_light_curve(seed=11, noise=NOISE, planet_radius=PLANET_RADIUS):
    """Return a two-minute cadence planet light curve around each transit."""
    t0s = np.arange(EPOCHS) * PLANET_PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, 2 / 1440) for t0 in t0s])
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = PLANET_PERIOD
    params.rp = planet_radius * R_earth_to_R_sun / STAR_RADIUS
    params.a = PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN / STAR_RADIUS
    params.inc = PLANET_INC
    params.ecc = 0.0
    params.w = 90.0
    params.u = LIMB_DARKENING
    params.limb_dark = 'quadratic'
    flux = batman.TransitModel(params, time).light_curve(params)
    flux = flux + np.random.default_rng(seed).normal(0, noise, len(time))
    duration = PLANET_PERIOD / np.pi * np.arcsin((1 + params.rp) / params.a)
    return t0s, time, flux, np.full(len(time), noise), duration


def execute(output_dir, inject_params=None, noise=NOISE, planet_radius=PLANET_RADIUS, semimajor_axis_bias=1.0,
            planet_inc=PLANET_INC, planet_inc_err=0.5, **kwargs):
    t0s, time, flux, flux_err, duration = planet_light_curve(noise=noise, planet_radius=planet_radius)
    emls = ExoMoonLeastSquares(
        str(output_dir), 2, STAR_MASS, STAR_RADIUS, LIMB_DARKENING, planet_radius, 0.05 * planet_radius, PLANET_MASS,
        PLANET_PERIOD, 0.0, duration, PLANET_SEMIMAJOR_AXIS * semimajor_axis_bias, planet_inc, planet_inc_err, 0.0,
        0.0, 90.0, 0.0, 1.0, 4.0, t0s, time, flux, flux_err, 5.5, period_grid_size=60, radius_grid_size=4, **kwargs)
    emls.run(plot=True, inject_params=inject_params)
    return emls


@pytest.fixture(scope='module')
def executions(tmp_path_factory):
    return (execute(tmp_path_factory.mktemp('emls_null')),
            execute(tmp_path_factory.mktemp('emls_moon'), MOON))


@pytest.mark.parametrize('index', [0, 1])
def test_execution_finishes_and_writes_results(executions, index):
    emls = executions[index]
    written = os.listdir(emls.object_dir)

    assert {'stats.png', 'residuals_map.png'} <= set(written)
    assert sum(name.startswith('SNR ') for name in written) == 5
    assert sum(name.startswith('Residuals ') for name in written) == 5
    assert sum(name.endswith('_moon_model.png') for name in written) == len(emls.bary_t0s)
    assert np.isfinite(emls.max_snr)
    assert np.isfinite(emls.max_sde)


def test_planet_only_execution_keeps_every_epoch(executions):
    assert len(executions[0].bary_t0s) == EPOCHS


def test_injected_moon_raises_the_detection_statistic(executions):
    null, moon = executions

    assert moon.max_snr > 5 * null.max_snr
    assert len(moon.bary_t0s) >= EPOCHS // 2


def test_detected_period_matches_the_expected_alias_of_the_injected_moon(executions):
    """A regression pin on this fixture, which detects the k=-5 alias of the moon period.

    Sampling once per planet orbit aliases the period. Chance also lands here
    in some runs without an orbit, so this is neither proof of an orbit nor a
    claim that the true period was recovered.
    """
    moon = executions[1]
    alias = 1 / (1 / MOON['moon_period'] - 5 / PLANET_PERIOD)

    error = abs(alias - moon.recovered_period) / alias

    assert error < 0.01, (f"detected P={moon.recovered_period:.4f} d, alpha={moon.recovered_alpha:.4f} rad, "
                          f"expected alias {alias:.4f} d of injected P={MOON['moon_period']} d")


def test_best_scenario_is_a_dimming(executions):
    assert executions[1].best_snr_radius > 0


def test_repeated_execution_gives_the_same_result(executions, tmp_path):
    null = executions[0]

    repeated = execute(tmp_path)

    assert repeated.max_snr == null.max_snr
    assert repeated.max_sde == null.max_sde
    assert repeated.recovered_period == null.recovered_period


def test_impossible_planet_geometry_stops_the_execution(tmp_path):
    emls_inputs = dict(planet_inc=85.0, planet_inc_err=0.1)

    with pytest.raises(PlanetModelError, match="Shared planet fit rejected"):
        execute(tmp_path, **emls_inputs)

    assert os.listdir(tmp_path) == []


def test_a_over_rstar_uncertainty_absorbs_a_biased_catalog_value(tmp_path):
    a_over_rstar = PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN / STAR_RADIUS

    emls = execute(tmp_path, semimajor_axis_bias=1.15, planet_a_over_rstar_err=0.1 * a_over_rstar)

    assert emls.shared_planet_params.a == pytest.approx(a_over_rstar, rel=0.03)
    assert emls.shared_fit_residual['status'] == 'ok'
    assert len(emls.bary_t0s) == EPOCHS


def test_biased_catalog_without_uncertainty_is_flagged(tmp_path):
    emls = execute(tmp_path, semimajor_axis_bias=1.15)

    assert emls.shared_fit_residual['status'] == 'structured'


def test_moon_run_keeps_every_epoch_and_only_the_moon_is_flagged(executions):
    null, moon = executions

    assert len(moon.bary_t0s) == EPOCHS
    assert null.shared_fit_residual['status'] == 'ok'
    assert moon.shared_fit_residual['status'] == 'structured'


def test_shallow_planet_execution_keeps_every_epoch(tmp_path):
    emls = execute(tmp_path, noise=2e-3, planet_radius=2.0, skip_fold_fit=True)

    assert len(emls.bary_t0s) == EPOCHS
    assert emls.subtraction_status['ephemeris_timing_epochs'] > 0
