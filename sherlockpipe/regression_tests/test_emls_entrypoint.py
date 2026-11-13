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

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun

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


def planet_light_curve(seed=11):
    """Return a two-minute cadence planet light curve around each transit."""
    t0s = np.arange(EPOCHS) * PLANET_PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, 2 / 1440) for t0 in t0s])
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = PLANET_PERIOD
    params.rp = PLANET_RADIUS * R_earth_to_R_sun / STAR_RADIUS
    params.a = PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN / STAR_RADIUS
    params.inc = PLANET_INC
    params.ecc = 0.0
    params.w = 90.0
    params.u = LIMB_DARKENING
    params.limb_dark = 'quadratic'
    flux = batman.TransitModel(params, time).light_curve(params)
    flux = flux + np.random.default_rng(seed).normal(0, NOISE, len(time))
    duration = PLANET_PERIOD / np.pi * np.arcsin((1 + params.rp) / params.a)
    return t0s, time, flux, np.full(len(time), NOISE), duration


def execute(output_dir, inject_params=None):
    t0s, time, flux, flux_err, duration = planet_light_curve()
    emls = ExoMoonLeastSquares(
        str(output_dir), 2, STAR_MASS, STAR_RADIUS, LIMB_DARKENING, PLANET_RADIUS, 0.5, PLANET_MASS, PLANET_PERIOD,
        0.0, duration, PLANET_SEMIMAJOR_AXIS, PLANET_INC, 0.5, 0.0, 0.0, 90.0, 0.0, 1.0, 4.0, t0s, time, flux,
        flux_err, 5.5, period_grid_size=60, radius_grid_size=4)
    emls.run(inject_params=inject_params)
    return emls


@pytest.fixture(scope='module')
def executions(tmp_path_factory):
    return (execute(tmp_path_factory.mktemp('emls_null')),
            execute(tmp_path_factory.mktemp('emls_moon'), MOON))


@pytest.mark.parametrize('index', [0, 1])
def test_execution_finishes_and_writes_results(executions, index):
    emls = executions[index]
    written = os.listdir(emls.object_dir)

    assert {'search_overview.png', 'snr_map.png', 'hypothesis_1.png'} <= set(written)
    assert 1 <= sum(name.startswith('hypothesis_') for name in written) <= emls.HYPOTHESIS_FIGURES
    assert len(os.listdir(os.path.join(emls.object_dir, 'epochs'))) == len(emls.bary_t0s)
    assert np.isfinite(emls.max_snr)
    assert np.isfinite(emls.max_sde)


def test_planet_only_execution_keeps_every_epoch(executions):
    assert len(executions[0].bary_t0s) == EPOCHS


def test_injected_moon_raises_the_detection_statistic(executions):
    null, moon = executions

    assert moon.max_snr > 5 * null.max_snr
    assert len(moon.bary_t0s) >= EPOCHS // 2


def test_detected_period_is_the_injected_one_or_its_alias(executions):
    """Sampling once per planet orbit aliases the period, 1/P' = 1/P + k/P_planet, so the detected
    period must be the injected one or one of its aliases (it was the k = -5 alias before the
    Domingos outer limit, and is the injected period since), and the reported alias family must
    hold the injected period (audit item H9). Neither is proof of an orbit."""
    moon = executions[1]
    injected = MOON['moon_period']
    aliases = [1 / (1 / injected + k / PLANET_PERIOD) for k in range(-30, 31)
               if 1 / injected + k / PLANET_PERIOD > 0]

    error = min(abs(alias - moon.recovered_period) / alias for alias in aliases)

    assert error < 0.01, (f"detected P={moon.recovered_period:.4f} d, alpha={moon.recovered_alpha:.4f} rad, "
                          f"not an alias of the injected P={injected} d")
    assert any(abs(alias['period'] - injected) / injected < 0.01 for alias in moon.aliases)


def test_best_scenario_is_a_dimming(executions):
    assert executions[1].best_snr_radius > 0


def test_repeated_execution_gives_the_same_result(executions, tmp_path):
    null = executions[0]

    repeated = execute(tmp_path)

    assert repeated.max_snr == null.max_snr
    assert repeated.max_sde == null.max_sde
    assert repeated.recovered_period == null.recovered_period
