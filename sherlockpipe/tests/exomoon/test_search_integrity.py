"""Guards on the behaviour of a complete EMLS search on synthetic light curves.

These hold on the code before and after a change, so they define what "the
search still works" means. Each run goes through ExoMoonLeastSquares.run with a
real planet fit. The moon is injected with the same timing model the search
uses, so these check the search machinery, not the timing physics. Pandora
injections are covered by the end-to-end tests.
"""
import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, M_earth_to_M_sun, R_earth_to_R_sun

STAR_RADIUS = 1.0
LIMB_DARKENING = [0.3, 0.2]
PLANET_PERIOD = 100.0
PLANET_RADIUS = 10.0
PLANET_MASS = 300.0
PLANET_SEMIMAJOR_AXIS = 0.4217
PLANET_INC = 89.95
EPOCHS = 12
NOISE = 3e-4
CADENCE = 5 / 1440
PERIOD_GRID = 120
MOON_PERIOD_INDEX = 100
MOON_ALPHA = 2 * np.pi * 8 / 25


def build(flux, time, flux_err, output_dir, t0s, duration):
    return ExoMoonLeastSquares(
        str(output_dir), 2, 1.0, STAR_RADIUS, LIMB_DARKENING, PLANET_RADIUS, 0.5, PLANET_MASS, PLANET_PERIOD, 0.0,
        duration, PLANET_SEMIMAJOR_AXIS, PLANET_INC, 0.5, 0.0, 0.0, 90.0,
        PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN * np.cos(np.deg2rad(PLANET_INC)), 1.0, 4.0, t0s, time, flux, flux_err, 5.5,
        period_grid_size=PERIOD_GRID, radius_grid_size=4, fit_rp_per_transit=False)


def planet_curve(seed):
    t0s = np.arange(EPOCHS) * PLANET_PERIOD
    time = np.concatenate([t0 + np.arange(-3.0, 3.0, CADENCE) for t0 in t0s])
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
    return t0s, time, flux, np.full(len(time), NOISE), duration, params


def moon_period():
    """Return a period on the search grid, wide enough to be identifiable."""
    limits = ExoMoonLeastSquares
    mass = PLANET_MASS * M_earth_to_M_sun
    widest = limits.au_to_period(mass, limits.max_moon_semimajor_axis(1.0, mass, PLANET_SEMIMAJOR_AXIS))
    grid = np.logspace(np.log10(limits.compute_roche_period(5.5)), np.log10(widest), PERIOD_GRID)
    return grid[MOON_PERIOD_INDEX]


def with_moon(flux, time, t0s, params, moon_radius):
    """Add a moon dip at the time the search model predicts for each epoch."""
    a_rsun = PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN
    b = a_rsun * np.cos(np.deg2rad(PLANET_INC))
    period = moon_period()
    a_moon = ExoMoonLeastSquares.compute_semimajor_axis(PLANET_MASS * M_earth_to_M_sun, period) * AU_TO_RSUN
    orbit = ExoMoonLeastSquares.compute_moon_orbit_transit_duration(STAR_RADIUS, a_rsun, PLANET_PERIOD, a_moon,
                                                                    4.0 * R_earth_to_R_sun, b)
    transit = ExoMoonLeastSquares.compute_transit_duration(STAR_RADIUS, a_rsun, PLANET_PERIOD, 4.0 * R_earth_to_R_sun, b)
    moon = batman.TransitParams()
    moon.per = PLANET_PERIOD
    moon.rp = moon_radius * R_earth_to_R_sun / STAR_RADIUS
    moon.a = params.a
    moon.inc = params.inc
    moon.ecc = 0.0
    moon.w = 90.0
    moon.u = LIMB_DARKENING
    moon.limb_dark = 'quadratic'
    flux = flux.copy()
    for t0 in t0s:
        tau = ExoMoonLeastSquares.compute_moon_ts(t0, orbit / 2, transit / 2)
        moon.t0 = ExoMoonLeastSquares.compute_moon_t0(t0s[0], t0, period, MOON_ALPHA, tau)[0]
        window = np.abs(time - t0) < 3.0
        flux[window] += batman.TransitModel(moon, time[window]).light_curve(moon) - 1
    return flux


def execute(output_dir, seed=3, moon_radius=None, error_scale=1.0):
    t0s, time, flux, flux_err, duration, params = planet_curve(seed)
    if moon_radius is not None:
        flux = with_moon(flux, time, t0s, params, moon_radius)
    emls = build(flux, time, flux_err * error_scale, output_dir, t0s, duration)
    emls.run()
    return emls


@pytest.fixture(scope='module')
def null_run(tmp_path_factory):
    return execute(tmp_path_factory.mktemp('null'))


@pytest.fixture(scope='module')
def moon_run(tmp_path_factory):
    return execute(tmp_path_factory.mktemp('moon'), moon_radius=3.0)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_noise_stays_below_detection_level(seed, tmp_path):
    emls = execute(tmp_path, seed=seed)

    assert emls.max_snr < 5
    assert len(emls.bary_t0s) == EPOCHS


def test_moon_is_found_at_its_true_period(moon_run):
    error = abs(moon_run.recovered_period - moon_period()) / moon_period()

    assert error < 0.01, f"found {moon_run.recovered_period:.4f} d, injected {moon_period():.4f} d"


def test_moon_stands_out_from_noise(null_run, moon_run):
    assert moon_run.max_snr > 10
    assert moon_run.max_snr > 3 * null_run.max_snr


def test_every_epoch_survives_with_a_moon(moon_run):
    assert len(moon_run.bary_t0s) == EPOCHS


def test_snr_grows_with_moon_depth(moon_run, tmp_path):
    larger = execute(tmp_path, moon_radius=4.0)

    assert larger.max_snr / moon_run.max_snr == pytest.approx((4.0 / 3.0) ** 2, rel=0.25)


def test_snr_and_period_do_not_depend_on_error_bar_scale(moon_run, tmp_path):
    scaled = execute(tmp_path, moon_radius=3.0, error_scale=3.0)

    assert scaled.max_snr == pytest.approx(moon_run.max_snr, rel=0.02)
    assert scaled.recovered_period == pytest.approx(moon_run.recovered_period, rel=0.01)
