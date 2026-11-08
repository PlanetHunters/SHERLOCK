"""Red noise, template amplitude, moon radius and period aliases: audit items H11, H13, H14 and H9.

H11: the SNR is divided by a red-noise factor measured on the moon-window time scale away from
the transits. H13: the template amplitude is a weighted least-squares value. H14: the radius
follows from that amplitude through the template set, so it holds at high impact parameter.
H9: the aliases of the selected period are listed with their SNR.
"""
import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon import test_search_integrity as guard

A_OVER_RSTAR, PERIOD, LD = 90.0, 100.0, [0.4, 0.2]


def geometry(impact):
    params = batman.TransitParams()
    params.t0, params.per, params.a = 10.0, PERIOD, A_OVER_RSTAR
    params.inc = np.rad2deg(np.arccos(impact / A_OVER_RSTAR))
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 0.0, LD, 'quadratic'
    return params


def fit_noiseless_moon(impact, radius):
    """Score a noiseless batman moon with the templates of the same geometry, as `search` does."""
    models = ExoMoonLeastSquares.build_moon_templates(geometry(impact), np.linspace(1, 4, 4), 1.0, 0.3)
    moon = geometry(impact)
    moon.t0, moon.rp = 0.0, radius * R_earth_to_R_sun
    time = np.arange(-0.5, 0.5, 1 / 1440)
    flux = batman.TransitModel(moon, time).light_curve(moon)
    half = ExoMoonLeastSquares.compute_transit_duration(1.0, A_OVER_RSTAR, PERIOD, 4 * R_earth_to_R_sun, impact) / 2
    inside = np.abs(time) <= half
    err = np.full(len(time), 1e-4)
    depth = np.mean(flux[~inside]) - np.mean(flux[inside])
    result = ExoMoonLeastSquares.calculate_residuals(time[inside], flux[inside], err[inside], time[~inside],
                                                     flux[~inside], err[~inside], 1 - depth, len(time), models, 1,
                                                     4, 1.0, 4)
    truth = 1 - flux[inside]
    amplitude = np.sum((1 - result[3]) * truth) / np.sum(truth ** 2)
    return result[2], amplitude


# --- H13 / H14 ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('impact', [0.0, 0.5, 0.8, 0.9])
@pytest.mark.parametrize('radius', [1.6, 2.3, 3.4])
def test_noiseless_moon_gives_its_amplitude_and_radius(impact, radius):
    # Before: the template peak was the mean depth (amplitude ~0.88) and r = sqrt(mean depth),
    # 0.81 of the truth at b = 0.9.
    fitted_radius, amplitude = fit_noiseless_moon(impact, radius)

    assert amplitude == pytest.approx(1.0, abs=0.01)
    assert fitted_radius == pytest.approx(radius, rel=0.03)


# --- H11 ---------------------------------------------------------------------------------------------------------

def ar1(time, sigma, tau, rng):
    out = np.zeros(len(time))
    steps = np.diff(time, prepend=time[0] - 1e9)
    for i in range(len(time)):
        phi = np.exp(-steps[i] / tau)
        out[i] = (phi * out[i - 1] if i else 0.0) + rng.normal(0, sigma * np.sqrt(1 - phi ** 2))
    return out


def red_noise_of(flux, time):
    t0s, _, _, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, np.full(len(time), 1e-3), '/tmp', t0s, duration)
    return emls.measure_red_noise(time, flux, 0.85, 0.3)


def test_white_noise_has_no_red_noise_factor():
    _, time, _, _, _, _ = guard.planet_curve(3)
    betas = [red_noise_of(1 + np.random.default_rng(seed).normal(0, 1e-3, len(time)), time) for seed in range(5)]

    assert np.median(betas) == pytest.approx(1.0, abs=0.15)


def test_correlated_noise_raises_the_factor():
    _, time, _, _, _, _ = guard.planet_curve(3)
    rng = np.random.default_rng(1)
    flux = 1 + rng.normal(0, 1e-3 * np.sqrt(0.5), len(time)) + ar1(time, 1e-3 * np.sqrt(0.5), 0.1, rng)

    assert red_noise_of(flux, time) > 2


def test_red_noise_needs_data_away_from_the_transits():
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, flux_err, '/tmp', t0s, duration)

    assert emls.measure_red_noise(time, flux, 3.5, 0.3) is None  # everything is within 3.5 d of a transit


def test_correlated_noise_rarely_reaches_snr_5(tmp_path):
    """The audit's case: 20 % of the variance in AR(1) noise with a 0.1 d e-folding, rms 1e-3, no dip.
    Before, the SNR passed 5 in 21 of 40 runs."""
    above = []
    for seed in range(8):
        t0s, time, _, _, duration, params = guard.planet_curve(seed + 100)
        rng = np.random.default_rng(seed)
        flux = batman.TransitModel(params, time).light_curve(params)
        flux = flux + rng.normal(0, 1e-3 * np.sqrt(0.8), len(time)) + ar1(time, 1e-3 * np.sqrt(0.2), 0.1, rng)
        emls = guard.build(flux, time, np.full(len(time), 1e-3), tmp_path, t0s, duration)
        emls.run()
        assert emls.red_noise_factor > 1
        above.append(emls.max_snr > 5)

    assert sum(above) == 0


# --- H9 ----------------------------------------------------------------------------------------------------------

def test_alias_family_lists_the_sampling_aliases(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    grid = np.logspace(np.log10(0.5), np.log10(40), 4000)
    true_period = 13.0
    spectrum = np.full(len(grid), 2.0)
    for k in (-1, 0, 2):
        spectrum[np.argmin(np.abs(grid - 1 / (1 / true_period + k / PERIOD)))] = 10.0 - abs(k)

    family = emls.alias_family(grid, spectrum, true_period)

    ks = [alias['k'] for alias in family]
    assert ks[:3] == [0, -1, 2]
    assert family[0]['snr_below_best'] == 0
    assert family[1]['snr_below_best'] == pytest.approx(1.0)
    for alias in family:
        assert 1 / alias['period'] == pytest.approx(1 / true_period + alias['k'] / PERIOD)
        assert grid[0] <= alias['period'] <= grid[-1]


def test_alias_family_holds_the_true_period_when_an_alias_is_selected(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    grid = np.logspace(np.log10(0.5), np.log10(40), 4000)
    true_period, selected = 13.0, 1 / (1 / 13.0 + 6 / PERIOD)  # |k| = 6 was seen in the audit

    family = emls.alias_family(grid, np.full(len(grid), 3.0), selected)

    assert any(alias['period'] == pytest.approx(true_period, rel=1e-9) for alias in family)
