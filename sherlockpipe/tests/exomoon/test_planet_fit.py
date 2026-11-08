"""Regression tests for the planet fit and epoch quality control: audit items B1, B4, C1, C4 and D1.

B1: the fold is fitted in (T14, b, rp), so a transit longer than the catalog a/R* allows is
fitted. B4: the moon stage uses the fitted transit geometry. C1: a Gaussian prior around the
ephemeris keeps low-SNR epochs from following the noise. C4: an epoch whose t0 stops at its
bound is dropped. D1: an epoch with a high post-subtraction RMS is flagged, not dropped.
"""
import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon.test_audit_fixes import LD, PERIOD, PLANET_RADIUS, SMA, run_to_search

CADENCE = 4 / 1440


def planet_params(a_over_rstar=SMA * AU_TO_RSUN, impact=0.0, rp=PLANET_RADIUS * R_earth_to_R_sun):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, PERIOD, rp, a_over_rstar
    params.inc = np.rad2deg(np.arccos(impact / a_over_rstar))
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, LD, 'quadratic'
    return params


def curve(params, epochs=12, noise=1e-4, shifts=None, seed=1, extra=None):
    """Transits of `params`, with optional per-epoch t0 shifts and an extra flux term per epoch."""
    t0s = np.arange(epochs) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, CADENCE) for t0 in t0s])
    flux = np.ones_like(time)
    for i, t0 in enumerate(t0s):
        window = np.abs(time - t0) < 3
        params.t0 = t0 + (0.0 if shifts is None else shifts[i])
        flux[window] = batman.TransitModel(params, time[window]).light_curve(params)
        if extra is not None:
            flux[window] += extra(time[window], params.t0) - 1
    flux = flux + np.random.default_rng(seed).normal(0, noise, len(time))
    return t0s, time, flux, np.full(len(time), noise)


def emls_for(tmp_path, data, duration, inc=90.0, impact=0.0, sma=SMA, **kwargs):
    t0s, time, flux, flux_err = data
    return ExoMoonLeastSquares(
        str(tmp_path), 1, 1.0, 1.0, LD, PLANET_RADIUS, 0.5, 300.0, PERIOD, 0.0, duration, sma, inc, 0.5, 0.0, 0.0,
        90.0, impact, 1.0, 4.0, t0s, time, flux, flux_err, 5.5, period_grid_size=40, radius_grid_size=4, **kwargs)


def t14(params):
    centred = batman.TransitParams()
    for key in ('per', 'rp', 'a', 'inc', 'ecc', 'w', 'u', 'limb_dark'):
        setattr(centred, key, getattr(params, key))
    centred.t0 = 0.0
    return ExoMoonLeastSquares.model_transit_duration(centred, 1.0)


def stacked_residual_peak(emls, result, t0s):
    """Largest binned residual of the subtracted, stacked transits, around one."""
    time, flux = result[0], result[1]
    phase = np.concatenate([time[np.abs(time - t0) < emls.planet_duration] - t0 for t0 in t0s])
    residual = np.concatenate([flux[np.abs(time - t0) < emls.planet_duration] for t0 in t0s])
    edges = np.arange(-emls.planet_duration, emls.planet_duration, 10 / 1440)
    index = np.digitize(phase, edges)
    binned = [np.median(residual[index == i]) for i in np.unique(index) if np.sum(index == i) > 20]
    return np.max(np.abs(np.array(binned) - 1))


# --- B1: the fold reaches transits longer than the catalog a/R* allows ---------------------------------------------

def test_fold_fits_a_transit_longer_than_the_catalog_allows(tmp_path):
    truth = planet_params(a_over_rstar=0.85 * SMA * AU_TO_RSUN)
    data = curve(truth)
    true_t14 = t14(truth)
    emls = emls_for(tmp_path, data, t14(planet_params()))

    result = emls.subtract_planet_transit(*data[1:])

    assert emls.planet_duration == pytest.approx(true_t14, rel=0.02)
    assert stacked_residual_peak(emls, result, data[0]) < 0.05 * truth.rp ** 2


# --- B4: the moon stage uses the fitted transit geometry ------------------------------------------------------------

def test_moon_durations_follow_the_fitted_impact_parameter(tmp_path):
    truth = planet_params(impact=0.7)
    data = curve(truth)
    emls = emls_for(tmp_path, data, t14(planet_params()))  # given as a central transit

    inputs = run_to_search(emls)

    assert emls.planet_impact_param == pytest.approx(0.7, abs=0.05)
    expected = ExoMoonLeastSquares.compute_transit_duration(1.0, SMA * AU_TO_RSUN, PERIOD, 4.0 * R_earth_to_R_sun,
                                                            0.7)
    assert inputs[0].moon_transit_duration == pytest.approx(expected, rel=0.02)


def test_hill_sphere_keeps_the_physical_semimajor_axis(tmp_path):
    truth = planet_params(a_over_rstar=0.9 * SMA * AU_TO_RSUN)
    data = curve(truth)
    emls = emls_for(tmp_path, data, t14(planet_params()))

    run_to_search(emls)

    assert emls.planet_semimajor_axis == SMA
    assert emls.transit_semimajor_axis() == pytest.approx(0.9 * SMA * AU_TO_RSUN, rel=0.03)


# --- C1: ephemeris prior on the per-epoch t0 -----------------------------------------------------------------------

def carter_sigma(params, noise):
    duration = t14(params)
    in_transit = duration / CADENCE
    q = np.sqrt(in_transit) * params.rp ** 2 / noise
    return duration / q * np.sqrt(ExoMoonLeastSquares.ingress_duration(params, duration) / (2 * duration))


def test_low_snr_epochs_stay_near_the_ephemeris(tmp_path):
    truth = planet_params()
    depth = truth.rp ** 2
    noise = np.sqrt(t14(truth) / CADENCE) * depth / 3  # SNR 3 per epoch
    data = curve(truth, epochs=30, noise=noise, seed=4)
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, min_transit_completeness=0)

    emls.subtract_planet_transit(*data[1:])

    # The audit asked for below 2x Carter; without the prior this fixture gives about 1.8x, with it
    # about 0.7x, so the test asks for below 1x.
    assert np.sqrt(np.mean(np.square(emls.ttvs))) < carter_sigma(truth, noise)


def test_ttvs_are_still_fitted_at_high_snr(tmp_path):
    truth = planet_params()
    noise = np.sqrt(t14(truth) / CADENCE) * truth.rp ** 2 / 30  # SNR 30 per epoch
    shifts = np.where(np.arange(12) % 2 == 0, 10, -10) / 1440
    data = curve(truth, noise=noise, shifts=shifts, seed=2)
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True)

    emls.subtract_planet_transit(*data[1:])

    assert np.median(np.abs(np.array(emls.ttvs) - shifts)) < 2 / 1440


# --- C4: epochs whose t0 stops at its bound are dropped ------------------------------------------------------------

def test_epochs_at_the_t0_bound_are_dropped(tmp_path):
    truth = planet_params()
    duration = t14(truth)
    shifts = np.linspace(0, duration, 12)  # drifts by one T14
    data = curve(truth, shifts=shifts)
    emls = emls_for(tmp_path, data, duration, skip_fold_fit=True)

    emls.subtract_planet_transit(*data[1:])

    dropped = {epoch['t0'] for epoch in emls.epochs_dropped_fit}
    assert all(t0 in dropped for t0, shift in zip(data[0], shifts) if shift > 0.6 * duration)
    bound = duration / 2
    assert all(abs(ttv) < bound * (1 - 1e-3) for ttv in emls.ttvs)
    assert set(emls.bary_t0s).isdisjoint(dropped)


# --- D1: a moon inside the transit is flagged, not dropped ---------------------------------------------------------

def test_moon_inside_the_transit_keeps_its_epochs(tmp_path):
    truth = planet_params()
    duration = t14(truth)
    moon = planet_params(rp=5.0 * R_earth_to_R_sun)

    def moon_flux(time, t0):
        moon.t0 = t0 + 0.25 * duration
        return batman.TransitModel(moon, time).light_curve(moon)

    shifts = np.zeros(12)
    shifts[5] = duration  # one epoch the fit cannot follow
    data = curve(truth, shifts=shifts, extra=moon_flux)
    emls = emls_for(tmp_path, data, duration, max_subtraction_rms_ratio=1.5)

    emls.subtract_planet_transit(*data[1:])

    assert len(emls.bary_t0s) >= 11
    assert data[0][5] not in emls.bary_t0s
    assert len(emls.epochs_flagged_rms) > 0


# --- B3: the fraction of the moon absorbed by the planet fit is measured -------------------------------------------

def moon_times(emls, search_input):
    """Moon mid-transit times of a hypothesis, one per epoch, as `run` computes them."""
    times = []
    for t0 in emls.bary_t0s:
        tau = ExoMoonLeastSquares.compute_moon_ts(t0, search_input.half_moon_orbit_transit_duration,
                                                  search_input.half_moon_transit_duration)
        times.append(ExoMoonLeastSquares.compute_moon_t0(emls.planet_t0, t0, search_input.moon_period,
                                                         search_input.moon_alpha, tau)[0])
    return times


def absorbed_fraction(tmp_path, offset_t14, radius_err, moon_radius=3.0):
    """Absorbed fraction for a moon whose largest offset from the planet is offset_t14 transit durations."""
    truth = planet_params()
    duration = t14(truth)
    planet_a = SMA * AU_TO_RSUN
    moon_a = offset_t14 * duration * 2 * np.pi * planet_a / PERIOD / AU_TO_RSUN
    moon_period = ExoMoonLeastSquares.au_to_period(300.0 * 3.0034893e-06, moon_a)
    data = curve(truth, epochs=24, noise=3e-4)
    emls = emls_for(tmp_path, data, duration)
    emls.planet_radius_err = radius_err
    hypothesis = emls.search_input(moon_period, 1.0, 300.0, moon_period, 0.0)
    flux = data[2] + emls.moon_signal(data[1], moon_times(emls, hypothesis), moon_radius) - 1
    emls.flux = flux
    before = __import__('copy').copy(emls)
    emls.time, emls.flux, emls.flux_err = emls.subtract_planet_transit(data[1], flux, data[3])
    hypothesis = emls.search_input(moon_period, 1.0, 300.0, moon_period, emls.planet_impact_param)
    emls.best_snr_radius = moon_radius
    return emls.measure_absorbed_fraction(before, hypothesis, moon_times(emls, hypothesis))


def test_planet_fit_absorbs_a_close_moon_more_than_a_wide_one(tmp_path):
    close = absorbed_fraction(tmp_path, 0.3, 0.5)
    wide = absorbed_fraction(tmp_path, 1.5, 0.5)

    assert 0.2 < close < 1.0
    assert wide < 0.5 * close


def test_tighter_radius_bound_absorbs_less(tmp_path):
    tight = absorbed_fraction(tmp_path, 0.3, 0.05)
    loose = absorbed_fraction(tmp_path, 0.3, 2.0)

    assert tight < loose
