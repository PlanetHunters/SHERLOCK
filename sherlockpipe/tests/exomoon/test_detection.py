"""Detection statistic, false-alarm probability, single events and recovery: audit items H7, H8, H10, H12, I1, I8.

H7/H8: the SNR is the detection statistic, and every reported value describes the hypothesis it
selects. H10: a per-target null from pseudo-epochs, windows of the same light curve away from the
transits, gives a false-alarm probability. H12: a leave-one-out SNR flags a signal carried by one
transit. I1: recovered needs a detection, matching moon times and the period, with aliases apart.
"""
import batman
import numpy as np
import pytest
from scipy import stats

import sherlockpipe.exomoon as exomoon
from sherlockpipe.exomoon import AU_TO_RSUN, ExoMoonLeastSquares, R_earth_to_R_sun
from sherlockpipe.tests.exomoon import test_search_integrity as guard

NULL_CADENCE = 10 / 1440


def continuous_curve(seed, epochs=6, moon_radius=None):
    """The guard planet with data over the whole orbit, so pseudo-epochs exist."""
    t0s = np.arange(epochs) * guard.PLANET_PERIOD
    time = np.arange(-50.0, t0s[-1] + 50.0, NULL_CADENCE)
    params = batman.TransitParams()
    params.t0, params.per = 0.0, guard.PLANET_PERIOD
    params.rp = guard.PLANET_RADIUS * R_earth_to_R_sun / guard.STAR_RADIUS
    params.a = guard.PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN / guard.STAR_RADIUS
    params.inc, params.ecc, params.w = guard.PLANET_INC, 0.0, 90.0
    params.u, params.limb_dark = guard.LIMB_DARKENING, 'quadratic'
    flux = batman.TransitModel(params, time).light_curve(params)
    flux = flux + np.random.default_rng(seed).normal(0, guard.NOISE, len(time))
    if moon_radius is not None:
        flux = guard.with_moon(flux, time, t0s, params, moon_radius)
    duration = guard.PLANET_PERIOD / np.pi * np.arcsin((1 + params.rp) / params.a)
    return t0s, time, flux, np.full(len(time), guard.NOISE), duration


def null_search(tmp_path, moon_radius=None, null_runs=30):
    t0s, time, flux, flux_err, duration = continuous_curve(3, moon_radius=moon_radius)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    emls.null_runs = null_runs
    emls.run()
    return emls


# --- H7 / H8: one hypothesis -------------------------------------------------------------------------------------

def test_best_hypothesis_is_the_grid_maximum(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    periods, alphas = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0])
    matrix = np.array([[1.0, np.nan, 2.0], [0.5, 4.0, 1.0]])
    emls.search = lambda search_input, return_lc=False: (0, 0, 0, -np.inf, False)  # refinement never wins

    assert emls.select_best(periods, alphas, matrix) == (1, 2.0, 1.0, 4.0)
    assert emls.select_best(periods, alphas, np.full((2, 3), np.nan)) is None


@pytest.fixture(scope='module')
def moon_run(tmp_path_factory):
    return guard.execute(tmp_path_factory.mktemp('moon'), moon_radius=3.0)


def test_reported_statistics_come_from_one_search(moon_run):
    emls = moon_run
    # The selected (P, alpha) reproduces the reported SNR and radius in a single search call.
    grid_max = emls.max_moon_period(emls.planet_mass)
    best = emls.search_input(emls.recovered_period, emls.recovered_alpha, emls.planet_mass, grid_max,
                             emls.planet_impact_param)
    result = emls.search(best, return_lc=True)
    assert result[3] == pytest.approx(emls.max_snr, rel=1e-9)
    assert result[2] == pytest.approx(emls.best_snr_radius, rel=1e-9)
    assert emls.sde is not None and emls.sde > 0


# --- H10: false-alarm probability --------------------------------------------------------------------------------

def test_false_alarm_follows_the_null_distribution(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    nulls = stats.gumbel_r.rvs(loc=3.0, scale=0.4, size=400, random_state=np.random.default_rng(1))
    emls.null_snrs = list(nulls)
    threshold = stats.gumbel_r.isf(0.01, loc=3.0, scale=0.4)

    fap, empirical = emls.false_alarm(threshold)
    assert fap == pytest.approx(0.01, rel=0.3)
    assert empirical == pytest.approx((np.sum(nulls >= threshold) + 1) / 401)
    assert emls.false_alarm(10.0)[0] < 1e-4
    emls.null_snrs = list(nulls[:exomoon.MIN_NULL_RUNS - 1])
    assert emls.false_alarm(threshold) == (None, None)


def test_pseudo_epochs_hold_neither_the_planet_nor_a_moon(tmp_path):
    t0s, time, flux, flux_err, duration = continuous_curve(3)
    emls = guard.build(np.ones_like(time), time, flux_err, tmp_path, t0s, duration)
    trim = emls.trim_half_window(np.max(emls.planet_mass_grid))
    search = 0.85
    offsets = emls.null_offsets(trim, trim + search)
    # Mark everything the planet or a searched moon could touch around each real transit: the
    # widest moon window, +-0.85 d here.
    marked = np.ones_like(time)
    for t0 in t0s:
        marked[np.abs(time - t0) < search] = 0.5
    emls.flux = marked

    valid = emls.null_valid_offsets(emls, trim, offsets)
    null_time, null_flux, _, epochs = emls.null_curve(emls, np.random.default_rng(0), trim, valid)
    planet = emls.null_planet_params()
    model = np.ones_like(null_time)
    for t0 in epochs:
        planet.t0 = t0
        window = np.abs(null_time - t0) < trim
        model[window] = ExoMoonLeastSquares.transit_model(planet, null_time[window]).light_curve(planet)
    assert list(epochs) == list(t0s)
    assert np.all(np.abs(null_flux / model - 1) < 1e-9)  # no marked point, and the planet sits on each epoch
    assert all(np.any(np.abs(null_time - t0) < trim / 2) for t0 in t0s)


def test_no_null_without_data_away_from_the_transits(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)  # +-3 d around each transit only
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    trim = emls.trim_half_window(np.max(emls.planet_mass_grid))
    valid = emls.null_valid_offsets(emls, trim, emls.null_offsets(trim, trim + 0.85))

    assert emls.null_curve(emls, np.random.default_rng(0), trim, valid) is None


def test_independent_offsets_count_only_data(tmp_path):
    t0s, time, flux, flux_err, duration = continuous_curve(3)
    keep = np.abs(((time + 50) % guard.PLANET_PERIOD) - 50) < 6.0  # data only within 6 d of each transit
    emls = guard.build(flux[keep], time[keep], flux_err[keep], tmp_path, t0s, duration)
    trim = emls.trim_half_window(np.max(emls.planet_mass_grid))
    offsets = emls.null_offsets(trim, trim + 0.85)

    assert ExoMoonLeastSquares.independent_offsets(offsets, trim) > 10  # room in the orbit
    valid = emls.null_valid_offsets(emls, trim, offsets)
    # but data for one window on each side of the transit
    assert max(ExoMoonLeastSquares.independent_offsets(v, trim) for v in valid) <= 2


@pytest.fixture(scope='module')
def noise_with_nulls(tmp_path_factory):
    return null_search(tmp_path_factory.mktemp('noise_nulls'))


@pytest.fixture(scope='module')
def moon_with_nulls(tmp_path_factory):
    return null_search(tmp_path_factory.mktemp('moon_nulls'), moon_radius=3.0)


def test_noise_is_not_a_detection(noise_with_nulls):
    emls = noise_with_nulls
    assert len(emls.null_snrs) == 30
    assert emls.null_independent_offsets >= 3
    assert emls.fap > 0.05
    assert emls.detected is False


def test_moon_is_a_detection(moon_with_nulls):
    emls = moon_with_nulls
    assert emls.max_snr > 10
    assert emls.fap < 1e-6
    assert emls.fap_empirical == pytest.approx(1 / 31)
    assert emls.detected is True
    assert emls.single_event is False


def test_given_nulls_are_used_without_new_searches(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    emls.null_runs = 50  # would fail: the fixture has no data away from the transits
    emls.run(null_snrs=np.linspace(2.0, 4.5, 40))

    assert len(emls.null_snrs) == 40
    assert emls.detected is False


# --- H12: single event -------------------------------------------------------------------------------------------

def test_single_event_is_flagged(tmp_path):
    """The audit's case: one 0.2 d dip of depth 5e-3 in epoch 5 of 12, inside the widest moon window."""
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    flux = flux.copy()
    dip = np.abs(time - (t0s[5] + 0.5)) < 0.1
    flux[dip] -= 5e-3
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    emls.run()

    assert emls.max_snr > 7
    assert emls.single_event is True
    assert emls.loo_snr_min < 0.5 * emls.max_snr
    assert len(emls.epoch_stats) == len(emls.bary_t0s)


def test_repeating_moon_is_not_a_single_event(moon_run):
    assert moon_run.single_event is False
    assert moon_run.loo_snr_min > 0.8 * moon_run.max_snr


# --- I1: recovery ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('found, expected', [
    (10.0, (True, False, 0)),
    (10.2, (True, False, 0)),
    # Kepler-1514 b injection: the k = 1 alias of a 10 d moon of a 217.83 d planet is 4.4 % away,
    # inside the 5 % tolerance, and was counted as the period.
    (9.5558, (False, True, 1)),
    (8.1299, (False, True, 5)),
])
def test_recovered_period_is_the_nearest_alias(found, expected):
    period_match, alias_match, k, _, _ = ExoMoonLeastSquares.classify_recovered_period(10.0, 217.83, found)

    assert (period_match, alias_match, k) == expected


def test_period_tolerance_still_applies_without_close_aliases():
    # A 10 d moon of a 30 d planet: the nearest aliases are 7.5 d and 15 d, so 11 d is the
    # injected period's family member but 10 % off, which is neither a match nor an alias.
    period_match, alias_match, k, _, error = ExoMoonLeastSquares.classify_recovered_period(10.0, 30.0, 11.0)

    assert k == 0 and error == pytest.approx(0.1)
    assert (period_match, alias_match) == (False, False)


def test_undetectable_moon_is_not_recovered(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(5)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    emls.run(inject_params=dict(moon_radius=0.02, moon_period=13.27, moon_mass=0.001, initial_alpha=0.52),
             null_snrs=np.linspace(2.0, 4.5, 40))

    assert emls.detected is False
    assert emls.recovered is False
    assert emls.recovered_alias is False


def test_recovery_needs_a_false_alarm_probability(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(5)
    emls = guard.build(flux, time, flux_err, tmp_path, t0s, duration)
    emls.run(inject_params=dict(moon_radius=3.0, moon_period=13.27, moon_mass=5.0, initial_alpha=0.52))

    assert emls.detected is None
    assert emls.recovered is None


def test_invalid_detection_settings_are_rejected(tmp_path):
    t0s, time, flux, flux_err, duration, _ = guard.planet_curve(3)
    with pytest.raises(ValueError, match='null_runs'):
        ExoMoonLeastSquares(*guard_args(tmp_path, t0s, time, flux, flux_err, duration), null_runs=-1)
    with pytest.raises(ValueError, match='fap_threshold'):
        ExoMoonLeastSquares(*guard_args(tmp_path, t0s, time, flux, flux_err, duration), fap_threshold=1.5)


def guard_args(output_dir, t0s, time, flux, flux_err, duration):
    return (str(output_dir), 2, 1.0, guard.STAR_RADIUS, guard.LIMB_DARKENING, guard.PLANET_RADIUS, 0.5,
            guard.PLANET_MASS, guard.PLANET_PERIOD, 0.0, duration, guard.PLANET_SEMIMAJOR_AXIS, guard.PLANET_INC, 0.5,
            0.0, 0.0, 90.0, guard.PLANET_SEMIMAJOR_AXIS * AU_TO_RSUN * np.cos(np.deg2rad(guard.PLANET_INC)), 1.0, 4.0,
            t0s, time, flux, flux_err, 5.5)
