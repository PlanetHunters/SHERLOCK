"""Light curves that mix cadences: per-point exposures and inverse-variance weights.

EMLS does not know the mission. Each point's exposure comes from the spacing to its nearest
point, the planet and moon models are averaged over each point's own exposure, and the moon
statistic, the joint planet fit and the red-noise factor weight each point by its error. With one
cadence and equal errors everything reduces to the unweighted single-exposure pipeline.
"""
import types

import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import AU_TO_RSUN, R_earth_to_R_sun, ExoMoonLeastSquares
from sherlockpipe.tests.exomoon.test_audit_fixes import LD, PERIOD, SMA, run_to_search, system
from sherlockpipe.tests.exomoon.test_exposure import integrated

MINUTE = 1 / 1440


def segments(plan, gap=3.0, start=0.0):
    """Times of consecutive segments [(cadence in minutes, days)], separated by gaps, with their exposures."""
    times, exposures, t = [], [], start
    for cadence, days in plan:
        segment = t + np.arange(int(days / (cadence * MINUTE))) * cadence * MINUTE
        times.append(segment)
        exposures.append(np.full(len(segment), cadence * MINUTE))
        t = segment[-1] + gap
    return np.concatenate(times), np.concatenate(exposures)


def planet_params(radius=10.0):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a, params.inc = 0.0, PERIOD, radius * R_earth_to_R_sun, SMA * AU_TO_RSUN, 89.9
    params.ecc, params.w, params.u, params.limb_dark = 0.0, 90.0, LD, 'quadratic'
    return params


@pytest.mark.parametrize("plan", [
    [(1, 4), (29.4, 6), (1, 3)],                 # Kepler short and long cadence
    [(2, 5), (10, 5), (30, 5), (200 / 60, 5)],   # TESS 2 min, 10 and 30 min FFIs, 200 s
    [(0.5, 2), (3, 3), (25, 6)],                 # cadences of no mission
])
def test_point_exposures_come_from_the_spacing_without_a_mission(plan):
    time, truth = segments(plan)
    # Missing points (flagged cadences) at random, one in ten.
    kept = np.random.default_rng(3).random(len(time)) > 0.1
    time, truth = time[kept], truth[kept]
    shuffled = np.random.default_rng(4).permutation(len(time))

    exposures = ExoMoonLeastSquares.infer_point_exposures(time[shuffled])

    np.testing.assert_allclose(exposures, truth[shuffled], rtol=1e-6)


def test_one_cadence_keeps_one_exposure(tmp_path):
    t0s = np.arange(4) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, 30 * MINUTE) for t0 in t0s])
    emls = system(tmp_path, curve=(t0s, time, np.ones_like(time), np.full(len(time), 1e-4), 0.25),
                  exposure_time='auto')

    assert emls.exposure_lookup is None
    assert emls.exposure_time == pytest.approx(30 * MINUTE, rel=1e-9)
    assert emls.exposure_at(time[:10]) == emls.exposure_time


def test_mixed_cadences_get_one_exposure_per_point(tmp_path):
    t0s = np.arange(6) * PERIOD
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, (1 if i % 2 else 30) * MINUTE) for i, t0 in enumerate(t0s)])
    truth = np.concatenate([np.full(len(np.arange(-2.0, 2.0, (1 if i % 2 else 30) * MINUTE)),
                                    (1 if i % 2 else 30) * MINUTE) for i in range(len(t0s))])
    noise = np.where(truth < 2 * MINUTE, 5.5e-4, 1e-4)
    emls = system(tmp_path, curve=(t0s, time, 1 + np.random.default_rng(1).normal(0, noise), noise, 0.25),
                  exposure_time='auto')

    assert emls.exposure_lookup is not None
    np.testing.assert_allclose(emls.exposure_at(time), truth, rtol=1e-6)
    # Shuffled subsets of the curve's own times find their points.
    pick = np.random.default_rng(2).choice(len(time), 500, replace=False)
    np.testing.assert_allclose(emls.exposure_at(time[pick]), truth[pick], rtol=1e-6)
    # Both cadences carry the same weight per day here; the templates take the one found first of the two.
    assert any(emls.exposure_time == pytest.approx(value, rel=1e-6) for value in (MINUTE, 30 * MINUTE))


def test_per_point_model_matches_the_model_of_each_cadence():
    params = planet_params()
    time = np.linspace(-0.3, 0.3, 3001)
    exposures = np.where(np.arange(len(time)) % 3 == 0, 30 * MINUTE, MINUTE)

    mixed = ExoMoonLeastSquares.transit_model(params, time, exposures).light_curve(params)

    for value in (MINUTE, 30 * MINUTE):
        index = exposures == value
        alone = ExoMoonLeastSquares.transit_model(params, time[index], value).light_curve(params)
        np.testing.assert_array_equal(mixed[index], alone)


def test_weighted_statistics_are_the_unweighted_ones_with_equal_errors():
    rng = np.random.default_rng(7)
    it_flux, oot_flux = 1 - 2e-4 + rng.normal(0, 1e-3, 80), 1 + rng.normal(0, 1e-3, 400)
    plain = (np.mean(oot_flux) - np.mean(it_flux)) / (np.std(oot_flux) * np.sqrt(1 / 80 + 1 / 400))

    weighted = ExoMoonLeastSquares.box_snr(it_flux, oot_flux, np.full(80, 3e-3), np.full(400, 3e-3))

    assert weighted == pytest.approx(plain, rel=1e-12)
    assert ExoMoonLeastSquares.box_snr(it_flux, oot_flux) == pytest.approx(plain, rel=1e-12)
    flux = np.concatenate((it_flux, oot_flux))
    inside = np.arange(len(flux)) < 80
    columns = np.column_stack((rng.normal(size=len(flux)), rng.normal(size=len(flux)),
                               (np.arange(len(flux)) % 4).astype(float)))
    unweighted_fit = ExoMoonLeastSquares.planet_fit(flux, inside, columns)
    weighted_fit = ExoMoonLeastSquares.planet_fit(flux, inside, columns, flux_err=np.full(len(flux), 2e-3))
    np.testing.assert_allclose(weighted_fit["model"], unweighted_fit["model"], rtol=1e-9, atol=1e-15)
    assert weighted_fit["inflation"] == pytest.approx(unweighted_fit["inflation"], rel=1e-9)


def test_weighted_box_snr_on_mixed_cadences_is_calibrated_and_more_sensitive():
    """1-minute points are 5.5 times noisier than 30-minute points; the plain mean lets them dominate."""
    rng = np.random.default_rng(11)
    sigma = np.r_[np.full(1800, 5.5e-4), np.full(60, 1e-4)]   # 30 h at 1 min and 30 h at 30 min
    inside = np.r_[np.arange(1800) % 300 < 60, np.arange(60) % 10 < 2]
    depth = 6e-5
    weighted, plain, null = [], [], []
    for _ in range(300):
        noise = rng.normal(0, sigma)
        flux = 1 + noise - depth * inside
        weighted.append(ExoMoonLeastSquares.box_snr(flux[inside], flux[~inside], sigma[inside], sigma[~inside]))
        plain.append(ExoMoonLeastSquares.box_snr(flux[inside], flux[~inside]))
        null.append(ExoMoonLeastSquares.box_snr((1 + noise)[inside], (1 + noise)[~inside], sigma[inside],
                                                sigma[~inside]))
    expected = depth / np.sqrt(1 / np.sum(1 / sigma[inside] ** 2) + 1 / np.sum(1 / sigma[~inside] ** 2))

    assert np.mean(weighted) == pytest.approx(expected, rel=0.1)
    assert np.mean(weighted) > 1.25 * np.mean(plain)
    assert np.std(null) == pytest.approx(1.0, abs=0.12)
    assert abs(np.mean(null)) < 0.15


def test_red_noise_of_white_noise_with_mixed_cadences_is_one():
    time, exposures = segments([(1, 30), (30, 60), (1, 30)], gap=1.0)
    sigma = np.where(exposures < 2 * MINUTE, 5.5e-4, 1e-4)
    flux = 1 + np.random.default_rng(5).normal(0, sigma)
    owner = types.SimpleNamespace(planet_t0=0.5, planet_period=1000.0)

    beta = ExoMoonLeastSquares.measure_red_noise(owner, time, flux, 0.1, 0.25, sigma)

    assert beta == pytest.approx(1.0, abs=0.1)


def test_errors_of_each_cadence_are_put_on_one_scale(tmp_path):
    t0s = np.arange(6) * PERIOD
    cadence = [1 if i % 2 else 30 for i in range(len(t0s))]
    time = np.concatenate([t0 + np.arange(-4.0, 4.0, c * MINUTE) for t0, c in zip(t0s, cadence)])
    exposures = np.concatenate([np.full(len(np.arange(-4.0, 4.0, c * MINUTE)), c * MINUTE) for c in cadence])
    true_sigma = np.where(exposures < 2 * MINUTE, 5.5e-4, 1e-4)
    # The 1-minute errors are given half their real size.
    given = np.where(exposures < 2 * MINUTE, 0.5 * 5.5e-4, 1e-4)
    flux = 1 + np.random.default_rng(8).normal(0, true_sigma)

    emls = system(tmp_path, curve=(t0s, time, flux, given, 0.25), exposure_time='auto')

    ratio = emls.flux_err / given
    assert np.median(ratio[exposures < 2 * MINUTE]) / np.median(ratio[exposures > 2 * MINUTE]) == \
        pytest.approx(2.0, rel=0.1)


def mixed_cadence_planet(tmp_path, exposure_time):
    """Fit and subtract a deep planet observed at 1 and 30 min; return the in-transit residuals in sigma."""
    params = planet_params(radius=15.0)
    t0s = np.arange(8) * PERIOD
    cadence = [1 if i % 2 else 30 for i in range(len(t0s))]
    time = np.concatenate([t0 + np.arange(-1.5, 1.5, c * MINUTE) for t0, c in zip(t0s, cadence)])
    exposures = np.concatenate([np.full(len(np.arange(-1.5, 1.5, c * MINUTE)), c * MINUTE) for c in cadence])
    sigma = np.where(exposures < 2 * MINUTE, 5.5e-4, 1e-4)
    flux = np.empty(len(time))
    for value in (MINUTE, 30 * MINUTE):
        index = exposures == value
        flux[index] = integrated(params, time[index], value)
    flux += np.random.default_rng(9).normal(0, sigma)
    duration = PERIOD / np.pi * np.arcsin((1 + params.rp) / params.a)
    emls = system(tmp_path, curve=(t0s, time, flux, sigma, duration), exposure_time=exposure_time,
                  min_cadences_per_t0=5)
    run_to_search(emls)
    in_transit = np.any([np.abs(emls.time - t0) < duration / 2 for t0 in emls.bary_t0s], axis=0)
    long_points = emls.exposure_at(emls.time) > 2 * MINUTE if emls.exposure_lookup is not None else \
        np.isin(np.round(emls.time, 9), np.round(time[exposures > 2 * MINUTE], 9))
    residual = (emls.flux - 1) / emls.flux_err
    return residual[in_transit & long_points], residual[in_transit & ~long_points]


def test_planet_on_mixed_cadences_is_subtracted_with_each_point_exposure(tmp_path):
    (tmp_path / 'per_point').mkdir()
    (tmp_path / 'one').mkdir()

    long_per_point, short_per_point = mixed_cadence_planet(tmp_path / 'per_point', 'auto')
    long_one, _ = mixed_cadence_planet(tmp_path / 'one', MINUTE)

    assert np.sqrt(np.mean(long_per_point ** 2)) < 1.2
    assert np.sqrt(np.mean(short_per_point ** 2)) < 1.2
    assert np.sqrt(np.mean(long_one ** 2)) > 1.5 * np.sqrt(np.mean(long_per_point ** 2))


def test_null_curves_keep_each_point_exposure(tmp_path):
    t0s = np.arange(6) * PERIOD
    cadence = [1 if i % 2 else 30 for i in range(len(t0s))]
    time = np.concatenate([t0 + np.arange(-6.0, 6.0, c * MINUTE) for t0, c in zip(t0s, cadence)])
    exposures = np.concatenate([np.full(len(np.arange(-6.0, 6.0, c * MINUTE)), c * MINUTE) for c in cadence])
    sigma = np.where(exposures < 2 * MINUTE, 5.5e-4, 1e-4)
    emls = system(tmp_path, curve=(t0s, time, 1 + np.random.default_rng(3).normal(0, sigma), sigma, 0.25),
                  exposure_time='auto')
    emls.bary_t0s = list(t0s)
    valid = [np.array([2.0, -2.0, 3.0]) for _ in t0s]

    null_time, _, _, epochs, null_exposures = emls.null_curve(emls, np.random.default_rng(0), 0.5, valid)

    for t0, c in zip(t0s, cadence):
        near = np.abs(null_time - t0) < 0.5
        assert np.any(near)
        np.testing.assert_allclose(null_exposures[near], c * MINUTE, rtol=1e-6)


def test_a_time_held_at_two_cadences_is_kept_once_at_the_shorter():
    from sherlockpipe.exomoon import combine_cadence_curves
    long_time, _ = segments([(29.4, 20), (29.4, 20)], gap=0.1)
    short_time = long_time[0] + 22 + np.arange(int(10 / MINUTE)) * MINUTE   # 10 days inside the second segment
    curves = [(long_time, np.full(len(long_time), 2.0), np.full(len(long_time), 1e-4)),
              (short_time, np.full(len(short_time), 1.0), np.full(len(short_time), 5e-4))]

    time, flux, flux_err = combine_cadence_curves(curves)

    overlap = (long_time >= short_time[0] - 15 * MINUTE) & (long_time <= short_time[-1] + 15 * MINUTE)
    assert np.count_nonzero(flux == 1.0) == len(short_time)
    assert np.count_nonzero(flux == 2.0) == len(long_time) - np.count_nonzero(overlap)
    assert np.all(np.diff(time) > 0) and len(flux_err) == len(time)
    np.testing.assert_allclose(ExoMoonLeastSquares.infer_point_exposures(time)[flux == 1.0], MINUTE, rtol=1e-6)


@pytest.mark.parametrize("epoch_depth", [False, True])
def test_sparse_planet_fit_is_the_dense_weighted_least_squares(epoch_depth):
    """planet_fit sums the normal equations over each point's non-zero columns; check against the dense design."""
    rng = np.random.default_rng(21)
    n, n_epochs = 3000, 12
    epoch = rng.integers(-1, n_epochs, n)
    columns = np.column_stack([rng.normal(size=n) * (epoch >= 0), rng.normal(size=n) * (epoch >= 0), epoch,
                               rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)]).astype(float)
    inside = rng.random(n) < 0.2
    flux_err = 1e-3 * (1 + 4 * rng.random(n))
    flux = 1 + rng.normal(0, flux_err)

    fit = ExoMoonLeastSquares.planet_fit(flux, inside, columns, epoch_depth=epoch_depth, flux_err=flux_err)

    used = epoch >= 0
    depth_terms = [np.where(epoch == e, columns[:, 0], 0.0) for e in range(n_epochs)] if epoch_depth \
        else [columns[:, 0]]
    t0_terms = [np.where(epoch == e, columns[:, 1], 0.0) for e in range(n_epochs) if np.any(used & (epoch == e))]
    design = np.column_stack([np.ones(n), inside.astype(float)] + depth_terms + [columns[:, 3], columns[:, 4],
                                                                              columns[:, 5]] + t0_terms)
    weights = 1 / flux_err ** 2
    normal = design.T @ (design * weights[:, None])
    coefficients = np.linalg.solve(normal, design.T @ (flux * weights))
    np.testing.assert_allclose(fit["model"], design[:, 2:] @ coefficients[2:], rtol=1e-8, atol=1e-12)
    inflation = np.sqrt(np.linalg.inv(normal)[1, 1] / (1 / weights[inside].sum() + 1 / weights[~inside].sum()))
    assert fit["inflation"] == pytest.approx(inflation, rel=1e-9)


def test_upper_clip_is_astropy_sigma_clip():
    from astropy.stats import sigma_clip
    rng = np.random.default_rng(31)
    for _ in range(200):
        values = rng.standard_t(3, size=rng.integers(10, 3000))
        expected = sigma_clip(values, sigma=5.0, sigma_lower=float('inf'), sigma_upper=3).mask
        np.testing.assert_array_equal(ExoMoonLeastSquares.upper_clip(values), expected)


def test_grid_search_on_the_fixed_stack_is_the_generic_split(tmp_path):
    """search() takes the fixed-stack path; the same hypothesis through the generic split gives the same stack."""
    t0s = np.arange(6) * PERIOD
    cadence = [1 if i % 2 else 30 for i in range(len(t0s))]
    time = np.concatenate([t0 + np.arange(-1.5, 1.5, c * MINUTE) for t0, c in zip(t0s, cadence)])
    exposures = np.concatenate([np.full(len(np.arange(-1.5, 1.5, c * MINUTE)), c * MINUTE) for c in cadence])
    sigma = np.where(exposures < 2 * MINUTE, 5.5e-4, 1e-4)
    params = planet_params(radius=10.0)
    flux = np.empty(len(time))
    for value in (MINUTE, 30 * MINUTE):
        index = exposures == value
        flux[index] = integrated(params, time[index], value)
    flux += np.random.default_rng(12).standard_t(4, len(time)) * sigma
    duration = PERIOD / np.pi * np.arcsin((1 + params.rp) / params.a)
    emls = system(tmp_path, curve=(t0s, time, flux, sigma, duration), exposure_time='auto', min_cadences_per_t0=5)
    inputs = run_to_search(emls)
    column = emls.search_planet_depth_column()

    for search_input in inputs[::max(1, len(inputs) // 60)]:
        stacked = emls.split_scenarios(search_input, emls.time, emls.flux, emls.flux_err, emls.bary_t0s, extra=column)
        generic = emls.split_scenarios(search_input, emls.time.copy(), emls.flux, emls.flux_err,
                                       list(emls.bary_t0s), extra=column)
        assert emls._search_stack_cache[1] is not None
        for index in (0, 1, 4):
            np.testing.assert_array_equal(stacked[index], generic[index])
        for index in (2, 3, 5, 6):
            np.testing.assert_allclose(stacked[index], generic[index], rtol=0, atol=1e-11)
        assert stacked[8] == pytest.approx(generic[8], rel=1e-9)
