"""Regression tests for audit items H1 (placeholder scores in the SDE spectrum) and H3 (depth gate).

H1: a hypothesis without a dip, or with a brightening, scores exactly as the null, because a moon
only dims the star. No placeholder value reaches the spectrum. H3: every positive dip is fitted,
including those shallower than min_radius, and a best candidate below the range is flagged.
"""
import logging

import numpy as np
import pytest

from sherlockpipe.exomoon import R_earth_to_R_sun, ExoMoonLeastSquares
from sherlockpipe.tests.exomoon import test_search_integrity as guard

SDE_CAP = np.sqrt(guard.PERIOD_GRID - 1)


def residuals_for(mean_in_window, min_radius=1.0):
    """Score a box-shaped scenario with the given in-window mean against a V-shaped template."""
    it_time = np.linspace(-0.1, 0.1, 200)
    it_flux = np.full(len(it_time), mean_in_window)
    oot_flux = np.ones(400)
    template_time = np.linspace(-0.2, 0.2, 401)
    template = 1 - 1e-4 * np.clip(1 - np.abs(template_time) / 0.2, 0, None)
    return ExoMoonLeastSquares.calculate_residuals(
        it_time, it_flux, np.full(len(it_time), 1e-4), np.linspace(0.2, 0.6, 400), oot_flux, np.full(400, 1e-4),
        mean_in_window, 600, [(1.0, template_time, template)], min_radius, 4.0, 1.0, 4)


@pytest.mark.parametrize('mean_in_window', [1.0, 1.0002])
def test_no_dip_or_a_brightening_scores_as_the_null(mean_in_window):
    residual, baseline, radius, model, unusable = residuals_for(mean_in_window)

    assert residual == baseline
    assert radius == 0.0
    assert not unusable
    np.testing.assert_allclose(model, 1.0)


def test_dip_shallower_than_the_smallest_moon_is_fitted():
    # A dip with the template's shape at 0.95^2 of its depth: a 0.95 R_earth moon for the
    # 1 R_earth template, below min_radius = 1 R_earth. The radius comes from the least-squares
    # amplitude through the template (audit items H13, H14).
    it_time = np.linspace(-0.1, 0.1, 200)
    template_time = np.linspace(-0.2, 0.2, 401)
    template = 1 - 1e-4 * np.clip(1 - np.abs(template_time) / 0.2, 0, None)
    it_flux = 1 - 0.95 ** 2 * (1 - np.interp(it_time, template_time, template))
    residual, baseline, radius, _, unusable = ExoMoonLeastSquares.calculate_residuals(
        it_time, it_flux, np.full(len(it_time), 1e-4), np.linspace(0.2, 0.6, 400), np.ones(400), np.full(400, 1e-4),
        float(np.mean(it_flux)), 600, [(1.0, template_time, template)], 1.0, 4.0, 1.0, 4)

    assert radius == pytest.approx(0.95, rel=1e-6)
    assert np.isfinite(residual) and residual < baseline
    assert not unusable


@pytest.mark.parametrize('seed', [407, 413])
def test_noise_sde_does_not_sit_at_the_single_period_cap(seed, tmp_path):
    """Seed 407 gave exactly sqrt(N - 1) before H1/H3, when one surviving period rose above a flat spectrum."""
    emls = guard.execute(tmp_path, seed=seed)

    assert abs(emls.max_sde - SDE_CAP) > 0.01
    assert emls.max_sde < SDE_CAP


def test_moon_sde_rises_above_the_noise_runs(tmp_path):
    """The residual SDE is a diagnostic only (audit item H7); a moon should still rank above noise."""
    folders = {name: tmp_path / name for name in ('noise1', 'noise2', 'noise3', 'moon')}
    for folder in folders.values():
        folder.mkdir()
    noise = [guard.execute(folders[f'noise{seed}'], seed=seed).max_sde for seed in (1, 2, 3)]
    moon = guard.execute(folders['moon'], moon_radius=3.0).max_sde

    assert moon > max(noise)


def test_shallow_moon_is_reported_and_flagged_below_the_range(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(guard, 'NOISE', 2e-5)

    with caplog.at_level(logging.WARNING):
        emls = guard.execute(tmp_path, moon_radius=0.8)

    assert 0 < emls.best_snr_radius < emls.min_radius
    assert emls.best_snr_radius_below_range
    assert abs(emls.recovered_period - guard.moon_period()) / guard.moon_period() < 0.01
    assert any('below the searched range' in record.getMessage() for record in caplog.records)
