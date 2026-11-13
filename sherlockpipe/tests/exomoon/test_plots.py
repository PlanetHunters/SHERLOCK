"""The EMLS figure module writes every figure and picks distinct SNR peaks."""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from sherlockpipe import exomoon_plots


def test_spectrum_peaks_skip_neighbours_of_the_same_peak_and_the_selected_period():
    periods = np.linspace(1, 10, 901)
    spectrum = np.exp(-((periods - 3) / 0.2) ** 2) * 5 + np.exp(-((periods - 7) / 0.2) ** 2) * 4 \
        + np.exp(-((periods - 5) / 0.2) ** 2) * 3
    peaks = exomoon_plots.spectrum_peaks(spectrum, periods, 2, exclude=[3.0], separation=0.1)
    assert [round(periods[index]) for index in peaks] == [7, 5]


def test_spectrum_peaks_ignore_nan():
    periods = np.linspace(1, 10, 10)
    spectrum = np.full(10, np.nan)
    spectrum[4] = 2.0
    assert exomoon_plots.spectrum_peaks(spectrum, periods, 3) == [4]


def test_alias_periods_are_the_sampling_aliases_inside_the_grid():
    aliases = dict(exomoon_plots.alias_periods(2.0, 10.0, 1.0, 5.0))
    assert np.isclose(aliases[1], 1 / (1 / 2 + 1 / 10))
    assert np.isclose(aliases[-1], 1 / (1 / 2 - 1 / 10))
    assert all(1.0 <= period <= 5.0 for period in aliases.values())


def test_every_figure_is_written(tmp_path):
    rng = np.random.default_rng(1)
    periods = np.linspace(1, 5, 50)
    alphas = np.linspace(0, 2 * np.pi, 8)
    matrix = rng.normal(3, 0.5, (len(alphas), len(periods)))
    spectrum = matrix.max(axis=0)
    exomoon_plots.search_overview_figure(tmp_path / "overview.png", periods, spectrum, 1 - spectrum / 100,
                                         spectrum - 3, spectrum - 3, 2.0, 4.0, snr_threshold=5.0,
                                         fap_threshold=0.01, aliases=exomoon_plots.alias_periods(2.0, 10.0, 1, 5))
    exomoon_plots.grid_map_figure(tmp_path / "map.png", periods, alphas, matrix, "SNR", "map", best=(2.0, 1.0),
                                  injected=(2.1, 7.0))
    time = np.linspace(-0.3, 0.3, 200)
    flux = 1 + rng.normal(0, 1e-4, 200)
    inside = np.abs(time) < 0.05
    exomoon_plots.hypothesis_figure(tmp_path / "hypothesis.png", "heading", time[inside], flux[inside],
                                    time[~inside], flux[~inside], np.full(np.sum(inside), 1 - 1e-4), 0.1, 20)
    nulls = rng.gumbel(3, 0.4, 100)
    exomoon_plots.null_distribution_figure(tmp_path / "nulls.png", nulls, 4.0, 0.1, 0.15, 0.1, 0.01, False, False)
    exomoon_plots.epoch_figure(tmp_path / "epoch.png", 0.0, 0.2, time, flux, time, np.ones(200), 0.05, "heading",
                               injected_time=time, injected_flux=np.ones(200), injected_t0=0.04,
                               timing_tolerance=0.03)
    t0s = np.arange(20) * 10.0
    share = np.where(np.arange(20) % 3 == 0, 2e-4, 0.0)
    exomoon_plots.epoch_depth_figure(tmp_path / "depths.png", t0s, rng.normal(0, 5e-5, 20) + share, share,
                                     np.full(20, 5e-5), 2.0, 1.0)
    written = sorted(os.listdir(tmp_path))
    assert written == sorted(["overview.png", "map.png", "hypothesis.png", "nulls.png", "epoch.png", "depths.png"])
    assert all(os.path.getsize(tmp_path / name) > 10000 for name in written)


def test_epoch_depth_figure_compares_the_changes_not_the_means(tmp_path):
    rng = np.random.default_rng(2)
    t0s = np.arange(30) * 10.0
    share = 3e-4 + np.where(np.arange(30) % 3 == 0, 2e-4, 0.0)
    errors = np.full(30, 3e-5)
    # Depths that follow the moon's changes, on another mean: the moon explains them.
    following = exomoon_plots.epoch_depth_figure(tmp_path / "a.png", t0s, -1e-4 + share - 3e-4
                                                 + rng.normal(0, 3e-5, 30), share, errors, 2.0, 1.0)
    assert following["correlation"] > 0.9
    assert following["chi2_with"] < 60 < following["chi2_without"]
    # Depths that change at other epochs: the moon makes it worse.
    other = exomoon_plots.epoch_depth_figure(tmp_path / "b.png", t0s, np.where(np.arange(30) % 3 == 1, 2e-4, 0.0)
                                             + rng.normal(0, 3e-5, 30), share, errors, 2.0, 1.0)
    assert other["correlation"] < 0
    assert other["chi2_with"] > other["chi2_without"]


def test_epochs_with_the_moon_over_the_planet_are_left_out(tmp_path):
    rng = np.random.default_rng(3)
    t0s = np.arange(30) * 10.0
    share = 3e-4 + np.where(np.arange(30) % 3 == 0, 2e-4, 0.0)
    depths = share + rng.normal(0, 3e-5, 30)
    hidden = np.arange(30) % 10 == 5
    depths[hidden] = 0.0
    errors = np.full(30, 3e-5)
    counted = exomoon_plots.epoch_depth_figure(tmp_path / "a.png", t0s, depths, share, errors, 2.0, 1.0,
                                               overlaps=hidden)
    assert counted["epochs"] == 27
    assert counted["correlation"] > 0.9
    assert exomoon_plots.epoch_depth_figure(tmp_path / "b.png", t0s, depths, share, errors, 2.0, 1.0)["correlation"] \
        < counted["correlation"]
