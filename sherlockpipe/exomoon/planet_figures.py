"""Diagnostic figures of the planet stage: the fold fit, each subtracted or masked epoch, and the TTVs.

They use the paper style and colour code of `sherlockpipe.exomoon.plots`.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sherlockpipe.exomoon.plots import (ALIAS_STYLE, BIN_COLOR, DATA_COLOR, INJECTED_COLOR, INJECTED_STYLE,
                                        MODEL_COLOR, MODEL_STYLE, PLANET_COLOR, REFERENCE_COLOR, _save, _style,
                                        legend_below, paper_figure, wrap)


class PlanetFigures:
    """Figures written by `PlanetModel.subtract_planet_transit` into object_dir."""

    def epoch_view_dir(self):
        """Return the directory of the per-epoch planet figures, created if needed."""
        view_dir = os.path.join(self.object_dir, "t0s_view")
        os.makedirs(view_dir, exist_ok=True)
        return view_dir

    @paper_figure
    def plot_fold_fit(self, fold, start_params, params_best, guess, theta):
        """Plot the fold with the input-shape model and the fitted one (fold_fit.png)."""
        model_guess = self.transit_model(start_params, fold.time, fold.exposure).light_curve(start_params)
        model_best = self.transit_model(params_best, fold.time, fold.exposure).light_curve(params_best)
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.scatter(fold.time * 24, (fold.flux - 1) * 1e6, color=DATA_COLOR, alpha=0.35, s=6, linewidths=0,
                   rasterized=True, label="Folded flux")
        order = np.argsort(fold.time)
        ax.plot(fold.time[order] * 24, (model_guess[order] - 1) * 1e6, color=INJECTED_COLOR, linestyle=INJECTED_STYLE,
                linewidth=2.5, label="Input shape")
        ax.plot(fold.time[order] * 24, (model_best[order] - 1) * 1e6, color=MODEL_COLOR, linestyle=MODEL_STYLE,
                linewidth=3, label="Fold fit")
        ax.set_title(wrap(f"Fold fit: T14 {guess[1] * 24:.2f} -> {theta[1] * 24:.2f} h, b {guess[2]:.2f} -> "
                          f"{theta[2]:.2f}, depth {guess[3] ** 2 * 1e6:.0f} -> {theta[3] ** 2 * 1e6:.0f} ppm"))
        ax.set_xlabel("Time from mid-transit (h)")
        ax.set_ylabel("Relative flux (ppm)")
        _style(ax)
        fig.tight_layout()
        legend_below(fig, ax)
        _save(fig, os.path.join(self.object_dir, "fold_fit.png"))

    @paper_figure
    def plot_subtracted_epoch(self, epoch_time, epoch_flux, model, residual, best_t0, rejected):
        """Plot one epoch with its planet model and after the subtraction (t0s_view/T0_*_sub.png,
        or *_rejected.png for a dropped epoch).
        """
        suffix = "_rejected" if rejected else "_sub"
        hours = (epoch_time - best_t0) * 24
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
        axs[0].scatter(hours, (epoch_flux - 1) * 1e6, color=DATA_COLOR, s=10, linewidths=0, rasterized=True,
                       label="Flux")
        axs[0].plot(hours, (model - 1) * 1e6, color=MODEL_COLOR, linestyle=MODEL_STYLE, linewidth=3,
                    label="Planet model")
        axs[0].set_title(wrap(f"Epoch at {best_t0:.3f} d{' (rejected)' if rejected else ''}: planet model"))
        axs[0].set_ylabel("Relative flux (ppm)")
        axs[1].scatter(hours, (residual - 1) * 1e6, color=BIN_COLOR, s=10, linewidths=0, rasterized=True)
        axs[1].axhline(0, color=REFERENCE_COLOR, linewidth=1)
        axs[1].set_title("After the subtraction")
        axs[1].set_xlabel("Time from the fitted mid-transit (h)")
        axs[1].set_ylabel("Residual (ppm)")
        for ax in axs:
            _style(ax)
        fig.tight_layout()
        legend_below(fig, axs)
        _save(fig, os.path.join(self.epoch_view_dir(), "T0_" + str(round(best_t0, 3)) + suffix + ".png"))

    @paper_figure
    def plot_masked_epoch(self, epoch_time, epoch_flux, model, centre, half_mask):
        """Plot one epoch before and after its transit is masked (t0s_view/T0_*_mask.png)."""
        hours = (epoch_time - centre) * 24
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
        ax1.scatter(hours, (epoch_flux - 1) * 1e6, color=DATA_COLOR, s=10, linewidths=0, rasterized=True,
                    label="Flux")
        ax1.plot(hours, (model - 1) * 1e6, color=MODEL_COLOR, linestyle=MODEL_STYLE, linewidth=3, label='Planet model')
        ax1.axvspan(-half_mask * 24, half_mask * 24, facecolor=PLANET_COLOR, alpha=0.15, edgecolor=PLANET_COLOR,
                    linewidth=1.5, label='Masked region')
        ax1.axvline(0, color=PLANET_COLOR, linestyle=ALIAS_STYLE, linewidth=2)
        ax1.set_title(wrap(f"Epoch at {centre:.3f} d, before the mask"))
        ax1.set_ylabel("Relative flux (ppm)")
        kept = np.abs(epoch_time - centre) > half_mask
        if np.any(kept):
            ax2.scatter(hours[kept], (epoch_flux[kept] - 1) * 1e6, color=BIN_COLOR, s=10, linewidths=0,
                        rasterized=True)
        ax2.set_title("After the mask")
        ax2.set_xlabel("Time from the mask centre (h)")
        ax2.set_ylabel("Relative flux (ppm)")
        for ax in (ax1, ax2):
            _style(ax)
        fig.tight_layout()
        legend_below(fig, (ax1, ax2))
        _save(fig, os.path.join(self.epoch_view_dir(), "T0_" + str(round(centre, 3)) + "_mask.png"))

    @paper_figure
    def plot_ttvs(self):
        """Plot the fitted mid-time offsets of the kept epochs (ttvs.png)."""
        if not self.object_dir:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.bary_t0s, np.asarray(self.ttvs) * 24 * 60, 'o', color=BIN_COLOR, markersize=8)
        ax.axhline(0, color=REFERENCE_COLOR, linewidth=1)
        ax.set_xlabel("Planet mid-transit time (d)")
        ax.set_ylabel("Timing offset (min)")
        ax.set_title("Fitted epoch times against the ephemeris")
        _style(ax)
        fig.tight_layout()
        _save(fig, os.path.join(self.object_dir, "ttvs.png"))

    @paper_figure
    def plot_ttv_periodogram(self, periods, power):
        """Plot the Lomb-Scargle periodogram of the TTVs (ttv_periodogram.png)."""
        if not self.object_dir:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(periods, power, color=REFERENCE_COLOR, linewidth=2)
        ax.set_xlabel("Period (d)")
        ax.set_ylabel("Lomb-Scargle power")
        ax.set_title("Periodogram of the timing offsets")
        ax.set_xscale('log')
        _style(ax)
        fig.tight_layout()
        _save(fig, os.path.join(self.object_dir, "ttv_periodogram.png"))
