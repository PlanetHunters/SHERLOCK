"""Diagnostic figures of the planet stage: the fold fit, each subtracted or masked epoch, and the TTVs."""
import os
import matplotlib.pyplot as plt
import numpy as np


class PlanetFigures:
    """Figures written by `PlanetModel.subtract_planet_transit` into object_dir."""

    def epoch_view_dir(self):
        """Return the directory of the per-epoch planet figures, created if needed."""
        view_dir = os.path.join(self.object_dir, "t0s_view")
        os.makedirs(view_dir, exist_ok=True)
        return view_dir

    def plot_fold_fit(self, fold, start_params, params_best, guess, theta):
        """Plot the fold with the input-shape model and the fitted one (fold_fit.png)."""
        model_guess = self.transit_model(start_params, fold.time, fold.exposure).light_curve(start_params)
        model_best = self.transit_model(params_best, fold.time, fold.exposure).light_curve(params_best)
        fig_fold, ax_fold = plt.subplots(1, 1, figsize=(10, 6))
        ax_fold.scatter(fold.time, fold.flux, color='gray', alpha=0.5, s=5)
        fold_order = np.argsort(fold.time)
        ax_fold.plot(fold.time[fold_order], model_guess[fold_order], 'b--', alpha=0.7, label='Initial guess')
        ax_fold.plot(fold.time[fold_order], model_best[fold_order], 'r-', alpha=1, label='Fold fit')
        ax_fold.set_title(f"Phase-folded fit: T14={guess[1]:.4f}->{theta[1]:.4f} d, "
                          f"b={guess[2]:.2f}->{theta[2]:.2f}, "
                          f"depth={guess[3]**2*1e6:.0f}->{theta[3]**2*1e6:.0f} ppm")
        ax_fold.set_xlabel('Time from T0 (days)')
        ax_fold.set_ylabel('Flux')
        ax_fold.legend()
        plt.savefig(os.path.join(self.object_dir, "fold_fit.png"))
        plt.close(fig_fold)

    def plot_subtracted_epoch(self, epoch_time, epoch_flux, model, residual, best_t0, rejected):
        """Plot one epoch with its planet model and after the subtraction (t0s_view/T0_*_sub.png,
        or *_rejected.png for a dropped epoch).
        """
        suffix = "_rejected" if rejected else "_sub"
        fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
        axs[0].scatter(epoch_time, epoch_flux, color='gray', alpha=1, rasterized=True, label="Flux")
        axs[0].plot(epoch_time, model)
        axs[0].set_title("Subtracting transit in t0 " + str(best_t0))
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Flux')
        axs[1].scatter(epoch_time, residual, color='gray', alpha=1, rasterized=True, label="Flux")
        axs[1].set_title("Subtracted transit in t0 " + str(best_t0))
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Flux')
        plt.savefig(os.path.join(self.epoch_view_dir(), "T0_" + str(round(best_t0, 3)) + suffix + ".png"))
        plt.close()

    def plot_masked_epoch(self, epoch_time, epoch_flux, model, centre, half_mask):
        """Plot one epoch before and after its transit is masked (t0s_view/T0_*_mask.png)."""
        fig_t, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.scatter(epoch_time, epoch_flux, color='gray', alpha=0.7, s=10)
        ax1.plot(epoch_time, model, color='orange', alpha=1, label='Planet model')
        ax1.axvspan(centre - half_mask, centre + half_mask, color='red', alpha=0.15, label='Masked region')
        ax1.axvline(centre, color='red', linestyle='--')
        ax1.set_title(f"T0={centre:.4f} (before mask)")
        ax1.set_ylabel('Flux')
        ax1.legend()
        kept = np.abs(epoch_time - centre) > half_mask
        if np.any(kept):
            ax2.scatter(epoch_time[kept], epoch_flux[kept], color='gray', alpha=0.7, s=10, label='After mask')
        ax2.set_title(f"T0={centre:.4f} (after mask)")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Flux')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.epoch_view_dir(), "T0_" + str(round(centre, 3)) + "_mask.png"))
        plt.close()

    def plot_ttvs(self):
        """Plot the fitted mid-time offsets of the kept epochs (ttvs.png)."""
        if not self.object_dir:
            return
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.ttvs)), self.ttvs, marker='o')
        plt.xticks(range(len(self.ttvs)), [f"{t0:.4f}" for t0 in self.bary_t0s], rotation=45)
        plt.xlabel("Barycentric T0 Values")
        plt.ylabel("TTV (days)")
        plt.title("Transit Timing Variations")
        plt.tight_layout()
        plt.savefig(os.path.join(self.object_dir, "ttvs.png"))
        plt.close()

    def plot_ttv_periodogram(self, periods, power):
        """Plot the Lomb-Scargle periodogram of the TTVs (ttv_periodogram.png)."""
        if not self.object_dir:
            return
        fig_ttv_per, ax_ttv = plt.subplots(1, 1, figsize=(10, 6))
        ax_ttv.plot(periods, power, color='gray')
        ax_ttv.set_xlabel("Period (days)")
        ax_ttv.set_ylabel("Lomb-Scargle Power")
        ax_ttv.set_title("TTV Periodogram")
        ax_ttv.set_xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.object_dir, "ttv_periodogram.png"))
        plt.close()
