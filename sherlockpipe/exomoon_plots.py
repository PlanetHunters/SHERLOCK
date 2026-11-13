"""Figures of the EMLS exomoon search (`sherlockpipe.exomoon`).

Every figure states what it shows in its title and axis labels, with units and rounded
values, so it can be read without the log. One colour code for all of them:

- data: grey points, with black binned means and their standard errors;
- EMLS hypothesis (template, selected period): orange;
- injected moon (Pandora) and its period aliases: green;
- planet transit: blue band;
- noise (nulls, detection threshold): purple.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DATA_COLOR = '0.55'
BIN_COLOR = 'black'
MODEL_COLOR = 'darkorange'
INJECTED_COLOR = 'forestgreen'
PLANET_COLOR = 'tab:blue'
NOISE_COLOR = 'tab:purple'
DPI = 130


def _style(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(direction='in', top=True, right=True)


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def binned(time, flux, bins):
    """Return bin centres, means and standard errors of the mean, for bins with two or more points."""
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    if len(time) < 2:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(np.min(time), np.max(time), bins + 1)
    means, _, _ = stats.binned_statistic(time, flux, statistic='mean', bins=edges)
    stds, _, _ = stats.binned_statistic(time, flux, statistic='std', bins=edges)
    counts, _, _ = stats.binned_statistic(time, flux, statistic='count', bins=edges)
    centres = (edges[1:] + edges[:-1]) / 2
    keep = counts >= 2
    return centres[keep], means[keep], stds[keep] / np.sqrt(counts[keep])


def _ppm(flux):
    return (np.asarray(flux, dtype=float) - 1) * 1e6


def hypothesis_figure(path, heading, it_time, it_flux, oot_time, oot_flux, model, moon_duration, epochs):
    """The light curve stacked at the moon times of one hypothesis, with its template.

    Top: the whole stacked moon window. Bottom: a zoom of +-1 moon transit duration.
    Times are relative to the moon mid-transit in each epoch.
    """
    time = np.concatenate([np.asarray(it_time, dtype=float), np.asarray(oot_time, dtype=float)])
    flux = np.concatenate([np.asarray(it_flux, dtype=float), np.asarray(oot_flux, dtype=float)])
    model_flux = np.concatenate([np.asarray(model, dtype=float), np.ones(len(oot_time))])
    order = np.argsort(time)
    time, flux, model_flux = time[order], flux[order], model_flux[order]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    zoom = np.abs(time) <= moon_duration
    for ax, mask, bins, title in ((axs[0], np.ones(len(time), dtype=bool), 80, "Whole moon window"),
                                  (axs[1], zoom, 40, "Zoom: +-1 moon transit duration")):
        if not np.any(mask):
            continue
        ax.scatter(time[mask], _ppm(flux[mask]), s=4, color=DATA_COLOR, alpha=0.4, rasterized=True,
                   label="Planet-subtracted flux")
        centres, means, errors = binned(time[mask], flux[mask], bins)
        ax.errorbar(centres, _ppm(means), yerr=errors * 1e6, fmt='o', markersize=3, color=BIN_COLOR,
                    elinewidth=0.8, label="Binned mean +- standard error")
        ax.plot(time[mask], _ppm(model_flux[mask]), color=MODEL_COLOR, linewidth=2, label="EMLS moon template")
        ax.axhline(0, color='0.3', linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"Time from the moon mid-transit (d), {epochs} epochs stacked")
        ax.set_ylabel("Relative flux (ppm)")
        _style(ax)
    axs[0].legend(fontsize=8, loc='lower left')
    fig.suptitle(heading, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def hypothesis_heading(rank_label, period, alpha, snr, radius, depth):
    return (f"{rank_label}: P = {period:.3f} d, alpha = {alpha:.2f} rad\n"
            f"SNR {snr:.2f}, radius {radius:.2f} R_earth, depth {depth * 1e6:.0f} ppm")


def alias_periods(moon_period, planet_period, period_min, period_max, max_k=3):
    """Return (k, period) of the sampling aliases 1/P' = 1/P + k/P_planet inside the grid, k != 0."""
    aliases = []
    for k in range(-max_k, max_k + 1):
        frequency = 1 / moon_period + k / planet_period
        if k != 0 and frequency > 0 and period_min <= 1 / frequency <= period_max:
            aliases.append((k, 1 / frequency))
    return aliases


def search_overview_figure(path, period_grid, snr_spectrum, residual_spectrum, sde_snr, sde_residuals,
                           best_period, best_snr, snr_threshold=None, fap_threshold=None, injected_period=None,
                           aliases=(), sde_selected=None, sde_peak=None):
    """SNR spectrum (the detection statistic) and the diagnostic spectra over the moon period grid."""
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1, 1]})
    axs[0].plot(period_grid, snr_spectrum, color='0.25', linewidth=0.6, rasterized=True,
                label="Best SNR over alpha at each period")
    axs[0].set_ylabel("SNR")
    axs[0].set_title("SNR spectrum (detection statistic)", fontsize=10)
    axs[1].plot(period_grid, residual_spectrum, color='0.4', linewidth=0.6, rasterized=True)
    axs[1].set_ylabel("Residual\n(relative to no moon)")
    axs[1].set_title("Least-squares residual, best alpha at each period (lower is better)", fontsize=10)
    axs[2].plot(period_grid, sde_snr, color='0.4', linewidth=0.6, rasterized=True, label="SDE of the SNR spectrum")
    axs[2].plot(period_grid, sde_residuals, color='tab:cyan', linewidth=0.6, alpha=0.7, rasterized=True,
                label="SDE of the residual spectrum")
    axs[2].set_ylabel("SDE")
    sde_values = []
    if sde_selected is not None:
        sde_values.append(f"SDE of the SNR spectrum at the selected period {sde_selected:.2f}")
    if sde_peak is not None:
        sde_values.append(f"peak SDE of the residual spectrum {sde_peak:.2f}")
    axs[2].set_title("Signal detection efficiency (diagnostic only)" +
                     (": " + ", ".join(sde_values) if sde_values else ""), fontsize=10)
    axs[2].set_xlabel("Moon period (d)")
    for ax in axs:
        ax.axvline(best_period, color=MODEL_COLOR, linewidth=1.5,
                   label=f"Selected P = {best_period:.3f} d (SNR {best_snr:.2f})")
        if injected_period is not None:
            ax.axvline(injected_period, color=INJECTED_COLOR, linestyle='--', linewidth=1.5,
                       label=f"Injected P = {injected_period:.3f} d")
        alias_color = INJECTED_COLOR if injected_period is not None else MODEL_COLOR
        alias_of = "injected" if injected_period is not None else "selected"
        for index, (k, alias) in enumerate(aliases):
            ax.axvline(alias, color=alias_color, linestyle=':', linewidth=1,
                       label=f"Sampling aliases of the {alias_of} P (k = +-1..3), not distinguishable from it"
                       if index == 0 else None)
        _style(ax)
    if snr_threshold is not None:
        axs[0].axhline(snr_threshold, color=NOISE_COLOR, linestyle='--', linewidth=1.2,
                       label=f"SNR at FAP = {fap_threshold:g} (Gumbel fit to the nulls)")
    axs[0].legend(fontsize=8, loc='upper right')
    axs[2].legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    _save(fig, path)


def grid_map_figure(path, period_grid, alpha_grid, matrix, colour_label, title, best=None, injected=None):
    """A statistic over the (moon period, initial phase) grid, with the selected and injected hypotheses."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mesh = ax.pcolormesh(period_grid, alpha_grid, np.ma.masked_invalid(matrix), cmap='viridis', shading='nearest',
                         rasterized=True)
    fig.colorbar(mesh, ax=ax, label=colour_label, pad=0.01)
    if best is not None:
        ax.scatter([best[0]], [best[1]], marker='o', s=90, facecolors='none', edgecolors=MODEL_COLOR, linewidths=2,
                   label=f"Selected: P = {best[0]:.3f} d, alpha = {best[1]:.2f} rad")
    if injected is not None:
        ax.scatter([injected[0]], [np.mod(injected[1], 2 * np.pi)], marker='x', s=90, color=INJECTED_COLOR,
                   linewidths=2, label=f"Injected: P = {injected[0]:.3f} d, alpha = {injected[1]:.2f} rad")
    ax.set_xlabel("Moon period (d)")
    ax.set_ylabel("Initial moon phase alpha (rad)")
    ax.set_title(title, fontsize=10)
    if best is not None or injected is not None:
        ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    _save(fig, path)


def null_distribution_figure(path, null_snrs, snr, fap, fap_upper, fap_empirical, fap_threshold, detected,
                             extrapolated):
    """The null SNR maxima, their Gumbel fit and the SNR of the selected hypothesis."""
    nulls = np.asarray([value for value in null_snrs if np.isfinite(value)], dtype=float)
    loc, scale = stats.gumbel_r.fit(nulls)
    threshold_snr = stats.gumbel_r.isf(fap_threshold, loc, scale)
    fig, ax = plt.subplots(figsize=(9, 5))
    upper = max(snr, threshold_snr, np.max(nulls)) * 1.1
    ax.hist(nulls, bins=20, density=True, color=NOISE_COLOR, alpha=0.35,
            label=f"SNR of {len(nulls)} null searches (pseudo-epochs, no moon)")
    grid = np.linspace(min(np.min(nulls), loc - 3 * scale), upper, 400)
    ax.plot(grid, stats.gumbel_r.pdf(grid, loc, scale), color=NOISE_COLOR, linewidth=2,
            label=f"Gumbel fit (loc {loc:.2f}, scale {scale:.2f})")
    ax.axvline(threshold_snr, color=NOISE_COLOR, linestyle='--', linewidth=1.2,
               label=f"SNR at FAP = {fap_threshold:g}: {threshold_snr:.2f}")
    ax.axvline(np.max(nulls), color=NOISE_COLOR, linestyle=':', linewidth=1.2,
               label=f"Largest null: {np.max(nulls):.2f} (below it the FAP is measured)")
    ax.axvline(snr, color=MODEL_COLOR, linewidth=2.5, label=f"Selected hypothesis: SNR {snr:.2f}")
    ax.set_xlim(grid[0], upper)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability density")
    verdict = "DETECTED" if detected else "NOT DETECTED"
    ax.set_title(f"False-alarm probability {fap:.2g} (95% bootstrap upper {fap_upper:.2g}, empirical "
                 f"{fap_empirical:.2g})\n{verdict}: the upper value must be below {fap_threshold:g}"
                 f"{'; extrapolated past the largest null' if extrapolated else ''}", fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    _style(ax)
    fig.tight_layout()
    _save(fig, path)


def epoch_figure(path, epoch_t0, planet_duration, time, flux, model_time, model_flux, recovered_t0, heading,
                 injected_time=None, injected_flux=None, injected_t0=None, timing_tolerance=None):
    """One planet transit after the planet subtraction, with the recovered and injected moon."""
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axvspan(-planet_duration / 2, planet_duration / 2, color=PLANET_COLOR, alpha=0.1,
               label=f"Planet transit (T14 = {planet_duration * 24:.1f} h)")
    if injected_t0 is not None and timing_tolerance is not None:
        ax.axvspan(injected_t0 - epoch_t0 - timing_tolerance, injected_t0 - epoch_t0 + timing_tolerance,
                   color=INJECTED_COLOR, alpha=0.08, label="Timing tolerance of the recovery")
    ax.scatter(time - epoch_t0, _ppm(flux), s=6, color=DATA_COLOR, alpha=0.6, label="Planet-subtracted flux")
    centres, means, errors = binned(time - epoch_t0, flux, max(10, int((np.ptp(time) if len(time) else 1) * 24 / 2)))
    ax.errorbar(centres, _ppm(means), yerr=errors * 1e6, fmt='o', markersize=3, color=BIN_COLOR, elinewidth=0.8,
                label="Binned mean (2 h) +- standard error")
    if injected_time is not None:
        order = np.argsort(injected_time)
        ax.plot(np.asarray(injected_time)[order] - epoch_t0, _ppm(np.asarray(injected_flux)[order]),
                color=INJECTED_COLOR, linewidth=1.8, label="Injected moon (Pandora)")
        if injected_t0 is not None:
            ax.axvline(injected_t0 - epoch_t0, color=INJECTED_COLOR, linestyle='--', linewidth=1,
                       label="Injected moon mid-transit")
    ax.plot(np.asarray(model_time) - epoch_t0, _ppm(model_flux), color=MODEL_COLOR, linewidth=2,
            label="EMLS template at the recovered time")
    ax.axvline(recovered_t0 - epoch_t0, color=MODEL_COLOR, linestyle=':', linewidth=1.2,
               label="Recovered moon mid-transit")
    ax.axhline(0, color='0.3', linewidth=0.6)
    ax.set_xlabel(f"Time from the planet mid-transit at {epoch_t0:.3f} d (d)")
    ax.set_ylabel("Relative flux (ppm)")
    ax.set_title(heading, fontsize=10)
    ax.legend(fontsize=7.5, loc='lower left', ncol=2)
    _style(ax)
    fig.tight_layout()
    _save(fig, path)


def spectrum_peaks(spectrum, period_grid, count, exclude=(), separation=0.02):
    """Return the indices of the highest peaks of a spectrum, at least `separation` (fractional) apart in period.

    Periods within the separation of a period in `exclude` (the selected one) are skipped too, so each
    returned index is a different peak, not a neighbour on the same one.
    """
    spectrum = np.nan_to_num(np.asarray(spectrum, dtype=float), nan=-np.inf)
    period_grid = np.asarray(period_grid, dtype=float)
    taken = [float(period) for period in exclude]
    peaks = []
    for index in np.argsort(spectrum)[::-1]:
        if len(peaks) >= count or not np.isfinite(spectrum[index]):
            break
        if all(abs(period_grid[index] - period) > separation * period for period in taken):
            peaks.append(int(index))
            taken.append(float(period_grid[index]))
    return peaks


def epoch_depth_figure(path, epoch_t0s, depths, moon_shares, errors, moon_period, moon_alpha, overlaps=None):
    """Each epoch's planet depth, measured without a moon, against the share of the selected moon.

    Both are changes of the planet depth column (positive = deeper). Their means are absorbed by
    the planet depth, so only their changes between epochs are compared: if the moon is real, the
    depth changes follow its share; spots or other stellar changes do not. The moon was selected
    on the stacked light curve, not on these depths, but the two share data, so a high
    correlation supports a moon without proving it. Epochs in `overlaps` may have the moon over the
    planet disk (for a coplanar orbit; the inclination is not known), where a real moon's dip would
    be hidden: they are drawn hollow and left out of the statistics.
    """
    epoch_t0s = np.asarray(epoch_t0s, dtype=float)
    depths = np.asarray(depths, dtype=float) * 1e6
    share = np.asarray(moon_shares, dtype=float) * 1e6
    errors = np.asarray(errors, dtype=float) * 1e6
    overlaps = np.zeros(len(depths), dtype=bool) if overlaps is None else np.asarray(overlaps, dtype=bool)
    finite = np.isfinite(depths) & np.isfinite(share) & np.isfinite(errors) & (errors > 0)
    epoch_t0s, depths, share, errors, overlaps = (epoch_t0s[finite], depths[finite], share[finite], errors[finite],
                                                  overlaps[finite])
    used = ~overlaps
    weights = 1 / errors[used] ** 2
    depth_mean = np.sum(weights * depths[used]) / np.sum(weights) if np.any(used) else 0.0
    share_mean = np.sum(weights * share[used]) / np.sum(weights) if np.any(used) else 0.0
    depth_change, share_change = depths - depth_mean, share - share_mean
    residual = depth_change - share_change
    chi2_without = float(np.sum((depth_change[used] / errors[used]) ** 2))
    chi2_with = float(np.sum((residual[used] / errors[used]) ** 2))
    correlation = float(np.corrcoef(depth_change[used], share_change[used])[0, 1]) \
        if np.sum(used) > 2 and np.std(share_change[used]) > 0 and np.std(depth_change[used]) > 0 else np.nan
    fig, axs = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    axs[0].errorbar(epoch_t0s[used], depth_change[used], yerr=errors[used], fmt='o', markersize=4, color=BIN_COLOR,
                    elinewidth=0.8, label="Planet depth of the epoch, fitted without a moon, minus the mean")
    axs[0].plot(epoch_t0s[used], share_change[used], 'D', markersize=5, color=MODEL_COLOR,
                label=f"Selected moon's share of it, minus its mean of {share_mean:.0f} ppm "
                      f"(P = {moon_period:.3f} d, alpha = {moon_alpha:.2f} rad)")
    if np.any(overlaps):
        axs[0].errorbar(epoch_t0s[overlaps], depth_change[overlaps], yerr=errors[overlaps], fmt='o', markersize=5,
                        markerfacecolor='none', color=BIN_COLOR, elinewidth=0.8,
                        label=f"Moon possibly over the planet disk (coplanar orbit; {int(np.sum(overlaps))} "
                              f"epochs, not counted)")
        axs[0].plot(epoch_t0s[overlaps], share_change[overlaps], 'D', markersize=5, markerfacecolor='none',
                    color=MODEL_COLOR)
    axs[0].axhline(0, color='0.3', linewidth=0.6)
    axs[0].set_ylabel("Planet depth change (ppm,\npositive = deeper)")
    axs[0].set_title(f"Depth changes between epochs against the moon: correlation {correlation:.2f}, chi2 "
                     f"{chi2_without:.0f} without the moon, {chi2_with:.0f} with it ({int(np.sum(used))} epochs)",
                     fontsize=10)
    axs[0].legend(fontsize=8, loc='best')
    axs[1].errorbar(epoch_t0s[used], residual[used], yerr=errors[used], fmt='o', markersize=3, color=DATA_COLOR,
                    elinewidth=0.8)
    axs[1].axhline(0, color='0.3', linewidth=0.6)
    axs[1].set_ylabel("Left after\nthe moon (ppm)")
    axs[1].set_title(f"Depth changes the moon does not explain: scatter {np.std(residual[used]):.0f} ppm, "
                     f"median error {np.median(errors[used]):.0f} ppm", fontsize=10)
    axs[1].set_xlabel("Planet mid-transit time (d)")
    for ax in axs:
        _style(ax)
    fig.tight_layout()
    _save(fig, path)
    return {"correlation": correlation, "chi2_without": chi2_without, "chi2_with": chi2_with,
            "epochs": int(np.sum(used))}
