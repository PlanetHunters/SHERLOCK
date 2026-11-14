"""Figures of the EMLS exomoon search (`sherlockpipe.exomoon`).

The figures are meant to go into a paper as they are: large text (20-22 pt at the saved size,
legible at one column width), thick lines and markers, and a colour-blind safe palette (Okabe &
Ito 2008), checked for protanopia, deuteranopia and tritanopia (OKLab distance with the Machado et
al. 2009 simulation). No meaning is carried by colour alone: every element also differs in line
style, marker or fill, and has a legend entry. One code for all figures:

- data: grey points, with black binned means and their standard errors;
- EMLS hypothesis (template, selected period): vermillion, solid;
- injected moon (Pandora) and its period aliases: blue, dashed (aliases dotted);
- planet transit: bluish-green band;
- noise (nulls, detection threshold): reddish purple, dash-dot.
"""
import functools
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Okabe & Ito (2008) colour-blind safe colours. The four colour roles pass the computable checks of
# a categorical palette on white (OKLab distance >= 15 for normal vision, >= 7.6 under simulated
# protanopia and deuteranopia; contrast >= 3:1).
DATA_COLOR = '0.6'
BIN_COLOR = 'black'
MODEL_COLOR = '#D55E00'     # vermillion
INJECTED_COLOR = '#0072B2'  # blue
PLANET_COLOR = '#009E73'    # bluish green
NOISE_COLOR = '#CC79A7'     # reddish purple
REFERENCE_COLOR = '0.25'
SECONDARY_COLOR = '#56B4E9'  # sky blue, only for a second spectrum drawn with another line style
COLORMAP = 'viridis'        # perceptually uniform and colour-blind safe

MODEL_STYLE = '-'
INJECTED_STYLE = '--'
ALIAS_STYLE = ':'
THRESHOLD_STYLE = '-.'

DPI = 200
TITLE_WIDTH = 50
LEGEND_WIDTH = 36

PAPER_RC = {
    'font.size': 20,
    'axes.titlesize': 19,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
    'axes.linewidth': 1.6,
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
    'xtick.major.width': 1.6,
    'ytick.major.width': 1.6,
    'xtick.major.size': 9,
    'ytick.major.size': 9,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.6',
    'errorbar.capsize': 0,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
}


def paper_figure(function):
    """Draw the figure of `function` with the paper style (PAPER_RC)."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with plt.rc_context(PAPER_RC):
            return function(*args, **kwargs)
    return wrapper


def wrap(text, width=TITLE_WIDTH):
    """Wrap a title or label so it fits the figure at the paper font size."""
    return "\n".join(textwrap.fill(line, width) for line in str(text).split("\n"))


def _readable_handles(legend_artist):
    """Show faint scatter points at full opacity and a visible size in the legend."""
    for handle in legend_artist.legend_handles:
        handle.set_alpha(1)
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([150])


def legend(ax, **kwargs):
    """Draw a legend inside an axis, with labels wrapped to LEGEND_WIDTH characters."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        _readable_handles(ax.legend(handles, [wrap(label, LEGEND_WIDTH) for label in labels], **kwargs))


def legend_below(fig, axes, ncol=2):
    """Draw one legend for all the axes below the figure, so it never covers the data.

    Call it after tight_layout; the saved figure grows to include it (savefig.bbox = tight).
    """
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        _readable_handles(fig.legend(handles, [wrap(label, LEGEND_WIDTH) for label in labels], loc='upper center',
                                     bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False))


def _style(ax):
    ax.grid(alpha=0.25, linewidth=0.8)


def _save(fig, path):
    fig.savefig(path)
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


def _data_points(ax, time, flux, label="Planet-subtracted flux", size=6, alpha=0.35):
    ax.scatter(time, flux, s=size, color=DATA_COLOR, alpha=alpha, rasterized=True, linewidths=0, label=label)


def _binned_points(ax, centres, means, errors, label):
    ax.errorbar(centres, means, yerr=errors, fmt='o', markersize=6, color=BIN_COLOR, elinewidth=1.5,
                markeredgecolor='white', markeredgewidth=0.8, label=label)


def _robust_limits(ax, values, model=None, margin=0.25):
    """Limit the y axis to the bulk of the points (0.5-99.5 per cent) and the model, so the signal fills the panel."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return
    low, high = np.percentile(values, [0.5, 99.5])
    if model is not None and len(model):
        low, high = min(low, np.nanmin(model)), max(high, np.nanmax(model))
    span = high - low
    if span > 0:
        ax.set_ylim(low - margin * span, high + margin * span)


@paper_figure
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
    fig, axs = plt.subplots(2, 1, figsize=(9, 9.5))
    zoom = np.abs(time) <= moon_duration
    for ax, mask, bins, title in ((axs[0], np.ones(len(time), dtype=bool), 60, "Whole moon window"),
                                  (axs[1], zoom, 30, r"Zoom: $\pm$1 moon transit duration")):
        if not np.any(mask):
            continue
        _data_points(ax, time[mask], _ppm(flux[mask]))
        centres, means, errors = binned(time[mask], flux[mask], bins)
        _binned_points(ax, centres, _ppm(means), errors * 1e6, r"Binned mean $\pm$ standard error")
        ax.plot(time[mask], _ppm(model_flux[mask]), color=MODEL_COLOR, linestyle=MODEL_STYLE, linewidth=3,
                label="EMLS moon template")
        ax.axhline(0, color=REFERENCE_COLOR, linewidth=1)
        _robust_limits(ax, _ppm(means) if len(means) else _ppm(flux[mask]), _ppm(model_flux[mask]), margin=1.0)
        ax.set_title(title)
        ax.set_xlabel("Time from the moon mid-transit (d)")
        ax.set_ylabel("Relative flux (ppm)")
        _style(ax)
    fig.suptitle(wrap(f"{heading}\n{epochs} epochs stacked"))
    fig.tight_layout()
    legend_below(fig, axs)
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


@paper_figure
def search_overview_figure(path, period_grid, snr_spectrum, residual_spectrum, sde_snr, sde_residuals,
                           best_period, best_snr, snr_threshold=None, fap_threshold=None, injected_period=None,
                           aliases=(), sde_selected=None, sde_peak=None):
    """SNR spectrum (the detection statistic) and the diagnostic spectra over the moon period grid."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    axs[0].plot(period_grid, snr_spectrum, color=REFERENCE_COLOR, linewidth=1, rasterized=True,
                label="Best SNR over alpha at each period")
    axs[0].set_ylabel("SNR")
    axs[0].set_title("SNR spectrum (detection statistic)")
    axs[1].plot(period_grid, residual_spectrum, color=REFERENCE_COLOR, linewidth=1, rasterized=True)
    axs[1].set_ylabel(wrap("Residual (rel. to no moon)", 16))
    axs[1].set_title("Least-squares residual (lower is better)")
    axs[2].plot(period_grid, sde_snr, color=REFERENCE_COLOR, linewidth=1, rasterized=True,
                label="SDE of the SNR spectrum")
    axs[2].plot(period_grid, sde_residuals, color=SECONDARY_COLOR, linewidth=1, linestyle='--', rasterized=True,
                label="SDE of the residual spectrum")
    axs[2].set_ylabel("SDE")
    sde_values = []
    if sde_selected is not None:
        sde_values.append(f"{sde_selected:.2f} at the selected period")
    if sde_peak is not None:
        sde_values.append(f"residual peak {sde_peak:.2f}")
    axs[2].set_title("SDE, diagnostic only" + (" (" + ", ".join(sde_values) + ")" if sde_values else ""))
    axs[2].set_xlabel("Moon period (d)")
    for ax in axs:
        ax.axvline(best_period, color=MODEL_COLOR, linestyle=MODEL_STYLE, linewidth=2.5,
                   label=f"Selected P = {best_period:.3f} d (SNR {best_snr:.1f})")
        if injected_period is not None:
            ax.axvline(injected_period, color=INJECTED_COLOR, linestyle=INJECTED_STYLE, linewidth=2.5,
                       label=f"Injected P = {injected_period:.3f} d")
        alias_color = INJECTED_COLOR if injected_period is not None else MODEL_COLOR
        alias_of = "injected" if injected_period is not None else "selected"
        for index, (k, alias) in enumerate(aliases):
            ax.axvline(alias, color=alias_color, linestyle=ALIAS_STYLE, linewidth=2,
                       label=f"Sampling aliases of the {alias_of} P" if index == 0 else None)
        _style(ax)
    if snr_threshold is not None:
        axs[0].axhline(snr_threshold, color=NOISE_COLOR, linestyle=THRESHOLD_STYLE, linewidth=2.5,
                       label=f"SNR at FAP = {fap_threshold:g}")
    fig.tight_layout()
    legend_below(fig, axs)
    _save(fig, path)


@paper_figure
def grid_map_figure(path, period_grid, alpha_grid, matrix, colour_label, title, best=None, injected=None):
    """A statistic over the (moon period, initial phase) grid, with the selected and injected hypotheses."""
    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(period_grid, alpha_grid, np.ma.masked_invalid(matrix), cmap=COLORMAP, shading='nearest',
                         rasterized=True)
    fig.colorbar(mesh, ax=ax, label=colour_label, pad=0.01)
    # White-edged markers stay visible on every colour of the map.
    if best is not None:
        ax.scatter([best[0]], [best[1]], marker='o', s=220, facecolors='none', edgecolors='white', linewidths=5)
        ax.scatter([best[0]], [best[1]], marker='o', s=220, facecolors='none', edgecolors=MODEL_COLOR, linewidths=2.5,
                   label=f"Selected: P = {best[0]:.3f} d, alpha = {best[1]:.2f} rad")
    if injected is not None:
        point = ([injected[0]], [np.mod(injected[1], 2 * np.pi)])
        ax.scatter(*point, marker='X', s=220, color=INJECTED_COLOR, edgecolors='white', linewidths=1.5,
                   label=f"Injected: P = {injected[0]:.3f} d, alpha = {injected[1]:.2f} rad")
    ax.set_xlabel("Moon period (d)")
    ax.set_ylabel(r"Moon phase $\alpha$ (rad)")
    ax.set_title(wrap(title))
    ax.tick_params(which='both', color='white')
    fig.tight_layout()
    legend_below(fig, ax, ncol=1)
    _save(fig, path)


@paper_figure
def null_distribution_figure(path, null_snrs, snr, fap, fap_upper, fap_empirical, fap_threshold, detected,
                             extrapolated):
    """The null SNR maxima, their Gumbel fit and the SNR of the selected hypothesis."""
    nulls = np.asarray([value for value in null_snrs if np.isfinite(value)], dtype=float)
    loc, scale = stats.gumbel_r.fit(nulls)
    threshold_snr = stats.gumbel_r.isf(fap_threshold, loc, scale)
    fig, ax = plt.subplots(figsize=(9, 6))
    upper = max(snr, threshold_snr, np.max(nulls)) * 1.1
    ax.hist(nulls, bins=20, density=True, color=NOISE_COLOR, alpha=0.35, edgecolor=NOISE_COLOR, linewidth=1,
            label=f"SNR of {len(nulls)} null searches")
    grid = np.linspace(min(np.min(nulls), loc - 3 * scale), upper, 400)
    ax.plot(grid, stats.gumbel_r.pdf(grid, loc, scale), color=NOISE_COLOR, linewidth=3, label="Gumbel fit")
    ax.axvline(threshold_snr, color=NOISE_COLOR, linestyle=THRESHOLD_STYLE, linewidth=2.5,
               label=f"SNR at FAP = {fap_threshold:g}: {threshold_snr:.1f}")
    ax.axvline(np.max(nulls), color=REFERENCE_COLOR, linestyle=ALIAS_STYLE, linewidth=2.5,
               label=f"Largest null: {np.max(nulls):.1f}")
    ax.axvline(snr, color=MODEL_COLOR, linestyle=MODEL_STYLE, linewidth=3.5,
               label=f"Selected hypothesis: SNR {snr:.1f}")
    ax.set_xlim(grid[0], upper)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Probability density")
    verdict = "detected" if detected else "not detected"
    ax.set_title(wrap(f"FAP {fap:.2g} (bootstrap upper {fap_upper:.2g}, empirical {fap_empirical:.2g}): "
                      f"{verdict} at {fap_threshold:g}{', extrapolated' if extrapolated else ''}"))
    _style(ax)
    fig.tight_layout()
    legend_below(fig, ax)
    _save(fig, path)


@paper_figure
def epoch_figure(path, epoch_t0, planet_duration, time, flux, model_time, model_flux, recovered_t0, heading,
                 injected_time=None, injected_flux=None, injected_t0=None, timing_tolerance=None):
    """One planet transit after the planet subtraction, with the recovered and injected moon."""
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axvspan(-planet_duration / 2, planet_duration / 2, facecolor=PLANET_COLOR, alpha=0.15, edgecolor=PLANET_COLOR,
               linewidth=1.5, label=f"Planet transit (T14 = {planet_duration * 24:.1f} h)")
    if injected_t0 is not None and timing_tolerance is not None:
        ax.axvspan(injected_t0 - epoch_t0 - timing_tolerance, injected_t0 - epoch_t0 + timing_tolerance,
                   facecolor='none', edgecolor=INJECTED_COLOR, hatch='//', alpha=0.5, linewidth=0,
                   label="Timing tolerance of the recovery")
    _data_points(ax, time - epoch_t0, _ppm(flux), size=10, alpha=0.5)
    centres, means, errors = binned(time - epoch_t0, flux, max(10, int((np.ptp(time) if len(time) else 1) * 24 / 2)))
    _binned_points(ax, centres, _ppm(means), errors * 1e6, r"Binned mean (2 h) $\pm$ standard error")
    if injected_time is not None:
        order = np.argsort(injected_time)
        ax.plot(np.asarray(injected_time)[order] - epoch_t0, _ppm(np.asarray(injected_flux)[order]),
                color=INJECTED_COLOR, linestyle=INJECTED_STYLE, linewidth=3, label="Injected moon (Pandora)")
        if injected_t0 is not None:
            ax.axvline(injected_t0 - epoch_t0, color=INJECTED_COLOR, linestyle=ALIAS_STYLE, linewidth=2,
                       label="Injected moon mid-transit")
    ax.plot(np.asarray(model_time) - epoch_t0, _ppm(model_flux), color=MODEL_COLOR, linestyle=MODEL_STYLE,
            linewidth=3, label="EMLS template at the recovered time")
    ax.axvline(recovered_t0 - epoch_t0, color=MODEL_COLOR, linestyle=THRESHOLD_STYLE, linewidth=2,
               label="Recovered moon mid-transit")
    ax.axhline(0, color=REFERENCE_COLOR, linewidth=1)
    ax.set_xlabel(f"Time from the planet mid-transit at {epoch_t0:.2f} d (d)")
    ax.set_ylabel("Relative flux (ppm)")
    ax.set_title(wrap(heading))
    _style(ax)
    fig.tight_layout()
    legend_below(fig, ax)
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


@paper_figure
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
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    axs[0].errorbar(epoch_t0s[used], depth_change[used], yerr=errors[used], fmt='o', markersize=7, color=BIN_COLOR,
                    elinewidth=1.5, label="Planet depth of the epoch, without a moon")
    axs[0].plot(epoch_t0s[used], share_change[used], 'D', markersize=8, color=MODEL_COLOR,
                markeredgecolor='white', markeredgewidth=0.8, label="Share of the selected moon")
    if np.any(overlaps):
        axs[0].errorbar(epoch_t0s[overlaps], depth_change[overlaps], yerr=errors[overlaps], fmt='o', markersize=8,
                        markerfacecolor='none', markeredgewidth=1.8, color=BIN_COLOR, elinewidth=1.5,
                        label=f"Moon possibly over the planet disc ({int(np.sum(overlaps))} epochs, not counted)")
        axs[0].plot(epoch_t0s[overlaps], share_change[overlaps], 'D', markersize=8, markerfacecolor='none',
                    markeredgewidth=1.8, color=MODEL_COLOR)
    axs[0].axhline(0, color=REFERENCE_COLOR, linewidth=1)
    axs[0].set_ylabel(wrap("Depth change (ppm, positive = deeper)", 20))
    axs[0].set_title(wrap(f"Correlation {correlation:.2f}; chi2 {chi2_without:.0f} without the moon, "
                          f"{chi2_with:.0f} with it ({int(np.sum(used))} epochs; moon P = {moon_period:.3f} d, "
                          f"alpha = {moon_alpha:.2f} rad, mean share {share_mean:.0f} ppm removed)", 56))
    axs[1].errorbar(epoch_t0s[used], residual[used], yerr=errors[used], fmt='o', markersize=6, color=REFERENCE_COLOR,
                    elinewidth=1.5)
    axs[1].axhline(0, color=REFERENCE_COLOR, linewidth=1)
    axs[1].set_ylabel(wrap("Left after the moon (ppm)", 14))
    axs[1].set_title(f"Scatter {np.std(residual[used]):.0f} ppm, median error {np.median(errors[used]):.0f} ppm")
    axs[1].set_xlabel("Planet mid-transit time (d)")
    for ax in axs:
        _style(ax)
    fig.tight_layout()
    legend_below(fig, axs, ncol=1)
    _save(fig, path)
    return {"correlation": correlation, "chi2_without": chi2_without, "chi2_with": chi2_with,
            "epochs": int(np.sum(used))}
