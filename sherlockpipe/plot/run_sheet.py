"""
Run sheet: one landscape page that summarises a signal found by a search Run.

It holds the folded transit with its residuals, the TLS SDE spectrum, the transit parameters with checks that flag
what looks problematic, and the epochs: what is shown for them depends on how many have data, because one panel per
epoch does not scale (there can be hundreds)::

    up to 24 epochs     one panel per epoch
    up to 150 epochs    compact thumbnails, framed when the epoch disagrees with the mean depth
    more than 150       stacked epoch map: consecutive epochs are stacked until each row has a minimum S/N

All figures are built with matplotlib ``Figure`` objects (no pyplot state) so this can run next to anything else.
"""
import logging
import textwrap

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#e3e7ec"
MODEL = "#2a6f97"
MODEL_LINE = "#c0392b"
ODD = "#d1495b"
OK = "#3a8f5c"
WARN = "#c98a1b"
BAD = "#c0392b"

#: Number of epochs with data up to which each layout is used.
EPOCH_PANELS_MAX = 24
THUMBNAILS_MAX = 150
#: Half width of the window around each transit, in transit durations.
WINDOW_DURATIONS = 3.5
#: Same for the folded transit, as wide as the existing Run plots (10 half durations).
FOLD_WINDOW_DURATIONS = 5.0
#: Minimum points inside and outside the transit for an epoch to count as "with data".
MIN_POINTS_IN_TRANSIT = 4
MIN_POINTS_OUT_OF_TRANSIT = 12
#: Stacked epoch map: S/N every row must reach, and the most rows it may have.
STACK_SNR_TARGET = 4.0
STACK_MAX_ROWS = 48
STACK_PHASE_BINS = 28
#: |z| of an epoch depth against the mean depth above which it is flagged.
Z_WARN = 2.0
Z_BAD = 3.0
#: (ok below, warn below) for quantities where bigger is worse; (ok from, warn from) for border score.
FLAG_THRESHOLDS = {"odd_even_sigma": (2.0, 3.0), "chi2": (1.5, 2.5), "fap": (0.01, 0.05),
                   "border_score": (0.8, 0.5)}
#: Gap in days between epochs above which a new observing segment starts (also at least 3 periods).
SEGMENT_GAP_DAYS = 5.0
MAX_SCATTER_POINTS = 60000

CAPTIONS = {
    1: "Each panel is one epoch: its points (grey), 15-min style bins with error bars (black) and the transit model (blue). "
       "The title gives the epoch number and its depth ± error in ppt; S/N is that depth over its error. The frame is "
       "orange when the epoch depth differs from the mean depth by more than 2σ and red by more than 3σ.",
    2: "Each thumbnail is one epoch, numbered in its corner: binned flux (black) and the transit model (blue). The frame is "
       "orange when the epoch depth differs from the mean depth by more than 2σ and red by more than 3σ; "
       "an unframed thumbnail is consistent with the signal even if it is noisy.",
    3: "How to read the stacked epoch map. Single epochs are mostly noise at this S/N, so each row stacks consecutive "
       "epochs of the same observing segment until the stack reaches S/N ≥ {target:g}. Columns are time from mid-transit "
       "(dotted lines: transit start and end) and the colour is the mean flux relative to the local baseline: red below "
       "it, blue above. A real transit appears as a red vertical band between the dotted lines in every row; a band that "
       "fades means the depth changes with time, a band shifted sideways means timing offsets, and a row without band is "
       "a stack that disagrees with the signal. Middle: the depth of each row ± error (dashed: mean depth; orange for "
       "|z| > 2 and red for |z| > 3). Right: the three most deviating stacks and the three with the highest S/N, folded.",
}


# --------------------------------------------------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------------------------------------------------

def clean_curve(time, flux, flux_err):
    """Drops the points without a valid time or flux and sorts by time. Missing errors are replaced by the scatter."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.full(len(time), np.nan) if flux_err is None else np.asarray(flux_err, dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]
    order = np.argsort(time, kind="stable")
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    if len(flux_err) and not np.isfinite(flux_err).any():
        flux_err = np.full(len(flux), np.nanstd(flux))
    return time, flux, flux_err


def transit_model(transit_result, period, duration, depth):
    """
    The transit shape as a function of the hours from mid-transit, relative to a baseline of 1. It is taken from the
    folded model of the search when it has one, and is a box of the given duration and depth otherwise.

    :param float depth: relative depth (not ppt)
    """
    results = getattr(transit_result, "results", None)
    phase = getattr(results, "model_folded_phase", None) if results is not None else None
    model = getattr(results, "model_folded_model", None) if results is not None else None
    if phase is not None and model is not None and len(phase) > 3 and len(phase) == len(model):
        hours = (np.asarray(phase, dtype=float) - 0.5) * period * 24
        values = np.asarray(model, dtype=float)
        valid = np.isfinite(hours) & np.isfinite(values)
        if valid.sum() > 3 and np.min(values[valid]) < 1:
            return lambda x: np.interp(x, hours[valid], values[valid], left=1.0, right=1.0)
    half = duration * 12
    return lambda x: np.where(np.abs(np.asarray(x, dtype=float)) < half, 1.0 - depth, 1.0)


def compute_epochs(time, flux, flux_err, period, t0, duration, depth_ppt):
    """
    Measures every epoch that has enough data inside and around the transit.

    Epochs are found with a binary search on the sorted time array, so the cost does not grow with the number of
    points times the number of epochs.

    :return: DataFrame with ``k, tc, depth, err, snr, z, n_in, base, lo, hi, segment`` (depths in ppt; ``lo``/``hi``
        are the slice of the arrays that belong to the epoch window)
    """
    columns = ["k", "tc", "depth", "err", "snr", "z", "n_in", "base", "lo", "hi", "segment"]
    if len(time) == 0 or not np.isfinite([period, t0, duration]).all() or period <= 0 or duration <= 0:
        return pd.DataFrame(columns=columns)
    half = WINDOW_DURATIONS * duration
    ks = np.arange(int(np.ceil((time[0] - t0 - half) / period)), int(np.floor((time[-1] - t0 + half) / period)) + 1)
    centres = t0 + ks * period
    lows = np.searchsorted(time, centres - half, side="left")
    highs = np.searchsorted(time, centres + half, side="right")
    rows = []
    for k, centre, lo, hi in zip(ks, centres, lows, highs):
        if hi - lo < MIN_POINTS_IN_TRANSIT + MIN_POINTS_OUT_OF_TRANSIT:
            continue
        offset = time[lo:hi] - centre
        inside = np.abs(offset) < duration / 2
        n_in = int(inside.sum())
        n_out = len(offset) - n_in
        if n_in < MIN_POINTS_IN_TRANSIT or n_out < MIN_POINTS_OUT_OF_TRANSIT:
            continue
        values = flux[lo:hi]
        base = float(np.median(values[~inside]))
        depth = base - values[inside].mean()
        err = np.sqrt(values[inside].std() ** 2 / n_in + values[~inside].std() ** 2 / n_out)
        if not np.isfinite(depth) or not np.isfinite(err) or err <= 0:
            continue
        rows.append((int(k), centre, depth * 1e3, err * 1e3, n_in, base, int(lo), int(hi)))
    epochs = pd.DataFrame(rows, columns=["k", "tc", "depth", "err", "n_in", "base", "lo", "hi"])
    if epochs.empty:
        return pd.DataFrame(columns=columns)
    epochs["snr"] = epochs["depth"] / epochs["err"]
    epochs["z"] = (epochs["depth"] - depth_ppt) / epochs["err"]
    gap = max(SEGMENT_GAP_DAYS, 3 * period)
    epochs["segment"] = (epochs["tc"].diff().fillna(0) > gap).cumsum().astype(int)
    return epochs[columns]


def mean_depth(epochs):
    """Inverse-variance mean depth of the epochs and its error, in ppt."""
    weights = 1 / epochs["err"] ** 2
    return float(np.sum(weights * epochs["depth"]) / weights.sum()), float(1 / np.sqrt(weights.sum()))


def stack_epochs(epochs, snr_target=STACK_SNR_TARGET, max_rows=STACK_MAX_ROWS):
    """
    Groups consecutive epochs of the same segment so that every group reaches ``snr_target`` (quadrature sum of the
    epoch S/N). A group that cannot reach it at the end of a segment joins the previous one of that segment. The target
    grows until there are at most ``max_rows`` groups.

    :return: ``(groups, snr_target)`` where groups is a list of lists of row positions in ``epochs``
    """
    snr = np.clip(epochs["snr"].to_numpy(), 0, None)
    segment = epochs["segment"].to_numpy()
    while True:
        groups, current, accumulated = [], [], 0.0
        for i in range(len(epochs)):
            if current and segment[i] != segment[current[-1]]:
                _close_group(groups, current, segment)
                current, accumulated = [], 0.0
            current.append(i)
            accumulated += snr[i] ** 2
            if np.sqrt(accumulated) >= snr_target:
                groups.append(current)
                current, accumulated = [], 0.0
        if current:
            _close_group(groups, current, segment)
        if len(groups) <= max_rows:
            return groups, snr_target
        snr_target *= 1.25


def _close_group(groups, current, segment):
    if groups and segment[groups[-1][0]] == segment[current[0]]:
        groups[-1] = groups[-1] + current
    else:
        groups.append(current)


def layout_tier(n_epochs):
    """1: one panel per epoch, 2: thumbnails, 3: stacked epoch map."""
    return 1 if n_epochs <= EPOCH_PANELS_MAX else 2 if n_epochs <= THUMBNAILS_MAX else 3


def flag(value, kind, sde_min=None):
    """'ok' / 'warn' / 'bad' for a value, or None when there is nothing to compare with."""
    if value is None or not np.isfinite(value):
        return None
    if kind == "border_score":
        good, poor = FLAG_THRESHOLDS[kind]
        return "ok" if value >= good else "warn" if value >= poor else "bad"
    good, poor = FLAG_THRESHOLDS[kind]
    return "ok" if value < good else "warn" if value < poor else "bad"


def parameter_groups(transit_result, epochs, n_listed, snr_min=None, sde_min=None):
    """
    The parameters table as ``[(group title, [(name, value text, flag or None), ...]), ...]``.
    Flags only appear where there is a threshold to compare with.
    """
    r = transit_result
    results = getattr(r, "results", None)
    depth_err = r.depth_err if np.isfinite(getattr(r, "depth_err", np.nan)) else None
    rp_rs = getattr(results, "rp_rs", None) if results is not None else None
    rp_rs = rp_rs if rp_rs is not None and np.isfinite(rp_rs) else np.sqrt(max(r.depth, 0) / 1e3)
    transit = [("Period", f"{r.period:.5f} ± {r.per_err:.0e} d" if np.isfinite(r.per_err) and r.per_err < 1e-4 else (f"{r.period:.5f} ± {r.per_err:.5f} d" if np.isfinite(r.per_err) else f"{r.period:.5f} d"), None),
               ("T0", f"{r.t0:.3f}", None),
               ("Depth", f"{r.depth:.3f}" + (f" ± {depth_err:.3f}" if depth_err else "") + " ppt", None),
               ("Duration", f"{r.duration * 1440:.0f} min", None), ("Rp/Rs", f"{rp_rs:.4f}", None)]
    sde_flag = None if sde_min is None else ("ok" if r.sde >= sde_min else "bad")
    snr_flag = None if snr_min is None else ("ok" if r.snr >= snr_min else "bad")
    detection = [("SDE", f"{r.sde:.1f}", sde_flag), ("SNR", f"{r.snr:.1f}", snr_flag),
                 ("FAP", f"{r.fap:.2g}" if np.isfinite(r.fap) else "–", flag(r.fap, "fap")), ("Border score", f"{r.border_score:.2f}" if np.isfinite(r.border_score) else "–",
                                                                flag(r.border_score, "border_score")),
                 ("Harmonic of another signal", "yes" if r.harmonic else "no", "warn" if r.harmonic else "ok")]
    consistency = [("Odd / even mismatch", f"{r.odd_even_mismatch:.1f} σ", flag(r.odd_even_mismatch, "odd_even_sigma"))]
    if len(epochs) > 1:
        chi2 = float((epochs["z"] ** 2).sum() / (len(epochs) - 1))
        outliers = int((epochs["z"].abs() > Z_BAD).sum())
        consistency += [("χ² between epochs", f"{chi2:.2f}", flag(chi2, "chi2")),
                        (f"Epochs with |z| > {Z_BAD:g}", f"{outliers} of {len(epochs)}",
                         "ok" if outliers == 0 else "warn" if outliers <= max(1, len(epochs) // 50) else "bad")]
    consistency.append(("Epochs with data", f"{len(epochs)} of {n_listed} listed", None))
    return [("Transit", transit), ("Detection", detection), ("Consistency", consistency)]


# --------------------------------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------------------------------

def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.ticklabel_format(axis="both", useOffset=False, style="plain") if ax.get_xscale() == "linear" else None


def binned(x, y, width, lo, hi, minimum=3):
    """Mean of y in bins of x, its standard error, and the bin centres (bins with fewer than ``minimum`` are dropped)."""
    edges = np.arange(lo, hi + width, width)
    index = np.digitize(x, edges) - 1
    out = []
    for i in range(len(edges) - 1):
        m = index == i
        if m.sum() >= minimum:
            out.append((x[m].mean(), y[m].mean(), y[m].std() / np.sqrt(m.sum())))
    return tuple(np.array(c) for c in zip(*out)) if out else (np.array([]),) * 3


def draw_fold(ax, ax_res, time, flux, epochs, period, t0, duration, depth, model, bin_hours):
    """Folded transit: points, odd and even binned curves when there are enough epochs, binned mean and model."""
    half = FOLD_WINDOW_DURATIONS * duration * 24
    phase = (((time - t0 + period / 2) % period) - period / 2) * 24
    inside = np.abs(phase) < half
    out_of_transit = inside & (np.abs(phase) > duration * 12)
    base = np.median(flux[out_of_transit]) if out_of_transit.any() else np.median(flux[inside])
    x, y = phase[inside], flux[inside] / base
    step = max(1, len(x) // MAX_SCATTER_POINTS)
    ax.scatter(x[::step], y[::step], s=1.2, color=MUTED, alpha=0.08, lw=0, rasterized=True)
    epoch_k = np.round((time[inside] - t0) / period).astype(int)
    if len(epochs) >= 20:
        for parity, colour, label in ((1, ODD, "odd"), (0, MODEL, "even")):
            m = (epoch_k % 2) == parity
            if m.sum() > 20:
                bx, by, _ = binned(x[m], y[m], bin_hours, -half, half)
                if len(bx):
                    ax.plot(bx, by, "-", lw=0.8, color=colour, alpha=0.9, label=label, zorder=3)
    bx, by, be = binned(x, y, bin_hours, -half, half)
    if len(bx):
        ax.errorbar(bx, by, be, fmt="o", ms=2.6, color=INK, lw=0.8, zorder=4, label="all")
    grid = np.linspace(-half, half, 500)
    ax.plot(grid, model(grid), color=MODEL_LINE, lw=1.4, zorder=5, label="model")
    low = min(1 - depth * 1.4, (by - be).min() - depth * 0.1) if len(bx) else 1 - depth * 1.4
    high = max(1 + depth * 0.6, (by + be).max() + depth * 0.1) if len(bx) else 1 + depth * 0.6
    ax.set_ylim(low, high)
    ax.set_xlim(-half, half)
    ax.set_ylabel("relative flux")
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper center", handlelength=1.2, columnspacing=0.8)
    ax.tick_params(labelbottom=False)
    residual = y - model(x)
    rx, ry, re = binned(x, residual, bin_hours, -half, half)
    if len(rx):
        ax_res.errorbar(rx, ry * 1e3, re * 1e3, fmt="o", ms=2, color=INK, lw=0.7)
    ax_res.axhline(0, color=MODEL_LINE, lw=0.8)
    ax_res.set_ylabel("resid. (ppt)", fontsize=6)
    ax_res.set_xlabel("hours from mid-transit")
    ax_res.set_xlim(-half, half)
    _style(ax)
    _style(ax_res)


def draw_spectrum(ax, periods, power, period, title, ylabel, harmonics_line=False):
    """The TLS SDE spectrum as the Run plots have always drawn it, larger: selected period and its harmonics."""
    ax.axvline(period, alpha=0.4, lw=3)
    for n in range(2, 10):
        ax.axvline(n * period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")
    ax.plot(periods, power, color="black", lw=0.5)
    ax.set_xlim(np.min(periods), np.max(periods))
    ax.set(xlabel="Period (days)", ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    _style(ax)


def draw_parameters(figure, spec, groups):
    sub = GridSpecFromSubplotSpec(1, len(groups), subplot_spec=spec, wspace=0.12)
    colours = {"ok": OK, "warn": WARN, "bad": BAD}
    symbols = {"ok": "●", "warn": "▲", "bad": "■"}
    for i, (title, rows) in enumerate(groups):
        ax = figure.add_subplot(sub[i])
        ax.axis("off")
        step = 1.0 / (len(rows) + 1.4)
        ax.text(0, 1.0, title.upper(), fontsize=6.5, color=MUTED, fontweight="bold", va="top", transform=ax.transAxes)
        ax.plot([0, 1], [1.0 - step * 0.75] * 2, color=GRID, lw=0.8, transform=ax.transAxes)
        y = 1.0 - step
        for name, value, state in rows:
            ax.text(0, y, name, fontsize=7, va="top", transform=ax.transAxes)
            ax.text(0.52, y, value, fontsize=7, fontweight="bold", va="top", transform=ax.transAxes)
            if state:
                ax.text(0.99, y, symbols[state], fontsize=8, color=colours[state], va="top", ha="right",
                        transform=ax.transAxes)
            y -= step


def _epoch_axes(ax, window_hours):
    ax.set_xlim(-window_hours, window_hours)
    ax.set_xticks([])
    ax.set_yticks([])


def _frame(ax, z, thick=1.4):
    colour = BAD if abs(z) > Z_BAD else WARN if abs(z) > Z_WARN else None
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(colour or GRID)
        spine.set_linewidth(thick if colour else 0.5)


def _epoch_curve(time, flux, epoch, duration):
    lo, hi = int(epoch.lo), int(epoch.hi)
    return (time[lo:hi] - epoch.tc) * 24, flux[lo:hi] / epoch.base


def draw_epoch_panels(figure, spec, time, flux, epochs, duration, depth, model, bin_hours):
    """Tier 1: one panel per epoch."""
    cols = 8
    rows = int(np.ceil(len(epochs) / cols))
    sub = GridSpecFromSubplotSpec(rows, cols, subplot_spec=spec, hspace=0.55, wspace=0.12)
    half = WINDOW_DURATIONS * duration * 24
    grid = np.linspace(-half, half, 200)
    for i, epoch in enumerate(epochs.itertuples()):
        ax = figure.add_subplot(sub[i // cols, i % cols])
        x, y = _epoch_curve(time, flux, epoch, duration)
        ax.scatter(x, y, s=1.6, color=MUTED, alpha=0.3, lw=0)
        bx, by, be = binned(x, y, bin_hours * 2, -half, half, minimum=2)
        if len(bx):
            ax.errorbar(bx, by, be, fmt="o", ms=1.8, color=INK, lw=0.5)
        ax.plot(grid, model(grid), color=MODEL, lw=1)
        ax.set_ylim(1 - depth * 4.5, 1 + depth * 3)
        _epoch_axes(ax, half)
        _frame(ax, epoch.z)
        ax.set_title(f"E{epoch.k}  {epoch.depth:.1f}±{epoch.err:.1f}", fontsize=5.5, loc="left", pad=1.5, color=INK)
        ax.text(0.97, 0.06, f"S/N {epoch.snr:.1f}", transform=ax.transAxes, ha="right", fontsize=5.2, color=MUTED)


def draw_thumbnails(figure, spec, time, flux, epochs, duration, depth, model, bin_hours):
    """Tier 2: compact thumbnails, framed only when the epoch disagrees with the mean depth."""
    cols = 16 if len(epochs) > 96 else 12
    rows = int(np.ceil(len(epochs) / cols))
    sub = GridSpecFromSubplotSpec(rows, cols, subplot_spec=spec, hspace=0.35, wspace=0.1)
    half = WINDOW_DURATIONS * duration * 24
    grid = np.linspace(-half, half, 100)
    for i, epoch in enumerate(epochs.itertuples()):
        ax = figure.add_subplot(sub[i // cols, i % cols])
        x, y = _epoch_curve(time, flux, epoch, duration)
        bx, by, _ = binned(x, y, bin_hours * 2, -half, half, minimum=2)
        if len(bx):
            ax.plot(bx, by, "-", lw=0.7, color=INK)
        ax.plot(grid, model(grid), color=MODEL, lw=0.7, alpha=0.8)
        ax.set_ylim(1 - depth * 4.5, 1 + depth * 3)
        _epoch_axes(ax, half)
        _frame(ax, epoch.z, thick=1.5)
        ax.text(0.04, 0.9, f"{epoch.k}", transform=ax.transAxes, fontsize=4.5, color=MUTED, va="top")


def draw_epoch_map(figure, spec, time, flux, epochs, duration, depth, depth_ppt, model, bin_hours):
    """Tier 3: stacked epoch map, depth of each stack and its most deviating / highest S/N stacks."""
    groups, target = stack_epochs(epochs)
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[2.2, 1.5, 2.6], wspace=0.16)
    ax_map, ax_depth = figure.add_subplot(sub[0]), figure.add_subplot(sub[1])
    half = WINDOW_DURATIONS * duration
    edges = np.linspace(-half, half, STACK_PHASE_BINS + 1)
    sums = np.zeros((len(epochs), STACK_PHASE_BINS))
    counts = np.zeros((len(epochs), STACK_PHASE_BINS))
    for i, epoch in enumerate(epochs.itertuples()):
        lo, hi = int(epoch.lo), int(epoch.hi)
        index = np.digitize(time[lo:hi] - epoch.tc, edges) - 1
        valid = (index >= 0) & (index < STACK_PHASE_BINS)
        values = flux[lo:hi] / epoch.base
        sums[i] = np.bincount(index[valid], weights=values[valid], minlength=STACK_PHASE_BINS)
        counts[i] = np.bincount(index[valid], minlength=STACK_PHASE_BINS)
    rows = []
    for g in groups:
        total = counts[g].sum(axis=0)
        row = np.where(total >= 3, sums[g].sum(axis=0) / np.maximum(total, 1), np.nan)
        members = epochs.iloc[g]
        value, error = mean_depth(members)
        rows.append({"row": row, "depth": value, "err": error, "z": (value - depth_ppt) / error, "snr": value / error,
                     "k0": int(members["k"].iloc[0]), "k1": int(members["k"].iloc[-1]), "n": len(g),
                     "segment": int(members["segment"].iloc[0]), "members": members})
    image = np.array([r["row"] for r in rows])
    norm = TwoSlopeNorm(vcenter=1.0, vmin=1 - depth * 3, vmax=1 + depth * 2)
    shown = ax_map.imshow(image, aspect="auto", cmap="RdBu", norm=norm, interpolation="nearest",
                          extent=[-half * 24, half * 24, len(rows), 0])
    ax_map.axvline(-duration * 12, color="k", lw=0.5, ls=":")
    ax_map.axvline(duration * 12, color="k", lw=0.5, ls=":")
    ax_map.set_xlabel("hours from mid-transit")
    ax_map.set_ylabel("stack of consecutive epochs")
    _style(ax_map)
    ax_map.grid(False)
    ax_map.set_title("Stacked epoch map", loc="left", fontsize=8, fontweight="bold", pad=11)
    ax_map.text(0, 1.005, f"{len(epochs)} epochs in {len(rows)} rows · S/N ≥ {target:.0f} per row",
                transform=ax_map.transAxes, fontsize=5.8, color=MUTED, va="bottom")
    colorbar = figure.colorbar(shown, ax=ax_map, pad=0.01, fraction=0.04)
    colorbar.ax.set_title("rel. flux", fontsize=5.5, pad=3)
    colorbar.ax.tick_params(labelsize=5.5)
    segments = np.array([r["segment"] for r in rows])
    starts = [0] + list(np.where(np.diff(segments) != 0)[0] + 1)
    for s in starts[1:]:
        ax_map.axhline(s, color="k", lw=0.6)
    ax_map.set_yticks([s + 0.5 for s in starts])
    ax_map.set_yticklabels([f"t={rows[s]['members']['tc'].iloc[0]:.0f}" for s in starts], fontsize=5.5)
    ax_map.tick_params(axis="x", labelsize=6.5)
    colours = [BAD if abs(r["z"]) > Z_BAD else WARN if abs(r["z"]) > Z_WARN else MODEL for r in rows]
    ax_depth.barh(np.arange(len(rows)) + 0.5, [r["depth"] for r in rows], xerr=[r["err"] for r in rows], height=0.8,
                  color=colours, ecolor=MUTED, error_kw={"lw": 0.5}, alpha=0.9)
    ax_depth.axvline(depth_ppt, color=INK, lw=0.8, ls="--")
    ax_depth.set_ylim(len(rows), 0)
    ax_depth.set_yticks([])
    ax_depth.spines["left"].set_visible(False)
    ax_depth.set_xlabel("depth per row (ppt)")
    ax_depth.set_title("Depth per row", loc="left", fontsize=8, fontweight="bold")
    _style(ax_depth)
    ax_depth.spines["left"].set_visible(False)
    deviating = sorted(range(len(rows)), key=lambda i: -abs(rows[i]["z"]))[:3]
    best = [i for i in sorted(range(len(rows)), key=lambda i: -rows[i]["snr"]) if i not in deviating][:3]
    extremes = [(i, "most deviating") for i in deviating] + [(i, "highest S/N") for i in best]
    sub2 = GridSpecFromSubplotSpec(2, 3, subplot_spec=sub[2], hspace=0.5, wspace=0.15)
    grid = np.linspace(-half * 24, half * 24, 200)
    for j, (i, label) in enumerate(extremes):
        r = rows[i]
        ax = figure.add_subplot(sub2[j // 3, j % 3])
        x = np.concatenate([(time[int(e.lo):int(e.hi)] - e.tc) * 24 for e in r["members"].itertuples()])
        y = np.concatenate([flux[int(e.lo):int(e.hi)] / e.base for e in r["members"].itertuples()])
        step = max(1, len(x) // 8000)
        ax.scatter(x[::step], y[::step], s=1.0, color=MUTED, alpha=0.12, lw=0, rasterized=True)
        bx, by, be = binned(x, y, bin_hours * 2, -half * 24, half * 24)
        if len(bx):
            ax.errorbar(bx, by, be, fmt="o", ms=1.8, color=INK, lw=0.5)
        ax.plot(grid, model(grid), color=MODEL, lw=1)
        ax.set_ylim(1 - depth * 3, 1 + depth * 2)
        _epoch_axes(ax, half * 24)
        _frame(ax, r["z"])
        title = f"E{r['k0']}–{r['k1']} ({r['n']}) · " + (f"z={r['z']:+.1f}" if label == "most deviating"
                                                         else f"S/N {r['snr']:.1f}")
        ax.set_title(title, fontsize=5.6, loc="left", pad=1.5, color=INK)
        ax.text(0.97, 0.06, label, transform=ax.transAxes, ha="right", fontsize=4.8, color=MUTED)
    return target


def draw_caption(figure, text):
    figure.text(0.06, 0.01, textwrap.fill(text, width=210), fontsize=6, color=MUTED, va="bottom", ha="left")


# --------------------------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------------------------

def save_run_sheet(object_id, plot_dir, file, time, flux, flux_err, transit_result, cadence, run_no, curve_label,
                   plot_harmonics=False, snr_min=None, sde_min=None, dpi=200):
    """
    Writes the run sheet of the signal of a Run.

    :param str object_id: the target id
    :param str plot_dir: directory where the sheet is stored
    :param str file: file name of the sheet
    :param time: time array of the light curve (days)
    :param flux: flux of the curve the signal was found in
    :param flux_err: flux errors of that curve (may be None)
    :param TransitResult transit_result: the search result of the signal
    :param cadence: cadence of the curve in seconds
    :param int run_no: the SHERLOCK run number
    :param str curve_label: name of the curve (e.g. ``SELECTED`` or a detrend)
    :param bool plot_harmonics: whether to add the harmonics spectrum below the SDE spectrum
    :param float snr_min: SNR threshold of the search, flags the SNR when it is not reached
    :param float sde_min: SDE threshold of the search, flags the SDE when it is not reached
    :return: the path of the written file
    """
    time, flux, flux_err = clean_curve(time, flux, flux_err)
    r = transit_result
    valid = np.isfinite([r.period, r.t0, r.duration, r.depth]).all() and r.period > 0 and r.duration > 0
    depth_rel = r.depth / 1e3 if valid else np.nan
    model = transit_model(r, r.period, r.duration, depth_rel) if valid else None
    epochs = compute_epochs(time, flux, flux_err, r.period, r.t0, r.duration, r.depth) if valid else \
        compute_epochs([], [], [], np.nan, np.nan, np.nan, np.nan)
    n_listed = len(r.t0s) if getattr(r, "t0s", None) is not None else len(epochs)
    tier = layout_tier(len(epochs))
    bin_hours = max(r.duration * 24 / 5, 3 * cadence / 3600) if valid else 0.1
    figure = Figure(figsize=(11.69, 8.27))
    outer = figure.add_gridspec(3, 1, height_ratios=[1.2, 0.42, 1.45], hspace=0.3, left=0.06, right=0.985, top=0.915,
                                bottom=0.12)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[2.9, 5.2], wspace=0.1)
    left = GridSpecFromSubplotSpec(2, 1, subplot_spec=top[0], height_ratios=[4, 1], hspace=0.05)
    ax_fold, ax_res = figure.add_subplot(left[0]), figure.add_subplot(left[1])
    if valid and len(time):
        draw_fold(ax_fold, ax_res, time, flux, epochs, r.period, r.t0, r.duration, depth_rel, model, bin_hours)
    else:
        for ax in (ax_fold, ax_res):
            ax.axis("off")
        ax_fold.text(0.5, 0.5, "No valid transit parameters", ha="center", va="center", transform=ax_fold.transAxes,
                     color=MUTED)
    ax_fold.set_title("Folded transit", loc="left", fontsize=8, fontweight="bold")
    results = getattr(r, "results", None)
    periods = getattr(results, "periods", None)
    power = getattr(results, "power", None)
    two_rows = bool(plot_harmonics and r.harmonic_spectrum is not None and periods is not None)
    if periods is not None and power is not None and len(periods):
        if two_rows:
            rows = GridSpecFromSubplotSpec(2, 1, subplot_spec=top[1], height_ratios=[3, 1.6], hspace=0.45)
            ax_sde = figure.add_subplot(rows[0])
            ax_h = figure.add_subplot(rows[1])
            draw_spectrum(ax_h, periods, r.harmonic_spectrum, r.period, "Harmonics power", "Harmonics power")
        else:
            ax_sde = figure.add_subplot(top[1])
        draw_spectrum(ax_sde, periods, power, r.period, "TLS SDE spectrum", "SDE")
    draw_parameters(figure, outer[1], parameter_groups(r, epochs, n_listed, snr_min, sde_min))
    figure.text(0.06, 0.962, f"{object_id} · Run {run_no} · curve {curve_label} · P = {r.period:.5f} d",
                fontsize=11, fontweight="bold", color=INK)
    figure.text(0.985, 0.965, f"{len(epochs)} epochs with data", fontsize=7.5, color=MUTED, ha="right")
    if len(epochs):
        if tier == 1:
            draw_epoch_panels(figure, outer[2], time, flux, epochs, r.duration, depth_rel, model, bin_hours)
            caption = CAPTIONS[1]
        elif tier == 2:
            draw_thumbnails(figure, outer[2], time, flux, epochs, r.duration, depth_rel, model, bin_hours)
            caption = CAPTIONS[2]
        else:
            target = draw_epoch_map(figure, outer[2], time, flux, epochs, r.duration, depth_rel, r.depth, model, bin_hours)
            caption = CAPTIONS[3].format(target=round(target, 1))
        draw_caption(figure, caption)
    else:
        ax = figure.add_subplot(outer[2])
        ax.axis("off")
        ax.text(0.5, 0.5, "No epoch has enough data inside and around the transit", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
    path = plot_dir + file
    figure.savefig(path, dpi=dpi)
    logging.info("Run sheet written to %s (%s epochs with data, layout %s)", path, len(epochs), tier)
    return path
