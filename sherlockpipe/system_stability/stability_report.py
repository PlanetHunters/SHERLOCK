"""Landscape PDF report of a system stability run (``stability_megno.csv``).

Whatever way the parameter space was defined (a grid in a properties file, free parameters, the posterior of a fit),
the result is a table with one row per scenario and one column per parameter of every body, so the report is built on
that table: the dimensions that vary are the swept ones, the others are fixed.

Pages:

1. the stability map of the two most typical dimensions (the eccentricities of two bodies when both are swept), the
   verdict, the MEGNO distribution and how fast the unstable scenarios are lost;
2. what drives the stability: the share of regular, chaotic and unstable scenarios at every value of every swept
   dimension;
3. more maps, for the other pairs of swept dimensions;
4. representative scenarios and where the run comes from.

The report is a convenience output: it is written by :func:`write_stability_report_guarded`, which never raises, and
every page is built in isolation (a page that fails becomes a stub that says why).
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7ec"
OK, WARN, BAD, BLUE, ORANGE, INFO = "#2f8f5b", "#d9a21b", "#c0392b", "#2a6f97", "#e08a2b", "#9aa5b1"
PLANET_COLOURS = [BLUE, "#c0568a", "#7a9a2e", "#8a6bbf", "#d17a22"]
PAGE_SIZE = (11.69, 8.27)
REGULAR_MEGNO = 2.5     # below it the orbits are regular (quasi-periodic): the MEGNO of a regular orbit tends to 2
UNSTABLE_MEGNO = 5.0    # the MEGNO is capped here: a simulation that reaches it, or escapes, is stopped
MAX_CELLS = 12          # at most this many cells per axis of a map
MAX_PROFILES = 6
MAX_MAPS = 6
RC = {"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
      "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8, "axes.titleweight": "bold"}
STATE_COLOURS = {"regular": OK, "chaotic": WARN, "unstable": BAD}
STATES = ["regular", "chaotic", "unstable"]
CMAP = LinearSegmentedColormap.from_list("megno", [OK, "#e9d36b", WARN, BAD])
PARAMETERS = [("eccentricity", "eccentricities", "e", ""), ("period", "periods", "P", "d"),
              ("mass", "masses", "mass", "M⊕"), ("inclination", "inclinations", "inc", "°"),
              ("omega", "arg_periastron", "ω", "°"), ("omega_big", "long_asc_node", "Ω", "°")]
BODY_NAMES = "bcdefgh"


def report_style():
    """The matplotlib defaults plus ``RC``, so the pages do not depend on what another library did to ``rcParams``."""
    defaults = {k: v for k, v in matplotlib.rcParamsDefault.items() if k != "backend"}
    return {**defaults, **RC}


@dataclass
class Dimension:
    """One column of the scenario table: a parameter of a body (or of the star)."""
    key: str
    label: str
    unit: str
    body: Optional[int]
    values: List
    swept: bool

    @property
    def title(self):
        return f"{self.label} {self.unit}".strip()


@dataclass
class RunInfo:
    """What the report says about how the run was made (everything optional)."""
    object_id: Optional[str] = None
    years: Optional[float] = None
    dt: Optional[float] = None
    repetitions: Optional[int] = None
    cpus: Optional[int] = None
    seed: Optional[int] = None
    properties_file: Optional[str] = None
    free_params: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- the table
def load_scenarios(csv_file):
    """
    Reads ``stability_megno.csv`` into one row per scenario with a column per parameter of every body.

    :return: the table and the number of bodies. Columns: ``star_mass``, ``p{i}_period``, ``p{i}_mass``, ``p{i}_eccentricity``,
        ``p{i}_inclination``, ``p{i}_omega``, ``p{i}_omega_big`` (angles may be the text ``rand``), the MEGNO columns and the
        derived ``state``, ``lost`` and ``crossing``.
    """
    raw = pd.read_csv(csv_file)
    table = pd.DataFrame({"star_mass": raw["star_mass"].astype(float)})
    bodies = 0
    for key, column, _, _ in PARAMETERS:
        if column not in raw:
            continue
        parts = raw[column].astype(str).str.split(",", expand=True)
        bodies = max(bodies, parts.shape[1])
        for i in range(parts.shape[1]):
            numeric = pd.to_numeric(parts[i], errors="coerce")
            table[f"p{i + 1}_{key}"] = numeric if numeric.notna().all() else parts[i]
    for column in ("megno", "last_year", "megno_min", "megno_max", "megno_std", "seed"):
        if column in raw:
            table[column] = raw[column]
    table["state"] = np.where(table.megno < REGULAR_MEGNO, "regular",
                              np.where(table.megno < UNSTABLE_MEGNO, "chaotic", "unstable"))
    table["lost"] = table.megno >= UNSTABLE_MEGNO
    table["crossing"] = crossing_orbits(table, bodies)
    return table, bodies


def crossing_orbits(table, bodies):
    """True where the orbits of two neighbouring bodies cross: a(1+e) of the inner is beyond a(1-e) of the outer.

    This is only a flag: a resonance can keep two planets apart in a configuration whose orbits cross."""
    crossing = pd.Series(False, index=table.index)
    needed = [f"p{i}_{k}" for i in range(1, bodies + 1) for k in ("period", "eccentricity")]
    if bodies < 2 or any(c not in table or not np.issubdtype(table[c].dtype, np.number) for c in needed):
        return crossing
    order = np.argsort(np.stack([table[f"p{i}_period"].median() for i in range(1, bodies + 1)]))
    for inner, outer in zip(order[:-1], order[1:]):
        a_in, a_out = table[f"p{inner + 1}_period"] ** (2 / 3), table[f"p{outer + 1}_period"] ** (2 / 3)
        crossing |= a_in * (1 + table[f"p{inner + 1}_eccentricity"]) >= a_out * (1 - table[f"p{outer + 1}_eccentricity"])
    return crossing


def dimensions(table, bodies):
    """Every parameter column as a :class:`Dimension`, flagged as swept when it takes more than one value."""
    result = []
    for key, _, label, unit in PARAMETERS:
        for i in range(1, bodies + 1):
            column = f"p{i}_{key}"
            if column not in table:
                continue
            values = sorted(table[column].unique(), key=str)
            result.append(Dimension(column, f"{label} ({BODY_NAMES[i - 1]})", unit, i, values, len(values) > 1))
    values = sorted(table["star_mass"].unique())
    result.append(Dimension("star_mass", "star mass", "M☉", None, values, len(values) > 1))
    return result


def axis_bins(series, max_bins=MAX_CELLS):
    """Cell index of every value of an axis, and the centre and label of every cell.

    A dimension with few values has a cell per value; one with many (free sampling) is cut in cells of equal count."""
    values = np.sort(series.unique())
    if len(values) <= max_bins:
        index = series.map({v: i for i, v in enumerate(values)}).to_numpy()
        return index, np.array(values, dtype=float), [_short(v) for v in values]
    edges = np.unique(np.quantile(series, np.linspace(0, 1, max_bins + 1)))
    index = np.clip(np.searchsorted(edges, series, side="right") - 1, 0, len(edges) - 2)
    centres = np.array([series[index == i].median() for i in range(len(edges) - 1)])
    return index, centres, [_short(c) for c in centres]


def _short(value):
    return f"{value:.3g}"


def sensitivity(table, dimension):
    """How much the share of non-regular (chaotic or unstable) scenarios changes across the values of a dimension
    (0 to 1)."""
    if not dimension.swept:
        return 0.0
    index, _, _ = axis_bins(table[dimension.key])
    share = (table.state != "regular").groupby(index).mean()
    return float(share.max() - share.min())


def hero_pair(table, swept: List[Dimension]) -> Optional[Tuple[Dimension, Dimension]]:
    """The dimensions of the main map: the eccentricities of two bodies when both are swept, otherwise the two most
    decisive swept dimensions."""
    eccs = [d for d in swept if d.key.endswith("_eccentricity")]
    if len(eccs) >= 2:
        return eccs[0], eccs[1]
    ranked = sorted(swept, key=lambda d: sensitivity(table, d), reverse=True)
    return (ranked[0], ranked[1]) if len(ranked) >= 2 else None


# ---------------------------------------------------------------------------------------------- drawing helpers
def _axes(fig, x, y, w, h, **kw):
    return fig.add_axes([x, y, w, h], **kw)


def _title(ax, text, note=None):
    ax.set_title(text, loc="left", pad=3)
    if note:
        ax.text(1, 1.025, note, transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)


def _head(fig, info, title, subtitle, page):
    fig.text(.03, .962, title, fontsize=12.5, fontweight="bold", color=INK)
    fig.text(.03, .940, subtitle, fontsize=7.5, color=MUTED)
    fig.text(.97, .962, f"{info.object_id or 'system stability'} · page {page}", fontsize=7, color=MUTED, ha="right")
    fig.add_artist(Line2D([.03, .97], [.93, .93], color=GRID, lw=.8))


def _new_figure():
    return Figure(figsize=PAGE_SIZE)


def _unavailable(ax, text="not available"):
    ax.axis("off")
    ax.text(.5, .5, text, ha="center", va="center", color=MUTED, fontsize=7)


def _edges(centres):
    if len(centres) == 1:
        return np.array([centres[0] - .5, centres[0] + .5])
    mid = (centres[1:] + centres[:-1]) / 2
    return np.concatenate([[centres[0] - (mid[0] - centres[0])], mid, [centres[-1] + (centres[-1] - mid[-1])]])


def draw_map(ax, table, x: Dimension, y: Dimension, annotate=True):
    """Median MEGNO of the scenarios in every cell of the plane of two dimensions (the others are marginalised).

    A cell is hatched when the orbits cross in all its scenarios."""
    xi, xc, xl = axis_bins(table[x.key])
    yi, yc, yl = axis_bins(table[y.key])
    frame = pd.DataFrame({"x": xi, "y": yi, "megno": table.megno.clip(upper=UNSTABLE_MEGNO),
                          "crossing": table.crossing, "n": 1})
    grouped = frame.groupby(["y", "x"]).agg(megno=("megno", "median"), crossing=("crossing", "mean"), n=("n", "sum"))
    grid = np.full((len(yc), len(xc)), np.nan)
    cross = np.zeros_like(grid)
    for (row, col), r in grouped.iterrows():
        grid[row, col] = r.megno
        cross[row, col] = r.crossing
    xe, ye = np.arange(len(xc) + 1) - .5, np.arange(len(yc) + 1) - .5
    ax.pcolormesh(xe, ye, grid, cmap=CMAP, norm=Normalize(1, UNSTABLE_MEGNO), edgecolors="white", linewidth=.8)
    for row in range(len(yc)):
        for col in range(len(xc)):
            if cross[row, col] >= .99:
                ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, fill=False, hatch="////", ec="white", lw=0, alpha=.45))
            if annotate and not np.isnan(grid[row, col]) and len(xc) * len(yc) <= 100:
                value = grid[row, col]
                ax.text(col, row, "≥5" if value >= UNSTABLE_MEGNO else f"{value:.1f}", ha="center", va="center",
                        fontsize=6.5, color="white" if value > 3.2 or value < 1.6 else INK)
    ax.set_xticks(range(len(xc)))
    ax.set_xticklabels(xl, fontsize=6.5)
    ax.set_yticks(range(len(yc)))
    ax.set_yticklabels(yl, fontsize=6.5)
    ax.set_xlim(-.5, len(xc) - .5)
    ax.set_ylim(-.5, len(yc) - .5)
    ax.set_xlabel(x.title)
    ax.set_ylabel(y.title)


# ---------------------------------------------------------------------------------------------- pages
def _verdict(table):
    share = {s: float((table.state == s).mean()) for s in STATES}
    if len(table) == 1:
        word = {"regular": "The scenario is regular", "chaotic": "The scenario is chaotic",
                "unstable": "The scenario is unstable"}[table.state.iloc[0]]
        return word, STATE_COLOURS[table.state.iloc[0]], "#fbe7e4" if table.state.iloc[0] == "unstable" else "#e6f3ec"
    if share["regular"] >= .9:
        return f"{share['regular'] * 100:.0f} % of the explored space is regular", OK, "#e6f3ec"
    if share["unstable"] >= .5:
        return f"{share['unstable'] * 100:.0f} % of the explored space is unstable", BAD, "#fbe7e4"
    return f"{share['regular'] * 100:.0f} % of the explored space is regular", WARN, "#fbf3dc"


def page_summary(table, dims, info, bodies):
    fig = _new_figure()
    swept = [d for d in dims if d.swept]
    years = f"{info.years:g} yr" if info.years else "?"
    _head(fig, info, "Stability of the system · map and summary",
          f"{len(table)} scenarios · {info.repetitions or '?'} repetitions · {years} · MEGNO < {REGULAR_MEGNO:g} regular, "
          f"{REGULAR_MEGNO:g} to {UNSTABLE_MEGNO:g} chaotic, ≥ {UNSTABLE_MEGNO:g} unstable or lost", 1)
    headline, colour, background = _verdict(table)
    ax = _axes(fig, .03, .845, .94, .075)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=.01", fc=background, ec="none"))
    ax.text(.012, .66, headline, fontsize=15, fontweight="bold", color=colour, va="center")
    share = {s: (table.state == s).mean() * 100 for s in STATES}
    ax.text(.012, .22, f"{share['regular']:.0f} % regular · {share['chaotic']:.0f} % chaotic · {share['unstable']:.0f} % unstable"
                       f" · {table.crossing.mean() * 100:.0f} % crossing orbits", fontsize=8, va="center")
    lost = table[table.lost]
    tiles = [("Regular scenarios", f"{int((table.state == 'regular').sum())}", f"of {len(table)}", OK)]
    if "last_year" in table and len(lost):
        tiles.append(("Median time of loss", f"{lost.last_year.median():.3g} yr", f"{len(lost)} scenarios lost", BAD))
    if table.crossing.any():
        tiles.append(("Regular if orbits do not cross", f"{(table[~table.crossing].state == 'regular').mean() * 100:.0f} %",
                      f"{int((~table.crossing).sum())} scenarios", OK))
    for i, (label, big, small, c) in enumerate(tiles):
        x = .5 + i * .17
        ax.text(x, .76, label, fontsize=6.5, color=MUTED, va="center")
        ax.text(x, .46, big, fontsize=14, fontweight="bold", color=c, va="center")
        ax.text(x, .15, small, fontsize=6.5, color=MUTED, va="center")
    pair = hero_pair(table, swept)
    ax = _axes(fig, .075, .27, .42, .51)
    if pair is not None:
        draw_map(ax, table, pair[0], pair[1])
        others = [d.title for d in swept if d.key not in (pair[0].key, pair[1].key)]
        _title(ax, f"{pair[0].title} vs {pair[1].title}",
               "median MEGNO over " + ", ".join(others) if others else "MEGNO · hatched: orbits cross")
        cax = _axes(fig, .50, .27, .008, .51)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(1, UNSTABLE_MEGNO), cmap=CMAP), cax=cax)
        cb.set_label("MEGNO", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        cb.ax.axhline(REGULAR_MEGNO, color=INK, lw=.8)
    elif swept:
        _profile(ax, table, swept[0])
        _title(ax, swept[0].title, "one swept dimension")
    else:
        _unavailable(ax, "a single scenario: nothing was swept")
    # card of the space explored
    fig.text(.03, .185, "SPACE EXPLORED", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.03, .52], [.18, .18], color=GRID, lw=.6))
    for i, d in enumerate(swept[:6]):
        y = .155 - i * .02
        fig.text(.03, y, d.title, fontsize=7, fontweight="bold")
        values = [v for v in d.values if not isinstance(v, str)]
        text = f"{len(d.values)} values" + (f" from {_short(min(values))} to {_short(max(values))}" if values else " (random)")
        fig.text(.17, y, text, fontsize=7, color=INK)
    fixed = [d for d in dims if not d.swept]
    if swept:
        fig.text(.36, .155, f"{len(fixed)} more dimensions fixed", fontsize=7, color=MUTED)
        fig.text(.36, .135, "x".join(str(len(d.values)) for d in swept) + f" = {int(np.prod([len(d.values) for d in swept]))} nominal scenarios",
                 fontsize=7, color=MUTED)
    # MEGNO histogram and survival
    a = _axes(fig, .60, .55, .17, .23)
    values = table.megno.clip(upper=UNSTABLE_MEGNO).to_numpy()
    bins = np.linspace(1, UNSTABLE_MEGNO + .2, 22)
    counts, _ = np.histogram(values, bins)
    for k, count in enumerate(counts):
        lo = bins[k]
        a.bar((bins[k] + bins[k + 1]) / 2, count, width=bins[1] - bins[0],
              color=OK if lo < REGULAR_MEGNO else (WARN if lo < UNSTABLE_MEGNO else BAD))
    a.axvline(REGULAR_MEGNO, color=INK, lw=.6, ls=":")
    a.set_xlabel("MEGNO (capped at 5)")
    a.set_ylabel("scenarios")
    a.grid(axis="y", color=GRID, lw=.4)
    _title(a, "MEGNO of every scenario")
    a = _axes(fig, .82, .55, .15, .23)
    if "last_year" in table and info.years:
        end = float(info.years)
        times = np.sort(np.where(table.lost, table.last_year, end * 1.01))
        grid = np.unique(np.concatenate([times[times <= end], [end]]))
        survival = [(times > t).mean() for t in grid]
        a.step(grid, survival, where="post", color=BAD, lw=1.3)
        a.set_xscale("log")
        a.set_ylim(0, 1.02)
        a.axhline(float((table.state == "regular").mean()), color=OK, lw=.8, ls="--")
        a.set_xlabel("years")
        a.set_ylabel("fraction still bound")
        a.grid(color=GRID, lw=.4)
        _title(a, "Survival of the scenarios")
    else:
        _unavailable(a, "no times of loss")
    notes = _notes(table)
    fig.text(.60, .47, "READ THIS", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.60, .97], [.465, .465], color=GRID, lw=.6))
    for i, text in enumerate(notes):
        fig.text(.60, .43 - i * .035, text, fontsize=7.2)
    fig.text(.03, .012, f"MEGNO tends to 2 for regular orbits and grows with chaos. Regular below {REGULAR_MEGNO:g}; a scenario that reaches "
                        f"{UNSTABLE_MEGNO:g}, escapes or has a close encounter is stopped and counted as unstable.", fontsize=6.5, color=MUTED)
    return fig


def _notes(table):
    notes = []
    if table.crossing.any():
        crossing = table[table.crossing]
        regular_crossing = (crossing.state == "regular").sum()
        notes.append(f"· {table.crossing.mean() * 100:.0f} % of the scenarios have crossing orbits; "
                     f"{regular_crossing} of them are still regular.")
        notes.append("  A crossing orbit is a hint, not a verdict: a resonance can protect it.")
    inside = table[~table.crossing] if table.crossing.any() else table
    if len(inside) and table.crossing.any():
        notes.append(f"· Without crossing: {(inside.state == 'regular').mean() * 100:.0f} % regular, "
                     f"{(inside.state == 'chaotic').mean() * 100:.0f} % chaotic, {(inside.state == 'unstable').mean() * 100:.0f} % unstable.")
    if "megno_std" in table and (table.megno_std > .5).any():
        notes.append(f"· {int((table.megno_std > .5).sum())} scenarios differ by more than 0.5 MEGNO between repetitions:")
        notes.append("  their random angles matter, raise the repetitions.")
    return notes or ["· Nothing special: every scenario behaves alike."]


def _profile(ax, table, dimension):
    """Share of regular, chaotic and unstable scenarios at every value (or cell) of a dimension."""
    index, centres, labels = axis_bins(table[dimension.key])
    shares = pd.crosstab(index, table.state, normalize="index").reindex(columns=STATES, fill_value=0)
    counts = pd.Series(index).value_counts().sort_index()
    bottom = np.zeros(len(shares))
    for state in STATES:
        ax.bar(np.arange(len(shares)), shares[state].to_numpy(), bottom=bottom, color=STATE_COLOURS[state], width=.72, label=state)
        bottom += shares[state].to_numpy()
    ax.set_xticks(np.arange(len(shares)))
    ax.set_xticklabels(labels, fontsize=6.5)
    for position, n in enumerate(counts.to_numpy()):
        ax.text(position, 1.02, f"n={n}", ha="center", fontsize=5.8, color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .5, 1])
    ax.set_ylabel("share of scenarios")
    ax.set_xlabel(dimension.title)
    ax.grid(axis="y", color=GRID, lw=.4)


def page_drivers(table, dims, info):
    swept = sorted([d for d in dims if d.swept], key=lambda d: sensitivity(table, d), reverse=True)
    fig = _new_figure()
    _head(fig, info, "What drives the stability",
          "For every swept dimension: the share of regular, chaotic and unstable scenarios at each value "
          "(the other dimensions are marginalised)", 2)
    if not swept:
        _unavailable(_axes(fig, .1, .2, .8, .6), "nothing was swept")
        return fig
    scores = [sensitivity(table, d) for d in swept]
    a = _axes(fig, .15, .74, .20, .15)
    a.barh(range(len(swept))[::-1], scores, color=BLUE, height=.6)
    a.set_yticks(range(len(swept))[::-1])
    a.set_yticklabels([d.title for d in swept])
    a.set_xlim(0, 1)
    a.set_xlabel("change of the non-regular share across the dimension")
    a.grid(axis="x", color=GRID, lw=.4)
    _title(a, "Sensitivity ranking")
    fig.text(.42, .85, "The ranking is the difference between the highest and the lowest share of non-regular (chaotic or unstable) scenarios over", fontsize=7.5)
    fig.text(.42, .83, "the values of each dimension. A dimension with many values is cut in cells of equal count.", fontsize=7.5)
    fig.text(.42, .79, f"Most decisive: {swept[0].title}.   Least decisive: {swept[-1].title}.", fontsize=8.5, fontweight="bold")
    for i, d in enumerate(swept[:MAX_PROFILES]):
        a = _axes(fig, .06 + (i % 2) * .48, .40 - (i // 2) * .31, .40, .20)
        _profile(a, table, d)
        _title(a, d.title, f"spread {scores[i]:.2f}")
        if i == 0:
            a.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.22))
    if len(swept) > MAX_PROFILES:
        fig.text(.97, .012, f"{len(swept) - MAX_PROFILES} less decisive dimensions are not drawn", fontsize=6.5, color=MUTED, ha="right")
    return fig


def page_maps(table, dims, info):
    swept = [d for d in dims if d.swept]
    pair = hero_pair(table, swept)
    fig = _new_figure()
    _head(fig, info, "More stability maps",
          "Median MEGNO over the other dimensions · green regular, red unstable · hatched: orbits that cross", 3)
    candidates = [(a, b) for i, a in enumerate(swept) for b in swept[i + 1:]]
    if pair is not None:
        candidates = [c for c in candidates if {c[0].key, c[1].key} != {pair[0].key, pair[1].key}]
    candidates.sort(key=lambda c: sensitivity(table, c[0]) + sensitivity(table, c[1]), reverse=True)
    if not candidates:
        _unavailable(_axes(fig, .1, .2, .8, .6), "the first page already shows the only pair of swept dimensions")
        return fig
    for i, (x, y) in enumerate(candidates[:MAX_MAPS]):
        ax = _axes(fig, .06 + (i % 3) * .315, .54 - (i // 3) * .44, .24, .33)
        draw_map(ax, table, x, y)
        _title(ax, f"{x.title} vs {y.title}")
    cax = _axes(fig, .96, .30, .008, .45)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(1, UNSTABLE_MEGNO), cmap=CMAP), cax=cax)
    cb.set_label("median MEGNO", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    if len(candidates) > MAX_MAPS:
        fig.text(.97, .012, f"{len(candidates) - MAX_MAPS} less decisive pairs are not drawn", fontsize=6.5, color=MUTED, ha="right")
    return fig


def _orbit(ax, row, bodies):
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(0, 0, "o", color=ORANGE, ms=7)
    for i in range(1, bodies + 1):
        period, ecc = row[f"p{i}_period"], row[f"p{i}_eccentricity"]
        omega = row.get(f"p{i}_omega", 0)
        omega = np.deg2rad(omega) if isinstance(omega, (int, float, np.floating, np.integer)) else 0.0
        semi = (period / 365.25) ** (2 / 3) * row["star_mass"] ** (1 / 3)
        radius = semi * (1 - ecc ** 2) / (1 + ecc * np.cos(theta))
        ax.plot(radius * np.cos(theta + omega), radius * np.sin(theta + omega), color=PLANET_COLOURS[(i - 1) % 5], lw=1.2,
                label=f"{BODY_NAMES[i - 1]}  P {period:.3g} d  e {ecc:.2f}")
    ax.set_aspect("equal")
    ax.axis("off")


def page_scenarios(table, dims, info, bodies):
    fig = _new_figure()
    _head(fig, info, "Representative scenarios", "The most regular, a median and the least stable scenario, drawn from their parameters", 4)
    ordered = table.sort_values(["megno", "last_year"], ascending=[True, False] if "last_year" in table else [True])
    picks = [("Most regular", ordered.iloc[0], OK), ("Middle of the ranking", ordered.iloc[len(ordered) // 2], WARN), ("Least stable", ordered.iloc[-1], BAD)]
    can_draw = all(f"p{i}_period" in table and f"p{i}_eccentricity" in table for i in range(1, bodies + 1)) and bodies > 0
    for i, (name, row, colour) in enumerate(picks):
        ax = _axes(fig, .04 + i * .325, .48, .29, .36)
        if can_draw:
            _orbit(ax, row, bodies)
            ax.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(.5, -.30))
        else:
            _unavailable(ax)
        ax.set_title(name, loc="left", color=colour)
        end = f"lost in year {row.last_year:.3g}" if row.lost and "last_year" in row else f"bound for {info.years:g} yr" if info.years else "bound"
        ax.text(1, 1.02, f"MEGNO {row.megno:.2f} · {end}", transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
    fig.text(.03, .34, "MOST REGULAR SCENARIOS (the full table is stability_megno.csv)", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.03, .62], [.335, .335], color=GRID, lw=.6))
    columns = [d for d in dims if d.swept][:6]
    headers = ["MEGNO"] + [d.title for d in columns] + (["lost at"] if "last_year" in table else [])
    xs = np.linspace(.03, .60, len(headers) + 1)[:-1]
    for header, x in zip(headers, xs):
        fig.text(x, .315, header, fontsize=6.3, color=MUTED)
    for j, (_, row) in enumerate(ordered.head(8).iterrows()):
        y = .29 - j * .024
        cells = [f"{row.megno:.2f}"] + [_short(row[d.key]) if not isinstance(row[d.key], str) else row[d.key] for d in columns]
        if "last_year" in table:
            cells.append(f"{row.last_year:.3g} yr" if row.lost else "never")
        for cell, x in zip(cells, xs):
            fig.text(x, y, cell, fontsize=7.2, color=OK if cell == cells[0] else INK)
    fig.text(.66, .34, "HOW THIS RUN WAS MADE", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.66, .97], [.335, .335], color=GRID, lw=.6))
    rows = [("Integration", f"{info.years:g} yr, dt {info.dt:g} of the shortest orbit" if info.years and info.dt else None),
            ("Repetitions", f"{info.repetitions} per scenario" if info.repetitions else None),
            ("Seed of the random angles", f"{info.seed}" if info.seed is not None else None),
            ("Free parameters", ", ".join(info.free_params) if info.free_params else None),
            ("Properties file", os.path.basename(info.properties_file) if info.properties_file else None),
            ("Workers", f"{info.cpus}" if info.cpus else None)]
    y = .31
    for key, value in rows:
        if value is None:
            continue
        fig.text(.66, y, key, fontsize=7, color=MUTED)
        fig.text(.80, y, value, fontsize=7, fontweight="bold")
        y -= .026
    fixed = [d for d in dims if not d.swept]
    fig.text(.66, y - .01, "NOT EXPLORED (fixed)", fontsize=6.5, color=MUTED, fontweight="bold")
    for k, d in enumerate(fixed[:12]):
        value = d.values[0]
        x = .66 + (k // 6) * .16
        fig.text(x, y - .04 - (k % 6) * .022, d.title, fontsize=6.8, color=MUTED)
        fig.text(x + .075, y - .04 - (k % 6) * .022, value if isinstance(value, str) else _short(value), fontsize=6.8)
    if len(fixed) > 12:
        fig.text(.66, .02, f"{len(fixed) - 12} more fixed dimensions in the CSV", fontsize=6.3, color=MUTED)
    return fig


# ---------------------------------------------------------------------------------------------- writer
def _stub(info, title, error, page):
    fig = _new_figure()
    _head(fig, info, title, "This page could not be built", page)
    fig.text(.5, .5, f"Not available: {error}", ha="center", color=MUTED, fontsize=9)
    return fig


def build_pages(table, bodies, info):
    """All the pages of the report; a page that fails is replaced by a stub that says why."""
    dims = dimensions(table, bodies)
    pages = []

    def add(title, builder):
        try:
            with matplotlib.rc_context(report_style()):
                pages.append(builder())
        except Exception as error:
            logging.exception("The page '%s' of the stability report failed", title)
            pages.append(_stub(info, title, f"{type(error).__name__}: {error}", len(pages) + 1))

    add("Map and summary", lambda: page_summary(table, dims, info, bodies))
    add("What drives the stability", lambda: page_drivers(table, dims, info))
    if sum(d.swept for d in dims) >= 3:
        add("More stability maps", lambda: page_maps(table, dims, info))
    add("Representative scenarios", lambda: page_scenarios(table, dims, info, bodies))
    return pages


def write_stability_report(csv_file, pdf_file, info: Optional[RunInfo] = None):
    """Writes the report (``.partial`` first, then moved into place) and returns its number of pages."""
    table, bodies = load_scenarios(csv_file)
    if len(table) == 0:
        raise ValueError("The stability results are empty")
    pages = build_pages(table, bodies, info or RunInfo())
    partial = pdf_file + ".partial"
    with matplotlib.rc_context({**report_style(), "pdf.fonttype": 42}), PdfPages(partial) as pdf:
        for fig in pages:
            pdf.savefig(fig)
    os.replace(partial, pdf_file)
    return len(pages)


def write_stability_report_guarded(csv_file, pdf_file, info: Optional[RunInfo] = None):
    """Like :func:`write_stability_report` but it never raises: the report must not break a stability run.

    :return: the number of pages, or None when the report could not be written
    """
    try:
        pages = write_stability_report(csv_file, pdf_file, info)
        logging.info("Stability report written with %s pages: %s", pages, pdf_file)
        return pages
    except (Exception, SystemExit):
        logging.exception("The stability report could not be written; the results are in %s", csv_file)
        partial = pdf_file + ".partial"
        if os.path.exists(partial):
            os.remove(partial)
        return None
