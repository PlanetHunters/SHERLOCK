"""Landscape PDF report of a Bayesian fit (``<object id>_fit.pdf``).

For every fitted candidate (SOI) the report has three pages: its summary (verdict, folded transit with the posterior
model, key parameters against their priors, including the stellar density), its consistency (odd against even fits and
every single transit) and its corner plot. One last page, shared by all the candidates, says whether the sampling
converged and whether the noise is understood. The allesfitter files stay as they are in the fit directory.

The report is a convenience output: :func:`write_fit_report_guarded` never raises, and every page is built in isolation
(a page that fails becomes a stub that says why).
"""
import datetime
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from sherlockpipe.bayesian_fit.fit_results import FitResults, model_curve

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7ec"
OK, WARN, BAD, BLUE, ORANGE, INFO = "#2f8f5b", "#d9a21b", "#c0392b", "#2a6f97", "#e08a2b", "#9aa5b1"
PAGE_SIZE = (11.69, 8.27)
EDGE_FRACTION = 0.05            # the outer part of a uniform prior
EDGE_MASS = 0.15                # a posterior with this share of its samples in that part piles up against the edge
# limits of a prior that are physical, so a posterior against them says nothing about the prior
NATURAL_EDGES = {"sbratio_lc": ("low", "high"), "cosi": ("low",)}

ESS_OK, ESS_WARN = 1000, 300    # effective samples of the nested sampling
RED_NOISE_WARN = 1.5            # beta factor of the residuals
DENSITY_TENSION_WARN = 3.0      # sigmas between the density of the star and the one of the orbit
ODD_EVEN_WARN = 3.0             # sigmas between the odd and the even depths
CHI2_TRANSIT_WARN = 1.5         # reduced chi square of a single transit
MAX_TRANSITS = 48
RC = {"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
      "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8, "axes.titleweight": "bold"}
STATUS_COLOURS = {"ok": OK, "warn": WARN, "fail": BAD, "info": INFO}
LABELS = {"rr": "Rp / R★", "rsuma": "(R★+Rp) / a", "cosi": "cos i", "epoch": "T0", "period": "P [d]", "sbratio_lc": "surface br.",
          "host_ldc_q1_lc": "q1", "host_ldc_q2_lc": "q2", "ln_err_flux_lc": "ln σ",
          "baseline_gp_matern32_lnsigma_flux_lc": "GP ln σ", "baseline_gp_matern32_lnrho_flux_lc": "GP ln ρ"}


def report_style():
    """The matplotlib defaults plus ``RC``, so the pages do not depend on what another library did to ``rcParams``."""
    defaults = {k: v for k, v in matplotlib.rcParamsDefault.items() if k != "backend"}
    return {**defaults, **RC}


@dataclass
class Check:
    """One line of a verdict: ``status`` is ok, warn, fail or info."""
    status: str
    label: str
    detail: str


# ---------------------------------------------------------------------------------------------- the checks
def percentiles(samples):
    return np.percentile(np.asarray(samples, dtype=float), [2.5, 16, 50, 84, 97.5])


def edge_flag(results: FitResults, name):
    """'low' or 'high' when the posterior of a parameter piles up against a hard limit of its prior, otherwise None.

    A posterior piles up when at least 15 % of its samples are in the outer 5 % of the prior range (a flat posterior
    would have 5 %). The epoch is skipped: allesfitter moves its origin to the middle of the data."""
    prior = results.priors.get(name)
    if prior is None or prior.bounds is None or name.endswith("_epoch") or name not in results.samples:
        return None
    low, high = prior.bounds
    span = high - low
    x = results.samples[name].to_numpy()
    if span <= 0:
        return None
    natural = NATURAL_EDGES.get(name.split("_", 2)[-1] if name.startswith("SOI_") else name, ())
    if "low" not in natural and (x < low + EDGE_FRACTION * span).mean() >= EDGE_MASS:
        return "low"
    if "high" not in natural and (x > high - EDGE_FRACTION * span).mean() >= EDGE_MASS:
        return "high"
    return None


def edge_flags(results: FitResults, planet=None):
    """The parameters (of a planet and the shared ones) whose posterior piles up against the edge of its prior."""
    flags = {}
    for name in results.samples.columns:
        if planet is not None and "_" in name and name.split("_")[0] == "SOI" and not name.startswith(planet + "_"):
            continue
        flag = edge_flag(results, name)
        if flag:
            flags[name] = flag
    return flags


def label_of(name, planet=None):
    key = name[len(planet) + 1:] if planet and name.startswith(planet + "_") else name
    return LABELS.get(key, key)


def red_noise_beta(residuals, bin_sizes=(5, 10, 20, 40)):
    """The red noise factor: how much larger the rms of the binned residuals is than the white noise expectation."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10 * max(bin_sizes):
        return np.nan
    sigma = np.std(r)
    ratios = []
    for n in bin_sizes:
        m = len(r) // n
        binned = r[:m * n].reshape(m, n).mean(axis=1)
        ratios.append(np.std(binned) / (sigma / np.sqrt(n) * np.sqrt(m / (m - 1))))
    return float(np.median(ratios))


def convergence_checks(results: FitResults) -> List[Check]:
    checks = []
    ess = results.ess
    status = "ok" if ess >= ESS_OK else ("warn" if ess >= ESS_WARN else "fail")
    checks.append(Check(status, "Effective samples", f"{ess:.0f}" + ("" if status == "ok" else f" (≥ {ESS_OK} recommended)")))
    checks.append(Check("ok" if results.logzerr < 0.5 else "warn", "Evidence error", f"ln Z = {results.logz:.2f} ± {results.logzerr:.2f}"))
    return checks


def noise_checks(results: FitResults) -> List[Check]:
    beta = red_noise_beta(results.residuals)
    if not np.isfinite(beta):
        return [Check("info", "Red noise", "not enough points to measure it")]
    return [Check("ok" if beta < RED_NOISE_WARN else "warn", "Red noise of the residuals", f"β = {beta:.2f}")]


def density_check(results: FitResults, planet) -> Optional[Check]:
    """The stellar density of the star (mass and radius) against the one of the orbit of a planet."""
    samples = results.derived_of(planet, "host_density")
    if samples is None or results.star_density is None or not np.isfinite(results.star_density[1]) or results.star_density[1] <= 0:
        return None
    q = percentiles(samples)
    sigma = np.hypot(results.star_density[1], (q[3] - q[1]) / 2)
    tension = abs(q[2] - results.star_density[0]) / sigma
    return Check("ok" if tension < DENSITY_TENSION_WARN else "warn", "Stellar density",
                 f"orbit {q[2]:.1f} vs star {results.star_density[0]:.1f} g/cm³ ({tension:.1f} σ)")


def _symmetric_error(row):
    return (abs(float(row["lower_error"])) + abs(float(row["upper_error"]))) / 2


def variant_value(variant, planet, parameter):
    """(median, error) of a fitted parameter of an odd or even fit, or None."""
    table = variant.table.set_index("#name")
    name = f"{planet}_{parameter}"
    if name not in table.index or str(table.loc[name, "lower_error"]) == "(fixed)":
        return None
    row = table.loc[name]
    return float(row["median"]), _symmetric_error(row)


def variant_edge(variant, planet, parameter):
    """True when the median of a parameter of an odd or even fit sits at the edge of its prior (it cannot be trusted)."""
    prior = variant.priors.get(f"{planet}_{parameter}")
    value = variant_value(variant, planet, parameter)
    if prior is None or prior.bounds is None or value is None:
        return False
    low, high = prior.bounds
    natural = NATURAL_EDGES.get(parameter, ())
    return (value[0] < low + 0.03 * (high - low) and "low" not in natural) or \
        (value[0] > high - 0.03 * (high - low) and "high" not in natural)



@dataclass
class ComparisonRow:
    """One parameter of the main fit against the odd and the even fits."""
    label: str
    fmt: str
    main: Optional[tuple]
    odd: Optional[tuple]
    even: Optional[tuple]
    sigma: Optional[float]

    @property
    def status(self):
        """ok below 2 sigma between the odd and the even fit, warn up to 3 and fail above; info when it cannot be told."""
        if self.sigma is None:
            return "info"
        return "ok" if self.sigma < 2 else ("warn" if self.sigma < ODD_EVEN_WARN else "fail")


# label, key of the derived samples of the main fit, how to find the row in a derived table, format
COMPARISON = [
    ("Rp / R★", None, None, "{:.4f}"),
    ("Depth [ppt]", "depth_tr_undil_lc", lambda p, n: p.startswith(f"Transit depth (undil.) {n};"), "{:.3f}"),
    ("Rp [R⊕]", "R_companion_(R_earth)", lambda p, n: p.startswith(f"Companion radius {n};") and "R_{\\oplus}" in p, "{:.2f}"),
    ("Impact parameter b", "b_tra", lambda p, n: p.startswith(f"Impact parameter {n};") and "occultation" not in p, "{:.2f}"),
    ("Total duration [h]", "T_tra_tot", lambda p, n: p.startswith(f"Total transit duration {n};"), "{:.3f}"),
    ("a / R★", "a/R_star", lambda p, n: p.startswith(f"Semi-major axis {n} over host radius"), "{:.2f}"),
    ("Stellar density [g/cm³]", "host_density", lambda p, n: p.startswith(f"Host density from orbit {n};"), "{:.2f}"),
]


def _derived_value(derived, finder, planet):
    """(value, symmetric error) of a row of a derived table found by ``finder``, or None."""
    if derived is None or finder is None or "#property" not in derived:
        return None
    for _, row in derived.iterrows():
        if finder(str(row["#property"]), planet):
            return float(row["value"]), _symmetric_error(row)
    return None


def comparison_rows(results: FitResults, planet) -> List[ComparisonRow]:
    """The main fit, the odd fit and the even fit side by side, and how far apart the odd and the even ones are."""
    rows = []
    odd, even = results.variants.get("odd"), results.variants.get("even")
    for label, sample_key, finder, fmt in COMPARISON:
        if sample_key is None:   # the radius ratio is a fitted parameter
            samples = results.samples.get(f"{planet}_rr")
            main = None if samples is None else _median_error(samples)
            values = [None if v is None else variant_value(v, planet, "rr") for v in (odd, even)]
        else:
            samples = results.derived_of(planet, sample_key)
            main = None if samples is None else _median_error(samples)
            values = [None if v is None else _derived_value(v.derived, finder, planet) for v in (odd, even)]
        if main is None and all(v is None for v in values):
            continue
        a, b = values
        sigma = None if a is None or b is None or np.hypot(a[1], b[1]) <= 0 else abs(a[0] - b[0]) / np.hypot(a[1], b[1])
        rows.append(ComparisonRow(label, fmt, main, a, b, sigma))
    return rows


def _median_error(samples):
    q = percentiles(samples)
    return float(q[2]), float((q[3] - q[1]) / 2)

def odd_even_check(results: FitResults, planet) -> Check:
    if "odd" not in results.variants or "even" not in results.variants:
        return Check("info", "Odd / even", "the odd and even fits were not run")
    odd, even = results.variants["odd"], results.variants["even"]
    pegged = [p for p in ("rr", "rsuma", "cosi") for v in (odd, even) if variant_edge(v, planet, p)]
    if pegged:
        return Check("warn", "Odd / even", "cannot be judged: their posteriors sit at the edge of the priors")
    rows = {r.label: r for r in comparison_rows(results, planet)}
    for label, name in (("Depth [ppt]", "depth"), ("Rp / R★", "Rp/R★")):
        row = rows.get(label)
        if row is not None and row.sigma is not None:
            return Check("ok" if row.sigma < ODD_EVEN_WARN else "warn", "Odd / even", f"{name} differs by {row.sigma:.1f} σ")
    return Check("info", "Odd / even", "no radius ratio or depth in the odd and even fits")


def planet_depth_samples(results, planet):
    return results.derived_of(planet, "depth_tr_undil_lc")


def transit_times(results: FitResults, planet):
    """The times of the transits of a planet that have data inside them."""
    period, epoch = results.median(f"{planet}_period"), results.median(f"{planet}_epoch")
    duration = results.derived_of(planet, "T_tra_tot")
    t14 = float(np.median(duration)) / 24 if duration is not None else 0.05
    t = results.time
    first, last = int(np.floor((t.min() - epoch) / period)), int(np.ceil((t.max() - epoch) / period))
    times = epoch + np.arange(first, last + 1) * period
    inside = np.array([(np.abs(t - x) < t14 / 2).sum() > 20 for x in times])
    return times[inside], t14, len(times)


def transit_chi2(results: FitResults, planet, times, t14):
    """The reduced chi square of every transit window (3 durations), with the noise taken from the residuals."""
    sigma = np.nanstd(results.residuals)   # the residuals have gaps (NaN) where the baseline could not be predicted
    out = []
    for x in times:
        r = results.residuals[np.abs(results.time - x) < 1.5 * t14]
        r = r[np.isfinite(r)]
        out.append(float(np.mean((r / sigma) ** 2)) if len(r) > 5 and sigma > 0 else np.nan)
    return np.array(out)


def planet_checks(results: FitResults, planet) -> List[Check]:
    checks = list(convergence_checks(results))
    flags = edge_flags(results, planet)
    if flags:
        checks.append(Check("warn", "Prior edges", ", ".join(label_of(n, planet) for n in flags)))
    else:
        checks.append(Check("ok", "Prior edges", "no posterior piles up against its prior"))
    density = density_check(results, planet)
    if density is not None:
        checks.append(density)
    checks.append(odd_even_check(results, planet))
    checks.extend(noise_checks(results))
    return checks


def verdict(checks: List[Check]):
    """Headline, colour and background of a list of checks."""
    bad = [c for c in checks if c.status == "fail"]
    warn = [c for c in checks if c.status == "warn"]
    if bad:
        return "The fit has problems: " + ", ".join(c.label.lower() for c in bad), BAD, "#fbe7e4"
    if warn:
        return f"The fit converged, {len(warn)} thing{'s' if len(warn) > 1 else ''} to check", WARN, "#fbf3dc"
    return "The fit converged and the checks are clean", OK, "#e6f3ec"


# ---------------------------------------------------------------------------------------------- drawing helpers
def _axes(fig, x, y, w, h, **kw):
    return fig.add_axes([x, y, w, h], **kw)


def _title(ax, text, note=None):
    ax.set_title(text, loc="left", pad=3)
    if note:
        ax.text(1, 1.025, note, transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)


def _head(fig, results, title, subtitle, page):
    fig.text(.03, .962, title, fontsize=12.5, fontweight="bold", color=INK)
    fig.text(.03, .940, subtitle, fontsize=7.5, color=MUTED)
    fig.text(.97, .962, f"{results.object_id} · page {page}", fontsize=7, color=MUTED, ha="right")
    fig.add_artist(Line2D([.03, .97], [.93, .93], color=GRID, lw=.8))


def _new_figure():
    return Figure(figsize=PAGE_SIZE)


def _checks_block(fig, checks, x, y_top, spacing, width=.45):
    """The checks of a verdict as rows: a colour dot (green ok, orange warning, red failure, grey information), what it
    is and what was found."""
    fig.text(x, y_top + .028, "CHECKS", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([x, x + width], [y_top + .022, y_top + .022], color=GRID, lw=.6))
    for i, check in enumerate(checks):
        y = y_top - i * spacing
        fig.patches.append(Circle((x + .006, y + .006), .0052, color=STATUS_COLOURS[check.status], transform=fig.transFigure))
        fig.text(x + .02, y, check.label, fontsize=7.2, fontweight="bold")
        detail = check.detail if len(check.detail) <= 58 else check.detail[:55] + "..."
        fig.text(x + .165, y, detail, fontsize=7.0, color=INK)


def _unavailable(ax, text="not available"):
    ax.axis("off")
    ax.text(.5, .5, text, ha="center", va="center", color=MUTED, fontsize=7)


def binned(x, y, lo, hi, n, minimum=5):
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = np.linspace(lo, hi, n + 1)
    index = np.digitize(x, edges) - 1
    cx, cy, ce = [], [], []
    for k in range(n):
        s = index == k
        if s.sum() >= minimum:
            cx.append(x[s].mean())
            cy.append(y[s].mean())
            ce.append(y[s].std() / np.sqrt(s.sum()))
    return np.array(cx), np.array(cy), np.array(ce)


def _subtitle(results: FitResults):
    s = results.settings
    parts = ["nested sampling (dynesty)", f"{results.niter} iterations", f"ln Z = {results.logz:.2f} ± {results.logzerr:.2f}",
             f"{results.ess:.0f} effective samples"]
    parts.append("circular orbit" if s.get("circular") else "eccentric orbit")
    baseline = str(s.get("baseline_flux_lc") or "")
    parts.append("GP baseline" if "GP" in baseline else (baseline.replace("_", " ") if baseline else "no baseline"))
    parts.append(f"{len(results.time)} points")
    return " · ".join(parts)


def _planet_model_curve(results, planet, time, row=None):
    """The model of one planet at the given times from the parameters of a sample (the median when ``row`` is None)."""
    get = (lambda n: results.median(n)) if row is None else (lambda n: float(row[n]))
    return model_curve(time, get(f"{planet}_rr"), get(f"{planet}_rsuma"), get(f"{planet}_cosi"), get(f"{planet}_epoch"),
                       get(f"{planet}_period"), get("host_ldc_q1_lc"), get("host_ldc_q2_lc"))


def _key_rows(results, planet):
    """(label, samples, prior key, format, prior band) of the key parameters of a planet."""
    d = lambda key: results.derived_of(planet, key)
    rows = [("Period [d]", results.samples[f"{planet}_period"], f"{planet}_period", "{:.6f}", None),
            ("T0 [BJD-2457000]", results.samples[f"{planet}_epoch"], None, "{:.4f}", None),
            ("Rp / R★", results.samples[f"{planet}_rr"], f"{planet}_rr", "{:.4f}", None)]
    for label, key, fmt in (("Depth [ppt]", "depth_tr_undil_lc", "{:.3f}"), ("Rp [R⊕]", "R_companion_(R_earth)", "{:.2f}"),
                            ("Impact parameter b", "b_tra", "{:.2f}"), ("a / R★", "a/R_star", "{:.2f}"),
                            ("Duration [h]", "T_tra_tot", "{:.2f}")):
        if d(key) is not None:
            rows.append((label, pd.Series(d(key)), None, fmt, None))
    if d("host_density") is not None:
        band = results.star_density if results.star_density and np.isfinite(results.star_density[1]) else None
        rows.append(("Stellar density [g/cm³]", pd.Series(d("host_density")), None, "{:.2f}", band))
    return rows


def _key_parameters(fig, results, planet, x0, y_top, row_height):
    fig.text(x0, y_top + .04, "KEY PARAMETERS · posterior over its prior (grey band: what the star says)", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([x0, .97], [y_top + .035, y_top + .035], color=GRID, lw=.6))
    for i, (label, samples, prior_key, fmt, band) in enumerate(_key_rows(results, planet)):
        y = y_top - i * row_height
        q = percentiles(samples)
        fig.text(x0, y, label, fontsize=7.2)
        fig.text(x0 + .14, y, fmt.format(q[2]) + f"  +{fmt.format(q[3] - q[2])} / −{fmt.format(q[2] - q[1])}", fontsize=7.2, fontweight="bold")
        ax = _axes(fig, x0 + .29, y - .008, .15, .028)
        ax.axis("off")
        prior = results.priors.get(prior_key) if prior_key else None
        bounds = prior.bounds if prior is not None else None
        span = q[4] - q[0]
        lo, hi = (bounds if bounds else (q[0] - .15 * span, q[4] + .15 * span))
        edges = np.linspace(lo, hi, 41)
        counts, _ = np.histogram(samples, bins=edges)
        if band is not None:
            ax.axvspan(band[0] - band[1], band[0] + band[1], color=GRID, lw=0)
            lo, hi = min(lo, band[0] - 2 * band[1]), max(hi, band[0] + 2 * band[1])
            edges = np.linspace(lo, hi, 61)
            counts, _ = np.histogram(samples, bins=edges)
        ax.fill_between((edges[1:] + edges[:-1]) / 2, 0, counts / max(counts.max(), 1), color=BLUE, alpha=.75, step="mid")
        ax.axvline(q[2], color=INK, lw=.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.1)
        if bounds:
            flagged = edge_flag(results, prior_key)
            ax.add_patch(Rectangle((lo, 0), hi - lo, 1.1, fill=False, ec=WARN if flagged else GRID, lw=1.2 if flagged else .6))
            if flagged:
                fig.text(x0 + .455, y, "!", fontsize=9, color=WARN, fontweight="bold", va="center")


# ---------------------------------------------------------------------------------------------- pages
def page_summary(results: FitResults, planet, page):
    fig = _new_figure()
    _head(fig, results, f"Bayesian fit · {planet}", _subtitle(results), page)
    checks = planet_checks(results, planet)
    headline, colour, background = verdict(checks)
    ax = _axes(fig, .03, .845, .94, .075)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=.01", fc=background, ec="none"))
    ax.text(.012, .66, headline, fontsize=15, fontweight="bold", color=colour, va="center")
    notes = [f"{c.label}: {c.detail}" for c in checks if c.status in ("warn", "fail")]
    if results.masked:
        notes.append("masked in the fit: " + ", ".join(results.masked))
    text = " · ".join(notes) if notes else "every check is clean"
    ax.text(.012, .22, text if len(text) <= 115 else text[:112] + "...", fontsize=7.2, va="center")
    depth = planet_depth_samples(results, planet)
    tiles = []
    if depth is not None:
        tiles.append(("Depth", f"{np.median(depth):.3f} ppt", f"depth / error {np.median(depth) / max(np.std(depth), 1e-12):.0f}"))
    radius = results.derived_of(planet, "R_companion_(R_earth)")
    if radius is not None:
        tiles.append(("Radius", f"{np.median(radius):.2f} R⊕", f"± {np.std(radius):.2f}"))
    tiles.append(("Period", f"{results.median(f'{planet}_period'):.6f} d", f"± {np.std(results.samples[f'{planet}_period']):.1e}"))
    duration, impact = results.derived_of(planet, "T_tra_tot"), results.derived_of(planet, "b_tra")
    if duration is not None:
        tiles.append(("Duration", f"{np.median(duration):.2f} h", "" if impact is None else f"b = {np.median(impact):.2f}"))
    for i, (label, big, small) in enumerate(tiles):
        x = .53 + i * .115
        ax.text(x, .76, label, fontsize=6.5, color=MUTED, va="center")
        ax.text(x, .46, big, fontsize=12, fontweight="bold", color=INK, va="center")
        ax.text(x, .15, small, fontsize=6.5, color=MUTED, va="center")
    # folded transit
    period, epoch = results.median(f"{planet}_period"), results.median(f"{planet}_epoch")
    duration_h = float(np.median(duration)) if duration is not None else 2.0
    half = max(3.4, 2.8 * duration_h)
    own = _planet_model_curve(results, planet, results.time)
    flux = results.detrended - (results.model - own)   # the other planets are removed
    phase = ((results.time - epoch + period / 2) % period - period / 2) * 24
    sel = np.abs(phase) < half
    a, ar = _axes(fig, .055, .515, .40, .27), _axes(fig, .055, .425, .40, .075)
    a.scatter(phase[sel][::6], flux[sel][::6], s=.8, color=MUTED, alpha=.12, lw=0, rasterized=True)
    bx, by, be = binned(phase[sel], flux[sel], -half, half, 44)
    a.errorbar(bx, by, be, fmt="o", ms=2.4, color=INK, lw=.6, zorder=3)
    xm = np.linspace(-half, half, 400)
    tm = epoch + xm / 24
    rng = np.random.default_rng(0)
    rows = rng.choice(len(results.samples), min(120, len(results.samples)), replace=False)
    draws = np.array([_planet_model_curve(results, planet, tm, results.samples.iloc[j]) for j in rows])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    median_model = np.median(draws, axis=0)
    a.fill_between(xm, lo, hi, color=BAD, alpha=.25, lw=0, label="95 % of the posterior")
    a.plot(xm, median_model, color=BAD, lw=1.3, label="median model")
    depth_scale = float(1 - median_model.min()) or 1e-3
    a.set_ylim(1 - 2.1 * depth_scale, 1 + 1.4 * depth_scale)
    a.set_ylabel("relative flux")
    a.grid(color=GRID, lw=.4)
    a.tick_params(labelbottom=False)
    a.legend(frameon=False, fontsize=6.5, loc="upper right")
    _title(a, "Folded transit and posterior model")
    ar.errorbar(bx, (by - np.interp(bx, xm, median_model)) * 1e3, be * 1e3, fmt="o", ms=2, color=INK, lw=.5)
    ar.axhline(0, color=BAD, lw=.7)
    ar.set_ylabel("res. (ppt)")
    ar.set_xlabel("hours from mid-transit")
    ar.grid(color=GRID, lw=.4)
    _key_parameters(fig, results, planet, .52, .735, .0475)
    _checks_block(fig, checks, .03, .345, .0315)
    fig.text(.52, .255, "LIMB DARKENING AND NOISE", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.52, .97], [.25, .25], color=GRID, lw=.6))
    shared = [("q1", "host_ldc_q1_lc"), ("q2", "host_ldc_q2_lc"), ("ln σ (white noise)", "ln_err_flux_lc"),
              ("GP ln σ", "baseline_gp_matern32_lnsigma_flux_lc"), ("GP ln ρ", "baseline_gp_matern32_lnrho_flux_lc")]
    for j, (label, key) in enumerate([s_ for s_ in shared if s_[1] in results.samples]):
        q = percentiles(results.samples[key])
        x, y = .52 + (j % 2) * .235, .22 - (j // 2) * .03
        fig.text(x, y, label, fontsize=7.2)
        fig.text(x + (.1 if j % 2 == 0 else .055), y, f"{q[2]:.3f}  +{q[3] - q[2]:.3f} / −{q[2] - q[1]:.3f}", fontsize=7.0, fontweight="bold")
    for k, line in enumerate(["Bands come from equally weighted samples of the sampling. An orange frame marks a posterior",
                              f"with at least {EDGE_MASS * 100:.0f} % of its samples in the outer {EDGE_FRACTION * 100:.0f} % of its prior range (it may be cut by",
                              "the prior). The grey band is the stellar density that the mass and radius of the star give."]):
        fig.text(.52, .105 - k * .017, line, fontsize=6.3, color=MUTED)
    return fig


def _comparison_table(fig, results, planet, x, y_top, width):
    """The main, odd and even fits side by side, with the difference between the odd and the even in sigmas."""
    rows = comparison_rows(results, planet)
    odd_even = "odd" in results.variants and "even" in results.variants
    cols = [x, x + .20 * width, x + .42 * width, x + .64 * width, x + .86 * width]
    fig.text(x, y_top + .042, "ODD AND EVEN FITS AGAINST THE MAIN FIT", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([x, x + width], [y_top + .036, y_top + .036], color=GRID, lw=.6))
    for label, cx, colour in (("parameter", cols[0], MUTED), ("all", cols[1] + .06 * width, INK), ("odd", cols[2] + .06 * width, BAD),
                              ("even", cols[3] + .06 * width, BLUE), ("odd − even", cols[4], MUTED)):
        fig.text(cx, y_top + .012, label, fontsize=6.4, color=colour, fontweight="bold")
    if not rows:
        fig.text(x, y_top - .03, "no parameters to compare", fontsize=7.2, color=MUTED)
        return
    for i, row in enumerate(rows):
        y = y_top - .02 - i * .036
        fig.text(cols[0], y, row.label, fontsize=7.2)
        for entry, cx, colour in ((row.main, cols[1], INK), (row.odd, cols[2], BAD), (row.even, cols[3], BLUE)):
            if entry is None:
                fig.text(cx + .06 * width, y, "–", fontsize=7.2, color=MUTED)
            else:
                fig.text(cx, y, row.fmt.format(entry[0]) + " ± " + row.fmt.format(entry[1]), fontsize=7.0, color=colour, fontweight="bold")
        if row.sigma is not None:
            fig.text(cols[4], y, f"{row.sigma:.1f} σ", fontsize=7.2, fontweight="bold")
            fig.patches.append(Circle((cols[4] + .085 * width, y + .006), .0052, color=STATUS_COLOURS[row.status], transform=fig.transFigure))
    if not odd_even:
        fig.text(x, y_top - .02 - len(rows) * .036 - .01, "The odd and even fits were not run.", fontsize=7, color=MUTED)


def _variant_panels(fig, results, planet):
    """Odd and even against the main fit for the depth, the radius ratio and the impact parameter."""
    rows = {r.label: r for r in comparison_rows(results, planet)}
    panels = [("Depth [ppt]", "Depth [ppt]"), ("Rp / R★", "Rp / R★"), ("Impact parameter b", "Impact parameter b")]
    for i, (key, title) in enumerate(panels):
        ax = _axes(fig, .56 + i * .145, .60, .115, .26)
        row = rows.get(key)
        if row is None or row.odd is None or row.even is None:
            _unavailable(ax, "not run" if "odd" not in results.variants else "not fitted")
            continue
        for x, (entry, colour, name) in enumerate(((row.main, INK, "all"), (row.odd, BAD, "odd"), (row.even, BLUE, "even"))):
            if entry is not None:
                ax.errorbar(x, entry[0], entry[1], fmt="o", color=colour, capsize=3, ms=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["all", "odd", "even"])
        ax.set_xlim(-.5, 2.5)
        ax.grid(axis="y", color=GRID, lw=.4)
        _title(ax, title, "" if row.sigma is None else f"{row.sigma:.1f} σ")


def page_consistency(results: FitResults, planet, page):
    fig = _new_figure()
    _head(fig, results, f"Consistency · {planet}", "Odd against even fits, and every transit against the posterior model", page)
    _comparison_table(fig, results, planet, .03, .83, .48)
    _variant_panels(fig, results, planet)
    check = odd_even_check(results, planet)
    times, t14, total = transit_times(results, planet)
    chi2 = transit_chi2(results, planet, times, t14)
    bad = int(np.nansum(chi2 > CHI2_TRANSIT_WARN))
    # the verdict of the odd / even test
    box = _axes(fig, .03, .50, .94, .05)
    box.axis("off")
    box.text(0, .78, f"ODD / EVEN: {check.detail}", fontsize=8.2, fontweight="bold", color=STATUS_COLOURS[check.status], va="center")
    if "odd" in results.variants:
        box.text(0, .18, "The odd fit keeps the 1st, 3rd, 5th... transit and the even fit the 2nd, 4th...; both use the period and the priors of the main fit. "
                         "Dots: green below 2 σ between odd and even, orange up to 3 σ, red above.", fontsize=6.4, color=MUTED, va="center")
    # chi square of every transit
    a = _axes(fig, .06, .31, .30, .15)
    if len(times):
        a.scatter(times, chi2, c=[BAD if c > CHI2_TRANSIT_WARN else BLUE for c in chi2], s=9, zorder=3)
        a.axhline(1.0, color=INK, lw=.6, ls=":")
        a.axhline(CHI2_TRANSIT_WARN, color=WARN, lw=.6, ls="--")
        a.set_xlabel("time of the transit (BTJD)", fontsize=6.5)
        a.set_ylabel("χ² per point", fontsize=6.5)
        a.grid(color=GRID, lw=.4)
    else:
        _unavailable(a, "no transit has data")
    _title(a, "Fit quality per transit", f"{bad} of {len(times)} > {CHI2_TRANSIT_WARN}")
    a = _axes(fig, .42, .31, .13, .15)
    a.axis("off")
    for k, (label, value) in enumerate([("transits with data", f"{len(times)}"), ("listed", f"{total}"), ("duration", f"{t14 * 24:.2f} h"),
                                        ("above χ² %.1f" % CHI2_TRANSIT_WARN, f"{bad}")]):
        a.text(0, .9 - k * .25, label, fontsize=7.0, color=MUTED)
        a.text(1, .9 - k * .25, value, fontsize=7.6, fontweight="bold", ha="right")
    # every transit, in a grid that fills the rest of the page
    shown = times[:MAX_TRANSITS]
    n = max(len(shown), 1)
    cols = min(16, max(4, int(np.ceil(n / 2)))) if n > 8 else max(4, n)
    rows_n = int(np.ceil(n / cols))
    fig.text(.03, .275, f"EVERY TRANSIT WITH DATA · {len(times)} of {total} · the posterior model in red, χ² per point in the corner (red above {CHI2_TRANSIT_WARN})",
             fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.03, .97], [.269, .269], color=GRID, lw=.6))
    top, bottom = .255, .045
    cell_h = min((top - bottom) / rows_n, .13)
    cell_w = .94 / cols
    own = _planet_model_curve(results, planet, results.time)
    flux = results.detrended - (results.model - own)
    for i, x in enumerate(shown):
        ax = _axes(fig, .03 + (i % cols) * cell_w + .003, top - (i // cols + 1) * cell_h + .012, cell_w - .006, cell_h - .024)
        s_ = np.abs(results.time - x) < 2.5 * t14
        hours = (results.time[s_] - x) * 24
        ax.plot(hours, flux[s_], ".", ms=.7, color=MUTED, alpha=.4)
        if hours.size > 8:
            bx, by, _ = binned(hours, flux[s_], hours.min(), hours.max(), 12, 3)
            ax.plot(bx, by, "-", color=INK, lw=.7)
        ax.plot(hours, own[s_], color=BAD, lw=.9)
        depth = max(1 - own[s_].min(), 1e-4)
        ax.set_ylim(1 - 4 * depth, 1 + 3 * depth)
        ax.set_xticks([])
        ax.set_yticks([])
        c = chi2[i]
        ax.set_title(f"#{i}", loc="left", fontsize=5.2, pad=1, color=MUTED)
        if np.isfinite(c):
            ax.text(.97, .05, f"{c:.1f}", transform=ax.transAxes, fontsize=5.4, ha="right", color=BAD if c > CHI2_TRANSIT_WARN else MUTED)
    if len(times) > MAX_TRANSITS:
        fig.text(.97, .015, f"First {MAX_TRANSITS} of {len(times)} transits shown", fontsize=6.3, color=MUTED, ha="right")
    fig.text(.03, .015, "The allesfitter files (results/ns_corner.pdf, ns_trace.pdf, ns_fit_*.pdf, the odd/ and even/ folders and the tables) stay as they are.",
             fontsize=6.2, color=MUTED)
    return fig


def corner_columns(results: FitResults, planet):
    """(label, samples, prior band) of the parameters of the corner plot: the radius ratio, the impact parameter, the
    stellar density, the duration, the depth and the period."""
    d = lambda key: results.derived_of(planet, key)
    columns = [("Rp/R★", results.samples[f"{planet}_rr"].to_numpy(), None)]
    for label, key in (("b", "b_tra"), ("ρ★ [g/cm³]", "host_density"), ("T14 [h]", "T_tra_tot"), ("depth [ppt]", "depth_tr_undil_lc")):
        samples = d(key)
        if samples is not None:
            band = results.star_density if key == "host_density" and results.star_density and np.isfinite(results.star_density[1]) else None
            columns.append((label, samples, band))
    period = results.samples[f"{planet}_period"].to_numpy()
    columns.append((f"P − {np.median(period):.5f} [10⁻⁶ d]", (period - np.median(period)) * 1e6, None))
    return columns


def page_corner(results: FitResults, planet, page):
    fig = _new_figure()
    _head(fig, results, f"Posteriors and correlations · {planet}",
          "Weighted samples of the nested sampling · diagonal: marginal posterior (68 % shaded; grey band: the stellar density of the star) · "
          "off diagonal: 68 and 95 % contours", page)
    columns = corner_columns(results, planet)
    n = len(columns)
    size = .108
    for i in range(n):
        for j in range(i + 1):
            ax = _axes(fig, .06 + j * (size + .008), .08 + (n - 1 - i) * (size + .012) * 1.17, size, size * 1.17)
            xi, yi = np.asarray(columns[j][1], float), np.asarray(columns[i][1], float)
            if i == j:
                counts, edges = np.histogram(xi, bins=40)
                q = percentiles(xi)
                if columns[i][2] is not None:
                    mu, sigma = columns[i][2]
                    ax.axvspan(mu - sigma, mu + sigma, color=GRID, lw=0)
                ax.fill_between((edges[1:] + edges[:-1]) / 2, 0, counts, color=BLUE, alpha=.7, step="mid")
                ax.axvspan(q[1], q[3], color=BLUE, alpha=.2)
                ax.axvline(q[2], color=INK, lw=.8)
                ax.set_yticks([])
            else:
                counts, xe, ye = np.histogram2d(xi, yi, bins=30)
                ax.pcolormesh(xe, ye, counts.T, cmap="Blues", rasterized=True)
                flat = np.sort(counts.ravel())[::-1]
                cumulative = np.cumsum(flat) / flat.sum()
                levels = sorted({flat[min(np.searchsorted(cumulative, p), len(flat) - 1)] for p in (.95, .68)})
                if len(levels) and levels[-1] > 0:
                    ax.contour((xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2, counts.T, levels=levels, colors=[INK], linewidths=.6)
            ax.tick_params(labelsize=5.5)
            if i == n - 1:
                ax.set_xlabel(columns[j][0], fontsize=6.5)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(columns[i][0], fontsize=6.5)
            elif j > 0:
                ax.set_yticklabels([])
    stack = np.vstack([np.asarray(c[1], float) for c in columns])
    corr = np.corrcoef(stack)
    pairs = sorted(((abs(corr[i, j]), columns[i][0], columns[j][0], corr[i, j]) for i in range(n) for j in range(i)), reverse=True)[:5]
    fig.text(.60, .86, "STRONGEST CORRELATIONS", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.60, .97], [.855, .855], color=GRID, lw=.6))
    for k, (_, a_, b_, c_) in enumerate(pairs):
        fig.text(.60, .83 - k * .027, f"{a_}  ×  {b_}", fontsize=7.2)
        fig.text(.93, .83 - k * .027, f"{c_:+.2f}", fontsize=7.2, fontweight="bold", color=WARN if abs(c_) > .8 else INK)
    density = results.derived_of(planet, "host_density")
    if density is not None and results.star_density is not None:
        q = percentiles(density)
        fig.text(.60, .68, "STELLAR DENSITY [g/cm³]", fontsize=6.5, color=MUTED, fontweight="bold")
        fig.add_artist(Line2D([.60, .97], [.675, .675], color=GRID, lw=.6))
        fig.text(.60, .648, "from the orbit", fontsize=7.2)
        fig.text(.75, .648, f"{q[2]:.2f} +{q[3] - q[2]:.2f} / −{q[2] - q[1]:.2f}", fontsize=7.2, fontweight="bold")
        fig.text(.60, .622, "from the star", fontsize=7.2)
        star = results.star_density
        fig.text(.75, .622, f"{star[0]:.2f} ± {star[1]:.2f}" if np.isfinite(star[1]) else f"{star[0]:.2f}", fontsize=7.2, fontweight="bold")
        combined = results.combined_density
        if combined is not None:
            qc = percentiles(combined)
            fig.text(.60, .596, "combined (all orbits)", fontsize=7.2)
            fig.text(.75, .596, f"{qc[2]:.2f} +{qc[3] - qc[2]:.2f} / −{qc[2] - qc[1]:.2f}", fontsize=7.2, fontweight="bold")
        fig.text(.60, .568, "The fit uses the density of the star as a prior on the orbit.", fontsize=6.5, color=MUTED)
    fig.text(.60, .52, "DERIVED PARAMETERS", fontsize=6.5, color=MUTED, fontweight="bold")
    fig.add_artist(Line2D([.60, .97], [.515, .515], color=GRID, lw=.6))
    derived_rows = [("Planet radius [R⊕]", "R_companion_(R_earth)", "{:.2f}"), ("Depth [ppt]", "depth_tr_undil_lc", "{:.3f}"),
                    ("Impact parameter b", "b_tra", "{:.2f}"), ("a / R★", "a/R_star", "{:.2f}"), ("Inclination [°]", "i", "{:.2f}"),
                    ("Total duration [h]", "T_tra_tot", "{:.3f}"), ("Full duration [h]", "T_tra_full", "{:.3f}"),
                    ("Semi-major axis [AU]", "a_(AU)", "{:.4f}"), ("Planet radius [R_jup]", "R_companion_(R_jup)", "{:.3f}")]
    shown = 0
    for label, key, fmt in derived_rows:
        samples = results.derived_of(planet, key)
        if samples is None:
            continue
        q = percentiles(samples)
        y = .488 - shown * .027
        fig.text(.60, y, label, fontsize=7.2)
        fig.text(.76, y, fmt.format(q[2]) + f"  +{fmt.format(q[3] - q[2])} / −{fmt.format(q[2] - q[1])}", fontsize=7.2, fontweight="bold")
        shown += 1
    return fig


def page_convergence(results: FitResults, page):
    fig = _new_figure()
    _head(fig, results, "Did the sampling converge and is the noise understood?",
          "Nested sampling diagnostics · residuals of the posterior median model, without the Gaussian process baseline", page)
    w = results.weights
    a = _axes(fig, .06, .55, .20, .33)
    a.plot(np.arange(len(results.logz_curve)), results.logz_curve, color=BLUE, lw=1.2)
    a.set_xlabel("iteration")
    a.set_ylabel("ln Z")
    a.grid(color=GRID, lw=.4)
    a.set_ylim(results.logz - 60, results.logz + 5)
    _title(a, "Evidence", f"{results.logz:.2f} ± {results.logzerr:.2f}")
    a = _axes(fig, .335, .55, .20, .33)
    a.scatter(-results.logvol, results.logl - results.logl.max(), c=w / w.max(), s=2, cmap="viridis_r", lw=0, rasterized=True)
    a.set_xlabel("−ln X (prior volume)")
    a.set_ylabel("ln L − max")
    a.set_ylim(-80, 3)
    a.grid(color=GRID, lw=.4)
    _title(a, "Likelihood vs volume", "colour: posterior weight")
    a = _axes(fig, .61, .55, .16, .33)
    a.hist(w * 1e3, bins=40, color=BLUE)
    a.set_xlabel("weight × 1000")
    a.set_ylabel("samples")
    _title(a, "Weights", f"ESS {results.ess:.0f}")
    a = _axes(fig, .82, .55, .15, .33)
    a.axis("off")
    checks = convergence_checks(results)
    facts = [("Effective samples", f"{results.ess:.0f}", checks[0].status), ("Iterations", f"{results.niter}", "ok"),
             ("Likelihood calls", f"{results.ncall:,}", "ok"), ("Sampling efficiency", f"{results.efficiency:.1f} %", "ok" if results.efficiency > 2 else "warn"),
             ("Information (nats)", f"{results.information:.1f}", "ok")]
    for k, (label, value, status) in enumerate(facts):
        a.text(0, .95 - k * .19, label, fontsize=7.2, color=MUTED)
        a.text(.86, .95 - k * .19, value, fontsize=7.8, fontweight="bold", ha="right")
        a.add_patch(Circle((.96, .965 - k * .19), .03, color=STATUS_COLOURS[status], transform=a.transAxes))
    _title(a, "Checks")
    residuals = results.residuals[np.isfinite(results.residuals)]
    sigma = np.std(residuals)
    a = _axes(fig, .06, .08, .20, .33)
    a.hist(residuals / sigma, bins=np.linspace(-5, 5, 61), density=True, color=BLUE, alpha=.8)
    x = np.linspace(-5, 5, 200)
    a.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=BAD, lw=1)
    a.set_xlabel("residual / σ")
    a.grid(color=GRID, lw=.4)
    _title(a, "Residual distribution", f"σ = {sigma * 1e6:.0f} ppm")
    a = _axes(fig, .335, .08, .20, .33)
    sizes = np.unique(np.logspace(0, 2.6, 18).astype(int))
    rms = []
    for k in sizes:
        m = len(residuals) // k
        rms.append(np.std(residuals[:m * k].reshape(m, k).mean(axis=1)) if m > 2 else np.nan)
    a.loglog(sizes, np.array(rms) * 1e6, "o-", color=INK, ms=3, lw=.8, label="residuals")
    a.loglog(sizes, sigma / np.sqrt(sizes) * 1e6, "--", color=BAD, label="white noise")
    a.set_xlabel("points per bin")
    a.set_ylabel("rms (ppm)")
    a.grid(color=GRID, lw=.4)
    a.legend(frameon=False, fontsize=6.5)
    beta = red_noise_beta(residuals)
    _title(a, "Red noise", "β = n/a" if not np.isfinite(beta) else f"β ≈ {beta:.2f}")
    a = _axes(fig, .61, .08, .16, .33)
    lags = np.arange(1, 200)
    centred = residuals - residuals.mean()
    acf = np.array([np.mean(centred[:-lag] * centred[lag:]) for lag in lags]) / np.var(centred)
    a.bar(lags, acf, color=BLUE, width=1)
    a.axhline(0, color=INK, lw=.5)
    a.set_xlabel("lag (cadences)")
    a.set_ylabel("autocorrelation")
    a.grid(color=GRID, lw=.4)
    _title(a, "Residual autocorrelation")
    a = _axes(fig, .82, .08, .15, .33)
    a.axis("off")
    a.text(0, .95, "NOISE MODEL", fontsize=6.5, color=MUTED, fontweight="bold")
    samples = results.samples
    rows = [("Points", f"{len(results.time)}")]
    if "ln_err_flux_lc" in samples:
        rows.insert(0, ("White noise", f"ln σ {np.median(samples['ln_err_flux_lc']):.2f}"))
    if "baseline_gp_matern32_lnsigma_flux_lc" in samples:
        rows.insert(1, ("GP kernel", "Matérn-3/2"))
        rows.insert(2, ("GP ln σ", f"{np.median(samples['baseline_gp_matern32_lnsigma_flux_lc']):.2f}"))
        rows.insert(3, ("GP ln ρ", f"{np.median(samples['baseline_gp_matern32_lnrho_flux_lc']):.2f}"))
    for k, (label, value) in enumerate(rows):
        a.text(0, .82 - k * .15, label, fontsize=7.2, color=MUTED)
        a.text(.5, .82 - k * .15, value, fontsize=7.4, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------------------------- writer
def _stub(results, title, error, page):
    fig = _new_figure()
    _head(fig, results, title, "This page could not be built", page)
    fig.text(.5, .5, f"Not available: {error}", ha="center", color=MUTED, fontsize=9)
    return fig


def build_pages(results: FitResults):
    """All the pages of the report; a page that fails is replaced by a stub that says why."""
    pages = []

    def add(title, builder):
        try:
            with matplotlib.rc_context(report_style()):
                pages.append(builder(len(pages) + 1))
        except Exception as error:
            logging.exception("The page '%s' of the fit report failed", title)
            pages.append(_stub(results, title, f"{type(error).__name__}: {error}", len(pages) + 1))

    for planet in results.planets:
        add(f"Summary {planet}", lambda n, p=planet: page_summary(results, p, n))
        add(f"Consistency {planet}", lambda n, p=planet: page_consistency(results, p, n))
        add(f"Corner {planet}", lambda n, p=planet: page_corner(results, p, n))
    add("Convergence and noise", lambda n: page_convergence(results, n))
    return pages


def write_fit_report(results: FitResults, pdf_file):
    """Writes the report (``.partial`` first, then moved into place) and returns its number of pages."""
    if not results.planets:
        raise ValueError("The fit has no fitted candidates")
    pages = build_pages(results)
    partial = pdf_file + ".partial"
    with matplotlib.rc_context({**report_style(), "pdf.fonttype": 42}), PdfPages(partial) as pdf:
        for fig in pages:
            pdf.savefig(fig)
        info = pdf.infodict()
        info["Title"] = f"Bayesian fit of {results.object_id}"
        info["CreationDate"] = datetime.datetime.now()
    os.replace(partial, pdf_file)
    return len(pages)


def write_fit_report_guarded(fit_dir, object_id, star_row, candidates, pdf_file=None):
    """
    Reads the fit and writes its report; it never raises, because the report must not break a fit.

    :return: the path of the PDF, or None when it could not be written
    """
    partial = None
    try:
        from sherlockpipe.bayesian_fit.fit_results import load_fit_results
        pdf_file = pdf_file or os.path.join(fit_dir, f"{object_id}_fit.pdf")
        partial = pdf_file + ".partial"
        results = load_fit_results(fit_dir, object_id, star_row, candidates)
        pages = write_fit_report(results, pdf_file)
        logging.info("Fit report written with %s pages: %s", pages, pdf_file)
        return pdf_file
    except (Exception, SystemExit):
        logging.exception("The fit report could not be written; the results are in %s", fit_dir)
        if partial and os.path.exists(partial):
            os.remove(partial)
        return None
