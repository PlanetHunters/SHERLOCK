"""
Target information report.

Creates ``<object_id>_info.pdf`` (A4 landscape) in the execution directory of a SHERLOCK target. It summarises what
happened before the transit search: star parameters, data used and the alternatives that were available, the
preprocessing steps and their measurable effect, the detrends, the periodicity of the light curve and the field of
view.

The report is built from the files SHERLOCK and its dependencies already write in the execution directory and from the
search configuration, so it can also be generated afterwards for an existing execution (see :func:`main`). Every
section is optional: a missing input skips that section and never prevents the rest of the report.
"""
import glob
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#e3e7ec"
ACCENT = "#2a6f97"
MASK_COLOR = "#d1495b"
USED_BG = "#dcebf5"
DETREND_COLORS = ["#2a6f97", "#3a8f5c", "#c98a1b", "#d1495b", "#7b5ea7", "#555555", "#1b9aaa", "#a44a3f"]
#: Product columns of the "alternative data" table, in display order.
AVAILABILITY_COLUMNS = ["SPOC", "SPOC TPF", "TESS-SPOC", "QLP", "TASOC", "TARS", "TESScut (FFI)"]
#: Sector count thresholds that decide how the "alternative data" table is laid out.
SINGLE_TABLE_MAX_SECTORS = 12
BLOCKS_TABLE_MAX_SECTORS = 24
TESS_BTJD_OFFSET = 2457000.0
LOGO_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "images", "sherlock3.png")


# --------------------------------------------------------------------------------------------------------------------
# Data layer: parsers and measurements. No PDF here, so it can be tested without rendering anything.
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """Preprocessing and search settings that the report needs, whatever their origin."""
    setup_rows: list = field(default_factory=list)
    initial_mask: list = field(default_factory=list)
    transit_mask: list = field(default_factory=list)
    outliers_sigma_upper: float = None
    outliers_sigma_lower: float = None
    initial_trim: float = None
    high_rms_enabled: bool = None
    high_rms_threshold: float = None
    high_rms_bin_hours: float = None
    oscillations_enabled: bool = None
    autodetrend_enabled: bool = None
    autodetrend_ratio: float = None
    autodetrend_method: str = None
    window_lengths: list = None
    detrend_method: str = None


def _glob(directory, pattern):
    """``glob`` inside a directory whose own name may hold glob characters: ids such as ``TIC1_[2]`` are common."""
    return glob.glob(os.path.join(glob.escape(directory), pattern))


def parse_report_log(text):
    """
    Reads the SHERLOCK report log of an execution. Only the last execution found in the file is considered.

    :param str text: the content of ``<object_id>_report.log``
    :return: a dict with ``availability``, ``user_definitions``, ``initial_mask``, ``transit_mask`` and
        ``window_range``.
    """
    start = text.rfind("SHERLOCK (Searching")
    text = text[start:] if start >= 0 else text
    user_definitions = {}
    if "USER DEFINITIONS" in text:
        block = text.split("USER DEFINITIONS", 1)[1].split("Retrieving star catalog", 1)[0]
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                user_definitions[key.strip()] = value.strip()
    initial_mask = [[float(a), float(b)] for a, b in
                    re.findall(r"Initial mask since day (\d+\.\d+) to day (\d+\.\d+)", text)]
    transit_mask = [{"P": float(p), "T0": float(t0), "D": float(d)} for p, t0, d in
                    re.findall(r"Transit mask with P=([\d.]+) d, T0=([\d.]+) d, Dur=([\d.]+) min", text)]
    window = dict(re.findall(r"wl/ks_(min|max): ([\d.]+)", text))
    return {"availability": parse_available_data(text), "user_definitions": user_definitions,
            "initial_mask": initial_mask, "transit_mask": transit_mask,
            "window_range": (float(window["min"]), float(window["max"])) if len(window) == 2 else None}


def parse_available_data(text):
    """
    Reads the ``There are <kind> data`` lines logged by lcbuilder.

    :param str text: log content
    :return: ``{sector: {column: {exposure_seconds, ...}}}`` where column is one of :data:`AVAILABILITY_COLUMNS`
        (or the author name for sources that are not in that list)
    """
    availability = {}
    pattern = r"There are (TPF|TessCut|LightCurve) data: TESS Sector (\d+), Year \d+, Author: ([^,]+), " \
              r"ExpTime: ([\d.]+)"
    for kind, sector, author, exposure in re.findall(pattern, text):
        column = {"TPF": "SPOC TPF", "TessCut": "TESScut (FFI)"}.get(kind, author.strip())
        availability.setdefault(int(sector), {}).setdefault(column, set()).add(int(round(float(exposure))))
    return availability


def read_used_products(object_dir):
    """
    Reads which products were used from ``apertures.yaml`` (sector -> author -> exposure).

    :return: ``{sector: {(author, exposure_seconds), ...}}``, empty if the file is missing
    """
    path = os.path.join(object_dir, "apertures.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        content = yaml.safe_load(f) or {}
    used = {}
    for sector, authors in (content.get("sectors") or {}).items():
        for author, exposures in (authors or {}).items():
            for exposure in (exposures or {}):
                used.setdefault(int(sector), set()).add((str(author), int(exposure)))
    return used


def availability_rows(availability, used):
    """
    Turns the availability dict into table rows, one per sector.

    :return: list of dicts ``{label, sectors, cells: {column: text}, used_columns: {column}, in_use}``
    """
    rows = []
    for sector in sorted(availability):
        cells = {}
        used_columns = set()
        for column in AVAILABILITY_COLUMNS:
            exposures = sorted(availability[sector].get(column, []))
            used_exposures = {e for a, e in used.get(sector, set()) if column == a}
            if exposures:
                cells[column] = " · ".join(f"[{e}]" if e in used_exposures else str(e) for e in exposures)
            else:
                cells[column] = "–"
            if used_exposures & set(exposures):
                used_columns.add(column)
        rows.append({"label": str(sector), "sectors": [sector], "cells": cells, "used_columns": used_columns,
                     "in_use": sector in used})
    return rows


def group_availability_rows(rows, max_rows=BLOCKS_TABLE_MAX_SECTORS):
    """
    Collapses consecutive sectors that have exactly the same availability and use into a single range row, so a target
    observed in dozens of sectors does not produce a huge table. Nothing is collapsed up to ``max_rows`` rows.
    """
    if len(rows) <= max_rows:
        return rows
    grouped = []
    for row in rows:
        signature = (tuple(sorted(row["cells"].items())), tuple(sorted(row["used_columns"])), row["in_use"])
        if grouped and grouped[-1]["signature"] == signature and grouped[-1]["sectors"][-1] + 1 == row["sectors"][0]:
            grouped[-1]["sectors"].append(row["sectors"][0])
        else:
            grouped.append({**row, "sectors": list(row["sectors"]), "signature": signature})
    for row in grouped:
        row.pop("signature", None)
        row["label"] = str(row["sectors"][0]) if len(row["sectors"]) == 1 \
            else f"{row['sectors'][0]}–{row['sectors'][-1]}"
    return grouped


def split_blocks(rows, single_max=SINGLE_TABLE_MAX_SECTORS, block_max=SINGLE_TABLE_MAX_SECTORS):
    """
    Splits rows into blocks: a single one up to ``single_max`` rows, otherwise blocks of at most ``block_max`` rows
    that are meant to be laid out side by side.
    """
    if len(rows) <= single_max:
        return [rows]
    size = int(np.ceil(len(rows) / np.ceil(len(rows) / block_max)))
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def assign_sectors(pre, lc):
    """Adds the ``sector`` column of ``lc.csv`` to the pre-detrend curve (nearest time, exact in practice)."""
    if "sector" not in lc.columns or lc["sector"].isna().all():
        pre = pre.copy()
        pre["sector"] = 0.0
        return pre
    sectors = lc[["time", "sector"]].dropna().sort_values("time")
    merged = pd.merge_asof(pre.sort_values("time"), sectors, on="time", direction="nearest", tolerance=0.01)
    merged["sector"] = merged["sector"].fillna(-1)
    return merged


def sector_rows(lc, pre, tess=True):
    """
    Per sector data summary. Only quantities that can be measured exactly from the execution files are reported.

    :return: list of dicts with ``sector, start, end, points, used, kept, rms_ppm, median_err_ppm``
    """
    rows = []
    pre = assign_sectors(pre, lc)
    lc = lc.copy()
    if "sector" not in lc.columns or lc["sector"].isna().all():
        lc["sector"] = 0.0
    for sector, group in lc.dropna(subset=["sector"]).groupby("sector"):
        used = pre[pre["sector"] == sector]
        rows.append({"sector": int(sector), "start": _format_time(group["time"].min(), tess),
                     "end": _format_time(group["time"].max(), tess), "points": len(group), "used": len(used),
                     "kept": 100 * len(used) / len(group) if len(group) else np.nan,
                     "rms_ppm": np.std(used["flux"]) * 1e6 if len(used) else np.nan,
                     "median_err_ppm": np.median(used["flux_err"]) * 1e6 if len(used) else np.nan})
    return rows


def _format_time(btjd, tess):
    if not tess:
        return f"{btjd:.2f}"
    from astropy.time import Time
    return Time(btjd + TESS_BTJD_OFFSET, format="jd").iso[:10]


def transit_mask(time, period, duration, t0):
    """Same definition as ``transitleastsquares.transit_mask`` (duration in days)."""
    return np.abs((time - t0 + 0.5 * period) % period - 0.5 * period) < 0.5 * duration


def measure_preprocessing(lc, pre, config):
    """
    Reproduces the point-removing steps that lcbuilder applies to ``lc.csv`` (written after the initial trim and
    before outlier clipping) in the same order, and counts what each one removes. The final count is compared with
    the real ``lc_predetrends.csv`` so anything not explained here (high RMS mask, NaN removal, smoothing...) is
    reported as such instead of being hidden.

    :return: list of ``(step, setting, removed_points)``; ``removed_points`` is None when it cannot be measured.
    """
    steps = []
    time = lc["time"].to_numpy()
    flux = lc["flux"].to_numpy()
    flux_err = lc["flux_err"].to_numpy()
    steps.append(("Data after initial trim", "" if config.initial_trim is None
                  else f"first {config.initial_trim:g} d kept", len(time)))
    if config.outliers_sigma_upper is not None or config.outliers_sigma_lower is not None:
        upper = config.outliers_sigma_upper if config.outliers_sigma_upper is not None else float("inf")
        lower = config.outliers_sigma_lower if config.outliers_sigma_lower is not None else float("inf")
        try:
            import lightkurve
            clipped = lightkurve.LightCurve(time=time, flux=flux, flux_err=flux_err) \
                .remove_outliers(sigma_lower=lower, sigma_upper=upper)
            removed = len(time) - len(clipped)
            time, flux, flux_err = clipped.time.value, clipped.flux.value, clipped.flux_err.value
        except Exception:
            logging.exception("Outliers could not be recomputed for the target info report")
            removed = None
        steps.append(("Outlier clipping", f"upper {upper:g} σ" + ("" if lower == float("inf") else f", lower {lower:g} σ"),
                      removed))
    for a, b in config.initial_mask or []:
        inside = (time >= a) & (time <= b)
        steps.append(("Initial mask", f"BTJD {a:.2f} – {b:.2f}", int(inside.sum())))
        time, flux, flux_err = time[~inside], flux[~inside], flux_err[~inside]
    for mask in config.transit_mask or []:
        inside = transit_mask(time, mask["P"], mask["D"] / 60 / 24, mask["T0"])
        steps.append(("Transit mask", f"P={mask['P']:g} d, T0={mask['T0']:g}, D={mask['D']:g} min", int(inside.sum())))
        time, flux, flux_err = time[~inside], flux[~inside], flux_err[~inside]
    unexplained = int(np.sum(~np.isnan(flux))) - len(pre)
    if unexplained != 0:
        steps.append(("Other steps", "high RMS mask, NaN removal, smoothing...", unexplained))
    return steps


def read_pulsations(object_dir):
    """Reads ``sa/signals.csv`` (stellar oscillations removed by lcbuilder) and the plots of each signal."""
    path = os.path.join(object_dir, "sa", "signals.csv")
    if not os.path.exists(path):
        return None
    signals = pd.read_csv(path)
    for index, row in signals.iterrows():
        folder = os.path.join(object_dir, "sa", str(int(row["number"])))
        signals.loc[index, "folded"] = os.path.join(folder, "folded_curve.png")
        signals.loc[index, "periodogram"] = os.path.join(folder, "Periodogram.png")
    return signals


def read_autodetrend(object_dir):
    """Returns ``(period_days, png_path)`` of the auto-detrend phase plot, or None if it was not produced."""
    for path in sorted(_glob(object_dir, "Phase_detrend_period_*.png")):
        match = re.search(r"_(\d+\.\d+)_days", os.path.basename(path))
        return (float(match.group(1)) if match else None), path
    return None


def read_rms_mask_plot(object_dir):
    """Path of the overall high RMS mask plot (the per time-window ones are not used), or None."""
    paths = [p for p in _glob(object_dir, os.path.join("rms_mask", "High_RMS_Mask_*.png"))
             if "_time_" not in os.path.basename(p)]
    return sorted(paths)[0] if paths else None


def config_from_sherlock_target(sherlock_target, window_lengths=None):
    """Builds the report configuration from a :class:`sherlockpipe.search.sherlock_target.SherlockTarget`."""
    info = sherlock_target.object_info
    lower = getattr(info, "lower_outliers_sigma", None)
    return ReportConfig(
        setup_rows=[("Detrend method", str(sherlock_target.detrend_method)),
                    ("Detrend models", str(sherlock_target.detrends_number)),
                    ("Period range (d)", f"{sherlock_target.period_min} – {sherlock_target.period_max}"),
                    ("Binning (min)", str(sherlock_target.bin_minutes)),
                    ("SNR / SDE thresholds", f"{sherlock_target.snr_min} / {sherlock_target.sde_min}"),
                    ("Signal scoring", str(sherlock_target.best_signal_algorithm)),
                    ("Oversampling", str(sherlock_target.oversampling))],
        initial_mask=[list(m) for m in (getattr(info, "initial_mask", None) or [])],
        transit_mask=list(getattr(info, "initial_transit_mask", None) or []),
        outliers_sigma_upper=getattr(info, "outliers_sigma", None),
        outliers_sigma_lower=None if lower is None or np.isinf(lower) else lower,
        initial_trim=getattr(info, "initial_trim", None),
        high_rms_enabled=getattr(info, "high_rms_enabled", None),
        high_rms_threshold=getattr(info, "high_rms_threshold", None),
        high_rms_bin_hours=getattr(info, "high_rms_bin_hours", None),
        oscillations_enabled=getattr(info, "reduce_simple_oscillations", None),
        autodetrend_enabled=getattr(info, "auto_detrend_enabled", None),
        autodetrend_ratio=getattr(info, "auto_detrend_ratio", None),
        autodetrend_method=getattr(info, "auto_detrend_method", None),
        window_lengths=None if window_lengths is None else [float(w) for w in window_lengths if float(w) > 0],
        detrend_method=sherlock_target.detrend_method)


def config_from_log(parsed_log, window_lengths=None):
    """Builds a (partial) report configuration from a parsed report log, for executions without the objects."""
    definitions = parsed_log["user_definitions"]
    wanted = [("Detrend method", "Detrend method"), ("Detrend models", "No of detrend models applied"),
              ("Minimum period (d)", "Minimum Period (d)"), ("Maximum period (d)", "Maximum Period (d)"),
              ("Binning (min)", "Binning size (min)"), ("SNR threshold", "Threshold limit for SNR"),
              ("SDE threshold", "Threshold limit for SDE"), ("Signal scoring", "Signal scoring algorithm"),
              ("Oversampling", "Oversampling")]
    sigma = definitions.get("Sigma threshold for upper outliers clipping")
    return ReportConfig(setup_rows=[(label, definitions[key]) for label, key in wanted if key in definitions],
                        initial_mask=parsed_log["initial_mask"], transit_mask=parsed_log["transit_mask"],
                        outliers_sigma_upper=float(sigma) if sigma else None, window_lengths=window_lengths,
                        detrend_method=definitions.get("Detrend method"))


def window_lengths_from_runs(object_dir, object_id):
    """Window lengths of the detrends, recovered from the ``Run_1_<ws|ks>=<value>_<id>.png`` file names."""
    values = []
    for path in _glob(object_dir, os.path.join("1", f"Run_1_*=*_{glob.escape(object_id)}.png")):
        match = re.search(r"_(?:ws|ks)=([\d.]+)_", os.path.basename(path))
        if match:
            values.append(float(match.group(1)))
    return sorted(values) or None


# --------------------------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------------------------

def _new_axes(figure, rows, cols, **kwargs):
    axes = figure.subplots(rows, cols, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=MUTED)
    return axes


def _binned(x, y, width):
    bins = np.floor((x - x.min()) / width).astype(int)
    frame = pd.DataFrame({"b": bins, "x": x, "y": y}).groupby("b").mean()
    return frame["x"].to_numpy(), frame["y"].to_numpy()


def _sector_groups(pre):
    return [(int(s), g) for s, g in pre.groupby("sector") if s >= 0 and len(g) > 0]


def figure_sector_curves(pre, initial_masks, path):
    """One panel per sector with the preprocessed curve, 20 min bins and the initial masks."""
    groups = _sector_groups(pre)
    figure = Figure(figsize=(13.2, 2.5), constrained_layout=True)
    axes = _new_axes(figure, 1, len(groups), sharey=True, squeeze=False)
    for ax, (sector, group) in zip(axes, groups):
        ax.scatter(group["time"], group["flux"], s=0.6, color=MUTED, alpha=0.12, lw=0, rasterized=True)
        bx, by = _binned(group["time"].to_numpy(), group["flux"].to_numpy(), 20 / 1440)
        ax.plot(bx, by, ".", ms=1.6, color=INK)
        for a, b in initial_masks or []:
            if b > group["time"].min() - 1 and a < group["time"].max() + 1:
                ax.axvspan(a, b, color=MASK_COLOR, alpha=0.18, lw=0)
        ax.set_title(f"Sector {sector}" if len(groups) > 1 or sector else "Light curve", fontsize=9, loc="left")
        ax.set_xlim(group["time"].min() - 0.5, group["time"].max() + 0.5)
    axes[0].set_ylabel("normalised flux", fontsize=8)
    low, high = np.nanpercentile(pre["flux"], [0.5, 99.5])
    axes[0].set_ylim(low - 0.001, high + 0.001)
    figure.savefig(path, dpi=170)


def read_detrends(object_dir, pre):
    """
    Reads ``lc_<i>.csv`` (i >= 1 are the detrended curves) and derives the trend each one removed.

    :return: list of ``(index, DataFrame)`` with ``time, flux`` (detrended), ``trend``
    """
    detrends = []
    for path in sorted(_glob(object_dir, "lc_[0-9]*.csv"),
                       key=lambda p: int(re.search(r"lc_(\d+)\.csv", p).group(1))):
        index = int(re.search(r"lc_(\d+)\.csv", path).group(1))
        if index == 0:
            continue
        frame = pd.read_csv(path)[["time", "flux"]].rename(columns={"flux": "detrended"})
        merged = pre[["time", "flux"]].merge(frame, on="time", how="inner")
        merged["trend"] = merged["flux"] / merged["detrended"]
        detrends.append((index, merged))
    return detrends


def figure_detrends(pre, detrends, window_lengths, path):
    """One panel per sector: the curve in 1 h bins and the trend that every detrend window removed from it."""
    groups = _sector_groups(pre)
    figure = Figure(figsize=(13.2, 2.7), constrained_layout=True)
    axes = _new_axes(figure, 1, len(groups), sharey=True, squeeze=False)
    labels = [f"{w:.2f} d" for w in window_lengths] if window_lengths and len(window_lengths) == len(detrends) \
        else [f"detrend {i}" for i, _ in detrends]
    for ax, (sector, group) in zip(axes, groups):
        t0, t1 = group["time"].min(), group["time"].max()
        bx, by = _binned(group["time"].to_numpy(), group["flux"].to_numpy(), 60 / 1440)
        ax.plot(bx, by, ".", ms=1.6, color=MUTED, alpha=0.6, rasterized=True)
        for (index, frame), color, label in zip(detrends, DETREND_COLORS * 4, labels):
            inside = frame[frame["time"].between(t0, t1)]
            if len(inside):
                tx, ty = _binned(inside["time"].to_numpy(), inside["trend"].to_numpy(), 60 / 1440)
                ax.plot(tx, ty, color=color, lw=1.1, label=label)
        ax.set_title(f"Sector {sector}" if len(groups) > 1 or sector else "Light curve", fontsize=9, loc="left")
    axes[0].set_ylabel("normalised flux", fontsize=8)
    low, high = np.nanpercentile(pre["flux"], [0.5, 99.5])
    axes[0].set_ylim(low - 0.0005, high + 0.0005)
    axes[-1].legend(frameon=False, fontsize=7, title="window length", title_fontsize=7, loc="upper right", ncol=2)
    figure.savefig(path, dpi=170)


def rasterize_pdf(pdf_path, png_path, dpi=130):
    """
    Renders the first page of a PDF to PNG with pypdfium2, which ships its own PDFium and needs no system library.

    :return: False, after a warning, if pypdfium2 is not installed
    """
    try:
        import pypdfium2
    except ImportError:
        logging.warning("pypdfium2 is not installed: the TRICERATOPS field of view PDFs are left out of the info report")
        return False
    document = pypdfium2.PdfDocument(pdf_path)
    try:
        document[0].render(scale=dpi / 72).to_pil().save(png_path)
    finally:
        document.close()
    return True


# --------------------------------------------------------------------------------------------------------------------
# PDF layer
# --------------------------------------------------------------------------------------------------------------------

def _styles():
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=8.5, leading=11, textColor=colors.HexColor(INK))
    return {"base": base,
            "small": ParagraphStyle("small", parent=base, fontSize=7.5, textColor=colors.HexColor(MUTED)),
            "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=9.5,
                                 textColor=colors.HexColor(ACCENT), spaceBefore=6, spaceAfter=3)}


def _table(rows, widths, header=True, extra_style=None):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8),
             ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor(GRID))]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(MUTED))]
    table.setStyle(TableStyle(style + (extra_style or [])))
    return table


def _image(path, width):
    from PIL import Image as PilImage
    with PilImage.open(path) as image:
        w, h = image.size
    return Image(path, width=width, height=width * h / w)


def _fmt(value, spec, empty="–"):
    return empty if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, spec)


class TargetInfoReport:
    """
    Builds ``<object_id>_info.pdf`` for one execution directory.

    :param str object_dir: the execution directory of the target
    :param str object_id: the SHERLOCK id of the target (e.g. ``TIC224298134_all``)
    :param SherlockTarget sherlock_target: optional, the configuration used. If it is not given the settings are read
        from the report log of the execution.
    :param list window_lengths: optional, the window lengths of the detrends
    """

    def __init__(self, object_dir, object_id, sherlock_target=None, window_lengths=None):
        self.object_dir = object_dir.rstrip("/")
        self.object_id = object_id
        self.sherlock_target = sherlock_target
        self.window_lengths = window_lengths
        self.styles = _styles()
        self.tmp_dir = None

    # ------------------------------------------------------------------ inputs
    def _read(self, name):
        path = os.path.join(self.object_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    def _log(self):
        path = os.path.join(self.object_dir, f"{self.object_id}_report.log")
        if not os.path.exists(path):
            return {"availability": {}, "user_definitions": {}, "initial_mask": [], "transit_mask": [],
                    "window_range": None}
        with open(path, errors="ignore") as f:
            return parse_report_log(f.read())

    def _load(self):
        self.lc = self._read("lc.csv")
        self.pre = self._read("lc_predetrends.csv")
        self.star = self._read("params_star.csv")
        self.parsed_log = self._log()
        windows = self.window_lengths or window_lengths_from_runs(self.object_dir, self.object_id)
        if self.sherlock_target is not None:
            self.config = config_from_sherlock_target(self.sherlock_target, windows)
        else:
            self.config = config_from_log(self.parsed_log, windows)
        self.tess = self.object_id.upper().startswith("TIC")
        self.used = read_used_products(self.object_dir)
        self.sectors = sector_rows(self.lc, self.pre, self.tess) if self.lc is not None and self.pre is not None else []
        if self.pre is not None and self.lc is not None:
            self.pre = assign_sectors(self.pre, self.lc)

    # ------------------------------------------------------------------ sections
    def _star_table(self):
        if self.star is None:
            return []
        s = self.star.iloc[0]

        def pm(prefix, nd=3):
            return f"{s[prefix]:.{nd}f} +{s[prefix + '_uerr']:.{nd}f}/-{s[prefix + '_lerr']:.{nd}f}"
        rows = [["Star", ""], ["Object", str(s["obj_id"])], ["RA / Dec (deg)", f"{s['ra']:.4f} / {s['dec']:.4f}"],
                ["R★ (Rsun)", pm("R_star")], ["M★ (Msun)", pm("M_star")],
                ["Teff (K)", f"{s['Teff_star']:.0f} ± {s['Teff_star_lerr']:.0f}"],
                ["log g", _fmt(s.get("logg"), ".2f")], ["Luminosity (Lsun)", _fmt(s.get("lum"), ".3f")],
                ["Limb darkening (a, b)", f"{s['ld_a']:.3f}, {s['ld_b']:.3f}"],
                ["V / J / H / K", " / ".join(_fmt(s.get(k), ".2f") for k in ("v", "j", "h", "k"))]]
        return [_table(rows, [4.1 * cm, 5.9 * cm])]

    def _setup_table(self):
        rows = [["Search setup", ""]] + [[k, v] for k, v in self.config.setup_rows]
        wl = self.parsed_log["window_range"]
        if wl:
            rows.append(["Detrend window range (d)", f"{wl[0]:.2f} – {wl[1]:.2f}"])
        return [_table(rows, [5.2 * cm, 4.8 * cm])] if len(rows) > 1 else []

    def _data_used_table(self, width):
        if not self.sectors:
            return []
        rows = [["Sector", "Start", "End", "Points", "Used", "Kept", "RMS (ppm)", "Median σ (ppm)"]]
        for r in self.sectors:
            rows.append([str(r["sector"]) if r["sector"] else "all", r["start"], r["end"], str(r["points"]),
                         str(r["used"]), _fmt(r["kept"], ".0f") + " %", _fmt(r["rms_ppm"], ".0f"),
                         _fmt(r["median_err_ppm"], ".0f")])
        n = len(rows[0])
        return [Paragraph("Data used", self.styles["h2"]), _table(rows, [width / n] * n),
                Spacer(1, 3), Paragraph("Points: after the initial trim. Used: after outlier clipping and masks. "
                                        "RMS and σ are measured on the used points.", self.styles["small"])]

    def _availability_flowables(self, width):
        availability = self.parsed_log["availability"]
        if not availability:
            return []
        rows = availability_rows(availability, self.used)
        rows = group_availability_rows(rows)
        blocks = split_blocks(rows)
        columns = [c for c in AVAILABILITY_COLUMNS if any(r["cells"].get(c, "–") != "–" for r in rows)]
        gap = 0.5 * cm
        block_width = (width - gap * (len(blocks) - 1)) / len(blocks)
        label_width = 1.5 * cm
        tables = []
        for block in blocks:
            data = [["Sector"] + columns]
            style = [("ALIGN", (1, 0), (-1, -1), "CENTER")]
            for r_index, row in enumerate(block, start=1):
                data.append([row["label"]] + [row["cells"].get(c, "–") for c in columns])
                if not row["in_use"]:
                    style.append(("TEXTCOLOR", (0, r_index), (-1, r_index), colors.HexColor("#a0a8b3")))
                for c_index, column in enumerate(columns, start=1):
                    if column in row["used_columns"]:
                        style += [("BACKGROUND", (c_index, r_index), (c_index, r_index), colors.HexColor(USED_BG)),
                                  ("FONTNAME", (c_index, r_index), (c_index, r_index), "Helvetica-Bold")]
            tables.append(_table(data, [label_width] + [(block_width - label_width) / len(columns)] * len(columns),
                                 extra_style=style))
        body = tables[0] if len(tables) == 1 else Table([tables], colWidths=[block_width + gap] * len(tables),
                                                        style=[("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                               ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                                               ("RIGHTPADDING", (0, 0), (-1, -1), 0)])
        note = "Exposure time in seconds, per product. Highlighted and [bracketed]: data used in this run. " \
               "Grey: sectors available but not used."
        if any(len(r["sectors"]) > 1 for r in rows):
            note += " Consecutive sectors with identical availability are shown as a range."
        return [Paragraph("Alternative data available", self.styles["h2"]), body, Spacer(1, 3),
                Paragraph(note, self.styles["small"])]

    def _steps_table(self, width):
        if self.lc is None or self.pre is None:
            return []
        steps = measure_preprocessing(self.lc, self.pre, self.config)
        rows = [["Preprocessing step", "Setting", "Points removed"]]
        for name, setting, removed in steps[1:]:
            rows.append([name, setting, "not measured" if removed is None else f"{removed}"])
        rows.insert(1, ["Data after initial trim", steps[0][1], f"{steps[0][2]} left"])
        rows.append(["Used for detrending", "", f"{len(self.pre)}"])
        extras = []
        pulsations = read_pulsations(self.object_dir)
        if pulsations is not None:
            extras.append(f"{len(pulsations)} stellar oscillation(s) reduced (see the oscillations page).")
        autodetrend = read_autodetrend(self.object_dir)
        if autodetrend is not None:
            period = f"{autodetrend[0]:.3f} d" if autodetrend[0] else "unknown period"
            extras.append(f"Auto-detrend applied with a strong period of {period}.")
        if self.config.high_rms_enabled and self.config.initial_mask:
            extras.append("High RMS mask is enabled but is not applied when an initial mask is set.")
        elif read_rms_mask_plot(self.object_dir):
            extras.append("High RMS mask applied (see the masks page).")
        flowables = [Paragraph("Preprocessing", self.styles["h2"]), _table(rows, [width * 0.3, width * 0.45, width * 0.25])]
        flowables += [Paragraph(e, self.styles["small"]) for e in extras]
        return flowables

    def _page_one(self, width):
        left = Table([[t] for t in self._star_table() + [Spacer(1, 8)] + self._setup_table()], colWidths=[10.2 * cm],
                     style=[("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0)])
        right_width = width - 10.6 * cm
        compact = len(self.sectors) <= SINGLE_TABLE_MAX_SECTORS
        right = []
        if compact:
            right += self._data_used_table(right_width) + self._availability_flowables(right_width)
        right += self._steps_table(right_width)
        right_table = Table([[f] for f in right] or [[Spacer(1, 1)]], colWidths=[right_width],
                            style=[("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 0)])
        story = [Table([[left, right_table]], colWidths=[10.6 * cm, right_width],
                       style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)])]
        if not compact:
            story += [PageBreak()] + self._data_used_table(width) + self._availability_flowables(width)
        return story + [PageBreak()]

    def _curves_page(self, width):
        if self.pre is None:
            return []
        story = []
        path = os.path.join(self.tmp_dir, "sector_curves.png")
        figure_sector_curves(self.pre, self.config.initial_mask, path)
        story += [Paragraph("Preprocessed light curve" + (" (red bands: initial masks)" if self.config.initial_mask else ""),
                            self.styles["h2"]), _image(path, width)]
        detrends = read_detrends(self.object_dir, self.pre)
        if detrends:
            path = os.path.join(self.tmp_dir, "detrends.png")
            figure_detrends(self.pre, detrends, self.config.window_lengths, path)
            story += [Paragraph("Trend removed by each detrend window", self.styles["h2"]), _image(path, width)]
            rows = [["Detrend", "RMS (ppm)", "RMS, 10 min bins (ppm)"],
                    ["none (preprocessed)", f"{np.std(self.pre['flux']) * 1e6:.0f}",
                     f"{np.std(_binned(self.pre['time'].to_numpy(), self.pre['flux'].to_numpy(), 10 / 1440)[1]) * 1e6:.0f}"]]
            labels = [f"{w:.4f} d" for w in self.config.window_lengths] \
                if self.config.window_lengths and len(self.config.window_lengths) == len(detrends) \
                else [f"detrend {i}" for i, _ in detrends]
            for (index, frame), label in zip(detrends, labels):
                rows.append([label, f"{np.std(frame['detrended']) * 1e6:.0f}",
                             f"{np.std(_binned(frame['time'].to_numpy(), frame['detrended'].to_numpy(), 10 / 1440)[1]) * 1e6:.0f}"])
            story += [Spacer(1, 4), _table(rows, [4.5 * cm, 3 * cm, 4.5 * cm])]
        return story + [PageBreak()]

    def _periodicity_page(self, width):
        names = ["Periodogram_Initial", "Periodogram_Final", "Autocorrelation_Initial", "Autocorrelation_Final"]
        paths = [os.path.join(self.object_dir, "periodicity", f"{n}_{self.object_id}.png") for n in names]
        if not all(os.path.exists(p) for p in paths):
            return []
        cell = (width - 0.6 * cm) / 2
        table = Table([[_image(paths[0], cell), _image(paths[1], cell)], [_image(paths[2], cell), _image(paths[3], cell)]],
                      colWidths=[cell + 0.3 * cm] * 2)
        return [Paragraph("Periodicity of the light curve: initial (left) and final (right)", self.styles["h2"]), table,
                PageBreak()]

    def _field_of_view_pages(self, width):
        story = []
        pdfs = sorted(_glob(self.object_dir, os.path.join("fov", "fov_sector_*.pdf")),
                      key=lambda p: int(re.search(r"fov_sector_(\d+)", p).group(1)))
        images = []
        for pdf in pdfs:
            sector = re.search(r"fov_sector_(\d+)", pdf).group(1)
            png = os.path.join(self.tmp_dir, f"fov_sector_{sector}.png")
            try:
                if rasterize_pdf(pdf, png):
                    images.append((sector, png))
            except Exception:
                logging.exception("Target info report: %s could not be rendered and was left out", pdf)
        if images:
            cell = (width - 0.4 * cm) / 2
            cells = [Table([[Paragraph(f"Sector {s}", self.styles["small"])], [_image(p, cell - 2.6 * cm)]],
                           colWidths=[cell]) for s, p in images]
            rows = [cells[i:i + 2] + [""] * (2 - len(cells[i:i + 2])) for i in range(0, len(cells), 2)]
            story += [Paragraph("Field of view per sector (TRICERATOPS): neighbouring stars, aperture (red) and pixel flux",
                                self.styles["h2"]),
                      Table(rows, colWidths=[cell + 0.2 * cm] * 2,
                            style=[("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]),
                      PageBreak()]
        pixels = sorted(_glob(self.object_dir, os.path.join("fov", "fov_Flux_pixels*.png")))
        if pixels:
            cell = 8.3 * cm
            cells = [_image(p, cell) for p in pixels]
            rows = [cells[i:i + 3] + [""] * (3 - len(cells[i:i + 3])) for i in range(0, len(cells), 3)]
            story += [Paragraph("Pixel light curves per sector", self.styles["h2"]),
                      Table(rows, colWidths=[8.5 * cm] * 3), PageBreak()]
        return story

    def _preprocessing_pages(self, width):
        story = []
        pulsations = read_pulsations(self.object_dir)
        if pulsations is not None:
            rows = [["#", "Period (s)", "Frequency (µHz)", "Amplitude", "Phase", "SNR"]]
            for _, r in pulsations.iterrows():
                rows.append([str(int(r["number"])), f"{r['period_s']:.2f}", f"{r['frequency_microHz']:.2f}",
                             f"{r['amplitude']:.2e}", f"{r['phase']:.2f}", f"{r['snr']:.1f}"])
            story += [Paragraph("Stellar oscillations reduced", self.styles["h2"]), _table(rows, [1.2 * cm] + [3.5 * cm] * 5)]
            cell = (width - 0.6 * cm) / 2
            for _, r in pulsations.iterrows():
                if os.path.exists(r["folded"]) and os.path.exists(r["periodogram"]):
                    story += [Spacer(1, 6), Paragraph(f"Signal {int(r['number'])}", self.styles["small"]),
                              Table([[_image(r["folded"], cell), _image(r["periodogram"], cell)]],
                                    colWidths=[cell + 0.3 * cm] * 2)]
            story.append(PageBreak())
        autodetrend = read_autodetrend(self.object_dir)
        if autodetrend is not None:
            period, path = autodetrend
            detail = ", ".join(x for x in [f"period {period:.3f} d" if period else None,
                                           f"ratio {self.config.autodetrend_ratio:g}" if self.config.autodetrend_ratio else None,
                                           f"method {self.config.autodetrend_method}" if self.config.autodetrend_method else None] if x)
            story += [Paragraph("Auto-detrend" + (f" ({detail})" if detail else ""), self.styles["h2"]),
                      _image(path, width * 0.5), PageBreak()]
        rms_plot = read_rms_mask_plot(self.object_dir)
        if rms_plot:
            story += [Paragraph("High RMS mask", self.styles["h2"]), _image(rms_plot, width * 0.55), PageBreak()]
        return story

    # ------------------------------------------------------------------ document
    def _header(self, canvas, doc):
        page_width, page_height = landscape(A4)
        canvas.saveState()
        if os.path.exists(LOGO_IMAGE):
            canvas.drawImage(LOGO_IMAGE, 1.2 * cm, page_height - 1.7 * cm, height=1.1 * cm, width=1.1 * cm,
                             mask="auto", preserveAspectRatio=True)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.setFillColor(colors.HexColor(INK))
        canvas.drawString(2.7 * cm, page_height - 1.15 * cm, f"{self.object_id} · Target information")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        try:
            version = sys.modules["sherlockpipe"].__version__
        except Exception:
            version = ""
        canvas.drawString(2.7 * cm, page_height - 1.55 * cm,
                          f"Data retrieval, preprocessing and detrending summary · SHERLOCK {version}")
        canvas.setStrokeColor(colors.HexColor(GRID))
        canvas.line(1.2 * cm, page_height - 1.9 * cm, page_width - 1.2 * cm, page_height - 1.9 * cm)
        canvas.drawRightString(page_width - 1.2 * cm, 0.7 * cm, f"page {doc.page}")
        canvas.restoreState()

    def create(self):
        """
        Writes the report.

        :return: the path of the PDF, or None if there was not enough data to build it
        """
        self._load()
        required = {"time", "flux", "flux_err"}
        if self.lc is None or self.pre is None:
            logging.warning("Target info report skipped: lc.csv or lc_predetrends.csv not found in %s", self.object_dir)
            return None
        if not required.issubset(self.lc.columns) or not required.issubset(self.pre.columns):
            logging.warning("Target info report skipped: lc.csv or lc_predetrends.csv lack the columns %s",
                            sorted(required))
            return None
        page_width, page_height = landscape(A4)
        width = page_width - 2.4 * cm
        path = os.path.join(self.object_dir, f"{self.object_id}_info.pdf")
        with tempfile.TemporaryDirectory() as tmp:
            self.tmp_dir = tmp
            story = []
            for section in (self._page_one, self._curves_page, self._periodicity_page, self._field_of_view_pages,
                            self._preprocessing_pages):
                try:
                    story += section(width)
                except Exception:
                    logging.exception("Target info report: section %s failed and was left out",
                                      getattr(section, "__name__", repr(section)))
            while story and isinstance(story[-1], PageBreak):
                story.pop()
            frame = Frame(1.2 * cm, 1.1 * cm, width, page_height - 3.2 * cm, id="main", leftPadding=0, rightPadding=0,
                          topPadding=0, bottomPadding=0)
            # Written aside and moved at the end: a failure never leaves a truncated PDF nor replaces a good one.
            partial_path = path + ".partial"
            try:
                doc = BaseDocTemplate(partial_path, pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm,
                                      topMargin=2.1 * cm, bottomMargin=1.1 * cm,
                                      pageTemplates=[PageTemplate(id="info", frames=frame, onPage=self._header)])
                doc.build(story)
                os.replace(partial_path, path)
            finally:
                if os.path.exists(partial_path):
                    os.remove(partial_path)
        logging.info("Target info report written to %s", path)
        return path


def write_target_info_report(object_dir, object_id, sherlock_target=None, window_lengths=None):
    """
    Writes the info report and never raises: a problem in the report must not stop a search.

    :return: the path of the PDF, or None if it could not be written
    """
    try:
        return TargetInfoReport(object_dir, object_id, sherlock_target, window_lengths).create()
    except (Exception, SystemExit):  # some libraries exit instead of raising; KeyboardInterrupt still goes through
        logging.exception("The target info report could not be written")
        return None


def main(argv=None):
    """Creates the info report of an existing execution: ``python -m sherlockpipe.search.target_info_report DIR``."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m sherlockpipe.search.target_info_report <execution_dir> [object_id]")
        return 1
    object_dir = os.path.abspath(argv[0])
    object_id = argv[1] if len(argv) > 1 else os.path.basename(object_dir)
    path = write_target_info_report(object_dir, object_id)
    print(path or "nothing written")
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
