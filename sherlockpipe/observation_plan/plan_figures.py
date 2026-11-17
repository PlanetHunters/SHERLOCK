"""
Figures of the observation plan report: the calendar of opportunities, the event card that compares the observatories
for one transit, and the legend of the cards. They are matplotlib ``Figure`` objects built without pyplot.
"""
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from astroplan import moon_illumination
from astropy.time import Time
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

from sherlockpipe.observation_plan.plan_data import night_curves, observable_mask

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#e3e7ec"
NIGHT = "#c5ccd6"
TWILIGHT = "#e8ecf1"
GREEN = "#2f8f5b"
RED = "#c0392b"
ORANGE = "#e08a2b"
UNCERTAINTY = "#f4c7c3"
MOON = "#b9a64a"
MOON_FILL = "#f3e7b8"
RC = {"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
      "axes.spines.top": False, "axes.spines.right": False}
#: Width of every figure in inches, so that they fill the landscape page.
WIDTH_INCHES = 10.7
#: How many of the best opportunities are numbered in the calendar.
NUMBERED_OPPORTUNITIES = 6
#: Height in inches of each observatory row of an event card.
CARD_ROW_INCHES = 0.75


def figure_calendar(enriched, summary, all_midtimes, since, until, numbered=NUMBERED_OPPORTUNITIES):
    """
    The calendar of opportunities: one row per observatory, every transit of the plan as a faint line and every
    observable event as a dot, sized and coloured by the observable fraction, with the best ones numbered. A strip at
    the bottom shows the moon illumination. An observatory without events says why.

    :param enriched: the result of :func:`plan_data.enrich_events`
    :param summary: the result of :func:`plan_data.observatory_summary`
    :param all_midtimes: astropy Time with every mid-transit of the plan (may be None)
    :param since: first day of the plan (anything pandas can read)
    :param until: last day of the plan
    """
    names = [entry["name"] for entry in summary]
    since, until = pd.Timestamp(since), pd.Timestamp(until)
    span_days = max((until - since).days, 1)
    scale = float(np.clip(45 / span_days, 0.15, 1.0))
    with matplotlib.rc_context(RC):
        figure = Figure(figsize=(WIDTH_INCHES, 0.4 * len(names) + 0.95))
        ax = figure.add_axes([0.12, 0.2, 0.86, 0.62])
        ypos = {name: len(names) - i for i, name in enumerate(names)}
        ax.set_ylim(0.3, len(names) + 0.7)
        ax.set_xlim(since - pd.Timedelta(days=1), until + pd.Timedelta(days=1))
        days = pd.date_range(since, until, freq="6h")
        illumination = moon_illumination(Time(list(days.to_pydatetime())))
        moon_axis = ax.twinx()
        moon_axis.set_ylim(0, 8)
        moon_axis.axis("off")
        moon_axis.fill_between(days, 0, illumination, color=MOON_FILL, lw=0, step="mid")
        moon_axis.text(until, 1.15, "moon illumination", fontsize=6.5, color="#9a8a4a", va="bottom", ha="right")
        if all_midtimes is not None and len(all_midtimes):
            for transit in all_midtimes.to_datetime():
                ax.axvline(transit, color=GRID, lw=0.6, zorder=0)
        colormap = matplotlib.colormaps["YlGn"]
        for _, event in enriched.iterrows():
            row = ypos.get(event["observatory"])
            if row is None:
                continue
            numbered_event = event["rank"] <= numbered
            ax.scatter(event["event"], row, s=(70 + 260 * event["observable"] ** 2) * scale,
                       color=colormap(0.25 + 0.75 * event["observable"]), edgecolor=INK if numbered_event else "none",
                       linewidth=1.0, zorder=3)
            if numbered_event:
                ax.text(event["event"], row, str(int(event["rank"])), ha="center", va="center", fontsize=6.5,
                        fontweight="bold", color="white", zorder=4)
        for entry in summary:
            if entry["reason"]:
                ax.text(since, ypos[entry["name"]] + 0.22, entry["reason"], fontsize=7.5, color=RED, va="center")
        ax.set_yticks([ypos[name] for name in names])
        ax.set_yticklabels([name.replace("-", " ") for name in names], fontsize=8)
        locator = mdates.WeekdayLocator(byweekday=0) if span_days <= 120 else mdates.MonthLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.tick_params(axis="x", labelsize=7)
        total = len(all_midtimes) if all_midtimes is not None else enriched["event"].nunique()
        ax.set_title(f"{total} transits in the plan, {enriched['event'].nunique()} observable from at least one "
                     f"observatory", loc="left", fontsize=8, color=MUTED)
    return figure


def _row_text(axis, event, tz_text):
    lines = [f"observable {event['observable'] * 100:.0f}%",
             f"alt at T0 {event['alt_t0']:.0f}°  ·  X {event['airmass']:.2f}" if np.isfinite(event["alt_t0"])
             else "no location: altitude unknown",
             f"moon {event['moon_phase'] * 100:.0f}% at {event['moon_dist']:.0f}°",
             f"local T0 {event['local_t0']:%H:%M} ({tz_text})"]
    for k, line in enumerate(lines):
        axis.text(0, 0.92 - k * 0.24, line, fontsize=7, va="top", color=INK, fontweight="bold" if k == 0 else "normal",
                  transform=axis.transAxes)


def figure_event_card(transit, rows, observers, target, min_altitude, moon_min_dist, moon_max_dist):
    """
    One transit seen from every observatory that can observe it, on a common UTC time axis: the night shaded, the
    altitude of the target, the part that is observable, the transit and the window of the timing uncertainty.

    :param transit: the ``event`` timestamp that identifies the transit
    :param rows: the enriched events of that transit, one per observatory, in the order to show them
    :param observers: ``{name: Observer or None}``
    :param target: the astroplan FixedTarget
    """
    rows = rows.reset_index(drop=True)
    with matplotlib.rc_context(RC):
        figure = Figure(figsize=(WIDTH_INCHES, CARD_ROW_INCHES * len(rows) + 0.45))
        spec = figure.add_gridspec(1, 1, left=0.07, right=0.985, top=0.99, bottom=0.12)[0]
        sub = GridSpecFromSubplotSpec(len(rows), 2, subplot_spec=spec, width_ratios=[3.6, 1.15], hspace=0.2, wspace=0.03)
        for i, event in rows.iterrows():
            ax, text = figure.add_subplot(sub[i, 0]), figure.add_subplot(sub[i, 1])
            text.axis("off")
            observer = observers.get(event["observatory"])
            error = pd.Timedelta(hours=float(event["err_h"]))
            tz_text = f"UTC{event['tz']:+.0f}" if float(event["tz"]).is_integer() else f"UTC{event['tz']:+.1f}"
            _row_text(text, event, tz_text)
            ax.set_ylim(0, 90)
            ax.set_yticks([0, 30, 60, 90])
            ax.set_ylabel("altitude (°)", fontsize=6.5)
            if observer is not None:
                curves = night_curves(observer, target, transit, 10.0, 3.0, 261, moon_min_dist, moon_max_dist)
                x = curves["time"]
                ax.fill_between(x, 0, 90, where=curves["sun"] < -12, color=NIGHT, lw=0, step="mid")
                ax.fill_between(x, 0, 90, where=(curves["sun"] >= -12) & (curves["sun"] < 0), color=TWILIGHT, lw=0,
                                step="mid")
                ax.axvspan(event["ingress"] - error, event["egress"] + error, color=UNCERTAINTY, alpha=0.55, lw=0, zorder=1)
                ax.axvspan(event["ingress"], event["egress"], color=ORANGE, alpha=0.55, lw=0, zorder=2)
                ax.axvline(event["event"], color=INK, lw=1, zorder=3)
                ax.axhline(min_altitude, color=RED, lw=0.8, ls="--", zorder=3)
                ax.plot(x, np.where(curves["moon"] > 0, curves["moon"], np.nan), color=MOON, lw=0.8, ls=":", zorder=3)
                ax.plot(x, curves["altitude"], color=MUTED, lw=1, zorder=4)
                ax.plot(x, np.where(observable_mask(curves, min_altitude), curves["altitude"], np.nan), color=GREEN,
                        lw=2.6, zorder=5)
                ax.set_xlim(x[0], x[-1])
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            else:
                ax.text(0.5, 0.5, "no location: only the moon constraint was applied", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED)
                ax.set_xticks([])
            ax.grid(axis="y", color="white", lw=0.6, zorder=0)
            ax.tick_params(labelsize=6.5, labelbottom=(i == len(rows) - 1 and observer is not None))
            ax.text(0.005, 0.93, event["observatory"].replace("-", " "), transform=ax.transAxes, fontsize=8,
                    fontweight="bold", va="top", zorder=6,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})
            if i == len(rows) - 1:
                ax.set_xlabel("time (UTC)", fontsize=7)
    return figure


def figure_legend():
    """The key of the colours of the event cards, drawn once for the whole report."""
    items = [(UNCERTAINTY, "T0 uncertainty window"), (ORANGE, "transit (ingress to egress)"),
             (GREEN, "observable: night, above the minimum altitude, moon far enough"), (NIGHT, "nautical night"),
             (TWILIGHT, "twilight"), (MOON, "moon altitude")]
    positions = [0, 0.15, 0.31, 0.62, 0.74, 0.84]
    with matplotlib.rc_context(RC):
        figure = Figure(figsize=(WIDTH_INCHES, 0.25))
        ax = figure.add_axes([0, 0, 1, 1])
        ax.axis("off")
        for x, (color, label) in zip(positions, items):
            ax.add_patch(Rectangle((x, 0.3), 0.012, 0.4, transform=ax.transAxes, color=color, lw=0))
            ax.text(x + 0.018, 0.5, label, transform=ax.transAxes, fontsize=6.5, va="center", color=MUTED)
    return figure
