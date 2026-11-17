import datetime
import os
import tempfile
from os import path

import numpy as np
import pandas as pd
from astroplan import FixedTarget
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

from sherlockpipe.observation_plan import plan_data, plan_figures

width, height = landscape(A4)
resources_dir = path.join(path.dirname(__file__))

INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#e3e7ec"
ACCENT = "#2a6f97"
GOOD, FAIR = "#d6efdf", "#f7ecc9"
#: How many events the shortlist of the first page shows.
SHORTLIST_SIZE = 10
#: Default number of event cards: the transits with the best opportunity.
DEFAULT_MAX_CARDS = 20


def format_with_errors(value, low, up):
    """
    ``value ± error`` with as many decimals as the smallest error needs plus one, or ``value +up −low`` when the
    errors are not the same.
    """
    errors = [float(e) for e in (low, up) if e is not None and np.isfinite(e) and e > 0]
    decimals = 3 if not errors else int(np.clip(-np.floor(np.log10(min(errors))) + 1, 0, 8))
    if not errors:
        return f"{value:.{decimals}f}"
    low, up = float(low), float(up)
    if abs(low - up) < 0.5 * 10 ** -decimals:
        return f"{value:.{decimals}f} ± {up:.{decimals}f}"
    return f"{value:.{decimals}f} +{up:.{decimals}f} −{low:.{decimals}f}"


def minutes_text(hours):
    return f"{hours * 60:.0f} min"


class ObservationReport:
    """
    Creates the pdf of the observation plan: a landscape document with the target and candidate, a calendar of the
    observation opportunities, a shortlist of the best ones, one event card per transit (the best ``max_cards``) that
    compares the observatories that can observe it, and an appendix with every observation window.
    """
    LOGO_IMAGE = resources_dir + "/../resources/images/sherlock3.png"
    ALERT_IMAGE = resources_dir + "/../resources/images/alert.png"

    def __init__(self, df_observatories, df, alert_date, object_id, name, working_path, ra, dec, t0, t0_low_err, t0_up_err,
                 period, period_low_err, period_up_err, duration, duration_low_err, duration_up_err, depth, depth_low_err,
                 depth_up_err, observable, min_dist, max_dist, min_altitude, max_days, v, j, h, k,
                 midtransit_times=None, since=None, max_cards=DEFAULT_MAX_CARDS):
        """Initialize the observation report.

        Parameters
        ----------
        df_observatories : DataFrame
            The observatories of the plan: name, tz, lat, lon and alt.
        df : DataFrame
            The observation events of the planner.
        alert_date : Time or str or None
            The date after which error propagation makes observations unreliable.
        object_id : str
            The target object identifier.
        name : str
            The name for the output report file.
        working_path : str
            The directory where the report will be saved.
        ra, dec : float
            Coordinates of the target in degrees.
        t0, period, duration, depth : float
            The candidate parameters (the duration in hours, the depth in ppt) with their lower and upper errors.
        observable : float
            The minimum transit observability fraction.
        min_dist, max_dist : float
            The minimum moon distance in degrees for new moon and for full moon.
        min_altitude : float
            The minimum altitude above the horizon in degrees.
        max_days : int
            The maximum number of days to search for observables.
        v, j, h, k : float
            The magnitudes of the target.
        midtransit_times : Time, optional
            Every mid-transit of the plan, so the transits that nobody can observe show in the calendar too.
        since : Time or str, optional
            The first day of the plan.
        max_cards : int
            The number of transits that get an event card, the best ones. Zero or less gives all of them.
        """
        self.df = df
        self.df_observatories = df_observatories.reset_index(drop=True)
        self.alert_date = alert_date
        self.object_id = object_id
        self.name = name
        self.working_path = working_path
        self.observable = observable
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.ra = ra
        self.dec = dec
        self.t0 = t0
        self.t0_low_err = t0_low_err
        self.t0_up_err = t0_up_err
        self.period = period
        self.period_low_err = period_low_err
        self.period_up_err = period_up_err
        self.duration = duration
        self.duration_low_err = duration_low_err
        self.duration_up_err = duration_up_err
        self.depth = depth
        self.depth_low_err = depth_low_err
        self.depth_up_err = depth_up_err
        self.min_altitude = min_altitude
        self.max_days = max_days
        self.v = v
        self.j = j
        self.h = h
        self.k = k
        self.midtransit_times = midtransit_times
        self.since = since
        self.max_cards = max_cards
        self.styles = self._styles()

    # ------------------------------------------------------------------ data
    def _prepare(self):
        self.enriched = plan_data.enrich_events(self.df, self.df_observatories, self.ra, self.dec)
        self.summary = plan_data.observatory_summary(self.df_observatories, self.enriched, self.dec, self.min_altitude)
        self.observers = plan_data.observers_of(self.df_observatories)
        self.target = FixedTarget(SkyCoord(self.ra, self.dec, unit="deg"))

    def _period_of_the_plan(self):
        """First and last day of the plan, from the date given or from the times that are known."""
        if self.since is not None:
            first = pd.Timestamp(self.since.datetime if hasattr(self.since, "datetime") else self.since)
        elif self.midtransit_times is not None and len(self.midtransit_times):
            first = pd.Timestamp(self.midtransit_times.to_datetime().min())
        elif len(self.enriched):
            first = self.enriched["event"].min()
        else:
            first = pd.Timestamp.now()
        first = first.normalize()
        return first, first + pd.Timedelta(days=int(self.max_days))

    # ------------------------------------------------------------------ styles and helpers
    @staticmethod
    def _styles():
        base = ParagraphStyle("base", fontName="Helvetica", fontSize=8.5, leading=11, textColor=colors.HexColor(INK))
        return {"base": base,
                "small": ParagraphStyle("small", parent=base, fontSize=7.5, leading=9.5, textColor=colors.HexColor(MUTED)),
                "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21),
                "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                                     textColor=colors.HexColor(ACCENT), spaceBefore=6, spaceAfter=3),
                "card": ParagraphStyle("card", parent=base, fontName="Helvetica-Bold", fontSize=9.5, leading=12,
                                       spaceBefore=4, spaceAfter=1),
                "alert": ParagraphStyle("alert", parent=base, textColor=colors.HexColor("#c0392b"))}

    @staticmethod
    def _table(rows, widths, header=True, font_size=8, extra=None):
        table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        style = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), font_size),
                 ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)), ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor(GRID))]
        if header:
            style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(MUTED))]
        table.setStyle(TableStyle(style + (extra or [])))
        return table

    @staticmethod
    def _image(figure, directory, file, dpi=200):
        file_path = os.path.join(directory, file)
        figure.savefig(file_path, dpi=dpi)
        figure_width, figure_height = figure.get_size_inches()
        frame_width = width - 2.4 * cm
        return Image(file_path, width=frame_width, height=frame_width * figure_height / figure_width)

    def _window_text(self, event):
        start, end = event["start_obs"], event["end_obs"]
        return f"{start:%H:%M}–{end:%H:%M}" if pd.notna(start) and pd.notna(end) else "–"

    def _event_row(self, event, with_rank=True, with_observatory=True):
        altitude = f"{event['alt_t0']:.0f}°" if np.isfinite(event["alt_t0"]) else "–"
        airmass = f"{event['airmass']:.2f}" if np.isfinite(event["airmass"]) else "–"
        tz = int(event["tz"]) if float(event["tz"]).is_integer() else event["tz"]
        row = []
        if with_rank:
            row.append(str(int(event["rank"])))
        row += [f"{event['event']:%a %d %b %Y  %H:%M}", f"{event['local_t0']:%H:%M} (UTC{tz:+})"]
        if with_observatory:
            row.append(event["observatory"].replace("-", " "))
        row += [f"{event['observable'] * 100:.0f}%", self._window_text(event), altitude, airmass,
                f"{event['moon_phase'] * 100:.0f}% · {event['moon_dist']:.0f}°", minutes_text(event["err_h"])]
        return row

    # ------------------------------------------------------------------ sections
    def create_header(self, canvas, doc):
        """
        Initializes the common header for all the pages.

        :param canvas: the report canvas
        :param doc: the report document
        """
        canvas.saveState()
        if os.path.exists(self.LOGO_IMAGE):
            canvas.drawImage(self.LOGO_IMAGE, x=1.2 * cm, y=height - 1.9 * cm, height=1.6 * cm, width=1.6 * cm,
                             preserveAspectRatio=True, mask="auto")
        canvas.setFont("Helvetica-Bold", 13)
        canvas.setFillColor(colors.HexColor(INK))
        canvas.drawString(3.2 * cm, height - 1.2 * cm, "%s · Ground-based observation plan" % self.name.replace("_", " "))
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(3.2 * cm, height - 1.65 * cm, "Sherlock observation plan: %s" % self.object_id)
        canvas.drawRightString(width - 1.2 * cm, height - 1.2 * cm,
                               datetime.datetime.now().strftime("%a, %d %B %Y, %H:%M:%S"))
        canvas.setStrokeColor(colors.HexColor(GRID))
        canvas.line(1.2 * cm, height - 2.1 * cm, width - 1.2 * cm, height - 2.1 * cm)
        canvas.restoreState()

    def create_footer(self, canvas, doc):
        """
        Initializes the common footer for all the pages

        :param canvas: the report canvas
        :param doc: the report document
        """
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(1.2 * cm, 0.6 * cm, "Powered by Astropy, Astroplan and ReportLab")
        canvas.drawRightString(width - 1.2 * cm, 0.6 * cm, "Page %s" % doc.page)
        canvas.restoreState()

    def _facts(self):
        coordinates = "RA %s   Dec %s" % (Angle(self.ra, u.deg).to_string(unit=u.hourangle, sep=':', precision=2),
                                          Angle(self.dec, u.deg).to_string(unit=u.deg, sep=':', precision=2))
        magnitudes = "V %.2f   J %.2f   H %.2f   K %.2f" % (self.v, self.j, self.h, self.k)
        candidate = "P = %s d   ·   T0 = %s   ·   duration %s h   ·   depth %s ppt" % (
            format_with_errors(self.period, self.period_low_err, self.period_up_err),
            format_with_errors(self.t0, self.t0_low_err, self.t0_up_err),
            format_with_errors(self.duration, self.duration_low_err, self.duration_up_err),
            format_with_errors(self.depth, self.depth_low_err, self.depth_up_err))
        if len(self.enriched):
            low, high = self.enriched["err_h"].min(), self.enriched["err_h"].max()
            timing = ("±%s to ±%s over the plan: the uncertainty of T0 and of the period grows with the transits "
                      "since T0" % (minutes_text(low), minutes_text(high)))
        else:
            timing = "no event to report"
        first, last = self._period_of_the_plan()
        constraints = ("plan from %s for %d days   ·   target at least %g° above the horizon   ·   moon at least %g° away at "
                       "new moon and %g° at full moon   ·   at least %.0f%% of the transit observable   ·   nautical "
                       "twilight" % (first.strftime("%d %b %Y"), int(self.max_days), self.min_altitude, self.min_dist,
                                     self.max_dist, self.observable * 100))
        rows = [[Paragraph("Target", self.styles["small"]), Paragraph(coordinates + "   ·   " + magnitudes, self.styles["base"])],
                [Paragraph("Candidate", self.styles["small"]), Paragraph(candidate, self.styles["base"])],
                [Paragraph("Timing", self.styles["small"]), Paragraph(timing, self.styles["base"])],
                [Paragraph("Constraints", self.styles["small"]), Paragraph(constraints, self.styles["base"])]]
        table = Table(rows, colWidths=[2.3 * cm, width - 2.4 * cm - 2.3 * cm])
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        return table

    def _shortlist(self):
        header = ["#", "Transit (UTC)", "Local T0", "Observatory", "Observable", "Window UTC", "Alt @T0", "Airmass",
                  "Moon", "T0 ±"]
        rows = [header] + [self._event_row(event) for _, event in self.enriched.head(SHORTLIST_SIZE).iterrows()]
        extra = []
        for index, (_, event) in enumerate(self.enriched.head(SHORTLIST_SIZE).iterrows(), start=1):
            if event["observable"] >= 0.9:
                extra.append(("BACKGROUND", (4, index), (4, index), colors.HexColor(GOOD)))
            elif event["observable"] >= 0.7:
                extra.append(("BACKGROUND", (4, index), (4, index), colors.HexColor(FAIR)))
        frame = width - 2.4 * cm
        widths = [0.06, 0.17, 0.12, 0.13, 0.08, 0.12, 0.07, 0.07, 0.11, 0.07]
        return self._table(rows, [frame * w for w in widths], extra=extra)

    def _alert(self):
        if self.alert_date is None:
            return []
        date = self.alert_date.isot[:16] if hasattr(self.alert_date, "isot") else str(self.alert_date)
        return [Spacer(1, 4),
                Paragraph("The plan stops at %s: from then on the error propagation of T0 and of the period makes the "
                          "transit time too imprecise for an observation to be planned." % date, self.styles["alert"])]

    def _summary_page(self, directory):
        first, last = self._period_of_the_plan()
        story = [self._facts(), Spacer(1, 4)]
        story += self._alert()
        figure = plan_figures.figure_calendar(self.enriched, self.summary, self.midtransit_times, first, last)
        story += [Paragraph("Observation opportunities", self.styles["h2"]), self._image(figure, directory, "calendar.png"),
                  Paragraph("Dot size and colour show the observable fraction of the transit window; the numbers are the "
                            "best opportunities of the shortlist. Faint lines are all the transits of the plan.",
                            self.styles["small"])]
        if len(self.enriched):
            story += [Paragraph("Shortlist: the best opportunities", self.styles["h2"]), self._shortlist(),
                      Paragraph("Ranked by %s." % plan_data.SCORE_DESCRIPTION, self.styles["small"])]
        else:
            story += [Spacer(1, 6), Paragraph("No transit is observable with these constraints. The observatories "
                                              "section of the appendix tells what limits each one.", self.styles["base"])]
        return story

    def _cards(self, directory):
        transits = plan_data.top_transits(self.enriched, self.max_cards)
        if not transits:
            return []
        all_transits = self.enriched["event"].nunique()
        story = [PageBreak(), Paragraph("Event cards: compare the observatories for each transit", self.styles["h2"]),
                 Paragraph("One card per transit that at least one observatory can observe, the best first. %s" % (
                     "All %d of them are shown." % all_transits if len(transits) == all_transits else
                     "The %d best of %d are shown; every window is in the appendix." % (len(transits), all_transits)),
                           self.styles["small"]),
                 self._image(plan_figures.figure_legend(), directory, "legend.png", dpi=150), Spacer(1, 2)]
        for index, transit in enumerate(transits, start=1):
            rows = self.enriched[self.enriched["event"] == transit].sort_values("rank")
            best = rows.iloc[0]
            title = "#%d · Transit %s · T0 %s UTC ± %s · %d observator%s" % (
                int(best["rank"]), transit.strftime("%a %d %b %Y"), transit.strftime("%H:%M"),
                minutes_text(rows["err_h"].max()), len(rows), "y" if len(rows) == 1 else "ies")
            figure = plan_figures.figure_event_card(transit, rows, self.observers, self.target, self.min_altitude,
                                                    self.min_dist, self.max_dist)
            story.append(KeepTogether([Paragraph(title, self.styles["card"]),
                                       self._image(figure, directory, "card_%d.png" % index)]))
        return story

    def _appendix(self):
        if len(self.enriched) == 0 and len(self.df_observatories) == 0:
            return []
        story = [PageBreak(), Paragraph("Appendix", self.styles["h2"]),
                 Paragraph("Observatories", self.styles["card"])]
        rows = [["Name", "Latitude (º)", "Longitude (º)", "Altitude (m)", "Observable transits", "Events / limit"]]
        for (_, observatory), entry in zip(self.df_observatories.iterrows(), self.summary):
            def number(key, spec):
                try:
                    value = float(observatory[key])
                except (TypeError, ValueError):
                    return "–"
                return format(value, spec) if np.isfinite(value) else "–"
            rows.append([observatory["name"].replace("-", " "), number("lat", ".4f"), number("lon", ".4f"),
                         number("alt", ".0f"), str(entry["transits"]), entry["reason"] or "%d event%s" % (entry["events"], "" if entry["events"] == 1 else "s")])
        frame = width - 2.4 * cm
        story.append(self._table(rows, [frame * w for w in (0.14, 0.09, 0.09, 0.09, 0.12, 0.47)]))
        if len(self.enriched):
            story += [Spacer(1, 8), Paragraph("Every observation window, by date", self.styles["card"])]
            header = ["#", "Transit (UTC)", "Local T0", "Observatory", "Observable", "Window UTC", "Alt @T0", "Airmass",
                      "Moon", "T0 ±"]
            ordered = self.enriched.sort_values(["event", "observatory"])
            data = [header] + [self._event_row(event) for _, event in ordered.iterrows()]
            widths = [0.06, 0.17, 0.12, 0.13, 0.08, 0.12, 0.07, 0.07, 0.11, 0.07]
            story.append(self._table(data, [frame * w for w in widths], font_size=7.5))
        return story

    # ------------------------------------------------------------------ document
    def create_report(self):
        """
        Creates the final report with the target and candidate, the calendar and shortlist of the observation
        opportunities, the event cards and the appendix, storing it in a pdf.

        :return: the path of the pdf
        """
        self._prepare()
        frame = Frame(1.2 * cm, 1.1 * cm, width - 2.4 * cm, height - 3.4 * cm, id='normal', leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        template = PageTemplate(id='UnaColumna', frames=frame, onPage=self.create_header, onPageEnd=self.create_footer)
        file_path = self.working_path + "/" + self.name + "_observation_plan.pdf"
        with tempfile.TemporaryDirectory() as directory:
            story = self._summary_page(directory) + self._cards(directory) + self._appendix()
            document = BaseDocTemplate(file_path, pagesize=landscape(A4), rightMargin=1.2 * cm, leftMargin=1.2 * cm,
                                       topMargin=2.3 * cm, bottomMargin=1.1 * cm, pageTemplates=template)
            document.build(story)
        return file_path
