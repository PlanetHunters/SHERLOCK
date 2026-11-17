import io
import unittest

import numpy as np
import pandas as pd
from astroplan import FixedTarget
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.figure import Figure

from sherlockpipe.observation_plan import plan_data, plan_figures
from sherlockpipe.tests.plan.fixtures import DEC, RA, offline_astropy, sample_events, sample_observatories


class FigureCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()
        cls.observatories = sample_observatories()
        cls.enriched = plan_data.enrich_events(sample_events(), cls.observatories, RA, DEC)
        cls.summary = plan_data.observatory_summary(cls.observatories, cls.enriched, DEC, 30)
        cls.observers = plan_data.observers_of(cls.observatories)
        cls.target = FixedTarget(SkyCoord(RA, DEC, unit="deg"))

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)

    @staticmethod
    def render(figure):
        """Saving proves that the figure can be drawn, and gives its size."""
        buffer = io.BytesIO()
        figure.savefig(buffer, dpi=60)
        return len(buffer.getvalue())


class TestCalendar(FigureCase):

    def _calendar(self, enriched=None, midtimes=None, since="2026-12-01", until="2027-01-15", **kwargs):
        enriched = self.enriched if enriched is None else enriched
        return plan_figures.figure_calendar(enriched, self.summary, midtimes, since, until, **kwargs)

    def test_it_is_a_figure_that_can_be_saved(self):
        figure = self._calendar()
        self.assertIsInstance(figure, Figure)
        self.assertGreater(self.render(figure), 1000)

    def test_there_is_one_row_per_observatory_including_those_without_events(self):
        ax = self._calendar().axes[0]
        self.assertEqual(["Calar Alto", "Teide OT", "Trappist South", "Remote"], [t.get_text() for t in ax.get_yticklabels()])

    def test_every_event_is_a_dot(self):
        ax = self._calendar().axes[0]
        self.assertEqual(len(self.enriched), len(ax.collections))

    def test_the_best_opportunities_are_numbered(self):
        ax = self._calendar(numbered=3).axes[0]
        numbers = sorted(t.get_text() for t in ax.texts if t.get_text().isdigit())
        self.assertEqual(["1", "2", "3"], numbers)

    def test_an_observatory_without_events_says_why(self):
        texts = [t.get_text() for t in self._calendar().axes[0].texts]
        self.assertTrue(any("never rises above 30°" in t for t in texts))

    def test_every_transit_of_the_plan_is_a_faint_line(self):
        midtimes = Time(["2026-12-01T12:00:00", "2026-12-02T12:00:00", "2026-12-03T12:00:00"])
        with_lines = self._calendar(midtimes=midtimes).axes[0]
        without = self._calendar().axes[0]
        self.assertEqual(3, len(with_lines.lines) - len(without.lines))

    def test_the_title_counts_the_transits_and_the_observable_ones(self):
        midtimes = Time(["2026-12-01T12:00:00", "2026-12-02T12:00:00", "2026-12-03T12:00:00"] * 10)
        title = self._calendar(midtimes=midtimes).axes[0].get_title(loc="left")
        self.assertIn("30 transits in the plan, 3 observable", title)

    def test_a_year_long_plan_and_a_plan_without_events_can_be_drawn(self):
        self.assertGreater(self.render(self._calendar(since="2026-12-01", until="2027-12-01")), 1000)
        empty = self._calendar(enriched=self.enriched.iloc[0:0])
        self.assertGreater(self.render(empty), 1000)


class TestEventCard(FigureCase):

    def _card(self, transit):
        rows = self.enriched[self.enriched["event"] == transit].sort_values("rank")
        return plan_figures.figure_event_card(transit, rows, self.observers, self.target, 30, 30, 70), rows

    @staticmethod
    def _axis(figure, rows, name):
        """The plot of an observatory: the rows are drawn in rank order, a plot and a text column each."""
        return figure.axes[2 * list(rows["observatory"]).index(name)]

    def test_the_height_grows_with_the_number_of_observatories(self):
        first = pd.Timestamp("2026-12-07T05:39:00")
        three, rows = self._card(first)
        one, _ = self._card(pd.Timestamp("2026-12-10T05:17:00"))
        self.assertEqual(3, len(rows))
        self.assertAlmostEqual(three.get_size_inches()[1] - one.get_size_inches()[1], 2 * plan_figures.CARD_ROW_INCHES)
        self.assertAlmostEqual(plan_figures.WIDTH_INCHES, three.get_size_inches()[0])

    def test_each_observatory_has_a_plot_and_a_text_column(self):
        figure, rows = self._card(pd.Timestamp("2026-12-07T05:39:00"))
        self.assertEqual(2 * len(rows), len(figure.axes))
        self.assertGreater(self.render(figure), 1000)

    def test_the_text_column_gives_the_numbers_an_observer_needs(self):
        figure, rows = self._card(pd.Timestamp("2026-12-07T05:39:00"))
        text = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
        self.assertIn("observable 65%", text)
        self.assertIn("alt at T0", text)
        self.assertIn("moon 3% at 90°", text)
        self.assertIn("local T0 06:39 (UTC+1)", text)
        self.assertIn("Calar Alto", text)

    def test_the_observable_part_of_the_night_is_drawn_in_green(self):
        figure, rows = self._card(pd.Timestamp("2026-12-07T05:39:00"))
        green = [line for line in self._axis(figure, rows, "Calar-Alto").lines if line.get_color() == plan_figures.GREEN]
        self.assertEqual(1, len(green))
        self.assertTrue(np.isfinite(green[0].get_ydata().astype(float)).any())

    def test_the_minimum_altitude_is_a_dashed_line(self):
        figure, rows = self._card(pd.Timestamp("2026-12-07T05:39:00"))
        dashed = [line for line in self._axis(figure, rows, "Calar-Alto").lines if line.get_linestyle() == "--"]
        self.assertEqual([30, 30], list(dashed[0].get_ydata()))

    def test_an_observatory_without_location_says_so_instead_of_drawing_a_curve(self):
        figure, rows = self._card(pd.Timestamp("2026-12-07T05:39:00"))
        remote = self._axis(figure, rows, "Remote")
        self.assertTrue(any("no location" in t.get_text() for t in remote.texts))
        self.assertEqual(0, len(remote.lines))

    def test_the_time_axis_is_only_labelled_on_the_last_row(self):
        figure, rows = self._card(pd.Timestamp("2026-12-10T05:17:00"))
        self.assertEqual("time (UTC)", figure.axes[0].get_xlabel())
        three, _ = self._card(pd.Timestamp("2026-12-07T05:39:00"))
        labels = [three.axes[2 * i].get_xlabel() for i in range(3)]
        self.assertEqual(["", "", "time (UTC)"], labels)


class TestLegend(unittest.TestCase):

    def test_the_legend_has_one_entry_per_colour_of_the_cards(self):
        figure = plan_figures.figure_legend()
        ax = figure.axes[0]
        self.assertEqual(6, len(ax.patches))
        self.assertEqual(6, len(ax.texts))
        self.assertIn("observable", " ".join(t.get_text() for t in ax.texts))


if __name__ == '__main__':
    unittest.main()
