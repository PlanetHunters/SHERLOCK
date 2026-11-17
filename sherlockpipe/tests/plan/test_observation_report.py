import os
import re
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from astropy.time import Time

from sherlockpipe.observation_plan.observation_report import (DEFAULT_MAX_CARDS, SHORTLIST_SIZE, ObservationReport,
                                                              format_with_errors, minutes_text)
from sherlockpipe.tests.plan.fixtures import (DEC, EVENT_COLUMNS, RA, offline_astropy, sample_events,
                                              sample_observatories)

try:
    import pypdfium2
except ImportError:  # pragma: no cover
    pypdfium2 = None


def make_report(directory, events=None, observatories=None, alert_date=None, **kwargs):
    return ObservationReport(sample_observatories() if observatories is None else observatories,
                             sample_events() if events is None else events, alert_date, "TIC 1", "TIC_1_b", directory, RA,
                             DEC, 1713.45, 0.002, 0.002, 1.49233, 2e-5, 2e-5, 0.69, 0.02, 0.02, 0.49, 0.03, 0.03, 0.5, 30,
                             70, 30, 45, 12.8, 9.5, 8.8, 8.6, **kwargs)


def pdf_pages(path):
    document = pypdfium2.PdfDocument(path)
    try:
        return [document[i].get_textpage().get_text_range() for i in range(len(document))]
    finally:
        document.close()


def many_events(count):
    """``count`` distinct transits, one per period, each seen from Calar Alto and Teide."""
    frames = []
    for night in range(count):
        frame = sample_events().iloc[[0, 1]].copy()
        for column in ("start_obs", "end_obs", "ingress", "egress", "midtime", "twilight_evening", "twilight_morning"):
            shifted = pd.to_datetime(frame[column]) + pd.Timedelta(days=1.49233 * night)
            frame[column] = shifted.dt.strftime("%Y-%m-%dT%H:%M:%S.000")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TestFormatting(unittest.TestCase):

    def test_symmetric_errors_are_written_with_plus_minus_and_one_decimal_more_than_the_error_needs(self):
        self.assertEqual("1.492330 ± 0.000020", format_with_errors(1.49233, 2e-5, 2e-5))
        self.assertEqual("1713.4500 ± 0.0020", format_with_errors(1713.45, 0.002, 0.002))

    def test_different_errors_are_written_separately(self):
        self.assertEqual("0.650 +0.050 −0.040", format_with_errors(0.65, 0.04, 0.05))

    def test_the_smallest_error_sets_the_decimals(self):
        self.assertEqual("10.0000 +0.1000 −0.0010", format_with_errors(10.0, 0.001, 0.1))

    def test_a_value_without_a_usable_error_has_three_decimals(self):
        self.assertEqual("1.250", format_with_errors(1.25, np.nan, np.nan))
        self.assertEqual("1.250", format_with_errors(1.25, None, 0))

    def test_minutes(self):
        self.assertEqual("51 min", minutes_text(0.85))


class ReportCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)


class TestShortlist(ReportCase):

    def test_the_shortlist_has_a_header_and_at_most_ten_rows(self):
        report = make_report(self.directory, events=many_events(10))
        report._prepare()
        self.assertEqual(20, len(report.enriched))
        self.assertEqual(SHORTLIST_SIZE + 1, len(report._shortlist()._cellvalues))

    def test_the_shortlist_follows_the_ranking(self):
        report = make_report(self.directory)
        report._prepare()
        rows = report._shortlist()._cellvalues
        self.assertEqual([str(i) for i in range(1, len(report.enriched) + 1)], [row[0] for row in rows[1:]])

    def test_moon_illumination_is_rounded_not_truncated(self):
        """It used to be truncated: (0.29 * 100) is 28.999999999999996."""
        events = sample_events()
        events["moon_phase"] = 0.29
        report = make_report(self.directory, events=events)
        report._prepare()
        self.assertTrue(all(row[8].startswith("29%") for row in report._shortlist()._cellvalues[1:]))

    def test_a_row_has_the_times_window_altitude_and_timing_error(self):
        report = make_report(self.directory)
        report._prepare()
        calar = [r for r in report._shortlist()._cellvalues if r[3] == "Calar Alto" and r[1].startswith("Mon 07 Dec")][0]
        self.assertEqual("Mon 07 Dec 2026  05:39", calar[1])
        self.assertEqual("06:39 (UTC+1)", calar[2])
        self.assertEqual("65%", calar[4])
        self.assertEqual("03:56–06:08", calar[5])
        self.assertTrue(re.match(r"^\d+°$", calar[6]))
        self.assertEqual("51 min", calar[9])

    def test_an_observatory_without_location_has_no_altitude_or_airmass(self):
        report = make_report(self.directory)
        report._prepare()
        remote = [r for r in report._shortlist()._cellvalues if r[3] == "Remote"][0]
        self.assertEqual(("–", "–"), (remote[6], remote[7]))


class TestPlanPeriod(ReportCase):

    def test_the_period_starts_on_the_given_day_and_lasts_the_days_of_the_plan(self):
        for since in ("2026-12-01T12:00:00", Time("2026-12-01T12:00:00")):
            report = make_report(self.directory, since=since)
            report._prepare()
            first, last = report._period_of_the_plan()
            self.assertEqual(pd.Timestamp("2026-12-01"), first)
            self.assertEqual(pd.Timestamp("2027-01-15"), last)

    def test_without_a_date_the_first_known_transit_is_used(self):
        report = make_report(self.directory, midtransit_times=Time(["2026-12-03T12:00:00", "2026-12-05T12:00:00"]))
        report._prepare()
        self.assertEqual(pd.Timestamp("2026-12-03"), report._period_of_the_plan()[0])
        report = make_report(self.directory)
        report._prepare()
        self.assertEqual(pd.Timestamp("2026-12-07"), report._period_of_the_plan()[0])


@unittest.skipIf(pypdfium2 is None, "pypdfium2 is needed to read the PDF")
class TestPdf(unittest.TestCase):
    """The PDFs are made once for the whole class: each one draws figures with astroplan."""

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()
        cls.directory = tempfile.mkdtemp()
        cls.paths, cls.pages = {}, {}
        for key, kwargs in (("default", {}), ("all_cards", {"max_cards": 0}), ("one_card", {"max_cards": 1}),
                            ("alert_text", {"alert_date": "2027-01-05T12:00:00"}),
                            ("alert_time", {"alert_date": Time("2027-01-05T12:00:00")}),
                            ("empty", {"events": sample_events().iloc[0:0]}),
                            ("single", {"events": sample_events()[sample_events()["observatory"] == "Calar-Alto"],
                                        "observatories": sample_observatories().iloc[[0]]})):
            folder = os.path.join(cls.directory, key)
            os.makedirs(folder)
            cls.paths[key] = make_report(folder, **kwargs).create_report()
            cls.pages[key] = pdf_pages(cls.paths[key])

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)
        shutil.rmtree(cls.directory)

    def _make(self, **kwargs):
        """A report that is not one of the shared ones: its path and the text of its pages."""
        folder = tempfile.mkdtemp(dir=self.directory)
        path = make_report(folder, **kwargs).create_report()
        return path, pdf_pages(path)

    def test_the_pdf_is_written_with_the_name_of_the_candidate_and_nothing_else_is_left(self):
        path = self.paths["default"]
        self.assertEqual(self.directory + "/default/TIC_1_b_observation_plan.pdf", path)
        self.assertEqual(["TIC_1_b_observation_plan.pdf"], os.listdir(self.directory + "/default"))

    def test_the_page_is_landscape(self):
        path = self.paths["default"]
        document = pypdfium2.PdfDocument(path)
        width, height = document[0].get_size()
        document.close()
        self.assertGreater(width, height)

    def test_every_page_has_the_title_in_its_header(self):
        pages = self.pages["default"]
        self.assertTrue(all("TIC 1 b · Ground-based observation plan" in page for page in pages))

    def test_the_first_page_has_the_target_the_candidate_the_constraints_the_calendar_and_the_shortlist(self):
        pages = self.pages["default"]
        first = pages[0]
        for fragment in ("12:21:21.46", "57:45:09.00", "V 12.80", "1.492330 ± 0.000020", "duration 0.690 ± 0.020 h",
                         "target at least 30° above the horizon", "at least 50% of the transit observable",
                         "Observation opportunities", "Shortlist: the best opportunities", "Calar Alto", "Teide OT"):
            self.assertIn(fragment, first)
        self.assertIn("Ranked by observable fraction", first)

    def test_the_timing_line_gives_the_growth_of_the_uncertainty(self):
        pages = self.pages["default"]
        self.assertIn("±51 min to ±65 min over the plan", pages[0])

    def test_an_observatory_that_cannot_see_the_target_is_explained_in_the_appendix(self):
        pages = self.pages["default"]
        text = "\n".join(pages)
        self.assertIn("Appendix", text)
        self.assertIn("Trappist South", text)
        self.assertIn("the target never rises above 30° (highest altitude 3°)", text)

    def test_the_appendix_writes_a_dash_for_what_an_observatory_does_not_have_and_counts_in_singular(self):
        pages = self.pages["default"]
        text = "\n".join(pages)
        self.assertNotIn("nan", text)
        remote = [line for line in text.splitlines() if line.startswith("Remote")][0]
        self.assertEqual("Remote – – – 1 1 event", remote.strip())

    def test_the_appendix_lists_every_window(self):
        pages = self.pages["default"]
        text = "\n".join(pages)
        self.assertIn("Every observation window, by date", text)
        for fragment in ("Mon 07 Dec 2026", "Thu 10 Dec 2026", "Sun 13 Dec 2026", "03:56–06:08", "03:12–06:50"):
            self.assertIn(fragment, text)

    def test_there_is_an_event_card_for_each_transit_with_its_number_and_the_observatories(self):
        pages = self.pages["default"]
        text = "\n".join(pages)
        self.assertIn("Event cards: compare the observatories for each transit", text)
        self.assertIn("Transit Mon 07 Dec 2026 · T0 05:39 UTC ± 65 min · 3 observatories", text)  # the largest error of its rows
        self.assertIn("Transit Thu 10 Dec 2026 · T0 05:17 UTC ± 51 min · 1 observatory", text)
        self.assertIn("All 3 of them are shown.", text)

    @staticmethod
    def _cards(pages):
        return len(re.findall(r"#\d+ · Transit ", "\n".join(pages)))

    def test_the_number_of_cards_is_limited_and_the_report_says_how_many_were_left_out(self):
        all_pages = self.pages["all_cards"]
        one_page = self.pages["one_card"]
        self.assertEqual(3, self._cards(all_pages))
        self.assertEqual(1, self._cards(one_page))
        self.assertIn("The 1 best of 3 are shown; every window is in the appendix.", "\n".join(one_page))

    def test_twenty_cards_are_the_default(self):
        self.assertEqual(20, DEFAULT_MAX_CARDS)
        events = many_events(25)
        report = make_report(self.directory, events=events)
        report._prepare()
        transits = report.enriched["event"].nunique()
        self.assertEqual(25, transits)
        _, pages = self._make(events=events)
        self.assertEqual(20, self._cards(pages))
        self.assertIn("The 20 best of %d are shown" % transits, "\n".join(pages))

    def test_the_alert_is_only_written_when_the_plan_reached_an_imprecise_date(self):
        without = self.pages["default"]
        self.assertNotIn("The plan stops at", "\n".join(without))
        with_text = self.pages["alert_text"]
        self.assertIn("The plan stops at 2027-01-05T12:00:00", "\n".join(with_text))
        self.assertIn("The plan stops at 2027-01-05T12:00", "\n".join(self.pages["alert_time"]))

    def test_a_plan_without_events_still_produces_a_pdf_that_says_so(self):
        text = "\n".join(self.pages["empty"])
        self.assertIn("No transit is observable with these constraints", text)
        self.assertNotIn("Event cards", text)
        self.assertIn("Trappist South", text)
        self.assertIn("no event to report", text)

    def test_a_single_observatory_plan_is_written(self):
        self.assertIn("Calar Alto", "\n".join(self.pages["single"]))


if __name__ == '__main__':
    unittest.main()
