import os
import re
import shutil
import tempfile
import unittest

import pandas as pd
from reportlab.platypus import Image, Paragraph

from sherlockpipe.observation_plan.observation_report import ObservationReport
from sherlockpipe.tests.plan.fixtures import EVENT_COLUMNS

try:
    import pypdfium2
except ImportError:  # pragma: no cover
    pypdfium2 = None

PNG_1X1 = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000001"
                        "e221bc330000000049454e44ae426082")


def events_frame():
    rows = [
        ["Calar-Alto", 1.0, "2026-12-07T03:56:26.673", "2026-12-07T06:08:42.673", "2026-12-07T05:18:02.673",
         "2026-12-07T05:59:26.673", "2026-12-07T05:38:44.673", "0:51", "0:51", "2026-12-06T17:53:46.000",
         "2026-12-07T06:09:13.000", 0.65, 0.03, 90],
        ["Teide-OT", 0.0, "2026-12-07T03:56:26.673", "2026-12-07T06:50:08.673", "2026-12-07T05:18:02.673",
         "2026-12-07T05:59:26.673", "2026-12-07T05:38:44.673", "0:51", "0:51", "2026-12-06T19:04:27.000",
         "2026-12-07T06:50:08.000", 0.96, 0.50, 111],
        ["Remote", 2.0, "2026-12-07T05:00:00.000", "2026-12-07T06:30:00.000", "2026-12-07T05:18:02.673",
         "2026-12-07T05:59:26.673", "2026-12-07T05:38:44.673", "1:05", "1:05", None, None, 1.0, 0.5, 45]]
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def observatories_frame():
    return pd.DataFrame({"name": ["Calar-Alto", "Teide-OT"], "tz": [1, 0], "lat": [37.2, 28.3], "lon": [-2.5, -16.5],
                         "alt": [2168, 2390]})


def make_report(directory, events, alert_date=None, name="TIC_1_b"):
    return ObservationReport(observatories_frame(), events, alert_date, "TIC 1", name, directory, 185.3394, 57.7525,
                             1713.45, 0.002, 0.002, 1.49233, 2e-5, 2e-5, 0.69, 0.02, 0.02, 0.49, 0.03, 0.03, 0.5, 30, 70,
                             30, 45, 12.8, 9.5, 8.8, 8.6)


def pdf_text(path):
    document = pypdfium2.PdfDocument(path)
    try:
        return "\n".join(document[i].get_textpage().get_text_range() for i in range(len(document))), len(document)
    finally:
        document.close()


class TestEventsTable(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        os.makedirs(self.directory + "/images")
        self.report = make_report(self.directory, events_frame())
        self.table = self.report.df_manipulations()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_the_table_has_the_columns_of_the_report(self):
        self.assertEqual(['Observatory', 'TZ', 'Event times', 'TT Error', 'Moon', 'Image'], list(self.table.columns))
        self.assertEqual(3, len(self.table))

    def test_observatory_names_lose_their_dashes_and_the_time_zone_is_an_integer_text(self):
        self.assertEqual(["Calar Alto", "Teide OT", "Remote"], list(self.table["Observatory"]))
        self.assertEqual(["1", "0", "2"], list(self.table["TZ"]))

    def test_timing_errors_are_shown_as_minus_and_plus(self):
        self.assertEqual("-0:51<br/>+0:51", self.table["TT Error"].iloc[0])
        self.assertEqual("-1:05<br/>+1:05", self.table["TT Error"].iloc[2])

    def test_moon_is_the_illumination_and_the_distance(self):
        self.assertEqual("3%<br/>90º", self.table["Moon"].iloc[0])
        self.assertEqual("50%<br/>111º", self.table["Moon"].iloc[1])

    @unittest.expectedFailure
    def test_moon_illumination_is_rounded_not_truncated(self):
        """Known bug: (0.29 * 100) is 28.999999999999996 and astype(int) truncates it to 28."""
        events = events_frame()
        events.loc[0, "moon_phase"] = 0.29
        table = make_report(self.directory, events).df_manipulations()
        self.assertEqual("29%<br/>90º", table["Moon"].iloc[0])

    def test_event_times_are_sorted_with_their_julian_day(self):
        text = self.table["Event times"].iloc[0]
        labels = re.findall(r"(TWE|SO|I|M|E|EO|TWM): ", text)
        self.assertEqual(["TWE", "SO", "I", "M", "E", "EO", "TWM"], labels)
        self.assertIn("M: <strong>2026-12-07T05:38:44 (2461381.74 JD)</strong>", text)

    def test_milliseconds_are_removed(self):
        self.assertNotIn(".673", "".join(self.table["Event times"]))

    def test_only_the_times_inside_the_night_are_bold(self):
        text = self.table["Event times"].iloc[0]
        self.assertNotIn("<strong>2026-12-06T17:53:46", text)   # evening twilight itself
        self.assertNotIn("<strong>2026-12-07T06:09:13", text)   # morning twilight itself
        self.assertEqual(5, text.count("<strong>"))             # SO, I, M, E, EO

    def test_missing_times_are_double_dashes_and_sort_first(self):
        text = self.table["Event times"].iloc[2]
        self.assertTrue(text.startswith('<font name="HELVETICA" size="8">TWE: --<br/>TWM: --<br/>SO:'))
        self.assertNotIn("<strong>", text)

    def test_the_image_cell_is_empty_text_when_the_image_does_not_exist(self):
        self.assertTrue(all(isinstance(cell, Paragraph) for cell in self.table["Image"]))

    def test_the_image_is_found_by_observatory_and_mid_time(self):
        with open(self.directory + "/images/Calar-Alto_2026-12-07T05:38:44.png", "wb") as f:
            f.write(PNG_1X1)
        table = make_report(self.directory, events_frame()).df_manipulations()
        self.assertIsInstance(table["Image"].iloc[0], Image)
        self.assertIsInstance(table["Image"].iloc[1], Paragraph)

    def test_no_events_give_an_empty_table_with_the_same_columns(self):
        table = make_report(self.directory, events_frame().iloc[0:0]).df_manipulations()
        self.assertEqual(0, len(table))
        self.assertEqual(['Observatory', 'TZ', 'Event times', 'TT Error', 'Moon', 'Image'], list(table.columns))

    def test_observatories_are_shown_without_their_time_zone(self):
        self.assertEqual(["name", "lat", "lon", "alt"], list(self.report.df_observatories.columns))


@unittest.skipIf(pypdfium2 is None, "pypdfium2 is needed to read the PDF")
class TestPdf(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        os.makedirs(self.directory + "/images")

    def tearDown(self):
        shutil.rmtree(self.directory)

    def _create(self, events=None, alert_date=None):
        report = make_report(self.directory, events_frame() if events is None else events, alert_date)
        report.create_report()
        return pdf_text(self.directory + "/TIC_1_b_observation_plan.pdf")

    def test_the_pdf_is_written_with_the_name_of_the_candidate(self):
        self._create()
        self.assertTrue(os.path.exists(self.directory + "/TIC_1_b_observation_plan.pdf"))

    def test_the_first_page_has_the_title_the_star_and_the_candidate(self):
        text, _ = self._create()
        self.assertIn("TIC 1 OBSERVATION PLAN", text)
        self.assertIn("12.8", text)                       # V magnitude in table 1
        self.assertIn("1.49233", text)                    # period in table 2

    def test_ra_and_dec_are_written_in_sexagesimal(self):
        text, _ = self._create()
        self.assertIn("12:21:21.46", text)
        self.assertIn("57:45:09.00", text)

    def test_the_observation_rows_are_in_the_pdf(self):
        text, _ = self._create()
        for fragment in ("Calar Alto", "Teide OT", "Remote", "2026-12-07T05:38:44"):
            self.assertIn(fragment, text)

    def test_the_constraints_table_is_in_the_pdf(self):
        text, _ = self._create()
        self.assertIn("Min altitude", text)
        self.assertIn("Min Moon Dist", text)

    def test_the_alert_is_only_written_when_the_plan_reached_an_imprecise_date(self):
        without, _ = self._create()
        self.assertNotIn("error propagation gets too wide", without)
        with_alert, _ = self._create(alert_date="2027-01-05T12:00:00")
        self.assertIn("error propagation gets too wide", with_alert)
        self.assertIn("2027-01-05T12:00:00", with_alert)

    def test_a_long_plan_continues_in_more_pages_with_the_header_repeated(self):
        many = pd.concat([events_frame()] * 8, ignore_index=True)
        text, pages = self._create(many)
        self.assertGreater(pages, 2)
        self.assertGreaterEqual(text.count("Sherlock observation plan: TIC 1"), pages)

    def test_a_plan_without_events_still_produces_a_pdf(self):
        text, pages = self._create(events_frame().iloc[0:0])
        self.assertGreaterEqual(pages, 1)
        self.assertIn("OBSERVATION PLAN", text)

    @unittest.expectedFailure
    def test_every_table_caption_is_numbered_once_and_in_order(self):
        """Known bug: the list of observatories is captioned "Table 3" (like the constraints) and nothing is "Table 4",
        while the introduction cites Tables 1, 3, 4 and 5."""
        text, _ = self._create()
        numbers = re.findall(r"Table (\d): [A-Z]", text)
        self.assertEqual(["1", "2", "3", "4", "5"], sorted(numbers))


if __name__ == '__main__':
    unittest.main()
