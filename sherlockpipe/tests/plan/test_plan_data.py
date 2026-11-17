import unittest

import numpy as np
import pandas as pd
from astroplan import FixedTarget, Observer
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from sherlockpipe.observation_plan import plan_data
from sherlockpipe.observation_plan.plan_data import (enrich_events, highest_altitude, hours_from_text, night_curves,
                                                     observable_mask, observatory_summary, quality, top_transits)
from sherlockpipe.tests.plan.fixtures import DEC, RA, EVENT_COLUMNS, offline_astropy, sample_events, sample_observatories


class TestQuality(unittest.TestCase):

    def test_the_score_multiplies_the_fraction_the_altitude_and_the_moon(self):
        self.assertAlmostEqual(1.0, quality(1.0, 60, 0.0))
        self.assertAlmostEqual(0.5, quality(1.0, 30, 0.0))
        self.assertAlmostEqual(0.5, quality(1.0, 60, 1.0))
        self.assertAlmostEqual(0.8 * 0.5 * 0.9, quality(0.8, 30, 0.2))

    def test_a_target_above_60_degrees_gets_no_extra_credit_and_one_below_the_horizon_none(self):
        self.assertAlmostEqual(quality(1.0, 60, 0.0), quality(1.0, 85, 0.0))
        self.assertEqual(0.0, quality(1.0, -5, 0.0))

    def test_an_unknown_altitude_does_not_penalise(self):
        self.assertAlmostEqual(0.9 * 0.75, quality(0.9, np.nan, 0.5))
        self.assertAlmostEqual(0.9 * 0.75, quality(0.9, None, 0.5))

    def test_hours_and_minutes_text_is_read_as_hours(self):
        self.assertAlmostEqual(0.85, hours_from_text("0:51"))
        self.assertAlmostEqual(3.0 + 5 / 60, hours_from_text("3:05"))

    def test_the_highest_altitude_depends_on_the_latitude_and_the_declination(self):
        self.assertEqual(90.0, highest_altitude(37.0, 37.0))
        self.assertAlmostEqual(90 - 57.7525 + 37.2236, highest_altitude(37.2236, 57.7525))
        self.assertAlmostEqual(2.99, highest_altitude(-29.2563, 57.7525), places=2)


class TestEnrichedEvents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()
        cls.events = sample_events()
        cls.observatories = sample_observatories()
        cls.enriched = enrich_events(cls.events, cls.observatories, RA, DEC)

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)

    def test_the_columns_are_the_planner_ones_plus_what_an_observer_needs(self):
        self.assertEqual(EVENT_COLUMNS + ["event", "err_h", "tz", "local_t0", "alt_t0", "airmass", "quality", "rank"],
                         list(self.enriched.columns))
        self.assertEqual(len(self.events), len(self.enriched))

    def test_the_input_is_not_modified(self):
        self.assertEqual(EVENT_COLUMNS, list(self.events.columns))
        self.assertIsInstance(self.events["midtime"].iloc[0], str)

    def test_the_events_are_ranked_from_the_best_opportunity(self):
        self.assertEqual(list(range(1, len(self.enriched) + 1)), list(self.enriched["rank"]))
        self.assertEqual(sorted(self.enriched["quality"], reverse=True), list(self.enriched["quality"]))

    def test_the_time_columns_are_datetimes_and_the_event_is_the_mid_time_rounded_to_the_minute(self):
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.enriched["midtime"]))
        self.assertTrue((self.enriched["event"] == self.enriched["midtime"].dt.round("min")).all())

    def test_the_local_time_adds_the_time_zone_of_the_observatory(self):
        calar = self.enriched[self.enriched["observatory"] == "Calar-Alto"].iloc[0]
        self.assertEqual(calar["event"] + pd.Timedelta(hours=1), calar["local_t0"])
        teide = self.enriched[self.enriched["observatory"] == "Teide-OT"].iloc[0]
        self.assertEqual(teide["event"], teide["local_t0"])

    def test_the_timing_error_is_in_hours(self):
        remote = self.enriched[self.enriched["observatory"] == "Remote"].iloc[0]
        self.assertAlmostEqual(1 + 5 / 60, remote["err_h"])

    def test_the_altitude_at_the_mid_transit_is_the_one_astroplan_gives(self):
        calar = self.enriched[self.enriched["observatory"] == "Calar-Alto"].iloc[0]
        site = Observer(latitude=37.2236 * u.deg, longitude=-2.5463 * u.deg, elevation=2168 * u.m)
        expected = site.altaz(Time(calar["midtime"].to_pydatetime(), scale="utc"),
                              FixedTarget(SkyCoord(RA, DEC, unit="deg"))).alt.deg
        self.assertAlmostEqual(float(expected), calar["alt_t0"], places=4)

    def test_the_airmass_follows_the_altitude(self):
        row = self.enriched[self.enriched["observatory"] == "Calar-Alto"].iloc[0]
        self.assertAlmostEqual(1 / np.sin(np.radians(row["alt_t0"])), row["airmass"], places=6)

    def test_an_observatory_without_location_has_no_altitude_and_is_not_penalised_for_it(self):
        remote = self.enriched[self.enriched["observatory"] == "Remote"].iloc[0]
        self.assertTrue(np.isnan(remote["alt_t0"]))
        self.assertTrue(np.isnan(remote["airmass"]))
        self.assertAlmostEqual(quality(1.0, np.nan, 0.5), remote["quality"])

    def test_a_target_below_the_horizon_has_no_airmass(self):
        below = enrich_events(sample_events(), pd.DataFrame({"name": ["Calar-Alto", "Teide-OT", "Remote"],
                                                            "tz": [1, 0, 2], "lat": [-80.0, -80.0, np.nan],
                                                            "lon": [0.0, 0.0, np.nan], "alt": [10, 10, np.nan]}), RA, DEC)
        row = below[below["observatory"] == "Calar-Alto"].iloc[0]
        self.assertLess(row["alt_t0"], 0)
        self.assertTrue(np.isnan(row["airmass"]))

    def test_no_events_give_an_empty_table_with_all_the_columns(self):
        empty = enrich_events(self.events.iloc[0:0], self.observatories, RA, DEC)
        self.assertEqual(0, len(empty))
        self.assertIn("rank", empty.columns)
        self.assertEqual([], top_transits(empty, 20))


class TestTopTransits(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()
        cls.enriched = enrich_events(sample_events(), sample_observatories(), RA, DEC)

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)

    def test_each_transit_appears_once_even_when_several_observatories_can_observe_it(self):
        transits = top_transits(self.enriched, 20)
        self.assertEqual(3, len(transits))
        self.assertEqual(len(transits), len(set(transits)))

    def test_the_transits_are_ordered_by_their_best_opportunity(self):
        transits = top_transits(self.enriched, 20)
        best = [self.enriched[self.enriched["event"] == t]["rank"].min() for t in transits]
        self.assertEqual(sorted(best), best)

    def test_the_number_of_cards_is_capped(self):
        self.assertEqual(2, len(top_transits(self.enriched, 2)))
        self.assertEqual(top_transits(self.enriched, 20)[:2], top_transits(self.enriched, 2))

    def test_zero_or_negative_gives_all_the_transits(self):
        self.assertEqual(3, len(top_transits(self.enriched, 0)))
        self.assertEqual(3, len(top_transits(self.enriched, -1)))
        self.assertEqual(3, len(top_transits(self.enriched, None)))


class TestObservatorySummary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()
        cls.enriched = enrich_events(sample_events(), sample_observatories(), RA, DEC)
        cls.summary = {entry["name"]: entry for entry in
                       observatory_summary(sample_observatories(), cls.enriched, DEC, 30)}

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)

    def test_an_observatory_with_events_has_no_reason_and_counts_them(self):
        self.assertIsNone(self.summary["Calar-Alto"]["reason"])
        self.assertEqual(2, self.summary["Calar-Alto"]["events"])
        self.assertEqual(2, self.summary["Calar-Alto"]["transits"])

    def test_an_observatory_that_never_sees_the_target_says_its_highest_altitude(self):
        reason = self.summary["Trappist-South"]["reason"]
        self.assertIn("never rises above 30°", reason)
        self.assertIn("highest altitude 3°", reason)

    def test_an_observatory_with_a_good_sky_but_no_events_blames_the_constraints(self):
        summary = observatory_summary(sample_observatories(), self.enriched.iloc[0:0], DEC, 30)
        calar = {e["name"]: e for e in summary}["Calar-Alto"]
        self.assertIn("night, altitude, moon and timing", calar["reason"])

    def test_an_observatory_without_location_and_without_events_says_so(self):
        summary = observatory_summary(sample_observatories(), self.enriched.iloc[0:0], DEC, 30)
        self.assertIn("no location", {e["name"]: e for e in summary}["Remote"]["reason"])


class TestNightCurves(unittest.TestCase):

    def setUp(self):
        self._offline = offline_astropy()
        self._offline.__enter__()
        self.site = Observer(latitude=37.2236 * u.deg, longitude=-2.5463 * u.deg, elevation=2168 * u.m)
        self.target = FixedTarget(SkyCoord(RA, DEC, unit="deg"))
        self.curves = night_curves(self.site, self.target, pd.Timestamp("2026-12-07T05:38:44"))

    def tearDown(self):
        self._offline.__exit__(None, None, None)

    def test_the_curves_cover_ten_hours_before_and_three_after_the_time(self):
        self.assertEqual(261, len(self.curves["time"]))
        self.assertAlmostEqual(13.0, (self.curves["time"][-1] - self.curves["time"][0]).total_seconds() / 3600, places=3)
        centre = pd.Timestamp("2026-12-07T05:38:44")
        self.assertAlmostEqual(10.0, (centre - pd.Timestamp(self.curves["time"][0])).total_seconds() / 3600, places=2)

    def test_the_window_starts_at_night_and_ends_after_the_morning_twilight(self):
        self.assertLess(self.curves["sun"][0], -12)     # 19:38 UTC, after the evening twilight of Calar Alto (17:53)
        self.assertGreater(self.curves["sun"][-1], -12)  # 08:38 UTC, after the morning twilight (06:09)

    def test_the_altitude_of_this_target_rises_towards_the_morning(self):
        self.assertGreater(self.curves["altitude"][-1], self.curves["altitude"][0])

    def test_the_observable_part_needs_night_altitude_and_moon(self):
        mask = observable_mask(self.curves, 30)
        self.assertTrue(mask.any())
        self.assertTrue((self.curves["sun"][mask] < -12).all())
        self.assertTrue((self.curves["altitude"][mask] >= 30).all())
        stricter = observable_mask(self.curves, 60)
        self.assertLessEqual(stricter.sum(), mask.sum())
        no_moon = dict(self.curves, moon_ok=np.zeros_like(self.curves["moon_ok"]))
        self.assertFalse(observable_mask(no_moon, 30).any())

    def test_the_moon_rule_is_the_one_of_the_planner(self):
        far = night_curves(self.site, self.target, pd.Timestamp("2026-12-07T05:38:44"), moon_min_dist=0, moon_max_dist=0)
        self.assertTrue(far["moon_ok"].all())
        near = night_curves(self.site, self.target, pd.Timestamp("2026-12-07T05:38:44"), moon_min_dist=181,
                            moon_max_dist=181)
        self.assertFalse(near["moon_ok"].any())


if __name__ == '__main__':
    unittest.main()
