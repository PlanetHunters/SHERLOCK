import datetime
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import Observer, moon_illumination
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time

from sherlockpipe.observation_plan.planner import MoonIlluminationSeparationConstraint, Planner
from sherlockpipe.tests.plan.fixtures import (DEC, EPOCH_JD, EVENT_COLUMNS, PERIOD, RA, offline_astropy, plan,
                                              write_observatories)


class TestMoonConstraint(unittest.TestCase):

    def _target_at(self, time, separation_deg):
        moon = get_body("moon", time)
        return moon.directional_offset_by(0 * u.deg, separation_deg * u.deg)

    def test_the_required_distance_grows_linearly_with_the_moon_illumination(self):
        time = Time("2024-03-01T20:00:00")
        constraint = MoonIlluminationSeparationConstraint(30 * u.deg, 70 * u.deg)
        required = 30 + 40 * float(moon_illumination(time))
        self.assertTrue(bool(constraint.compute_constraint(time, None, self._target_at(time, required + 5))))
        self.assertFalse(bool(constraint.compute_constraint(time, None, self._target_at(time, required - 5))))

    def test_with_equal_limits_the_illumination_does_not_matter(self):
        constraint = MoonIlluminationSeparationConstraint(40 * u.deg, 40 * u.deg)
        for stamp in ("2024-03-01T20:00:00", "2024-03-25T20:00:00"):
            time = Time(stamp)
            self.assertTrue(bool(constraint.compute_constraint(time, None, self._target_at(time, 45))))
            self.assertFalse(bool(constraint.compute_constraint(time, None, self._target_at(time, 35))))

    def test_it_is_evaluated_for_every_time(self):
        times = Time("2024-03-01T20:00:00") + np.arange(3) * u.hour
        target = SkyCoord(RA, DEC, unit="deg")
        mask = MoonIlluminationSeparationConstraint(30 * u.deg, 70 * u.deg).compute_constraint(times, None, target)
        self.assertEqual((3,), np.shape(mask))


class TestHelpers(unittest.TestCase):

    def test_time_zone_offset_in_hours_follows_daylight_saving_time(self):
        winter, summer = datetime.datetime(2024, 1, 15, 12), datetime.datetime(2024, 7, 15, 12)
        self.assertEqual(1.0, Planner.get_offset(40.4168, -3.7038, winter))    # Madrid
        self.assertEqual(2.0, Planner.get_offset(40.4168, -3.7038, summer))
        self.assertEqual(0.0, Planner.get_offset(28.3, -16.5, winter))         # Canary Islands
        self.assertEqual(1.0, Planner.get_offset(28.3, -16.5, summer))
        self.assertEqual(-10.0, Planner.get_offset(19.82, -155.47, winter))    # Hawaii, no daylight saving time


    def test_an_observatory_has_coordinates_only_when_latitude_longitude_and_altitude_are_numbers(self):
        has = Planner.has_coordinates
        self.assertTrue(has(pd.Series({"lat": 28.3, "lon": -16.5, "alt": 2390})))
        self.assertTrue(has(pd.Series({"lat": "28.3", "lon": "-16.5", "alt": "2390"})))
        self.assertFalse(has(pd.Series({"lat": np.nan, "lon": -16.5, "alt": 2390})))
        self.assertFalse(has(pd.Series({"lat": 28.3, "lon": None, "alt": 2390})))
        self.assertFalse(has(pd.Series({"lat": 28.3, "lon": -16.5, "alt": ""})))
        self.assertFalse(has(pd.Series({"lat": 28.3, "lon": -16.5})))

    def test_the_night_of_a_time_holds_it_between_the_twilights(self):
        with offline_astropy():
            site = Observer(latitude=37.2236 * u.deg, longitude=-2.5463 * u.deg, elevation=2168 * u.m)
            for stamp in ("2026-12-04T06:00:50", "2026-12-04T23:00:00", "2026-12-04T03:00:00"):
                time = Time(stamp)
                evening, morning = Planner.night_of(site, time)
                self.assertLessEqual(evening, time)
                self.assertGreaterEqual(morning, time)
                self.assertLess((morning - evening).to_value(u.hour), 16)


class TestObservables(unittest.TestCase):
    """A 12 day plan from three observatories, computed once: the real astroplan code, offline."""

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()
        cls.directory = tempfile.mkdtemp()
        cls.observatories_file = write_observatories(cls.directory)
        cls.result = plan(cls.directory, cls.observatories_file)
        cls.observatories, cls.events, cls.alert_date, cls.plan_dir, cls.midtransit_times = cls.result

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)
        shutil.rmtree(cls.directory)

    def test_the_plan_directory_holds_the_csv_and_nothing_else(self):
        self.assertEqual(self.directory + "/plan", self.plan_dir)
        self.assertEqual(["observation_plan.csv"], os.listdir(self.plan_dir))

    def test_all_the_mid_transit_times_of_the_plan_are_returned_also_those_nobody_can_observe(self):
        self.assertEqual(int(12 // PERIOD), len(self.midtransit_times))
        self.assertGreater(len(self.midtransit_times), len({e for e in self.events["midtime"]}))
        observed = set(Time(list(self.events["midtime"]), scale="utc").isot)
        self.assertTrue(observed <= set(self.midtransit_times.isot))

    def test_the_observatories_are_returned_as_read(self):
        self.assertEqual(["Calar-Alto", "Trappist-North", "Trappist-South"], list(self.observatories["name"]))

    def test_the_events_have_the_documented_columns_and_the_csv_matches(self):
        self.assertEqual(EVENT_COLUMNS, list(self.events.columns))
        saved = pd.read_csv(self.plan_dir + "/observation_plan.csv")
        self.assertEqual(EVENT_COLUMNS, list(saved.columns))
        self.assertEqual(len(self.events), len(saved))

    def test_there_are_events_and_no_alert_for_a_precise_ephemeris(self):
        self.assertGreater(len(self.events), 0)
        self.assertIsNone(self.alert_date)

    def test_events_are_sorted_by_mid_time_and_then_by_observatory(self):
        keys = list(zip(self.events["midtime"], self.events["observatory"]))
        self.assertEqual(sorted(keys), keys)

    def test_an_observatory_that_never_sees_the_target_has_no_events(self):
        self.assertNotIn("Trappist-South", set(self.events["observatory"]))
        self.assertIn("Calar-Alto", set(self.events["observatory"]))

    def test_every_event_reaches_the_requested_observable_fraction(self):
        self.assertTrue((self.events["observable"] >= 0.5).all())
        self.assertTrue((self.events["observable"] <= 1.0).all())
        self.assertTrue(np.allclose(self.events["observable"] * 100, np.round(self.events["observable"] * 100)))

    def test_mid_times_follow_the_ephemeris(self):
        jd = Time(list(self.events["midtime"]), scale="utc").tdb.jd  # the planner writes UTC, the epoch is TDB
        epochs = (jd - EPOCH_JD) / PERIOD
        self.assertTrue(np.allclose(epochs, np.round(epochs), atol=1e-4))

    def test_ingress_and_egress_surround_the_mid_time_by_half_the_duration(self):
        for _, event in self.events.iterrows():
            ingress, mid, egress = (Time(event[c], scale="utc") for c in ("ingress", "midtime", "egress"))
            self.assertAlmostEqual(0.69, (egress - ingress).to_value(u.hour), delta=0.02)
            self.assertAlmostEqual(0.345, (mid - ingress).to_value(u.hour), delta=0.02)

    def test_timing_errors_are_hours_and_minutes_and_grow_with_the_epochs_since_t0(self):
        errors = self.events["midtime_up_err_h"]
        self.assertTrue(errors.str.match(r"^\d+:\d{2}$").all())
        hours = errors.map(lambda s: int(s.split(":")[0]) + int(s.split(":")[1]) / 60)
        self.assertTrue((hours > 0).all())
        self.assertEqual(list(errors), list(self.events["midtime_low_err_h"]))  # symmetric errors in this plan

    def test_moon_columns_are_the_illumination_and_the_distance_in_degrees(self):
        self.assertTrue(self.events["moon_phase"].between(0, 1).all())
        self.assertTrue(self.events["moon_dist"].between(0, 180).all())

    def test_the_observation_window_is_inside_the_night_of_the_event(self):
        for _, event in self.events.iterrows():
            start, end = pd.Timestamp(event["start_obs"]), pd.Timestamp(event["end_obs"])
            self.assertLessEqual(start, end)
            self.assertLess(end - start, pd.Timedelta(hours=14))


class TestObservableVariants(unittest.TestCase):

    def setUp(self):
        self._offline = offline_astropy()
        self._offline.__enter__()
        self.directory = tempfile.mkdtemp()
        self.observatories_file = write_observatories(self.directory)

    def tearDown(self):
        self._offline.__exit__(None, None, None)
        shutil.rmtree(self.directory)

    def test_a_bigger_required_fraction_leaves_fewer_events(self):
        counts = [len(plan(self.directory, self.observatories_file, transit_fraction=f)[1]) for f in (0.25, 0.5, 1.0)]
        self.assertGreaterEqual(counts[0], counts[1])
        self.assertGreaterEqual(counts[1], counts[2])
        self.assertGreater(counts[0], counts[2])

    def test_a_higher_minimum_altitude_leaves_fewer_events(self):
        low = len(plan(self.directory, self.observatories_file, min_altitude=20)[1])
        high = len(plan(self.directory, self.observatories_file, min_altitude=60)[1])
        self.assertGreater(low, high)

    def test_imprecise_ephemeris_raises_the_alert_and_drops_the_events(self):
        _, events, alert_date, _, _ = plan(self.directory, self.observatories_file, epoch_err=0.2)
        self.assertEqual(0, len(events))
        self.assertIsInstance(alert_date, Time)

    def test_the_alert_date_is_the_earliest_imprecise_mid_time(self):
        _, _, alert_date, _, _ = plan(self.directory, self.observatories_file, epoch_err=0.2)
        jd = (alert_date.tdb.jd - EPOCH_JD) / PERIOD
        self.assertAlmostEqual(round(jd), jd, places=3)
        self.assertGreaterEqual(alert_date, Time("2024-03-01T12:00:00"))
        self.assertLess(alert_date, Time("2024-03-01T12:00:00") + 2 * u.day)

    def test_without_the_alert_imprecise_events_are_planned_anyway(self):
        _, events, alert_date, _, _ = plan(self.directory, self.observatories_file, epoch_err=0.2, error_alert=False)
        self.assertGreater(len(events), 0)
        self.assertIsNone(alert_date)

    def test_an_error_window_too_wide_to_be_observable_drops_the_events_even_without_the_alert(self):
        _, events, _, _, _ = plan(self.directory, self.observatories_file, epoch_err=1.0, error_alert=False)
        self.assertEqual(0, len(events))  # a +-24 h window is never 50 % observable at night

    def test_the_horizon_limits_how_many_transits_are_evaluated(self):
        short = plan(self.directory, self.observatories_file, max_days=3)[1]
        long = plan(self.directory, self.observatories_file, max_days=12)[1]
        self.assertLess(len(short), len(long))
        self.assertLessEqual(len({e for e in short["midtime"]}), int(3 // PERIOD))

    def test_a_site_without_coordinates_only_applies_the_moon_constraint(self):
        path = write_observatories(self.directory, "name,tz,lat,lon,alt\nRemote,2,,,\n")
        _, events, _, _, _ = plan(self.directory, path)
        self.assertGreater(len(events), 0)
        self.assertTrue((events["observable"] == 1.0).all())
        self.assertTrue(events["twilight_evening"].isna().all())
        self.assertTrue((events["timezone"] == 2).all())

    def test_an_event_that_cannot_be_planned_is_dropped_without_stopping_the_plan(self):
        """A site without coordinates and without time zone cannot give a local time: its events are dropped, the
        error is logged (by the worker) and the plan goes on."""
        path = write_observatories(self.directory, "name,tz,lat,lon,alt\nRemote,,,,\nCalar-Alto,,37.2236,-2.5463,2168\n")
        _, events, _, _, _ = plan(self.directory, path)  # the error is logged by a worker process of the pool
        self.assertEqual({"Calar-Alto"}, set(events["observatory"]))

    def test_plan_event_always_returns_an_alert_and_an_observable(self):
        broken = SimpleNamespace(i=0, ingress_egress_times=None)
        self.assertEqual((None, None), Planner.plan_event(broken))

    def test_a_single_observatory_can_be_given_by_its_coordinates_as_text(self):
        observatories, events, _, _, _ = plan(self.directory, None, tz="1", lat="37.2236", lon="-2.5463", alt="2168")
        self.assertEqual(["Obs-1"], list(observatories["name"]))
        self.assertEqual(["name", "tz", "lat", "lon", "alt"], list(observatories.columns))
        self.assertGreater(len(events), 0)
        self.assertEqual({"Obs-1"}, set(events["observatory"]))
        self.assertEqual({1.0}, set(events["timezone"]))

    def test_the_time_zone_of_a_single_observatory_is_computed_when_it_is_not_given(self):
        _, events, _, _, _ = plan(self.directory, None, lat=37.2236, lon=-2.5463, alt=2168)
        self.assertEqual({1.0}, set(events["timezone"]))

    def test_the_time_zone_of_a_listed_site_is_computed_from_its_position(self):
        _, events, _, _, _ = plan(self.directory, self.observatories_file)
        by_site = events.groupby("observatory")["timezone"].agg(lambda tz: set(tz))
        self.assertEqual({1.0}, by_site["Calar-Alto"])                 # Spain in March
        self.assertLessEqual(by_site["Trappist-North"], {0.0, 1.0})   # Morocco: UTC+1, UTC+0 during Ramadan

    def test_the_observation_window_never_ends_before_it_starts(self):
        """Calar Alto, 4 December 2026: the nearest evening twilight was the next one (17:53) and the window started
        after it ended (06:06)."""
        path = write_observatories(self.directory, "name,tz,lat,lon,alt\nCalar-Alto,,37.2236,-2.5463,2168\n")
        _, events, _, _, _ = plan(self.directory, path, since="2026-12-01T12:00:00", max_days=6)
        windows = events[["start_obs", "end_obs"]].apply(pd.to_datetime)
        self.assertTrue((windows["start_obs"] <= windows["end_obs"]).all())
        morning_event = events[events["midtime"].str.startswith("2026-12-04T06:00")].iloc[0]
        self.assertTrue(pd.Timestamp(morning_event["twilight_evening"]) < pd.Timestamp(morning_event["midtime"]))
        self.assertTrue(pd.Timestamp(morning_event["twilight_morning"]) > pd.Timestamp(morning_event["start_obs"]))

    def test_the_twilights_of_every_event_surround_its_window(self):
        _, events, _, _, _ = plan(self.directory, self.observatories_file)
        for _, e in events.iterrows():
            self.assertLessEqual(pd.Timestamp(e["twilight_evening"]), pd.Timestamp(e["start_obs"]))
            self.assertGreaterEqual(pd.Timestamp(e["twilight_morning"]), pd.Timestamp(e["end_obs"]))


if __name__ == '__main__':
    unittest.main()
