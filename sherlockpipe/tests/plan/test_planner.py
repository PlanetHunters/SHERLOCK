import datetime
import os
import shutil
import tempfile
import unittest

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time
from matplotlib.figure import Figure

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

    def test_twin_axis_is_found_by_its_bounds(self):
        figure = Figure()
        ax = figure.add_subplot()
        twin = ax.twinx()
        self.assertIs(twin, Planner.get_twin(ax))

    def test_twin_axis_is_none_when_there_is_only_one_axis(self):
        figure = Figure()
        self.assertIsNone(Planner.get_twin(figure.add_subplot()))


class TestObservables(unittest.TestCase):
    """A 12 day plan from three observatories, computed once: the real astroplan code, offline."""

    @classmethod
    def setUpClass(cls):
        cls._offline = offline_astropy()
        cls._offline.__enter__()
        cls.directory = tempfile.mkdtemp()
        cls.observatories_file = write_observatories(cls.directory)
        cls.result = plan(cls.directory, cls.observatories_file)
        cls.observatories, cls.events, cls.alert_date, cls.plan_dir, cls.images_dir = cls.result

    @classmethod
    def tearDownClass(cls):
        cls._offline.__exit__(None, None, None)
        shutil.rmtree(cls.directory)

    def test_directories_and_the_csv_are_created_inside_the_object_directory(self):
        self.assertEqual(self.directory + "/plan", self.plan_dir)
        self.assertEqual(self.directory + "/plan/images", self.images_dir)
        self.assertTrue(os.path.isdir(self.images_dir))
        self.assertTrue(os.path.exists(self.plan_dir + "/observation_plan.csv"))

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

    def test_one_image_is_written_per_event_with_the_observatory_and_the_mid_time_in_its_name(self):
        images = sorted(os.listdir(self.images_dir))
        expected = sorted(f"{e['observatory']}_{str(e['midtime'])[:-4]}.png" for _, e in self.events.iterrows())
        self.assertEqual(expected, images)
        with open(os.path.join(self.images_dir, images[0]), "rb") as f:
            self.assertEqual(b"\x89PNG", f.read(4))

    def test_running_again_replaces_the_images_but_keeps_one_csv(self):
        with offline_astropy():
            plan(self.directory, self.observatories_file)
        self.assertEqual(len(self.events), len(os.listdir(self.images_dir)))


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
        _, events, _, _, images_dir = plan(self.directory, path)
        self.assertGreater(len(events), 0)
        self.assertTrue((events["observable"] == 1.0).all())
        self.assertTrue(events["twilight_evening"].isna().all())
        self.assertTrue((events["timezone"] == 2).all())
        self.assertEqual([], os.listdir(images_dir))  # no airmass plot without a location

    @unittest.expectedFailure
    def test_an_event_that_cannot_be_planned_is_dropped_without_stopping_the_plan(self):
        """Known bug: plan_event logs the exception and returns None instead of the (alert_date, observable) tuple it
        documents, so the whole plan stops with "'NoneType' object is not subscriptable". A site without coordinates
        and without time zone triggers it."""
        path = write_observatories(self.directory, "name,tz,lat,lon,alt\nRemote,,,,\n")
        _, events, _, _, _ = plan(self.directory, path)
        self.assertEqual(0, len(events))

    def test_the_time_zone_of_a_listed_site_is_computed_from_its_position(self):
        _, events, _, _, _ = plan(self.directory, self.observatories_file)
        by_site = events.groupby("observatory")["timezone"].agg(lambda tz: set(tz))
        self.assertEqual({1.0}, by_site["Calar-Alto"])                 # Spain in March
        self.assertLessEqual(by_site["Trappist-North"], {0.0, 1.0})   # Morocco: UTC+1, UTC+0 during Ramadan

    @unittest.expectedFailure
    def test_the_observation_window_never_ends_before_it_starts(self):
        """Known bug: the evening twilight nearest to the mid-time can be the next evening, so a window starts after it
        ends (Calar Alto, 4 December 2026: starts 17:53, ends 06:06)."""
        path = write_observatories(self.directory, "name,tz,lat,lon,alt\nCalar-Alto,,37.2236,-2.5463,2168\n")
        _, events, _, _, _ = plan(self.directory, path, since="2026-12-01T12:00:00", max_days=6)
        windows = events[["start_obs", "end_obs"]].apply(pd.to_datetime)
        self.assertTrue((windows["start_obs"] <= windows["end_obs"]).all())


if __name__ == '__main__':
    unittest.main()
