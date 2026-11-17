"""The classes to run the plan task"""
import logging
from multiprocessing import Pool

from astroplan import moon_illumination, Constraint
import os
import astroplan
from astropy.coordinates import SkyCoord, get_body
import astropy.units as u
from astroplan import EclipsingSystem
import pandas as pd
import numpy as np
from astroplan import (Observer, FixedTarget, AtNightConstraint, AltitudeConstraint)

from astropy.time import Time, TimeDelta
from timezonefinder import TimezoneFinder
from pytz import timezone, utc


class MoonIlluminationSeparationConstraint(Constraint):
    """
    Constrain the distance between the Earth's moon and some targets.
    """

    def __init__(self, min_dist, max_dist):
        """
        Parameters
        ----------
        min_dist : `~astropy.units.Quantity`
            Minimum moon distance when moon illumination is 0
        max_dist : `~astropy.units.Quantity`
            Maximum moon distance when moon illumination is 1
        """
        self.min_dist = min_dist
        self.max_dist = max_dist

    def compute_constraint(self, times, observer, targets):
        """
        Computes the observability of the moon given a minimum distance to the moon between self.min_dist (for
        illumination = 0) and self.max_dist (for illumination = 1) by interpolating an intermediate distance from those
        two values following a linear regression.

        :param times: the times to compute the constraint for
        :param observer: the observer to compute the constraint for
        :param targets: the list of targets to compute the constraint for
        :return: the positive mask for target being observable for the given times and observer given the constraint is matched
        """
        # removed the location argument here, which causes small <1 deg
        # inaccuracies, but it is needed until astropy PR #5897 is released
        # which should be astropy 1.3.2
        moon = get_body("moon", times)
        # note to future editors - the order matters here
        # moon.separation(targets) is NOT the same as targets.separation(moon)
        # the former calculates the separation in the frame of the moon coord
        # which is GCRS, and that is what we want.
        moon_separation = moon.separation(targets)
        illumination = moon_illumination(times)
        min_dist = self.min_dist.value + (self.max_dist.value - self.min_dist.value) * illumination
        mask = min_dist <= moon_separation.degree
        return mask


class PlannerInput:
    """Input data class for a single observation planning event."""

    def __init__(self, observatory_row, midtransit_time, observer_site, midtransit_times, ingress_egress_times,
                 constraints, moon_for_midtransit_times, moon_dist_midtransit_times, moon_phase_midtransit_times,
                 transits_since_epoch, midtransit_time_low_err, midtransit_time_up_err, low_err_delta,
                 up_err_delta, i, plan_dir, target, min_altitude, transit_fraction, baseline,
                 error_alert) -> None:
        """Initialize the planner input.

        Parameters
        ----------
        observatory_row : Series
            The observatory data row from the observatories DataFrame.
        midtransit_time : Time
            The midtransit time for this event.
        observer_site : Observer
            The astroplan Observer object for the observatory.
        midtransit_times : Time
            Array of all midtransit times.
        ingress_egress_times : list
            List of ingress/egress time pairs.
        constraints : list
            List of astroplan observation constraints.
        moon_for_midtransit_times : SkyCoord
            The moon coordinates at midtransit times.
        moon_dist_midtransit_times : Quantity
            The moon-target angular distance at each midtransit time.
        moon_phase_midtransit_times : ndarray
            The moon illumination phase at each midtransit time.
        transits_since_epoch : ndarray
            The number of transits since the epoch for each midtransit.
        midtransit_time_low_err : ndarray
            Lower error of midtransit times in hours.
        midtransit_time_up_err : ndarray
            Upper error of midtransit times in hours.
        low_err_delta : TimeDelta
            The lower error as an astropy time delta.
        up_err_delta : TimeDelta
            The upper error as an astropy time delta.
        i : int
            The index of this midtransit in the array.
        plan_dir : str
            The output directory for plan artifacts.
        target : FixedTarget
            The astroplan FixedTarget object.
        min_altitude : float
            The minimum altitude in degrees.
        transit_fraction : float
            The required transit observable fraction.
        baseline : float
            The baseline time in hours.
        error_alert : bool
            Whether to raise an error alert when uncertainty is too large.
        """
        self.observatory_row = observatory_row
        self.midtransit_time = midtransit_time
        self.observer_site = observer_site
        self.midtransit_times = midtransit_times
        self.ingress_egress_times = ingress_egress_times
        self.constraints = constraints
        self.moon_for_midtransit_times = moon_for_midtransit_times
        self.moon_dist_midtransit_times = moon_dist_midtransit_times
        self.moon_phase_midtransit_times = moon_phase_midtransit_times
        self.transits_since_epoch = transits_since_epoch
        self.midtransit_time_low_err = midtransit_time_low_err
        self.midtransit_time_up_err = midtransit_time_up_err
        self.low_err_delta = low_err_delta
        self.up_err_delta = up_err_delta
        self.i = i
        self.plan_dir = plan_dir
        self.target = target
        self.min_altitude = min_altitude
        self.transit_fraction = transit_fraction
        self.baseline = baseline
        self.error_alert = error_alert


class Planner:
    """
    Class containing the main method to generate the observation plan events.
    """

    @staticmethod
    def create_observation_observables(object_id, object_dir, ra, dec, since, name, epoch, epoch_low_err, epoch_up_err,
                                       period, period_low_err, period_up_err, duration,
                                       observatories_file, timezone, latitude, longitude, altitude,
                                       max_days, min_altitude, moon_min_dist, moon_max_dist, transit_fraction, baseline,
                                       error_alert=True, time_unit='jd', cpus=os.cpu_count() - 1):
        """
        Computes the observation windows for the given target parameters.

        :param object_id: the candidate id
        :param object_dir: the candidate directory
        :param ra: right ascension of the target
        :param dec: declination of the target
        :param since: starting plan date
        :param name: the name given to the candidate
        :param epoch: the candidate epoch
        :param epoch_low_err: the candidate epoch's lower error
        :param epoch_up_err: the candidate epoch's upper error
        :param period: the candidate period
        :param period_low_err: the candidate period's lower error
        :param period_up_err: the candidate period's upper error
        :param duration: the candidate duration
        :param observatories_file: the file containing the observatories file (csv format)
        :param timezone: the timezone of the observatory (if observatories_file=None)
        :param latitude: the latitude of the observatory (if observatories_file=None)
        :param longitude: the longitude of the observatory (if observatories_file=None)
        :param altitude: the altitude of the observatory (if observatories_file=None)
        :param max_days: the maximum number of days to compute the observables
        :param min_altitude: the minimum altitude of the target above the horizon
        :param moon_min_dist: the minimum moon distance for moon illumination = 0
        :param moon_max_dist: the minimum moon distance for moon illumination = 1
        :param transit_fraction: the minimum transit observability (0.25 for at least ingress/egress, 0.5 for ingress/egress + midtime, 1 for ingress, egress and midtime).
        :param baseline: the required baseline in hours.
        :param error_alert: whether to create the alert date to signal imprecise observations
        :param time_unit: the unit of the light curve data
        :return: the generated data and target folders observatories_df, observables_df, alert_date, plan_dir, midtransit_times
        :return: observatories_df containing the observatories used for the computation
        :return: observables_df containing all the observation windows that passed the plan
        :return: alert_date in case the plan reached a date where the observation uncertainty was too high
        :return: midtransit_times all the mid-transit times of the plan, also those that nobody can observe
        """
        if observatories_file is not None:
            observatories_df = pd.read_csv(observatories_file, comment='#')
        else:
            observatories_df = pd.DataFrame([{'name': 'Obs-1', 'tz': timezone, 'lat': latitude, 'lon': longitude,
                                              'alt': altitude}])
            for column in ('tz', 'lat', 'lon', 'alt'):
                observatories_df[column] = pd.to_numeric(observatories_df[column])
        # TODO probably convert epoch to proper JD
        primary_eclipse_time = Time(epoch, format=time_unit, scale="tdb")
        target = FixedTarget(SkyCoord(str(ra) + ' ' + str(dec), unit=(u.deg, u.deg)))
        n_transits = int(max_days // period)
        system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time, orbital_period=u.Quantity(period, unit="d"),
                                 duration=u.Quantity(duration, unit="h"), name=name)
        observables_df = pd.DataFrame(columns=['observatory', 'timezone', 'start_obs', 'end_obs', 'ingress', 'egress',
                                               'midtime', "midtime_up_err_h", "midtime_low_err_h", 'twilight_evening',
                                               'twilight_morning', 'observable', 'moon_phase', 'moon_dist'])
        plan_dir = object_dir + "/plan"
        if not os.path.exists(plan_dir):
            os.mkdir(plan_dir)
        alert_date = None
        midtransit_times = system.next_primary_eclipse_time(since, n_eclipses=n_transits)
        ingress_egress_times = system.next_primary_ingress_egress_time(since, n_eclipses=n_transits)
        moon_for_midtransit_times = get_body("moon", midtransit_times)
        moon_dist_midtransit_times = moon_for_midtransit_times.separation(SkyCoord(ra, dec, unit="deg"))
        moon_phase_midtransit_times = np.round(astroplan.moon_illumination(midtransit_times), 2)
        transits_since_epoch = np.round((midtransit_times - primary_eclipse_time).jd / period)
        midtransit_time_low_err = np.round(
            (((transits_since_epoch * period_low_err) ** 2 + epoch_low_err ** 2) ** (1 / 2)) * 24, 2)
        midtransit_time_up_err = np.round(
            (((transits_since_epoch * period_up_err) ** 2 + epoch_up_err ** 2) ** (1 / 2)) * 24, 2)
        low_err_delta = TimeDelta(midtransit_time_low_err * 3600, format='sec')
        up_err_delta = TimeDelta(midtransit_time_up_err * 3600, format='sec')
        planner_inputs = []
        for index, observatory_row in observatories_df.iterrows():
            if not Planner.has_coordinates(observatory_row):
                observer_site = None
                constraints = []
            else:
                observer_site = Observer(latitude=observatory_row["lat"], longitude=observatory_row["lon"],
                                         elevation=u.Quantity(observatory_row["alt"], unit="m"))
                constraints = [AtNightConstraint.twilight_nautical(), AltitudeConstraint(min=min_altitude * u.deg)]
            constraints = constraints + [MoonIlluminationSeparationConstraint(min_dist=moon_min_dist * u.deg,
                                                                max_dist=moon_max_dist * u.deg)]
            for i, midtransit_time in enumerate(midtransit_times):
                planner_inputs = planner_inputs + [
                    PlannerInput(observatory_row, midtransit_time, observer_site, midtransit_times,
                                 ingress_egress_times,
                                 constraints, moon_for_midtransit_times, moon_dist_midtransit_times,
                                 moon_phase_midtransit_times,
                                 transits_since_epoch, midtransit_time_low_err, midtransit_time_up_err, low_err_delta,
                                 up_err_delta, i, plan_dir, target, min_altitude, transit_fraction, baseline,
                                 error_alert)]
        with Pool(processes=cpus) as pool:
            alert_dates_and_observables = pool.map(Planner.plan_event, planner_inputs)
        alert_dates = [item[0] for item in alert_dates_and_observables]
        alert_date = alert_dates[0] if len(alert_dates) > 0 else None
        for each_alert_date in alert_dates:
            if each_alert_date is not None and each_alert_date < alert_date:
                alert_date = each_alert_date
        observables = [item[1] for item in alert_dates_and_observables]
        for observable in observables:
            if observable is not None:
                observables_df = pd.concat([observables_df, pd.DataFrame([observable])], ignore_index=True)

        observables_df = observables_df.sort_values(["midtime", "observatory"], ascending=True)
        observables_df.to_csv(plan_dir + "/observation_plan.csv", index=False)
        print("Observation plan created in directory: " + object_dir)
        return observatories_df, observables_df, alert_date, plan_dir, midtransit_times

    @staticmethod
    def has_coordinates(observatory_row):
        """Whether an observatory row has a usable latitude, longitude and altitude."""
        try:
            return all(np.isfinite(float(observatory_row[key])) for key in ("lat", "lon", "alt"))
        except (TypeError, ValueError, KeyError):
            return False

    @staticmethod
    def night_of(observer_site, time):
        """
        The nautical night that contains a time: the evening twilight before it and the morning twilight after it.

        :param observer_site: the astroplan Observer
        :param time: a time that lies inside a night
        :return: the evening twilight and the morning twilight, both astropy Time
        """
        evening = observer_site.twilight_evening_nautical(time, which="previous")
        morning = observer_site.twilight_morning_nautical(time, which="next")
        return evening, morning

    @staticmethod
    def plan_event(planner_input: PlannerInput):
        """Compute the observability of a single transit event.

        Evaluates whether a transit event is observable given the constraints and returns the observability data.

        Parameters
        ----------
        planner_input : PlannerInput
            The input data for the observation event.

        Returns
        -------
        tuple
            A tuple of (alert_date, observable_dict) where alert_date is a Time
            object if the event uncertainty is too large (or None), and
            observable_dict contains the observation details (or None if not observable). It is (None, None) when the
            event cannot be planned because of an error, which is logged.
        """
        try:
            ingress = planner_input.ingress_egress_times[planner_input.i][0]
            egress = planner_input.ingress_egress_times[planner_input.i][1]
            lowest_ingress = ingress - planner_input.low_err_delta[planner_input.i]
            highest_egress = egress + planner_input.up_err_delta[planner_input.i]
            if planner_input.error_alert and (highest_egress - lowest_ingress).jd > 0.33:
                return (planner_input.midtransit_time, None)
            baseline_low = lowest_ingress - planner_input.baseline * u.hour
            baseline_up = highest_egress + planner_input.baseline * u.hour
            transit_times = baseline_low + (baseline_up - baseline_low) * np.linspace(0, 1, 100)
            if planner_input.observer_site is None:
                observable_transit_times = np.full(transit_times.shape, True)
            else:
                observable_transit_times = astroplan.is_event_observable(planner_input.constraints,
                                                                         planner_input.observer_site,
                                                                         planner_input.target,
                                                                         times=transit_times)[0]
            observable_transit_times_true = np.argwhere(observable_transit_times)
            observable = len(observable_transit_times_true) / 100
            if observable < planner_input.transit_fraction:
                return (None, None)
            start_obs = transit_times[observable_transit_times_true[0]][0]
            end_obs = transit_times[observable_transit_times_true[len(observable_transit_times_true) - 1]][0]
            twilight_evening = None
            twilight_morning = None
            if planner_input.observer_site is not None:
                # The night is the one that holds the observable samples, not the one nearest to the mid-time, which
                # can be the next one and left the window starting after it ended.
                twilight_evening, twilight_morning = Planner.night_of(planner_input.observer_site, start_obs)
                if twilight_evening > start_obs:
                    start_obs = twilight_evening
                if twilight_morning < end_obs:
                    end_obs = twilight_morning
            moon_dist = round(planner_input.moon_dist_midtransit_times[planner_input.i].degree)
            moon_phase = planner_input.moon_phase_midtransit_times[planner_input.i]
            # TODO get is_event_observable for several parts of the transit (ideally each 5 mins) to get the proper observable percent. Also with baseline
            if planner_input.observatory_row["tz"] is not None and not np.isnan(planner_input.observatory_row["tz"]):
                observer_timezone = planner_input.observatory_row["tz"]
            else:
                observer_timezone = Planner.get_offset(planner_input.observatory_row["lat"],
                                                       planner_input.observatory_row["lon"],
                                                       planner_input.midtransit_time.datetime)
            observable_dict = {"observatory": planner_input.observatory_row["name"],
                               "timezone": observer_timezone, "ingress": ingress.isot,
                               "start_obs": start_obs.isot, "end_obs": end_obs.isot,
                               "egress": egress.isot, "midtime": planner_input.midtransit_time.isot,
                               "midtime_up_err_h":
                                   str(int(planner_input.midtransit_time_up_err[planner_input.i] // 1)) + ":" +
                                   str(int(planner_input.midtransit_time_up_err[planner_input.i] % 1 * 60)).zfill(2),
                               "midtime_low_err_h":
                                   str(int(planner_input.midtransit_time_low_err[planner_input.i] // 1)) + ":" +
                                   str(int(planner_input.midtransit_time_low_err[planner_input.i] % 1 * 60)).zfill(2),
                               "twilight_evening": twilight_evening.isot if twilight_evening is not None else None,
                               "twilight_morning": twilight_morning.isot if twilight_morning is not None else None,
                               "observable": observable, "moon_phase": moon_phase,
                               "moon_dist": moon_dist}
            return (None, observable_dict)
        except Exception:
            logging.exception("Error when planning one observation event.")
            return (None, None)

    @staticmethod
    def get_offset(lat, lng, datetime):
        """
        Returns a location's time zone offset from UTC in minutes.

        :param lat: geographical latitude
        :param lng: geographical longitude
        :param datetime: the UTC time
        """
        tf = TimezoneFinder()
        tz_target = timezone(tf.certain_timezone_at(lng=lng, lat=lat))
        if tz_target is None:
            return None
        today_target = tz_target.localize(datetime)
        today_utc = utc.localize(datetime)
        return (today_utc - today_target).total_seconds() / 3600
