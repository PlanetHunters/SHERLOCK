"""Shared helpers for the tests of the plan module (observation planner and report)."""
import contextlib
import os

from astropy.time import Time
from astropy.utils import iers

from sherlockpipe.observation_plan.planner import Planner

#: Target of every test: TIC 224298134 b, P = 1.49233 d, a 0.69 h transit, northern sky (Dec +57.75).
RA, DEC = 185.3394, 57.7525
EPOCH_JD, EPOCH_ERR = 2458713.45, 0.002
PERIOD, PERIOD_ERR, DURATION_H = 1.49233, 2e-5, 0.69

OBSERVATORIES_CSV = """name,tz,lat,lon,alt
Calar-Alto,,37.2236,-2.5463,2168
Trappist-North,,31.2061,-7.8664,2751
Trappist-South,,-29.2563,-70.738,2375
"""

EVENT_COLUMNS = ['observatory', 'timezone', 'start_obs', 'end_obs', 'ingress', 'egress', 'midtime', 'midtime_up_err_h',
                 'midtime_low_err_h', 'twilight_evening', 'twilight_morning', 'observable', 'moon_phase', 'moon_dist']


@contextlib.contextmanager
def offline_astropy():
    """Astropy would try to download the IERS tables; the bundled ones cover the dates of these tests."""
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        yield


def write_observatories(directory, text=OBSERVATORIES_CSV):
    path = os.path.join(directory, "observatories.csv")
    with open(path, "w") as f:
        f.write(text)
    return path


def plan(object_dir, observatories_file, since="2024-03-01T12:00:00", max_days=12, epoch_err=EPOCH_ERR,
         period_err=PERIOD_ERR, min_altitude=30, transit_fraction=0.5, baseline=0.5, error_alert=True, epoch=EPOCH_JD,
         tz=None, lat=None, lon=None, alt=None, cpus=2):
    """Runs Planner.create_observation_observables with the target of these tests."""
    return Planner.create_observation_observables(
        "TIC 224298134", object_dir, RA, DEC, Time(since, scale="utc"), "TIC 224298134 b", epoch, epoch_err, epoch_err,
        PERIOD, period_err, period_err, DURATION_H, observatories_file, tz, lat, lon, alt, max_days, min_altitude, 30, 70,
        transit_fraction, baseline, error_alert, 'jd', cpus)
