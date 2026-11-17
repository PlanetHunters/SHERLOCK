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


def sample_events():
    """Events of three transits with their windows, as the planner writes them (UTC, ISO text)."""
    import pandas as pd

    def row(observatory, tz, mid, start, end, observable, moon, dist, evening, morning, err="0:51"):
        base = pd.Timestamp(mid)
        ingress, egress = base - pd.Timedelta(minutes=20.7), base + pd.Timedelta(minutes=20.7)
        f = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S.000")
        return [observatory, tz, f(start), f(end), f(ingress), f(egress), f(base), err, err,
                None if evening is None else f(evening), None if morning is None else f(morning), observable, moon, dist]

    rows = [
        row("Calar-Alto", 1.0, "2026-12-07T05:38:44", "2026-12-07T03:56:26", "2026-12-07T06:08:42", 0.65, 0.03, 90,
            "2026-12-06T17:53:46", "2026-12-07T06:09:13"),
        row("Teide-OT", 0.0, "2026-12-07T05:38:44", "2026-12-07T03:56:26", "2026-12-07T06:50:08", 0.96, 0.29, 111,
            "2026-12-06T19:04:27", "2026-12-07T06:50:08"),
        row("Remote", 2.0, "2026-12-07T05:38:44", "2026-12-07T05:00:00", "2026-12-07T06:30:00", 1.0, 0.5, 45, None, None,
            "1:05"),
        row("Calar-Alto", 1.0, "2026-12-10T05:16:39", "2026-12-10T03:34:21", "2026-12-10T06:11:25", 0.77, 0.01, 111,
            "2026-12-09T17:54:06", "2026-12-10T06:11:29"),
        row("Teide-OT", 0.0, "2026-12-13T04:55:20", "2026-12-13T03:12:00", "2026-12-13T06:50:41", 0.93, 0.15, 125,
            "2026-12-12T19:05:00", "2026-12-13T06:52:00", "0:52"),
    ]
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def sample_observatories():
    """Two northern observatories, one that never sees the target, and one without location."""
    import numpy as np
    import pandas as pd
    return pd.DataFrame({"name": ["Calar-Alto", "Teide-OT", "Trappist-South", "Remote"], "tz": [1, 0, -4, 2],
                         "lat": [37.2236, 28.3, -29.2563, np.nan], "lon": [-2.5463, -16.5, -70.738, np.nan],
                         "alt": [2168, 2390, 2375, np.nan]})
