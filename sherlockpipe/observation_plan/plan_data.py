"""
Data behind the observation plan report: what the planner found, enriched with what an observer needs to choose among
the opportunities (altitude and airmass at the mid-transit, local time, a ranking) and the curves of a night.
"""
import numpy as np
import pandas as pd
import astropy.units as u
from astroplan import FixedTarget, Observer, moon_illumination
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time

from sherlockpipe.observation_plan.planner import Planner

#: Altitude (degrees) above which a higher target stops adding to the score of an opportunity.
SCORE_ALTITUDE_CAP = 60.0
SCORE_DESCRIPTION = "observable fraction × min(altitude, 60°) / 60° × (1 − ½ · moon illumination)"
#: Sun altitude below which it is nautical night, the same limit the planner uses.
NIGHT_SUN_ALTITUDE = -12.0
TIME_COLUMNS = ("start_obs", "end_obs", "ingress", "egress", "midtime", "twilight_evening", "twilight_morning")


def hours_from_text(text):
    """Hours of an ``H:MM`` text, the format in which the planner writes the timing errors."""
    hours, minutes = str(text).split(":")
    return int(hours) + int(minutes) / 60


def observer_of(observatory_row):
    """The astroplan Observer of an observatory row, or None when it has no usable location."""
    if not Planner.has_coordinates(observatory_row):
        return None
    return Observer(latitude=float(observatory_row["lat"]) * u.deg, longitude=float(observatory_row["lon"]) * u.deg,
                    elevation=float(observatory_row["alt"]) * u.m)


def observers_of(observatories):
    """``{name: Observer or None}`` for every observatory of the plan."""
    return {row["name"]: observer_of(row) for _, row in observatories.iterrows()}


def quality(observable, altitude, moon_illumination_fraction):
    """
    The score of an opportunity, between 0 and 1: ``observable × min(altitude, 60°) / 60° × (1 − ½ · illumination)``.
    An unknown altitude (observatory without location) does not penalise.
    """
    altitude_factor = 1.0 if altitude is None or not np.isfinite(altitude) else \
        float(np.clip(altitude, 0, SCORE_ALTITUDE_CAP)) / SCORE_ALTITUDE_CAP
    return float(observable) * altitude_factor * (1 - 0.5 * float(moon_illumination_fraction))


def enrich_events(events, observatories, ra, dec):
    """
    Adds to the events of the plan what an observer needs to choose among them.

    New columns: ``event`` (mid-transit rounded to the minute, it identifies the transit), ``err_h`` (timing error in
    hours), ``tz`` (hours from UTC), ``local_t0``, ``alt_t0`` and ``airmass`` at the mid-transit (NaN for an observatory
    without location), ``quality`` and ``rank`` (1 is the best opportunity). The time columns become datetimes.

    :param events: the observables of the planner
    :param observatories: the observatories of the plan (name, lat, lon, alt)
    :return: a new DataFrame sorted by rank
    """
    columns = list(events.columns) + ["event", "err_h", "tz", "local_t0", "alt_t0", "airmass", "quality", "rank"]
    if len(events) == 0:
        return pd.DataFrame(columns=columns)
    enriched = events.copy().reset_index(drop=True)
    for column in TIME_COLUMNS:
        enriched[column] = pd.to_datetime(enriched[column])
    enriched["event"] = enriched["midtime"].dt.round("min")
    enriched["err_h"] = enriched["midtime_up_err_h"].map(hours_from_text)
    enriched["tz"] = pd.to_numeric(enriched["timezone"], errors="coerce").fillna(0.0)
    enriched["local_t0"] = enriched["event"] + pd.to_timedelta(enriched["tz"], unit="h")
    enriched["alt_t0"] = np.nan
    target = FixedTarget(SkyCoord(ra, dec, unit="deg"))
    observers = observers_of(observatories)
    for name, group in enriched.groupby("observatory"):
        observer = observers.get(name)
        if observer is None:
            continue
        times = Time(list(group["midtime"].dt.to_pydatetime()), scale="utc")
        enriched.loc[group.index, "alt_t0"] = observer.altaz(times, target).alt.deg
    altitude = enriched["alt_t0"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        airmass = np.where(altitude > 0, 1 / np.sin(np.radians(np.where(altitude > 0, altitude, 90))), np.nan)
    enriched["airmass"] = airmass
    enriched["quality"] = [quality(o, a, m) for o, a, m in zip(enriched["observable"], altitude, enriched["moon_phase"])]
    enriched = enriched.sort_values(["quality", "event", "observatory"], ascending=[False, True, True]) \
        .reset_index(drop=True)
    enriched["rank"] = np.arange(1, len(enriched) + 1)
    return enriched[columns]


def top_transits(enriched, max_cards):
    """
    The transits that deserve an event card: those observable from at least one observatory, best first.

    :param enriched: the result of :func:`enrich_events`
    :param int max_cards: how many at most; zero or less means all of them
    :return: list of ``event`` timestamps
    """
    if len(enriched) == 0:
        return []
    best = enriched.sort_values("rank").drop_duplicates("event")
    transits = list(best["event"])
    return transits if max_cards is None or max_cards <= 0 else transits[:max_cards]


def highest_altitude(latitude, dec):
    """The highest altitude (degrees) a target of declination ``dec`` reaches from latitude ``latitude``."""
    return 90.0 - abs(float(latitude) - float(dec))


def observatory_summary(observatories, enriched, dec, min_altitude):
    """
    What happened with each observatory, so that one without events is explained instead of silently missing.

    :return: list of dicts ``name, events, transits, reason`` where reason is None for an observatory with events
    """
    summary = []
    for _, row in observatories.iterrows():
        mine = enriched[enriched["observatory"] == row["name"]] if len(enriched) else enriched
        reason = None
        if len(mine) == 0:
            if not Planner.has_coordinates(row):
                reason = "no location: nothing could be planned"
            elif highest_altitude(row["lat"], dec) < min_altitude:
                reason = (f"the target never rises above {min_altitude:g}° "
                          f"(highest altitude {highest_altitude(row['lat'], dec):.0f}°)")
            else:
                reason = "no transit meets the night, altitude, moon and timing constraints"
        summary.append({"name": row["name"], "events": len(mine), "transits": mine["event"].nunique() if len(mine) else 0,
                        "reason": reason})
    return summary


def night_curves(observer, target, centre, before_hours=10.0, after_hours=3.0, points=261, moon_min_dist=30.0,
                 moon_max_dist=70.0):
    """
    The target, the sun and the moon seen from an observatory around a time.

    :param observer: the astroplan Observer
    :param target: the FixedTarget
    :param centre: the time the window is built around (pandas Timestamp, datetime or astropy Time)
    :return: dict with ``time`` (datetimes), ``altitude``, ``sun`` and ``moon`` (degrees) and ``moon_ok``: whether the
        moon is far enough for its illumination, with the planner's rule
    """
    times = Time(centre) + np.linspace(-before_hours, after_hours, points) * u.hour
    altitude = observer.altaz(times, target).alt.deg
    sun = observer.sun_altaz(times).alt.deg
    moon_altitude = observer.moon_altaz(times).alt.deg
    separation = get_body("moon", times).separation(target.coord).deg
    required = moon_min_dist + (moon_max_dist - moon_min_dist) * moon_illumination(times)
    return {"time": times.to_datetime(), "altitude": altitude, "sun": sun, "moon": moon_altitude,
            "moon_ok": separation >= required}


def observable_mask(curves, min_altitude):
    """Where the target is observable: nautical night, above the minimum altitude and with the moon far enough."""
    return (curves["altitude"] >= min_altitude) & (curves["sun"] < NIGHT_SUN_ALTITUDE) & curves["moon_ok"]
