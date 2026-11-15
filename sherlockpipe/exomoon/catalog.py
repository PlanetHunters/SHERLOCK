"""Catalog driver helpers: planet and moon parameters of KOI and TOI rows, the dynamical
filters (Hill-Roche, Barnes & O'Brien tidal survival, companions) and the light-curve cadences.
"""
import functools
import logging
import os
import astropy.units as u
from lcbuilder.helper import LcbuilderHelper
import numpy as np
import astropy.constants as ac
from sherlockpipe.exomoon.constants import AU_TO_M, AU_TO_RSUN, G, M_earth_to_M_sun, Mearth_to_kg, Msolar_to_kg
from sherlockpipe.exomoon.search import ExoMoonLeastSquares
from sherlockpipe.exomoon.tides import (TIDAL_K2_Q_MAX_RADIUS, TIDAL_SURVIVAL_AGE_GYR, TIDAL_SURVIVAL_K2_Q_FACTOR,  # noqa: F401
                                        max_surviving_moon_mass, max_surviving_moon_radius, moon_radius_from_mass,
                                        planet_k2_over_q)


def combine_cadence_curves(curves):
    """Combine light curves of one star taken at different cadences, each time kept once.

    A mission archive can hold the same photons at two cadences, as Kepler's short and long
    cadence of one quarter, so appending the curves counts them twice. The curves are added from
    the shortest cadence (each curve's median point spacing) to the longest, and a point is
    dropped when an already added point of a shorter cadence lies within half its spacing.
    Nothing about the mission is assumed, but each curve must hold one cadence.

    Parameters
    ----------
    curves : list of tuple
        (time, flux, flux_err) of each cadence; flux_err may be None in all of them.

    Returns
    -------
    tuple
        (time, flux, flux_err) sorted by time.
    """
    def cadence(curve):
        spacing = np.diff(np.sort(np.asarray(curve[0], dtype=float)))
        spacing = spacing[spacing > 0]
        return float(np.median(spacing)) if len(spacing) else np.inf

    kept_time, kept_flux, kept_err, dropped = [], [], [], []
    for curve in sorted((curve for curve in curves if len(curve[0]) > 0), key=cadence):
        time = np.asarray(curve[0], dtype=float)
        keep = np.ones(len(time), dtype=bool)
        if kept_time:
            shorter = np.sort(np.concatenate(kept_time))
            position = np.clip(np.searchsorted(shorter, time), 1, max(len(shorter) - 1, 1))
            distance = np.minimum(np.abs(time - shorter[position - 1]), np.abs(shorter[np.minimum(position, len(shorter) - 1)] - time))
            keep = distance > cadence(curve) / 2
        dropped.append((cadence(curve), int(np.count_nonzero(~keep))))
        kept_time.append(time[keep])
        kept_flux.append(np.asarray(curve[1], dtype=float)[keep])
        kept_err.append(None if curve[2] is None else np.asarray(curve[2], dtype=float)[keep])
    if not kept_time:
        return np.array([]), np.array([]), None
    time = np.concatenate(kept_time)
    order = np.argsort(time, kind="stable")
    errors = None if any(err is None for err in kept_err) else np.concatenate(kept_err)[order]
    if any(count for _, count in dropped):
        logging.info("Cadences combined: %s", "; ".join(f"{spacing * 1440:.2f} min, {count} points dropped as "
                                                           f"observed at a shorter cadence" for spacing, count in dropped))
    return time[order], np.concatenate(kept_flux)[order], errors


def catalog_csv_path(catalog):
    """Return the path of a local catalog table.

    Parameters
    ----------
    catalog : str
        Catalog name, such as "tois" or "kois".

    Returns
    -------
    str
        Path inside the SHERLOCK directory of the user.
    """
    return os.path.join(os.path.expanduser('~'), '.sherlockpipe', f"{catalog}.csv")


def emls_output_dir(target_name, object_of_interest):
    """Return the output directory of one planet.

    Parameters
    ----------
    target_name : str
        Name of the star.
    object_of_interest : str
        Identifier of the planet, which keeps planets of one star apart.

    Returns
    -------
    str
        Directory path.
    """
    return f"{target_name}_{object_of_interest}_EMLS/"


def mean_planet_mass(radius):
    """Return a planet mass in Earth masses from the Chen & Kipping (2017) mean mass-radius relation.

    The inverse of `moon_radius_from_mass` below the giant regime, and one Jupiter mass from
    `GIANT_RADIUS` up, as `catalog_planet_mass`. It is instant and within about 10 % of the
    forecast median, so the catalog filter uses it on thousands of rows; the targets searched
    get the forecast.
    """
    from sherlockpipe.exomoon.run import GIANT_RADIUS, JUPITER_MASS
    if radius is None or not np.isfinite(radius) or radius <= 0:
        return np.nan
    if radius >= GIANT_RADIUS:
        return JUPITER_MASS
    transition_mass = 2.04
    transition_radius = 1.008 * transition_mass ** 0.279
    if radius <= transition_radius:
        return float((radius / 1.008) ** (1 / 0.279))
    return float(transition_mass * (radius / transition_radius) ** (1 / 0.589))


def catalog_planet_mass(radius, radius_err=None):
    """Return the planet mass in Earth masses the catalog driver uses when the catalog has none.

    It is the same guess as the EMLS command (`sherlockpipe.exomoon_run.guess_planet_mass`): the
    Chen & Kipping (2017) forecast, and one Jupiter mass in the giant regime, where the radius does
    not constrain the mass. The mass sets the Hill sphere, so the widest stable moon orbit and the
    moon period grid.

    Parameters
    ----------
    radius : float
        Planet radius in Earth radii.
    radius_err : float, optional
        Radius uncertainty in Earth radii.

    Returns
    -------
    float
        Mass in Earth masses, NaN for a radius that is not finite and positive.
    """
    if radius is None or not np.isfinite(radius) or radius <= 0:
        return np.nan
    # Catalog radius errors can reach the radius itself, and the forecaster then samples radii
    # outside its model range and returns nothing: the error is capped at half the radius, and
    # the default 10 % is tried last.
    if radius_err is not None and np.isfinite(radius_err) and radius_err > 0:
        radius_err = min(float(radius_err), 0.5 * float(radius))
    else:
        radius_err = None
    return _cached_catalog_planet_mass(float(radius), radius_err)


@functools.lru_cache(maxsize=None)
def _cached_catalog_planet_mass(radius, radius_err):
    """The forecast of `catalog_planet_mass`, cached: it samples the forecaster (about 0.06 s a call)
    and catalogs repeat radius and error pairs."""
    from sherlockpipe.exomoon.run import guess_planet_mass
    for error in (radius_err, None):
        try:
            return guess_planet_mass(radius, error, warn=False)
        except ValueError:
            continue
    return np.nan


# Moon radius range of the catalog search, as fractions of the planet radius.
CATALOG_MIN_MOON_RADIUS_FRACTION = 1 / 20


CATALOG_MAX_MOON_RADIUS_FRACTION = 1 / 2


def catalog_planet_orbit(row, catalog):
    """Return the eccentricity and argument of periastron in degrees of a catalog row.

    They come from the KOI catalog (koi_eccen, koi_longp) when present, else the orbit is
    circular; the eccentricity sets the Domingos outer limit of the moon orbits.
    """
    ecc = row.get('koi_eccen', np.nan) if catalog == "kois" else np.nan
    ecc = float(ecc) if ecc is not None and np.isfinite(ecc) else 0.0
    arg_periastron = row.get('koi_longp', np.nan) if catalog == "kois" and ecc > 0 else np.nan
    arg_periastron = float(arg_periastron) if arg_periastron is not None and np.isfinite(arg_periastron) else 0.0
    return ecc, arg_periastron


def prepare_koi_table(tois_df):
    """Add the columns the EMLS driver needs to a KOI table.

    Parameters
    ----------
    tois_df : pandas.DataFrame
        KOI table.

    Returns
    -------
    pandas.DataFrame
        The table with the derived columns.
    """
    tois_df["Planet Radius (R_Earth) err"] = tois_df["koi_prad_err1"].fillna(0.1 * tois_df["Planet Radius (R_Earth)"])
    if "Predicted Mass (M_Earth)" not in tois_df.columns:
        # Left empty: the catalog filter uses the mean relation (`mean_planet_mass`) and the targets
        # searched the forecast (`catalog_planet_mass`), since forecasting thousands of rows takes
        # several minutes (about 0.3 s a row).
        tois_df["Predicted Mass (M_Earth)"] = np.nan
    if "Stellar Mass (M_Sun)" not in tois_df.columns:
        g = 10 ** tois_df["Stellar log(g) (cm/s^2)"] / 100
        r_m = tois_df["koi_srad"] * ac.R_sun.value
        tois_df["Stellar Mass (M_Sun)"] = g * r_m ** 2 / (G * Msolar_to_kg)
    if "Stellar Radius (R_Sun)" not in tois_df.columns:
        tois_df["Stellar Radius (R_Sun)"] = tois_df["koi_srad"]
    tois_df["Planet SNR"] = tois_df["koi_model_snr"]
    tois_df["planet_impact"] = tois_df["koi_impact"].fillna(0)
    return tois_df


def koi_planet_inclination(impact_param, star_mass, star_radius, planet_period):
    """Return the inclination of a KOI in degrees from its impact parameter.

    Parameters
    ----------
    impact_param : float
        Impact parameter in stellar radii.
    star_mass : float
        Stellar mass in solar masses.
    star_radius : float
        Stellar radius in solar radii.
    planet_period : float
        Planet period in days.

    Returns
    -------
    float
        Inclination in degrees, at most 90.
    """
    a_rsun = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period) * AU_TO_RSUN
    return min(np.rad2deg(np.arccos(np.clip(impact_param * star_radius / a_rsun, 0, 1))), 90)


def catalog_planet_params(row, catalog, star_mass, star_radius, first_time, bjd_offset):
    """Return the planet and moon-radius inputs of EMLS for one catalog row.

    Parameters
    ----------
    row : pandas.Series
        Catalog row, after `prepare_koi_table` for KOIs.
    catalog : str
        Catalog name, ``"kois"`` or a TOI-like catalog.
    star_mass : float
        Stellar mass in solar masses.
    star_radius : float
        Stellar radius in solar radii.
    first_time : float
        First time of the light curve, used to move the epoch into the data.
    bjd_offset : float
        Offset between the catalog epoch and the light curve time.

    Returns
    -------
    dict
        Keyword values named as the `ExoMoonLeastSquares` arguments, plus
        "max_surviving_moon_radius". The moon radius range is Rp/20 to Rp/2; the largest moon that
        survives tidal migration with the generous margin (`max_surviving_moon_radius`) is only
        reported, because the bound is too uncertain to cut the search.
    """
    planet_radius = row['Planet Radius (R_Earth)']
    planet_period = row['Period (days)']
    planet_t0 = row['Epoch (BJD)'] - bjd_offset
    planet_t0 = first_time + (planet_t0 - first_time) % planet_period
    planet_mass = row['Predicted Mass (M_Earth)']
    if planet_mass is None or np.isnan(planet_mass):
        planet_mass = catalog_planet_mass(planet_radius, row.get('Planet Radius (R_Earth) err'))
    planet_ecc, planet_arg_periastron = catalog_planet_orbit(row, catalog)
    # The tidal survival bound (Barnes & O'Brien 2002, generous margin) is reported, not used to
    # cut the radius range; the catalog filter has already dropped targets where even it is below
    # the smallest moon searched.
    min_radius = planet_radius * CATALOG_MIN_MOON_RADIUS_FRACTION
    surviving_radius = max_surviving_moon_radius(star_mass, planet_mass, planet_radius, planet_period, planet_ecc)
    max_radius = planet_radius * CATALOG_MAX_MOON_RADIUS_FRACTION
    planet_semimajor_axis = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period)
    if catalog == "kois":
        planet_impact_param = float(row["planet_impact"])
        planet_inc = koi_planet_inclination(planet_impact_param, star_mass, star_radius, planet_period)
        planet_inc_err = 0.5
    else:
        planet_inc = 90
        planet_inc_err = 3
        planet_impact_param = LcbuilderHelper.convert_from_to(planet_semimajor_axis, u.au, u.R_sun) / star_radius * \
            np.cos(np.deg2rad(planet_inc))
    return {
        'planet_radius': planet_radius,
        'planet_radius_err': row['Planet Radius (R_Earth) err'],
        'planet_period': planet_period,
        'planet_t0': planet_t0,
        'planet_duration': row['Duration (hours)'] / 24,
        'planet_inc': planet_inc,
        'planet_inc_err': planet_inc_err,
        'planet_ecc': planet_ecc,
        'planet_ecc_err': 0.0,
        'planet_arg_periastron': planet_arg_periastron,
        'planet_mass': planet_mass,
        'planet_semimajor_axis': planet_semimajor_axis,
        'planet_impact_param': planet_impact_param,
        'min_radius': min_radius,
        'max_radius': max_radius,
        'max_surviving_moon_radius': surviving_radius,
    }
