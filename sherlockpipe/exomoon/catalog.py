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


# Moon survival against tidal migration (Barnes & O'Brien 2002): assumed system age, and the
# planet radius in Earth radii above which k2/Q stays at its Jupiter value.
TIDAL_SURVIVAL_AGE_GYR = 5.0


TIDAL_K2_Q_MAX_RADIUS = 11.2


def planet_k2_over_q(planet_radius):
    """Return the planet tidal efficiency k2/Q from its radius in Earth radii.

    Empirical Solar System relation of Teachey, Kipping & Schmitt (2018, HEK VI, Fig. 1),
    log10(k2/Q) = -2.90 - 0.187 Rp/R_earth, fitted to the Moon, the Earth, Neptune and Jupiter.
    Above Jupiter's radius it is kept at Jupiter's value instead of extrapolated.
    """
    return 10 ** (-2.90 - 0.187 * min(float(planet_radius), TIDAL_K2_Q_MAX_RADIUS))


def max_surviving_moon_mass(star_mass, planet_mass, planet_radius, planet_period, planet_ecc=0.0,
                            age_gyr=TIDAL_SURVIVAL_AGE_GYR):
    """Return the largest moon mass in Earth masses that survives tidal migration for the system age.

    Barnes & O'Brien (2002, ApJ 575, 1087) Eq. 7 solved for the moon mass, as in their Eq. 8:
    a moon is lost once tides carry it across the region between the planet and the widest
    stable orbit a_crit, which takes T = (2/13) a_crit^(13/2) sqrt(M_p / G) / (3 (k2/Q)_p M_m R_p^5),
    so M_m,max = (2/13) a_crit^(13/2) sqrt(M_p) / (3 (k2/Q)_p T R_p^5 sqrt(G)). a_crit is the outer
    moon limit EMLS searches up to (`ExoMoonLeastSquares.max_moon_semimajor_axis`, Domingos et al.
    2006), and k2/Q comes from `planet_k2_over_q`. HEK VI (Teachey et al. 2018) and Kipping & Yahalomi (2023)
    apply it to the Kepler catalog the same way. It is an order-of-magnitude bound: k2/Q is an
    empirical Solar System relation, the age is assumed, and the mass goes as a_crit^6.5.

    Parameters
    ----------
    star_mass : float
        Stellar mass in solar masses.
    planet_mass, planet_radius : float
        Planet mass in Earth masses and radius in Earth radii.
    planet_period : float
        Planet period in days.
    planet_ecc : float, optional
        Planet eccentricity, default 0.
    age_gyr : float, optional
        System age in Gyr, default TIDAL_SURVIVAL_AGE_GYR.

    Returns
    -------
    float
        Mass in Earth masses, NaN when an input is not finite and positive.
    """
    values = (star_mass, planet_mass, planet_radius, planet_period, age_gyr)
    if not all(value is not None and np.isfinite(value) and value > 0 for value in values):
        return np.nan
    planet_semimajor_axis = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period)
    a_crit = ExoMoonLeastSquares.max_moon_semimajor_axis(star_mass, planet_mass * M_earth_to_M_sun,
                                                         planet_semimajor_axis, planet_ecc) * AU_TO_M
    mass = planet_mass * Mearth_to_kg
    radius = planet_radius * ac.R_earth.value
    age = age_gyr * 1e9 * 365.25 * 86400
    moon_mass = 2 / 13 * a_crit ** 6.5 * np.sqrt(mass) / (3 * planet_k2_over_q(planet_radius) * age * radius ** 5 *
                                                          np.sqrt(G))
    return float(moon_mass / Mearth_to_kg)


def moon_radius_from_mass(moon_mass):
    """Return a moon radius in Earth radii from its mass in Earth masses.

    Mean mass-radius relation of Chen & Kipping (2017, ApJ 834, 17, Table 1): R = 1.008 M^0.279
    for rocky bodies up to 2.04 M_earth, then R proportional to M^0.589 (continuous at the
    transition). Bodies more massive than 131.6 M_earth keep that radius.
    """
    if moon_mass is None or not np.isfinite(moon_mass) or moon_mass <= 0:
        return 0.0
    transition = 2.04
    rocky = 1.008 * min(moon_mass, transition) ** 0.279
    if moon_mass <= transition:
        return float(rocky)
    return float(rocky * (min(moon_mass, 131.6) / transition) ** 0.589)


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
        "max_surviving_moon_radius". The moon radius range is Rp/20 to the smaller of Rp/2 and
        the largest moon that survives tidal migration (`max_surviving_moon_mass`).
    """
    planet_radius = row['Planet Radius (R_Earth)']
    planet_period = row['Period (days)']
    planet_t0 = row['Epoch (BJD)'] - bjd_offset
    planet_t0 = first_time + (planet_t0 - first_time) % planet_period
    planet_mass = row['Predicted Mass (M_Earth)']
    if planet_mass is None or np.isnan(planet_mass):
        planet_mass = catalog_planet_mass(planet_radius, row.get('Planet Radius (R_Earth) err'))
    planet_ecc, planet_arg_periastron = catalog_planet_orbit(row, catalog)
    # The largest moon is the smaller of half the planet radius and the largest moon that
    # survives tidal migration (Barnes & O'Brien 2002, as in HEK VI); the catalog filter has
    # already dropped targets where it is below the smallest moon searched.
    min_radius = planet_radius * CATALOG_MIN_MOON_RADIUS_FRACTION
    surviving_radius = moon_radius_from_mass(max_surviving_moon_mass(star_mass, planet_mass, planet_radius,
                                                                     planet_period, planet_ecc))
    max_radius = max(min_radius, min(planet_radius * CATALOG_MAX_MOON_RADIUS_FRACTION, surviving_radius))
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
