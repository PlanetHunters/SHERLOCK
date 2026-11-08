"""Run the EMLS exomoon search on a candidate of a SHERLOCK results directory."""
import json
import logging
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
from lcbuilder.helper import LcbuilderHelper

from sherlockpipe import __version__
from sherlockpipe.exomoon import AU_TO_RSUN, R_earth_to_R_sun, ExoMoonLeastSquares
from sherlockpipe.system_stability.mr_forecast import MrForecast

MASK_DURATION_FACTOR = 2
DEFAULT_RADIUS_ERR_FRACTION = 0.1
DEFAULT_INC_ERR = 0.5
MASS_FORECAST_SEED = 0
# Above this radius the Chen & Kipping relation is nearly flat in mass, so a radius does not
# constrain the mass (the forecast median exceeds 6 Jupiter masses at 10-12 R_earth).
GIANT_RADIUS = 8.0
JUPITER_MASS = 317.8
MAX_EXPOSURE_SECONDS = 86400
CADENCE_TOLERANCE = 0.05
CADENCE_MIN_RUN = 5
MIXED_CADENCE_SHARE = 0.05


def read_candidate(object_dir, candidate):
    """Return the row of candidates.csv for a 1-based candidate number.

    Parameters
    ----------
    object_dir : str
        SHERLOCK results directory.
    candidate : int
        Candidate number, starting at 1.

    Returns
    -------
    tuple
        (candidates DataFrame, the candidate row).

    Raises
    ------
    ValueError
        If the candidate number is not in the file.
    """
    candidates = pd.read_csv(os.path.join(object_dir, "candidates.csv"))
    if candidate < 1 or candidate > len(candidates):
        raise ValueError(f"Candidate {candidate} not found: {object_dir}/candidates.csv has {len(candidates)} rows")
    return candidates, candidates.iloc[candidate - 1]


def _derived_value(derived, pattern):
    rows = derived[derived["#property"].str.contains(pattern, regex=True)]
    if len(rows) == 0:
        return None, None
    row = rows.iloc[0]
    error = None
    if "lower_error" in row and "upper_error" in row:
        error = float(np.nanmax([row["lower_error"], row["upper_error"]]))
    return float(row["value"]), error


def _fitted_value(table, suffix):
    rows = table[table["#name"].astype(str).str.endswith(suffix)]
    if len(rows) == 0:
        return None
    return float(rows.iloc[0]["median"])


def _fitted_error(table, suffix):
    """Return the larger of the lower and upper errors of a fitted parameter, or None if not positive."""
    rows = table[table["#name"].astype(str).str.endswith(suffix)]
    if len(rows) == 0 or "lower_error" not in rows or "upper_error" not in rows:
        return None
    error = float(np.nanmax([rows.iloc[0]["lower_error"], rows.iloc[0]["upper_error"]]))
    return error if np.isfinite(error) and error > 0 else None


def read_fit_parameters(fit_dir):
    """Read the planet parameters of a SHERLOCK fit directory (allesfitter results).

    Parameters
    ----------
    fit_dir : str
        The ``fit_[n]/main`` directory.

    Returns
    -------
    dict or None
        Parameters that were found, in EMLS units, or None if there are no results.
    """
    ns_file = os.path.join(fit_dir, "results", "ns_table.csv")
    derived_file = os.path.join(fit_dir, "results", "ns_derived_table.csv")
    if not (os.path.exists(ns_file) and os.path.exists(derived_file)):
        return None
    table = pd.read_csv(ns_file)
    derived = pd.read_csv(derived_file)
    params = {}
    for key, suffix in (("period", "_period"), ("t0", "_epoch")):
        value = _fitted_value(table, suffix)
        if value is not None:
            params[key] = value
            error = _fitted_error(table, suffix)
            if error is not None:
                params[f"{key}_err"] = error
    f_c, f_s = _fitted_value(table, "_f_c"), _fitted_value(table, "_f_s")
    if f_c is not None and f_s is not None:
        params["ecc"] = f_c ** 2 + f_s ** 2
        params["arg_periastron"] = float(np.degrees(np.arctan2(f_s, f_c))) % 360 if params["ecc"] > 0 else 0.0
    radius, radius_err = _derived_value(derived, r"^Companion radius [^;]*; \$R_[^$]*\$ \(\$\\mathrm\{R_\{\\oplus\}\}\$\)")
    if radius is not None:
        params["radius"], params["radius_err"] = radius, radius_err
    a_au, _ = _derived_value(derived, r"^Semi-major axis [^;]*; \$a_[^$]*\$ \(AU\)")
    if a_au is not None:
        params["a"] = a_au
    inc, inc_err = _derived_value(derived, r"^Inclination ")
    if inc is not None:
        params["inc"], params["inc_err"] = inc, inc_err
    impact, _ = _derived_value(derived, r"^Impact parameter (?!occultation)")
    if impact is not None:
        params["impact"] = impact
    duration_h, _ = _derived_value(derived, r"^Total transit duration ")
    if duration_h is not None:
        params["duration"] = duration_h / 24
    return params


def guess_planet_mass(radius, radius_err=None):
    """Guess the planet mass from its radius.

    Below GIANT_RADIUS this is the Chen & Kipping (2017) forecast, with a fixed seed
    because the forecaster samples at random. From GIANT_RADIUS up the radius does not
    constrain the mass, so one Jupiter mass is used and a warning recommends --planet_mass.

    Parameters
    ----------
    radius : float
        Planet radius in Earth radii.
    radius_err : float, optional
        Radius uncertainty in Earth radii. 10 % of the radius when missing.

    Returns
    -------
    float
        Mass in Earth masses.
    """
    if radius >= GIANT_RADIUS:
        logging.warning("Planet radius %.2f R_earth is in the giant regime, where the radius does not constrain the "
                        "mass. Using one Jupiter mass (%.1f M_earth); provide --planet_mass for a better moon "
                        "period range.", radius, JUPITER_MASS)
        return JUPITER_MASS
    if radius_err is None or not np.isfinite(radius_err) or radius_err <= 0:
        radius_err = 0.1 * radius
    state = np.random.get_state()
    try:
        np.random.seed(MASS_FORECAST_SEED)
        forecast = MrForecast.Rstat2M(radius, radius_err)
    finally:
        np.random.set_state(state)
    if forecast is None:
        raise ValueError(f"The mass forecaster cannot handle a radius of {radius} R_earth; provide --planet_mass")
    return float(forecast[0])


def mask_candidates(time, flux, flux_err, candidates, numbers):
    """Remove the transits of other candidates from the light curve.

    Each transit is masked over twice its duration, as in the SHERLOCK vetting.

    Parameters
    ----------
    time, flux, flux_err : numpy.ndarray
        Light curve.
    candidates : pandas.DataFrame
        The candidates.csv table (duration in minutes).
    numbers : iterable of int
        1-based candidate numbers to mask.

    Returns
    -------
    tuple
        (time, flux, flux_err) without the masked transits.
    """
    for number in numbers:
        row = candidates.iloc[number - 1]
        duration = row["duration"] / 60 / 24
        before = len(time)
        time, flux, flux_err = LcbuilderHelper.mask_transits(time, flux, row["period"],
                                                             duration * MASK_DURATION_FACTOR, row["t0"],
                                                             flux_err=flux_err)
        logging.info("Masked candidate %d (P=%.5f d, T0=%.5f, D=%.2f h): %d points removed", number, row["period"],
                     row["t0"], duration * 24, before - len(time))
    return time, flux, flux_err


def find_cadences(spacing_s, min_run=CADENCE_MIN_RUN, tolerance=CADENCE_TOLERANCE):
    """Find the cadences of a curve from its point spacings.

    Spacings within the relative tolerance of the shortest one of a cluster form that cluster. A spacing counts towards
    a cadence only inside a run of at least min_run consecutive spacings of the same cluster, so
    the isolated longer spacings left by removed points and gaps are not taken for a cadence.

    Parameters
    ----------
    spacing_s : array_like
        Positive spacings between consecutive points, in time order, in seconds.
    min_run : int, optional
        Shortest run of equal spacings that counts as a cadence.
    tolerance : float, optional
        Relative spacing difference that still counts as the same cadence.

    Returns
    -------
    list of dict
        One {"seconds", "points_share", "time_share"} per cadence, most points first. The shares are
        fractions of the spacings in cadence runs, by count and by summed duration.
    """
    spacing_s = np.asarray(spacing_s, dtype=float)
    if len(spacing_s) == 0:
        return []
    order = np.argsort(spacing_s)
    labels = np.empty(len(spacing_s), dtype=int)
    start, label = 0, 0
    while start < len(order):
        # Each cluster spans at most the tolerance from its shortest spacing, so a continuum of
        # irregular spacings does not chain into one cluster.
        end = np.searchsorted(spacing_s[order], spacing_s[order[start]] * (1 + tolerance), side="right")
        labels[order[start:end]] = label
        start, label = end, label + 1
    run_starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    run_lengths = np.diff(np.r_[run_starts, len(labels)])
    regular = np.repeat(run_lengths >= min_run, run_lengths)
    if not np.any(regular):
        return []
    total_points, total_time = np.sum(regular), np.sum(spacing_s[regular])
    cadences = []
    for label in np.unique(labels[regular]):
        members = regular & (labels == label)
        cadences.append({"seconds": float(np.median(spacing_s[members])),
                         "points_share": float(np.sum(members) / total_points),
                         "time_share": float(np.sum(spacing_s[members]) / total_time)})
    return sorted(cadences, key=lambda cadence: -cadence["points_share"])


def significant_cadences(cadences):
    """Return the cadences that hold more than MIXED_CADENCE_SHARE of the points or of the time."""
    return [cadence for cadence in cadences
            if max(cadence["points_share"], cadence["time_share"]) > MIXED_CADENCE_SHARE]


def _describe_cadences(cadences):
    return "; ".join(f"{cadence['seconds']:.1f} s: {100 * cadence['points_share']:.0f}% of points, "
                     f"{100 * cadence['time_share']:.0f}% of time" for cadence in cadences)


def estimate_exposure_time(time, lc_file, exposure_time=None):
    """Return the exposure time of the light-curve points in seconds, where it came from, and the cadences.

    The exposure is the median spacing between consecutive points of the whole curve. Mission
    exposures span their cadence, and a binned point spans its bin, so the spacing measures the
    exposure without the SHERLOCK configuration, whose EXPTIME only selects the downloaded data.
    A curve that mixes cadences gets its median spacing, which follows the cadence with the most
    points, and a warning names every cadence that holds more than MIXED_CADENCE_SHARE of the
    points or of the time. With more than about half of the points removed at random, the median
    can be twice the cadence; the warning then names both.

    Parameters
    ----------
    time : array_like
        Light-curve times in days.
    lc_file : str
        Light-curve file, named in the source and the messages.
    exposure_time : float, optional
        Exposure time in seconds given by the user, which takes precedence.

    Returns
    -------
    tuple
        (seconds or None, source, cadences). None means an instantaneous model. cadences is the
        list from find_cadences.

    Raises
    ------
    ValueError
        If the exposure is not a positive number of seconds up to one day.
    """
    if exposure_time is not None and not (np.isfinite(exposure_time) and 0 < exposure_time <= MAX_EXPOSURE_SECONDS):
        raise ValueError(f"--exposure_time must be a positive number of seconds up to {MAX_EXPOSURE_SECONDS}, "
                         f"got {exposure_time}")
    time = np.asarray(time, dtype=float)
    spacing_s = np.diff(time[np.argsort(time, kind="stable")]) * 86400
    spacing_s = spacing_s[spacing_s > 0]
    cadences = find_cadences(spacing_s)
    found = _describe_cadences(cadences) or "none"
    if exposure_time is not None:
        exposure_time = float(exposure_time)
        logging.info("Exposure time %.1f s given by the user; cadences in %s: %s", exposure_time, lc_file, found)
        if cadences and not any(abs(exposure_time - cadence["seconds"]) <= CADENCE_TOLERANCE * cadence["seconds"]
                                for cadence in cadences):
            logging.warning("--exposure_time %.1f s matches none of the cadences of %s (%s); check its unit",
                            exposure_time, lc_file, found)
        return exposure_time, "user", cadences
    if len(spacing_s) == 0:
        logging.warning("%s has fewer than two distinct times, so the exposure cannot be measured: the planet and "
                        "moon models are instantaneous", lc_file)
        return None, "none (instantaneous model)", cadences
    seconds = float(np.median(spacing_s))
    if seconds > MAX_EXPOSURE_SECONDS:
        raise ValueError(f"The median point spacing of {lc_file} is {seconds:.0f} s, longer than a day, so it is not "
                         f"an exposure; pass --exposure_time")
    logging.info("Exposure time %.1f s from the median point spacing of %s; cadences: %s", seconds, lc_file, found)
    others = [cadence for cadence in significant_cadences(cadences)
              if abs(cadence["seconds"] - seconds) > CADENCE_TOLERANCE * seconds]
    if others:
        logging.warning("%s mixes cadences (%s). Every point is modelled with the median %.1f s, so the points at "
                        "%s are modelled with the wrong exposure. Pass --exposure_time to choose another one",
                        lc_file, found, seconds, ", ".join(f"{cadence['seconds']:.1f} s" for cadence in others))
    elif not cadences:
        logging.warning("%s has no regular cadence, so the median spacing %.1f s may not be its exposure. Pass "
                        "--exposure_time to set it", lc_file, seconds)
    return seconds, f"median point spacing of {os.path.basename(lc_file)}", cadences


def epochs_in_span(t0, period, time):
    """Return every transit epoch of the ephemeris within the data span, observed or not."""
    first = np.floor((np.min(time) - t0) / period)
    last = np.ceil((np.max(time) - t0) / period)
    epochs = t0 + np.arange(first, last + 1) * period
    return [float(epoch) for epoch in epochs if np.min(time) - period / 2 <= epoch <= np.max(time) + period / 2]


def build_inputs(object_dir, candidate, mask=None, planet_mass=None, min_moon_radius=None, max_moon_radius=None,
                 curve=None, lc_file=None, use_fit=True, exposure_time=None):
    """Collect the light curve and the star, planet and moon-range inputs of an EMLS run.

    Parameters
    ----------
    object_dir : str
        SHERLOCK results directory.
    candidate : int
        1-based candidate number to search moons around.
    mask : iterable of int, optional
        1-based candidate numbers whose transits are masked.
    planet_mass : float, optional
        Planet mass in Earth masses. Forecast from the radius when missing.
    min_moon_radius, max_moon_radius : float, optional
        Moon radius range in Earth radii. Defaults to Rp/20 and Rp/2.
    curve : int, optional
        Detrended curve number (lc_<curve>.csv). Defaults to the curve the candidate was found in.
    lc_file : str, optional
        Light curve file with time, flux and flux_err columns. Overrides curve.
    use_fit : bool, optional
        Use the fit_[n]/main results when they exist, default True.
    exposure_time : float, optional
        Exposure time in seconds. The median point spacing of the light curve when missing.

    Returns
    -------
    dict
        Inputs, the light curve arrays and where each value came from.
    """
    candidates, row = read_candidate(object_dir, candidate)
    mask = sorted(set(mask or []))
    for number in mask:
        if number == candidate:
            raise ValueError(f"Candidate {candidate} cannot mask itself")
        if number < 1 or number > len(candidates):
            raise ValueError(f"Mask candidate {number} not found in candidates.csv ({len(candidates)} rows)")
    star = pd.read_csv(os.path.join(object_dir, "params_star.csv")).iloc[0]
    star_mass, star_radius = float(star["mass"]), float(star["radius"])
    if not (np.isfinite(star_mass) and np.isfinite(star_radius)):
        raise ValueError("params_star.csv has no finite stellar mass and radius")
    ld = [float(star["ld_a"]), float(star["ld_b"])]
    if not np.all(np.isfinite(ld)):
        raise ValueError("params_star.csv has no finite limb darkening coefficients ld_a, ld_b")
    sources = {}
    planet = {"period": float(row["period"]), "t0": float(row["t0"]), "duration": float(row["duration"]) / 60 / 24,
              "radius": float(row["rad_p"]), "radius_err": np.nan, "a": float(row["a"]), "ecc": 0.0,
              "arg_periastron": 0.0}
    for key in planet:
        sources[key] = "candidates.csv"
    fit_dir = os.path.join(object_dir, f"fit_[{candidate}]", "main")
    fit = read_fit_parameters(fit_dir) if use_fit else None
    if fit:
        for key, value in fit.items():
            if value is not None and np.isfinite(value):
                planet[key] = value
                sources[key] = fit_dir
    if not np.isfinite(planet["a"]) or planet["a"] <= 0:
        planet["a"] = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet["period"])
        sources["a"] = "Kepler's third law"
    a_over_rstar = planet["a"] * AU_TO_RSUN / star_radius
    rp_over_rstar = planet["radius"] * R_earth_to_R_sun / star_radius
    if "inc" not in planet and np.isfinite(planet.get("impact", np.nan)):
        # A fit with an impact parameter but no inclination row: invert b = a cos i (1 - e^2) / (1 + e sin w).
        distance_factor = (1 - planet["ecc"] ** 2) / (1 + planet["ecc"] * np.sin(np.radians(planet["arg_periastron"])))
        planet["inc"] = float(np.degrees(np.arccos(np.clip(planet["impact"] / (a_over_rstar * distance_factor), 0, 1))))
        sources["inc"] = "from the fitted impact parameter"
    if "inc" in planet:
        if "impact" not in planet:
            planet["impact"] = a_over_rstar * np.cos(np.radians(planet["inc"]))
            sources["impact"] = "a/R* cos(inc)"
        if not np.isfinite(planet.get("inc_err", np.nan)):
            planet["inc_err"] = DEFAULT_INC_ERR
            sources["inc_err"] = "default"
    else:
        # The candidates.csv duration comes from the search, not from a transit fit, and is
        # usually shorter than T14, so it does not give a reliable impact parameter. Start
        # from a central transit and let the planet fold fit cover every transiting inclination.
        min_inc = ExoMoonLeastSquares.min_transit_inclination(a_over_rstar, rp_over_rstar, planet["ecc"],
                                                              planet["arg_periastron"])
        planet["inc"], planet["impact"], planet["inc_err"] = 90.0, 0.0, 90.0 - min_inc
        if not use_fit:
            reason, advice = "the fit is ignored (--no_fit)", "Drop --no_fit to use it."
        elif fit is None:
            reason, advice = f"no {fit_dir}/results found", "Run sherlockpipe.fit first."
        else:
            reason, advice = (f"the {fit_dir} results have no inclination or impact parameter",
                              "Check the fit's ns_derived_table.csv.")
        logging.warning("No fitted inclination for candidate %d: %s. The fold fit starts from a central transit with "
                        "a/R* from %s, and EMLS bounds a/R* to 2%%, so the planet geometry can be biased. %s",
                        candidate, reason, sources["a"], advice)
        sources["inc"] = sources["impact"] = sources["inc_err"] = \
            "no fit: central start, fold fit free down to the grazing limit"
    if not np.isfinite(planet["radius_err"]):
        planet["radius_err"] = DEFAULT_RADIUS_ERR_FRACTION * planet["radius"]
        sources["radius_err"] = "default: 10% of the radius"
    if planet_mass is None:
        planet_mass = guess_planet_mass(planet["radius"], planet["radius_err"])
        sources["mass"] = ("Jupiter mass (giant radius)" if planet["radius"] >= GIANT_RADIUS
                           else "Chen & Kipping (2017) forecast")
    else:
        sources["mass"] = "user"
    planet["mass"] = float(planet_mass)
    moon = {"min_radius": planet["radius"] / 20 if min_moon_radius is None else float(min_moon_radius),
            "max_radius": planet["radius"] / 2 if max_moon_radius is None else float(max_moon_radius)}
    if not 0 < moon["min_radius"] < moon["max_radius"]:
        raise ValueError(f"Moon radius range must satisfy 0 < min < max, got {moon['min_radius']} and "
                         f"{moon['max_radius']}")
    if lc_file is None:
        curve = int(row["curve"]) if curve is None else int(curve)
        lc_file = os.path.join(object_dir, f"lc_{curve}.csv")
        if curve == 0:
            logging.warning("lc_0.csv is the curve before SHERLOCK detrending")
    lc = pd.read_csv(lc_file)
    time, flux, flux_err = lc["time"].to_numpy(float), lc["flux"].to_numpy(float), lc["flux_err"].to_numpy(float)
    finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[finite], flux[finite], flux_err[finite]
    exposure_seconds, sources["exposure_time"], cadences = estimate_exposure_time(time, lc_file, exposure_time)
    time, flux, flux_err = mask_candidates(time, flux, flux_err, candidates, mask)
    if len(time) == 0:
        raise ValueError("No light curve points remain after masking")
    return {"star": {"mass": star_mass, "radius": star_radius, "ld": ld}, "planet": planet, "moon": moon,
            "exposure_time": exposure_seconds, "cadences": cadences,
            "mask": mask, "lc_file": os.path.abspath(lc_file), "sources": sources,
            "t0s": epochs_in_span(planet["t0"], planet["period"], time),
            "time": time, "flux": flux, "flux_err": flux_err}


def _json_safe(value):
    """Replace non-finite floats with None, recursively, so the results are strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _close_file_logging():
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def _setup_logging(log_file):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def run_emls(object_dir, candidate, mask=None, planet_mass=None, min_moon_radius=None, max_moon_radius=None,
             period_grid_size=10000, radius_grid_size=20, cpus=None, curve=None, lc_file=None, use_fit=True,
             max_moon_density=5.5, subtract_mode='subtract', exposure_time=None, ttv_prior_minutes=None,
             null_runs=100, fap_threshold=0.01):
    """Run EMLS for one candidate of a SHERLOCK results directory.

    Outputs go to ``<object_dir>/emls_<candidate>``: the EMLS plots, ``emls.log`` and
    ``emls_results.json`` with the inputs, their sources and the search results. The run
    is written to ``emls_<candidate>.partial`` and only replaces previous results when it
    succeeds. A failed run leaves its log, with the error, in the partial directory.

    Parameters
    ----------
    object_dir : str or None
        SHERLOCK results directory. Defaults to the current directory.
    candidate : int
        1-based candidate number.
    period_grid_size, radius_grid_size : int, optional
        EMLS grid sizes, default 10000 and 20.
    cpus : int, optional
        Processes for the search. Defaults to all CPUs but one.
    max_moon_density : float, optional
        Maximum moon density in g/cm3 for the Roche limit, default 5.5.
    subtract_mode : str, optional
        'subtract' or 'mask' for the planet transit, default 'subtract'.
    exposure_time : float, optional
        Exposure time in seconds. The median point spacing of the light curve when missing.
    ttv_prior_minutes : float, optional
        Timing variation in minutes allowed around the linear ephemeris, on top of the
        ephemeris uncertainty at each epoch, in the prior on the per-epoch t0. Default None: the largest planet TTV
        a searched moon can cause. 0 removes the prior.
    null_runs : int, optional
        Searches on pseudo-epochs, windows of the light curve away from the transits, that form
        the null distribution of the SNR for the false-alarm probability, default 100. 0 skips it.
    fap_threshold : float, optional
        False-alarm probability below which the best hypothesis is a detection, default 0.01.
    Other parameters are passed to `build_inputs`.

    Returns
    -------
    dict
        The content written to emls_results.json.
    """
    object_dir = os.getcwd() if object_dir is None else object_dir
    cpus = max(1, (os.cpu_count() or 2) - 1) if cpus is None else int(cpus)
    final_dir = os.path.join(object_dir, f"emls_{candidate}")
    output_dir = final_dir + ".partial"
    for path in (final_dir, output_dir):
        if os.path.islink(path):
            raise ValueError(f"{path} is a symbolic link; remove it before running EMLS")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    _setup_logging(os.path.join(output_dir, "emls.log"))
    logging.info("Starting EMLS for candidate %d of %s", candidate, object_dir)
    try:
        results = _run(object_dir, candidate, output_dir, mask, planet_mass, min_moon_radius, max_moon_radius,
                       period_grid_size, radius_grid_size, cpus, curve, lc_file, use_fit, max_moon_density,
                       subtract_mode, exposure_time, ttv_prior_minutes, null_runs, fap_threshold)
    except Exception:
        logging.exception("EMLS failed for candidate %d; previous results, if any, are kept in %s", candidate,
                          final_dir)
        raise
    finally:
        _close_file_logging()
    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    os.rename(output_dir, final_dir)
    results["output_dir"] = final_dir
    return results


def _run(object_dir, candidate, output_dir, mask, planet_mass, min_moon_radius, max_moon_radius, period_grid_size,
         radius_grid_size, cpus, curve, lc_file, use_fit, max_moon_density, subtract_mode, exposure_time,
         ttv_prior_minutes=None, null_runs=0, fap_threshold=0.01):
    inputs = build_inputs(object_dir, candidate, mask, planet_mass, min_moon_radius, max_moon_radius, curve, lc_file,
                          use_fit, exposure_time)
    star, planet, moon = inputs["star"], inputs["planet"], inputs["moon"]
    for key in ("period", "t0", "duration", "radius", "a", "inc", "impact", "ecc", "mass"):
        logging.info("Planet %s = %s (%s)", key, planet[key], inputs["sources"].get(key))
    logging.info("Light curve %s, %d points, %d epochs in its span; moon radius %.3f-%.3f R_earth", inputs["lc_file"],
                 len(inputs["time"]), len(inputs["t0s"]), moon["min_radius"], moon["max_radius"])
    emls = ExoMoonLeastSquares(
        output_dir, cpus, star["mass"], star["radius"], star["ld"], planet["radius"], planet["radius_err"],
        planet["mass"], planet["period"], planet["t0"], planet["duration"], planet["a"], planet["inc"],
        planet["inc_err"], planet["ecc"], 0.0, planet["arg_periastron"], planet["impact"],
        moon["min_radius"], moon["max_radius"], inputs["t0s"], inputs["time"], inputs["flux"], inputs["flux_err"],
        max_moon_density, period_grid_size=period_grid_size, radius_grid_size=radius_grid_size,
        subtract_mode=subtract_mode,
        exposure_time=None if inputs["exposure_time"] is None else inputs["exposure_time"] / 86400,
        ttv_prior_width="auto" if ttv_prior_minutes is None else (ttv_prior_minutes / 1440 if ttv_prior_minutes
                                                                  else None),
        planet_t0_err=planet.get("t0_err"), planet_period_err=planet.get("period_err"), null_runs=null_runs,
        fap_threshold=fap_threshold)
    # EMLS keeps an epoch only with at least min_cadences_per_t0 points within two durations.
    kept = list(emls.bary_t0s)
    window = planet["duration"] * 2
    min_points = int(emls.min_cadences_per_t0)
    skipped = [{"t0": t0, "points": int(np.sum(np.abs(inputs["time"] - t0) < window))}
               for t0 in inputs["t0s"] if t0 not in kept]
    for epoch in skipped:
        logging.info("Epoch T0=%.4f skipped: %d points within two transit durations, fewer than %d", epoch["t0"],
                     epoch["points"], min_points)
    logging.info("%d of %d epochs have at least %d points within two transit durations", len(kept),
                 len(inputs["t0s"]), min_points)
    emls.run(plot=True)
    results = {
        "object_dir": os.path.abspath(object_dir), "candidate": candidate, "light_curve": inputs["lc_file"],
        "masked_candidates": inputs["mask"], "star": star,
        "exposure_time_s": inputs["exposure_time"],
        "cadences": inputs["cadences"],
        "mixed_cadence": len(significant_cadences(inputs["cadences"])) > 1,
        "planet": planet,
        "moon_radius_range": moon, "sources": inputs["sources"], "period_grid_size": period_grid_size,
        "radius_grid_size": radius_grid_size, "epochs_in_span": len(inputs["t0s"]),
        "min_points_per_epoch": min_points, "epochs_kept": len(kept), "epochs_skipped": skipped,
        "epochs_used": len(emls.bary_t0s), "epochs_dropped_fit": emls.epochs_dropped_fit,
        "epochs_dropped_coverage": emls.epochs_dropped_coverage,
        "ttv_prior_minutes": None if emls.ttv_prior_width_used is None else emls.ttv_prior_width_used * 1440,
        "ttv_prior_source": "auto: largest TTV of a searched moon" if ttv_prior_minutes is None
        else ("none" if not ttv_prior_minutes else "user"),
        "ephemeris_uncertainty_source": emls.ephemeris_source,
        "epochs_flagged_rms": emls.epochs_flagged_rms, "epochs_flagged_rp": emls.epochs_flagged_rp,
        "planet_fit_absorbed_fraction": emls.absorbed_fraction,
        "fold_fit": {"status": emls.fold_fit_status, "duration": emls.planet_duration,
                     "impact": emls.planet_impact_param, "inc": emls.planet_inc,
                     "a_over_rstar": emls.planet_a_over_rstar},
        "sherlockpipe_version": __version__,
        "max_snr": emls.max_snr, "sde": emls.sde, "max_sde": emls.max_sde,
        "best_period": emls.recovered_period, "best_alpha": emls.recovered_alpha,
        "fap": emls.fap, "fap_empirical": emls.fap_empirical, "fap_threshold": emls.fap_threshold,
        "detected": emls.detected,
        "null_runs_requested": null_runs, "null_runs_completed": len(emls.null_snrs or []),
        "null_independent_offsets": emls.null_independent_offsets, "null_snrs": emls.null_snrs,
        "alias_family": (emls.aliases or [])[:10],
        "red_noise_factor": emls.red_noise_factor, "red_noise_measured": emls.red_noise_measured,
        "single_event": emls.single_event, "loo_snr_min": emls.loo_snr_min, "epoch_stats": emls.epoch_stats,
        "best_radius": float(emls.best_snr_radius),
        "best_radius_above_range": bool(emls.best_snr_radius_above_range),
        "best_radius_below_range": bool(emls.best_snr_radius_below_range),
        "notes": "max_snr is the detection statistic: it selects the best period and alpha, and sde, best_radius "
                 "and the epoch statistics describe that same hypothesis (audit items H7, H8). fap is the "
                 "probability that noise alone gives this SNR, from a Gumbel fit to the SNR of null searches on "
                 "pseudo-epochs of the same light curve away from the transits; fap_empirical is (k+1)/(N+1). "
                 "detected is fap < fap_threshold, or null without enough null runs. single_event flags a "
                 "signal that one transit carries (leaving one out keeps less than half the SNR); it does not "
                 "change detected. sde (SNR spectrum at the selected period) and max_sde (residual spectrum) "
                 "are diagnostics that do not separate moons from noise. best_radius inverts the "
                 "least-squares template amplitude through the template set (audit items H13, H14). The "
                 "planet fit can absorb part of a moon inside the planet transit; see "
                 "planet_fit_absorbed_fraction.",
    }
    with open(os.path.join(output_dir, "emls_results.json"), "w") as file:
        json.dump(_json_safe(results), file, indent=2, allow_nan=False)
    logging.info("EMLS finished: SNR %.2f at P=%.4f d, alpha=%.3f rad, FAP %s, detected=%s, single event=%s",
                 emls.max_snr, emls.recovered_period, emls.recovered_alpha,
                 "n/a" if emls.fap is None else f"{emls.fap:.2g}", emls.detected, emls.single_event)
    return results
