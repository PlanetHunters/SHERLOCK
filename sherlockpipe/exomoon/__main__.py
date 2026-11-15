"""Catalog driver: search moons around the KOI or TOI catalog targets that pass the dynamical filters.

Example: ``python -m sherlockpipe.exomoon --catalog kois``
"""
import logging
import os
import sys
from lcbuilder.helper import LcbuilderHelper
import numpy as np
from lcbuilder.lcbuilder_class import LcBuilder
import pandas as pd
from sherlockpipe.exomoon.catalog import CATALOG_MIN_MOON_RADIUS_FRACTION, TIDAL_SURVIVAL_AGE_GYR, TIDAL_SURVIVAL_K2_Q_FACTOR, catalog_csv_path, catalog_planet_orbit, catalog_planet_params, combine_cadence_curves, emls_output_dir, max_surviving_moon_radius, mean_planet_mass, prepare_koi_table
from sherlockpipe.exomoon.constants import AU_TO_RSUN, M_earth_to_M_sun, R_earth_to_R_sun
from sherlockpipe.exomoon.search import ExoMoonLeastSquares

# The search set-up the validation runs used (controls, synthetic planets, injections): EMLS detrends
# the undetrended curve itself, with a biweight of DETREND_DURATIONS transit durations and the 'auto'
# mask, so the planet and a moon stay out of the trend, and NULL_RUNS pseudo-epoch searches give the
# false-alarm probability. A pre-flattened curve would let the trend take part of a moon.
DETREND_DURATIONS = 3
NULL_RUNS = 100
DRIVER_CPUS = 4


if __name__ == '__main__':

    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger()
    while len(logger.handlers) > 0:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    
    lc_builder = LcBuilder()
    max_moon_density = 5.5
    catalog = "tois"
    if "--catalog" in sys.argv:
        idx = sys.argv.index("--catalog")
        if idx + 1 < len(sys.argv):
            catalog = sys.argv[idx + 1]
    csv_path = catalog_csv_path(catalog)
    tois_df = pd.read_csv(csv_path)
    cadence = [120]
    author = 'SPOC'
    if catalog == "kois":
        author = 'Kepler'
        # Short cadence where a quarter has it, long cadence elsewhere; quarters held at both are
        # kept once, at short cadence (see drop_overlapping_cadences), and EMLS models every point
        # with its own exposure and weight.
        cadence = [60, 1800]
        tois_df = prepare_koi_table(tois_df)
    confirmed_tois_df = tois_df.loc[(tois_df['Disposition'] == 'CP') | (tois_df['Disposition'] == 'KP') | (tois_df['Disposition'] == 'CONFIRMED') | (tois_df['Disposition'] == 'CANDIDATE')]
    companion_tois_df = confirmed_tois_df
    confirmed_tois_df = confirmed_tois_df.loc[confirmed_tois_df['Planet SNR'] > 15]
    confirmed_tois_df = confirmed_tois_df.sort_values(by='Planet SNR', ascending=False)
    # Moon dynamical stability filter
    def hill_roche_stable(row, period, p_radius, p_mass, star_mass):
        """Check that the stable prograde moon region extends beyond the Roche limit.

        The outer limit is the one EMLS searches up to, a_max = 0.4895 (1 - 1.0305 e_p) r_H
        (Domingos, Winter & Yokoyama 2006, MNRAS 373, 1227), with the circular Hill radius
        r_H = a_p (M_p / 3 M_star)^(1/3) (`max_moon_semimajor_axis`). The inner limit is the fluid
        Roche limit r_Roche = 2.44 R_p (rho_p / rho_m)^(1/3) for an Earth-like moon,
        rho_m = 5.5 g cm^-3. For a circular orbit, a_max > r_Roche is r_H > 2.04 r_Roche. It replaces
        an r_H > 2 r_Roche margin that had no source and ignored the eccentricity."""
        ecc, _ = catalog_planet_orbit(row, catalog)
        a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, period)
        try:
            a_max = ExoMoonLeastSquares.max_moon_semimajor_axis(star_mass, p_mass * M_earth_to_M_sun, a, ecc)
        except ValueError:
            return False
        r_roche = ExoMoonLeastSquares.compute_roche_radius(p_radius, p_mass, max_moon_density)
        return a_max > r_roche * R_earth_to_R_sun / AU_TO_RSUN

    def planet_planet_stable(a1, m1_earth, a2, m2_earth, star_mass, n_hill=5):
        """Check if two planets are separated by n_hill mutual Hill radii.

        Two planets are dynamically packed (and moons can be ejected) if
        their separation is less than K × R_mHill where K = 3-5
        (Gladman 1993, Icarus 106, 247; Chambers et al. 1996, Icarus 119, 261).

        The mutual Hill radius is:
            R_mHill = (a1 + a2)/2 × ((M1 + M2) / 3M_star)^(1/3)

        Masses in Earth masses, semi-major axes in AU, star_mass in M_sun."""
        delta_a = abs(a1 - a2)
        m_total = (m1_earth + m2_earth) * M_earth_to_M_sun
        r_mhill = (a1 + a2) / 2 * (m_total / (3 * star_mass)) ** (1 / 3)
        return delta_a >= n_hill * r_mhill

    def moon_tidal_stable(row, period, p_mass, p_radius, star_mass):
        """Check if a moon at least as large as the smallest one searched survives tidal migration.

        Barnes & O'Brien (2002) Eq. 8 with the generous margin of `sherlockpipe.exomoon.tides`
        (k2/Q ten times lower, 1 Gyr), with the moon radius from the Chen & Kipping (2017) relation;
        HEK VI (Teachey et al. 2018) filters the Kepler catalog the same way with the nominal bound.
        The margin keeps targets that a planet less dissipative than the Solar System relation, or a
        younger system, would let keep a moon. It replaces a P > 10 d M^(1/6) age^(1/4) rule that
        could not be traced to its cited source (Heller et al. 2014)."""
        ecc, _ = catalog_planet_orbit(row, catalog)
        surviving = max_surviving_moon_radius(star_mass, p_mass, p_radius, period, ecc)
        return surviving >= p_radius * CATALOG_MIN_MOON_RADIUS_FRACTION

    keep_mask = []
    n_hillroche, n_tidal, n_companion = 0, 0, 0
    for _, r in confirmed_tois_df.iterrows():
        star_mass = r['Stellar Mass (M_Sun)']
        p_mass = r['Predicted Mass (M_Earth)']
        p_radius = r['Planet Radius (R_Earth)']
        period = r['Period (days)']
        if np.isnan(star_mass) or star_mass <= 0:
            keep_mask.append(False)
            continue
        if np.isnan(p_mass) or p_mass <= 0:
            # The mean relation, not the forecast: it is applied to thousands of rows.
            p_mass = mean_planet_mass(p_radius)
        a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, period)
        p_mass_sun = p_mass * M_earth_to_M_sun
        moon_stable = hill_roche_stable(r, period, p_radius, p_mass, star_mass)
        if not moon_stable:
            n_hillroche += 1
            keep_mask.append(False)
            continue
        if not moon_tidal_stable(r, period, p_mass, p_radius, star_mass):
            n_tidal += 1
            keep_mask.append(False)
            continue
        # Every planet of the system counts, not only those that passed the SNR cut: a faint
        # companion destabilises moons as much as a bright one.
        sisters = companion_tois_df[companion_tois_df['Object Id'] == r['Object Id']]
        companion_ok = True
        for _, sis in sisters.iterrows():
            if sis['OI'] == r['OI'] or not np.isfinite(sis['Period (days)']) \
                    or not np.isfinite(sis['Planet Radius (R_Earth)']):
                continue
            sis_mass = sis['Predicted Mass (M_Earth)']
            if np.isnan(sis_mass) or sis_mass <= 0:
                sis_mass = mean_planet_mass(sis['Planet Radius (R_Earth)'])
            sis_a = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, sis['Period (days)'])
            if not planet_planet_stable(a, p_mass, sis_a, sis_mass, star_mass):
                companion_ok = False
                break
        if not companion_ok:
            n_companion += 1
        keep_mask.append(companion_ok)
    confirmed_tois_df = confirmed_tois_df[keep_mask]
    logging.info("Dynamical filter: Hill-Roche=%d, Tidal (Barnes & O'Brien 2002, %.0f Gyr, k2/Q x %.1f)=%d, "
                 "Companion=%d removed, %d remain",
                 n_hillroche, TIDAL_SURVIVAL_AGE_GYR, TIDAL_SURVIVAL_K2_Q_FACTOR, n_tidal, n_companion,
                 len(confirmed_tois_df))
    #confirmed_tois_df = confirmed_tois_df.loc[confirmed_tois_df['Object Id'] == 'TIC 309792357']
    confirmed_tois_df = confirmed_tois_df.iloc[0:100]
    for index, row in confirmed_tois_df.iterrows():
        target_name = row['Object Id']
        logging.info("Processing target: %s", target_name)
        object_dir = emls_output_dir(target_name, row['OI'])
        if not os.path.exists(object_dir):
            os.mkdir(object_dir)
        try:
            # One build per cadence: lcbuilder sorts a multi-cadence curve by time, after which a
            # quarter held at two cadences cannot be told apart (see combine_cadence_curves).
            curves, lc_build = [], None
            for single_cadence in cadence:
                try:
                    object_info = lc_builder.build_object_info(target_name=target_name, author=[author], sectors="all", file=None, cadence=[single_cadence],
                                                  initial_mask=[], initial_transit_mask=None, star_info=None, aperture=None,
                                                  eleanor_corr_flux="pdcsap_flux", outliers_sigma=3, high_rms_enabled=False,
                                                  high_rms_threshold=1.5, high_rms_bin_hours=4, smooth_enabled=False, binning=0,
                                                  auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                                                  auto_detrend_period=None, prepare_algorithm=None, reduce_simple_oscillations=False,
                                                  oscillation_snr_threshold=4, oscillation_amplitude_threshold=0.1, oscillation_ws_scale=60,
                                                  oscillation_min_period=0.002, oscillation_max_period=0.2)
                    cadence_build = lc_builder.build(object_info, object_dir)
                except Exception as error:
                    logging.warning(f"No {single_cadence} s light curve for {target_name}: {error}")
                    continue
                lc_build = lc_build or cadence_build
                curves.append((cadence_build.lc.time.value, cadence_build.lc.flux.value,
                               cadence_build.lc.flux_err.value))
            if lc_build is None:
                raise ValueError(f"No light curve for {target_name} at cadences {cadence}")
            star_mass = lc_build.star_info.mass
            star_radius = lc_build.star_info.radius
            times, flux, flux_err = combine_cadence_curves(curves)
            ld_coeffs = lc_build.star_info.ld_coefficients
            bjd_offset = 2454833 if catalog == "kois" else 2457000
            companions_df = tois_df.loc[(tois_df['Object Id'] == target_name) & (tois_df['OI'] != row['OI'])]
            for companion_index, companion_row in companions_df.iterrows():
                logging.info(f"Masking companion {companion_row['OI']}")
                companion_t0 = companion_row['Epoch (BJD)'] - bjd_offset
                companion_period = companion_row['Period (days)']
                companion_duration = companion_row['Duration (hours)'] / 24 * 2
                times, flux, flux_err = LcbuilderHelper.mask_transits(times, flux, companion_period, companion_duration,
                                                                     companion_t0, flux_err=flux_err)
            # star_mass = 0.88
            # star_radius = 0.878413
            # times = np.linspace(planet_t0 - 5, planet_t0 + 1000, 1005 * 60 * 24 // 2)
            # flux = np.random.normal(1, 0.001, len(times))
            # flux_err = np.random.normal(0.001, 0.0001, len(times))
            # ld_coeffs = [0.4136, 0.1999]
            planet = catalog_planet_params(row, catalog, star_mass, star_radius, times[0], bjd_offset)
            planet_radius = planet['planet_radius']
            planet_radius_err = planet['planet_radius_err']
            planet_period = planet['planet_period']
            planet_t0 = planet['planet_t0']
            planet_duration = planet['planet_duration']
            planet_inc = planet['planet_inc']
            planet_inc_err = planet['planet_inc_err']
            planet_ecc = planet['planet_ecc']
            planet_ecc_err = planet['planet_ecc_err']
            planet_arg_periastron = planet['planet_arg_periastron']
            planet_mass = planet['planet_mass']
            planet_semimajor_axis = planet['planet_semimajor_axis']
            planet_impact_param = planet['planet_impact_param']
            min_radius = planet['min_radius']
            max_radius = planet['max_radius']
            logging.info("Moon radius range %.2f-%.2f R_earth", min_radius, max_radius)
            # planet_radius = 11.8142656578
            # planet_radius_err = 1.00
            # planet_period = 141.834025
            # planet_duration = 8.681 / 24
            # planet_inc = 89.903
            # planet_inc_err = 0.5
            # planet_ecc = 0.212
            # planet_ecc_err = 0.022
            # planet_arg_periastron = 0
            # planet_mass = 408.727
            # min_radius = 1
            # max_radius = 3
            flux_err = np.ma.filled(flux_err, np.ma.median(flux_err))
            t0s = [i for i in np.arange(planet_t0, np.max(times), planet_period)]
            emls = ExoMoonLeastSquares(object_dir, DRIVER_CPUS, star_mass, star_radius, ld_coeffs,
                                       planet_radius, planet_radius_err, planet_mass, planet_period, planet_t0, planet_duration,
                                       planet_semimajor_axis, planet_inc, planet_inc_err, planet_ecc, planet_ecc_err, planet_arg_periastron, planet_impact_param,
                                       min_radius, max_radius, t0s, times, flux, flux_err, max_moon_density,
                                       period_grid_size=10000, radius_grid_size=20, subtract_mode='subtract', fit_rp_per_transit=False,
                                       min_transit_completeness=0.75, max_subtraction_rms_ratio=1.5,
                                       null_runs=NULL_RUNS, detrend_window=DETREND_DURATIONS * planet_duration,
                                       detrend_mask='auto', exposure_time='auto')
            # emls.flux = emls.inject_moon(emls.time, emls.flux, emls.flux_err, emls.bary_t0s, planet_semimajor_axis,
            #                             planet_ecc, moon_radius, moon_period, moon_mass, plot=True, initial_alpha=moon_alpha,
            #                              inject_planet=False, inject_moon=False)
            p_m_relationship = 5
            emls.run(plot=True,
                    #  inject_params={
                    #     'moon_radius': planet_radius / p_m_relationship,
                    #     'moon_period': planet_duration * 20,
                    #     'initial_alpha': 1.0,
                    #     'moon_mass': (planet_radius / p_m_relationship) ** 1.5,
                    # }
            )
        except Exception as e:
            logging.exception(f"Error in run for target {object_dir}")
