import itertools
import logging
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

from numba import njit
from pytransit import QuadraticModel
import batman
import ellc
import numpy as np
import astropy.constants as ac
import astropy.units as u
import wotan
from lcbuilder.lcbuilder_class import LcBuilder
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

G = 6.674e-11  # m3 kg-1 s-2
AU_TO_RSUN = 215.032
Msolar_to_kg = 2.e30
Mearth_to_kg = 5.972e24
M_earth_to_M_sun = Mearth_to_kg / Msolar_to_kg
R_earth_to_R_sun = 0.009175

class ExoMoonLeastSquares:
    def __init__(self, cpus, star_mass, star_radius, ab, planet_radius, planet_period, planet_t0, planet_duration, planet_semimajor_axis, planet_inc, planet_ecc,
             planet_arg_periastron, planet_impact_param, min_radius, max_radius, t0s, time, flux,
             period_grid_size=2000):
        self.cpus = cpus
        self.star_mass = star_mass
        self.star_radius = star_radius
        self.ab = ab
        self.planet_radius = planet_radius
        self.planet_period = planet_period
        self.planet_t0 = planet_t0
        self.planet_duration = planet_duration
        self.planet_semimajor_axis = planet_semimajor_axis
        self.planet_inc = planet_inc
        self.planet_ecc = planet_ecc
        self.planet_arg_periastron = planet_arg_periastron
        self.planet_impact_param = planet_impact_param
        self.time = time
        self.flux = flux
        self.t0s = t0s
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.period_grid_size = period_grid_size

    @staticmethod
    def compute_semimajor_axis(major_mass, minor_period):
        period_seconds = minor_period * 24. * 3600.
        mass_kg = major_mass * Msolar_to_kg
        a1 = (G * mass_kg * period_seconds ** 2 / 4. / (np.pi ** 2)) ** (1. / 3.)
        return a1 / 1.496e11

    @staticmethod
    def compute_hill_radius(major_mass, minor_mass, semimajor_axis, eccentricity=0):
        """
        @param major_mass: The main body mass
        @param minor_mass: The minor body mass
        @param semimajor_axis: The minor body semimajor axis in AU.
        @param eccentricity: the planet eccentricity
        @return: the hill radius of the minor body in the same units than the semimajor_axis
        """
        return AU_TO_RSUN * semimajor_axis * (1 - eccentricity) * (minor_mass / (3 * major_mass) ** (1 / 3))

    @staticmethod
    def au_to_period(mass, au):
        """
        Calculates the orbital period for the semi-major axis assuming a circular orbit.
        @param mass: the stellar mass
        @param au: the semi-major axis in astronomical units.
        @return: the period in days
        """
        mass_kg = mass * 2.e30
        a = au * 1.496e11
        return ((a ** 3) * 4 * (np.pi ** 2) / G / mass_kg) ** (1. / 2.) / 3600 / 24

    @staticmethod
    @njit
    def compute_transit_duration(star_radius,
                                 transiting_body_semimajor_axis, transit_period, transiting_body_radius,
                                 impact_parameter=0):
        """

        @param star_radius: star radius
        @param transiting_body_semimajor_axis: orbit semimajor axis
        @param transit_period: in days
        @param transiting_body_radius: transiting body radius
        @param impact_parameter:
        @return:
        @rtype:
        """
        return transit_period / np.pi * np.arcsin(np.sqrt((star_radius + transiting_body_radius) ** 2 - (impact_parameter * star_radius) ** 2) / transiting_body_semimajor_axis)
        #return 2 * moon_semimajor_axis / (planet_semimajor_axis * 2 * np.pi) * planet_period

    @staticmethod
    def compute_moon_period_grid(min, max, mode="lin", samples=10000):
        if "log" == mode:
            return np.logspace(math.log(min, 10), math.log(max, 10), samples, base=10)
        else:
            return np.linspace(min, max, samples)

    @staticmethod
    @njit
    def tokenize_transits_moon_orbit_ranges(time, flux, star_mass, star_radius, planet_mass, planet_radius,
                                            planet_period, planet_t0, planet_duration, moon_semimajor_axis, planet_semimajor_axis,
                                            moon_period, moon_eccentrictiy=0,
                                            moon_arg_periastron=0, moon_inclination=90,
                                            planet_eccentricity=0, planet_arg_periastron=0, planet_inclination=90,
                                            planet_impact_param=0):
        moon_orbit_transit_duration = ExoMoonLeastSquares.compute_transit_duration(star_radius, planet_semimajor_axis * AU_TO_RSUN,
                                                                    planet_period, moon_semimajor_axis * AU_TO_RSUN,
                                                                    planet_impact_param)
        # TODO we probably need to define left_transit_length and right_transit_length depending on moon orbit parameters
        moon_orbit_tokens = [[t0, t0 - planet_duration / 2, t0 - moon_orbit_transit_duration / 2, t0 + moon_orbit_transit_duration / 2] for t0 in self.t0s]
        return moon_orbit_tokens

    def subtract_planet_transit(self, ab, star_radius, star_mass, time, flux, planet_radius, planet_t0,
                                planet_period, planet_inc=90):
        P1 = planet_period * u.day
        a = np.cbrt((ac.G * star_mass * u.M_sun * P1 ** 2) / (4 * np.pi ** 2)).to(u.au)
        model = ellc.lc(
            t_obs=time,
            radius_1=(star_radius * u.R_sun).to(u.au) / a,  # star radius convert from AU to in units of a
            radius_2=(planet_radius * u.R_earth).to(u.au) / a,
            # convert from Rearth (equatorial) into AU and then into units of a
            sbratio=0,
            incl=planet_inc,
            light_3=0,
            t_zero=planet_t0,
            period=planet_period,
            a=None,
            q=1e-6,
            f_c=None, f_s=None,
            ldc_1=ab, ldc_2=None,
            gdc_1=None, gdc_2=None,
            didt=None,
            domdt=None,
            rotfac_1=1, rotfac_2=1,
            hf_1=1.5, hf_2=1.5,
            bfac_1=None, bfac_2=None,
            heat_1=None, heat_2=None,
            lambda_1=None, lambda_2=None,
            vsini_1=None, vsini_2=None,
            t_exp=None, n_int=None,
            grid_1='default', grid_2='default',
            ld_1='quad', ld_2=None,
            shape_1='sphere', shape_2='sphere',
            spots_1=None, spots_2=None,
            exact_grav=False, verbose=1)
        return flux - model + 1

    @staticmethod
    @njit
    def compute_moon_transit_scenarios(time, flux, planet_t0, moon_initial_alpha, moon_period, moon_orbit_ranges,
                                       moon_orbit_transit_length, moon_transit_duration):
        #TODO need to take into account "prograde" or "retrograde" orbit
        orbit_scenarios = []
        for moon_orbit_range in moon_orbit_ranges:
            t0 = moon_orbit_range[0]
            t1 = moon_orbit_range[1]
            phase_delta = (t0 - planet_t0) % moon_period * 2 * np.pi
            alpha = (moon_initial_alpha + phase_delta) % (2 * np.pi)
            time_alpha = np.cos(alpha) * moon_orbit_transit_length / 2
            moon_t1 = t1 + time_alpha
            time_args = np.argwhere((time > moon_t1) & (time < moon_t1 + moon_transit_duration))
            #TODO we'd need to fill measurement gaps (detected from the time array)
            time_moon_transit = time[time_args]
            flux_moon_transit = flux[time_args]
            time_moon_transit = time_moon_transit - (moon_t1 + moon_transit_duration / 2)
            # fig_transit, axs = plt.subplots(1, 1, figsize=(8, 8))
            # axs.plot(time_moon_transit, flux_moon_transit, color='gray', alpha=1, rasterized=True,
            #          label="Flux Transit ")
            # axs.set_title("Residuals")
            # axs.set_xlabel('Time')
            # axs.set_ylabel('Flux')
            # fig_transit.show()
            if len(time_moon_transit) > 0:
                orbit_scenarios.append([alpha, time_moon_transit, flux_moon_transit])
        return orbit_scenarios

    def search(self, search_input, return_lc=False):
        logging.info("Searching for period=%.2fd and alpha=%.2frad", search_input.moon_period, search_input.moon_alpha)
        moon_semimajor_axis = self.compute_semimajor_axis(planet_mass * M_earth_to_M_sun, search_input.moon_period)
        moon_orbit_transit_length = self.compute_transit_duration(self.star_radius,
                                                                  self.planet_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_period,
                                                                  moon_semimajor_axis * AU_TO_RSUN,
                                                                  self.planet_impact_param)
        moon_transit_length = self.compute_transit_duration(self.star_radius, self.planet_semimajor_axis * AU_TO_RSUN,
                                                            self.planet_period, 1 * R_earth_to_R_sun,
                                                            self.planet_impact_param)
        moon_orbit_ranges = self.tokenize_transits_moon_orbit_ranges(self.time, self.flux,
                                                                     self.star_mass, self.star_radius, planet_mass,
                                                                     self.planet_radius, self.planet_period,
                                                                     self.planet_t0, self.planet_duration, moon_semimajor_axis,
                                                                     self.planet_semimajor_axis,
                                                                     search_input.moon_period, search_input.moon_ecc,
                                                                     search_input.moon_arg_periastron, search_input.moon_inc, self.planet_ecc,
                                                                     self.planet_arg_periastron, self.planet_inc,
                                                                     self.planet_impact_param)
        transit_scenarios = self.compute_moon_transit_scenarios(self.time, self.flux, self.planet_t0, search_input.moon_alpha,
                                                                search_input.moon_period, moon_orbit_ranges,
                                                                moon_orbit_transit_length, moon_transit_length)
        scenario_time = []
        scenario_flux = []
        for normalized_moon_transit_scenario in transit_scenarios:
            scenario_time = np.concatenate((scenario_time, normalized_moon_transit_scenario[1].flatten()))
            scenario_flux = np.concatenate((scenario_flux, normalized_moon_transit_scenario[2].flatten()))
        sorted_time_args = np.argsort(scenario_time)
        scenario_time = scenario_time[sorted_time_args]
        scenario_flux = scenario_flux[sorted_time_args]
        outliers_args = ExoMoonLeastSquares.remove_outliers(scenario_flux, sigma_lower=float('inf'), sigma_upper=3)
        scenario_time = scenario_time[~outliers_args].flatten()
        scenario_flux = scenario_flux[~outliers_args].flatten()
        residual_calculation, residual_radius, residual_model = self.calculate_residuals(scenario_time, scenario_flux, self.model)
        if return_lc:
            return residual_calculation, residual_radius, scenario_time, scenario_flux, residual_model
        else:
            return residual_calculation, residual_radius

    @staticmethod
    @njit
    def downsample(array, npts: int):
        interpolated = interp1d(np.arange(len(array)), array, axis=0, fill_value='extrapolate')
        downsampled = interpolated(np.linspace(0, len(array), npts))
        return downsampled

    @staticmethod
    @njit
    def calculate_residuals(time, flux, model_sample, min_radius, max_radius):
        model_sample = ExoMoonLeastSquares.downsample(model_sample, len(time))
        # TODO adjusting model to minimum flux value this might get improved by several scalations of min_flux
        best_residual = np.inf
        best_radius = min_radius
        best_model = model_sample
        for radius in np.linspace(min_radius, max_radius, 10):
            depth = ((radius * R_earth_to_R_sun) ** 2) / star_radius ** 2
            flux_at_middle = 1 - depth
            model_sample_scaled = np.copy(model_sample)
            model_sample_scaled[model_sample_scaled < 1] = model_sample_scaled[model_sample_scaled < 1] * (
                        flux_at_middle / np.min(model_sample))
            radius_residuals = np.sum((flux - model_sample_scaled) ** 2) ** 0.5 #/ depth
            if radius_residuals < best_residual:
                best_residual = radius_residuals / radius
                best_radius = radius
                best_model = model_sample_scaled
        # fig_transit, axs = plt.subplots(1, 1, figsize=(8, 8))
        # axs.plot(time, flux, color='gray', alpha=1, rasterized=True, label="Flux Transit ")
        # axs.plot(time, model_sample_scaled, color='red', alpha=1, rasterized=True, label="Flux Transit ")
        # axs.set_title("Residuals")
        # axs.set_xlabel('Time')
        # axs.set_ylabel('Flux')
        # fig_transit.show()
        return best_residual, best_radius, best_model

    @staticmethod
    def remove_outliers(
        flux, sigma=5.0, sigma_lower=None, sigma_upper=None, **kwargs
    ):
        # The import time for `sigma_clip` is somehow very slow, so we use
        # a local import here.
        from astropy.stats.sigma_clipping import sigma_clip

        # First, we create the outlier mask using AstroPy's sigma_clip function
        with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
            warnings.simplefilter("ignore")
            outlier_mask = sigma_clip(
                data=flux,
                sigma=sigma,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                **kwargs,
            ).mask
        # Second, we return the masked light curve and optionally the mask itself
        return outlier_mask

    def inject_moon(self, time, flux, t0s, planet_mass, planet_semimajor_axis, planet_ecc, moon_radius, moon_period, initial_alpha=0):
        logging.info("Injecting moon with radius of  %.2fR_e, %.2fdays and %.2frad", moon_radius, moon_period, initial_alpha)
        moon_semimajor_axis = self.compute_semimajor_axis(planet_mass * M_earth_to_M_sun, moon_radius)
        moon_transit_duration = self.compute_transit_duration(self.star_radius,
                                                              planet_semimajor_axis * AU_TO_RSUN,
                                                              self.planet_period,
                                                              moon_radius * R_earth_to_R_sun,
                                                              self.planet_impact_param)
        moon_orbit_transit_duration = self.compute_transit_duration(self.star_radius,
                                                                    planet_semimajor_axis * AU_TO_RSUN,
                                                                    self.planet_period,
                                                                    moon_semimajor_axis * AU_TO_RSUN,
                                                                    self.planet_impact_param)
        first_t0 = t0s[0]
        subtracted_planet_flux = self.subtract_planet_transit(self.ab, self.star_radius, self.star_mass, time,
                                                              flux, self.planet_radius, self.planet_t0,
                                                              self.planet_period, self.planet_inc)
        for t0 in t0s:
            phase_delta = ((t0 - first_t0) % moon_period) * 2 * np.pi
            moon_phase = (initial_alpha + phase_delta) % (2 * np.pi)
            moon_tau = np.cos(moon_phase)
            moon_t0 = t0 + moon_tau * moon_orbit_transit_duration / 2
            time_transit = time[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)]
            if len(time_transit) == 0:
                continue
            ma = batman.TransitParams()
            ma.t0 = moon_t0  # time of inferior conjunction
            ma.per = self.planet_period  # orbital period, use Earth as a reference
            ma.rp = moon_radius * R_earth_to_R_sun / self.star_radius  # planet radius (in units of stellar radii)
            ma.a = planet_semimajor_axis * AU_TO_RSUN / self.star_radius  # semi-major axis (in units of stellar radii)
            ma.inc = 90  # orbital inclination (in degrees)
            ma.ecc = planet_ecc  # eccentricity
            ma.w = 0  # longitude of periastron (in degrees)
            ma.u = self.ab  # limb darkening coefficients
            ma.limb_dark = "quadratic"  # limb darkening model
            m = batman.TransitModel(ma, time_transit)  # initializes model
            model = m.light_curve(ma)  # calculates light curve
            fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
            axs[0].plot(time_transit, flux[
                (moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)],
                        color='gray', alpha=1, rasterized=True, label="Flux")
            axs[0].set_title(
                "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(moon_phase))
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Flux')
            axs[0].plot(time_transit, model, color='red', alpha=1, rasterized=True, label="Model")
            flux[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)] = \
                flux[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)] + model - 1
            subtracted_planet_flux[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)] = \
                subtracted_planet_flux[(moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)] + model - 1
            axs[1].plot(time_transit, flux[
                (moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)],
                        color='gray', alpha=1, rasterized=True, label="Flux")
            axs[1].set_title(
                "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(moon_phase))
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Flux')
            axs[1].plot(time_transit, model, color='red', alpha=1, rasterized=True, label="Model")
            axs[2].plot(time_transit, subtracted_planet_flux[
                (moon_t0 - moon_transit_duration / 2 < time) & (time < moon_t0 + moon_transit_duration / 2)])
            axs[2].set_title(
                "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(moon_phase))
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Flux')
            axs[2].plot(time_transit, model, color='red', alpha=1, rasterized=True, label="Model")
            fig_transit.show()
        return flux

    def run(self):
        planet_mass_grid = self.planet_mass_grid
        moon_inc_grid = self.moon_inc_grid
        moon_ecc_grid = self.moon_ecc_grid
        moon_arg_periastron_grid = self.moon_arg_periastron_grid
        self.flux = self.subtract_planet_transit(self.ab, self.star_radius, self.star_mass, self.time, self.flux,
                                            self.planet_radius, self.planet_t0, self.planet_period, self.planet_inc)
        time_model = np.arange(0, 1, 0.0001)
        ma = batman.TransitParams()
        ma.t0 = 0.5  # time of inferior conjunction
        ma.per = 1  # orbital period, use Earth as a reference
        ma.rp = self.min_radius * R_earth_to_R_sun / self.star_radius  # planet radius (in units of stellar radii)
        ma.a = self.planet_semimajor_axis * AU_TO_RSUN / self.star_radius  # semi-major axis (in units of stellar radii)
        ma.inc = 90  # orbital inclination (in degrees)
        ma.ecc = planet_ecc  # eccentricity
        ma.w = 0  # longitude of periastron (in degrees)
        ma.u = self.ab  # limb darkening coefficients
        ma.limb_dark = "quadratic"  # limb darkening model
        m = batman.TransitModel(ma, time_model)  # initializes model
        self.model = m.light_curve(ma)
        self.model = self.model[self.model < 1]
        # baseline_model = np.full(len(model), 1)
        # model = np.append(baseline_model, model)
        # model = np.append(model, baseline_model)
        search_inputs = []
        for planet_mass in planet_mass_grid:
            min_period = 0.5 # TODO compute this value somehow
            max_period = self.au_to_period(planet_mass * M_earth_to_M_sun, self.compute_hill_radius(self.star_mass, planet_mass * M_earth_to_M_sun, self.planet_semimajor_axis))
            period_grid = self.compute_moon_period_grid(min_period, max_period, samples=self.period_grid_size, mode="log")
            for moon_inc in moon_inc_grid:
                for moon_ecc in moon_ecc_grid:
                    for moon_arg_periastron in moon_arg_periastron_grid:
                        for moon_period in period_grid:
                            for moon_initial_alpha in np.linspace(0, np.pi * 2 - np.pi * 2 / 25, 25):
                                #TODO moon_orbit_ranges should use moon_radius ?
                                search_inputs.append(SearchInput(moon_period, moon_initial_alpha, moon_ecc, moon_inc, moon_arg_periastron))
        with Pool(processes=self.cpus) as pool:
            all_residuals = pool.map(self.search, search_inputs)
        best_residuals_per_scenarios = []
        for i in np.arange(0, len(search_inputs)):
            all_residual = all_residuals[i]
            residuals = all_residual[0]
            radius = all_residual[1]
            moon_period = search_inputs[i].moon_period
            moon_initial_alpha = search_inputs[i].moon_alpha
            best_residuals_per_scenarios.append([moon_period, moon_initial_alpha, residuals, radius])
        best_residuals_per_scenarios = np.array(best_residuals_per_scenarios)
        best_residuals_per_scenarios = best_residuals_per_scenarios[np.argsort(np.array([best_residual_per_scenarios[2] for best_residual_per_scenarios in best_residuals_per_scenarios]).flatten())]
        for i in np.arange(0, 15):
            logging.info("Best residual for period %s, alpha %s: Residual->%s, Radius->%s",
                         best_residuals_per_scenarios[i][0],
                         best_residuals_per_scenarios[i][1], best_residuals_per_scenarios[i][2],
                         best_residuals_per_scenarios[i][3])
            residuals, radius, scenario_time, scenario_flux, model = self.search(SearchInput(best_residuals_per_scenarios[i][0],
                                                                                      best_residuals_per_scenarios[i][1],
                                                                                      0, 90, 0),
                                                                          return_lc=True)
            fig_transit, axs = plt.subplots(1, 1, figsize=(12, 12))
            axs.scatter(scenario_time, scenario_flux,
                     color='gray', alpha=0.4, rasterized=True, label="Flux")
            bin_means, bin_edges, binnumber = stats.binned_statistic(scenario_time,
                                                                     scenario_flux,
                                                                     statistic='mean', bins=25)
            bin_stds, _, _ = stats.binned_statistic(scenario_time,
                                                    scenario_flux, statistic='std', bins=25)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            axs.errorbar(bin_centers, bin_means, yerr=bin_stds / 2, xerr=bin_width / 2, marker='o', markersize=4,
                         color='darkorange', alpha=1, linestyle='none')
            axs.scatter(scenario_time, model,
                     color='red', alpha=1, rasterized=True, label="Model")
            axs.set_title(
                "Moon period " + str(best_residuals_per_scenarios[i][0]) + " with alpha="
                + str(best_residuals_per_scenarios[i][1]) + " and residual " + str(best_residuals_per_scenarios[i][2]))
            axs.set_xlabel('Time')
            axs.set_ylabel('Flux')
            fig_transit.show()


class SearchInput:
    def __init__(self, moon_period, moon_alpha, moon_ecc, moon_inc, moon_arg_periastron) -> None:
        self.moon_period = moon_period
        self.moon_alpha = moon_alpha
        self.moon_ecc = moon_ecc
        self.moon_inc = moon_inc
        self.moon_arg_periastron = moon_arg_periastron
        self.min_radius = min_radius


formatter = logging.Formatter('%(message)s')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
target_name = "TIC 350618622"
object_dir = target_name + "_EMLS"
lc_builder = LcBuilder()
object_info = lc_builder.build_object_info(target_name=target_name, author=None, sectors="all", file=None, cadence=120,
                              initial_mask=None, initial_transit_mask=None, star_info=None, aperture=None,
                              eleanor_corr_flux="pdcsap_flux", outliers_sigma=3, high_rms_enabled=False,
                              high_rms_threshold=1.5, high_rms_bin_hours=4, smooth_enabled=False,
                              auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                              auto_detrend_period=None, prepare_algorithm=None, reduce_simple_oscillations=False,
                              oscillation_snr_threshold=4, oscillation_amplitude_threshold=0.1, oscillation_ws_scale=60,
                              oscillation_min_period=0.002, oscillation_max_period=0.2)
if not os.path.exists(object_dir):
    os.mkdir(object_dir)
lc_build = lc_builder.build(object_info, object_dir)
star_mass = lc_build.star_info.mass
star_radius = lc_build.star_info.radius
ab = lc_build.star_info.ld_coefficients
times = lc_build.lc.time.value
flux = lc_build.lc.flux.value
flux = wotan.flatten(times, flux, method="biweight", window_length=0.5)
planet_radius = 11.8
planet_period = 52.97818
planet_t0 = 1376.0535
planet_duration = 4.452 / 24
planet_inc = 89
planet_ecc = 0.01
planet_arg_periastron = 0
t0s = [i for i in np.arange(planet_t0, np.max(times), planet_period)]
planet_mass = 133.4886
planet_impact_param = 0.42
planet_semimajor_axis = ExoMoonLeastSquares.compute_semimajor_axis(star_mass, planet_period)
min_radius = 2
max_radius = 8

P1 = planet_period * u.day
a = np.cbrt((ac.G * star_mass * u.M_sun * P1 ** 2) / (4 * np.pi ** 2)).to(u.au)
model = ellc.lc(
    t_obs=times,
    radius_1=(star_radius * u.R_sun).to(u.au) / a,  # star radius convert from AU to in units of a
    radius_2=(planet_radius * u.R_earth).to(u.au) / a,
    # convert from Rearth (equatorial) into AU and then into units of a
    sbratio=0,
    incl=planet_inc,
    light_3=0,
    t_zero=planet_t0,
    period=planet_period,
    a=None,
    q=1e-6,
    f_c=None, f_s=None,
    ldc_1=ab, ldc_2=None,
    gdc_1=None, gdc_2=None,
    didt=None,
    domdt=None,
    rotfac_1=1, rotfac_2=1,
    hf_1=1.5, hf_2=1.5,
    bfac_1=None, bfac_2=None,
    heat_1=None, heat_2=None,
    lambda_1=None, lambda_2=None,
    vsini_1=None, vsini_2=None,
    t_exp=None, n_int=None,
    grid_1='default', grid_2='default',
    ld_1='quad', ld_2=None,
    shape_1='sphere', shape_2='sphere',
    spots_1=None, spots_2=None,
    exact_grav=False, verbose=1)

for t0 in t0s:
    fig_transit, axs = plt.subplots(1, 1, figsize=(12, 12))
    axs.scatter(times[(times > t0 - planet_duration) & (times < t0 + planet_duration)],
             flux[(times > t0 - planet_duration) & (times < t0 + planet_duration)],
             color='gray', alpha=1, rasterized=True, label="Flux")
    axs.plot(times[(times > t0 - planet_duration) & (times < t0 + planet_duration)],
             model[(times > t0 - planet_duration) & (times < t0 + planet_duration)],
             color='red', alpha=1, rasterized=True, label="Flux")
    axs.set_title("Planet transit at t0=" + str(t0))
    axs.set_xlabel('Time')
    axs.set_ylabel('Flux')
    fig_transit.show()

emls = ExoMoonLeastSquares(7, star_mass, star_radius, ab, planet_radius, planet_period, planet_t0, planet_duration,
                           planet_semimajor_axis, planet_inc, planet_ecc, planet_arg_periastron, planet_impact_param,
                           min_radius, max_radius, t0s, times, flux, period_grid_size=10000)
moon_radius = 3
moon_period = 2
emls.flux = emls.inject_moon(emls.time, emls.flux, t0s, planet_mass, planet_semimajor_axis, planet_ecc, moon_radius, moon_period)
emls.planet_mass_grid = [planet_mass]
emls.moon_inc_grid = [90]
emls.moon_ecc_grid = [0]
emls.moon_arg_periastron_grid = [0]
emls.run()
