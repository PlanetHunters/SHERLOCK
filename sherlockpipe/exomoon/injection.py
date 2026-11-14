"""Pandora moon injection, the absorbed-fraction test and the recovery of an injected moon.
"""
import copy
import logging
import tempfile
import astropy.units as u
from lcbuilder.helper import LcbuilderHelper
import pandoramoon as pandora
import numpy as np


class MoonInjection:
    """Pandora moon injection, the absorbed-fraction test and the recovery of an injected moon.
    """

    RECOVERY_PERIOD_TOLERANCE = 0.05
    MAX_ALIAS_ORDER = 30

    def inject_moon(self, time, flux, flux_err, t0s, planet_semimajor_axis, planet_ecc, moon_radius,
                    moon_period, moon_mass, initial_alpha=0, plot=False, inject_planet=False, inject_moon=True):
        """Inject a moon transit signal into the light curve.

        Parameters
        ----------
        time : array_like
            Time array in days.
        flux : array_like
            Normalized flux array.
        flux_err : array_like
            Flux uncertainty array.
        t0s : list
            List of barycentric transit times.
        planet_semimajor_axis : float
            Planet semimajor axis in AU.
        planet_ecc : float
            Planet orbital eccentricity.
        moon_radius : float
            Moon radius in Earth radii.
        moon_period : float
            Moon orbital period in days.
        moon_mass : float
            Moon mass in Earth masses.
        initial_alpha : float, optional
            Initial moon orbital phase in radians, default 0.
        plot : bool, optional
            If True, generate injection plots, default False.
        inject_planet : bool, optional
            If True, inject planet transit as well, default False.
        inject_moon : bool, optional
            If True, inject moon transit signal, default True.

        Returns
        -------
        numpy.ndarray
            Flux array with injected moon signal.
        """
        logging.info("Injecting moon with radius of  %.2fR_e, %.2fdays and %.2frad", moon_radius, moon_period, initial_alpha)
        star_radius_m = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.m)
        star_radius_rearth = LcbuilderHelper.convert_from_to(self.star_radius, u.R_sun, u.R_earth)
        params = pandora.model_params()
        params.R_star = star_radius_m  # [m]
        params.u1 = self.ab[0]
        params.u2 = self.ab[1]
        # planet parameters
        params.per_bary = self.planet_period  # [days]
        params.a_bary = self.injection_a_over_rstar()  # [R_star]
        params.r_planet = self.planet_radius / star_radius_rearth  # [R_star]
        params.b_bary = self.planet_impact_param  # [0..1]
        params.t0_bary = self.planet_t0  # [days]
        params.t0_bary_offset = 0.001  # [days]
        params.M_planet = LcbuilderHelper.convert_from_to(self.planet_mass, u.M_earth, u.kg)  # [kg]
        params.w_bary = self.planet_arg_periastron  # [deg]
        params.ecc_bary = self.planet_ecc  # [0..1]
        # moon parameters
        params.r_moon = moon_radius / star_radius_rearth  # [R_star]
        params.per_moon = moon_period  # [days]
        params.tau_moon = (params.t0_bary / moon_period - (initial_alpha + np.pi) / (2 * np.pi)) % 1
        params.Omega_moon = 0  # [0..180]
        params.i_moon = self.planet_inc  # [0..180]
        params.e_moon = 0  # [0..1]
        params.w_moon = self.planet_arg_periastron  # [deg]
        params.M_moon = LcbuilderHelper.convert_from_to(moon_mass, u.M_earth, u.kg)  # [kg]
        # times params
        # params.epochs = 3  # [int]
        # params.epoch_duration = 2  # [days]
        # params.cadences_per_day = 48  # [int]
        params.epoch_distance = self.planet_period  # [days]
        params.supersampling_factor = 1  # [int]
        params.occult_small_threshold = 0.1  # [0..1]
        params.hill_sphere_threshold = 1
        model = pandora.moon_model(params)
        injected_flux_total, injected_flux_planet, injected_flux_moon = \
            self.pandora_light_curve(model, time, self.exposure_at(time))
        self.injected_flux_moon_pandora = injected_flux_moon
        self.injected_flux_planet_pandora = injected_flux_planet
        self.injected_moon_period = moon_period
        self.injected_moon_alpha = initial_alpha
        # model.video(
        #     time=pandora_time,
        #     limb_darkening=True,
        #     teff=3200,
        #     planet_color="black",
        #     moon_color="black",
        #     ld_circles=100
        # )
        if inject_moon:
            flux = flux - (1 - injected_flux_moon)
        if inject_planet:
            flux = flux - (1 - injected_flux_planet)
        # for t0 in t0s:
        #     max_moon_t0_tau = self.compute_moon_ts(t0, moon_orbit_transit_duration / 2, moon_transit_duration / 2)
        #     moon_t0, moon_phase = self.compute_moon_t0(self.planet_t0, t0, moon_period, initial_alpha, max_moon_t0_tau)
        #     #time_transit_mask = (moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)
        #     time_transit_mask = (t0 - moon_orbit_transit_duration < time) & (time < t0 + moon_orbit_transit_duration)
        #     time_transit = time[time_transit_mask]
        #     ma = batman.TransitParams()
        #     ma.t0 = moon_t0  # time of inferior conjunction
        #     ma.per = self.planet_period  # orbital period, use Earth as a reference
        #     ma.rp = moon_radius * R_earth_to_R_sun / self.star_radius  # planet radius (in units of stellar radii)
        #     ma.a = planet_semimajor_axis * AU_TO_RSUN / self.star_radius  # semi-major axis (in units of stellar radii)
        #     ma.inc = self.planet_inc  # orbital inclination (in degrees)
        #     ma.ecc = planet_ecc  # eccentricity
        #     ma.w = 0  # longitude of periastron (in degrees)
        #     ma.u = self.ab  # limb darkening coefficients
        #     ma.limb_dark = "quadratic"  # limb darkening model
        #     m = batman.TransitModel(ma, time_transit)  # initializes model
        #     model_moon = m.light_curve(ma)  # calculates light curve
        #     # flux[(moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)] = \
        #     #     flux[(moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)] - (1 - model_moon)
        #     subtracted_planet_flux[time_transit_mask] = \
        #         subtracted_planet_flux[time_transit_mask] - (1 - injected_flux_moon[time_transit_mask])
        #     if plot:
        #         fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
        #         axs[0].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[0].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[0].set_xlabel('Time')
        #         axs[0].set_ylabel('Flux')
        #         axs[0].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[0].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[1].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[1].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[1].set_xlabel('Time')
        #         axs[1].set_ylabel('Flux')
        #         axs[1].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[1].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[2].scatter(time_transit, subtracted_planet_flux[time_transit_mask])
        #         axs[2].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[2].set_xlabel('Time')
        #         axs[2].set_ylabel('Flux')
        #         axs[2].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[2].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon.png")
        #         plt.close()
        #         time_transit_mask = (moon_t0 - moon_transit_duration < time) & (time < moon_t0 + moon_transit_duration)
        #         time_transit = time[time_transit_mask]
        #         # limb darkening model
        #         m = batman.TransitModel(ma, time_transit)  # initializes model
        #         model_moon = m.light_curve(ma)
        #         fig_transit, axs = plt.subplots(3, 1, figsize=(16, 10))
        #         axs[0].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[0].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[0].set_xlabel('Time')
        #         axs[0].set_ylabel('Flux')
        #         axs[0].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[0].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[1].scatter(time_transit, flux[time_transit_mask],
        #                        color='gray', alpha=1, rasterized=True, label="Flux")
        #         axs[1].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[1].set_xlabel('Time')
        #         axs[1].set_ylabel('Flux')
        #         axs[1].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[1].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         axs[2].scatter(time_transit, subtracted_planet_flux[time_transit_mask])
        #         axs[2].set_title(
        #             "Injected transit in t0 " + str(t0) + " with moon t0=" + str(moon_t0) + " and phase " + str(
        #                 moon_phase))
        #         axs[2].set_xlabel('Time')
        #         axs[2].set_ylabel('Flux')
        #         axs[2].plot(time_transit, model_moon, color='red', alpha=1, rasterized=True, label="Model")
        #         axs[2].plot(time_transit, injected_flux_moon[time_transit_mask],
        #                     color='orange', alpha=1, rasterized=True, label="Pandora Model")
        #         plt.savefig(self.object_dir + "/T0_" + str(round(t0, 3)) + "_moon_focus.png")
        #         plt.close()
        return flux

    @classmethod
    def find_period_alias(cls, moon_period, planet_period, found_period, start_error):
        """Find the alias of a moon period closest to a found period.

        Parameters
        ----------
        moon_period : float
            True moon period in days.
        planet_period : float
            Planet period in days.
        found_period : float
            Detected period in days.
        start_error : float
            Relative error an alias must beat.

        Returns
        -------
        tuple
            (alias_period, k, relative_error). alias_period and k are None if
            no alias beats start_error, and relative_error is then start_error.
        """
        alias_period, alias_k, min_error = None, None, start_error
        for k in range(-cls.MAX_ALIAS_ORDER, cls.MAX_ALIAS_ORDER + 1):
            frequency = 1 / moon_period + k / planet_period
            if frequency <= 0:
                continue
            error = abs(1 / frequency - found_period) / moon_period
            if error < min_error:
                alias_period, alias_k, min_error = 1 / frequency, k, error
        return alias_period, alias_k, min_error

    @classmethod
    def classify_recovered_period(cls, moon_period, planet_period, found_period):
        """Classify a found period as the injected moon period, one of its sampling aliases, or neither.

        The found period is assigned to the nearest member of the alias family
        1/P' = 1/P + k/P_planet, |k| up to MAX_ALIAS_ORDER, and must lie within
        RECOVERY_PERIOD_TOLERANCE of it. A fixed tolerance around the injected period alone
        counts a neighbouring alias as the period whenever the alias spacing, about
        P^2 / P_planet, is below the tolerance: a 10 d moon of a 218 d planet has its k = 1
        alias 4.4 % away.

        Parameters
        ----------
        moon_period : float
            Injected moon period in days.
        planet_period : float
            Planet period in days.
        found_period : float
            Recovered period in days.

        Returns
        -------
        tuple
            (period_match, alias_match, k, alias_period, relative_error): k is the order of the
            nearest alias (0 for the injected period) and relative_error its distance to the
            found period over the moon period. period_match needs k = 0, alias_match k != 0.
        """
        alias_period, k, error = cls.find_period_alias(moon_period, planet_period, found_period,
                                                                       np.inf)
        close = k is not None and error < cls.RECOVERY_PERIOD_TOLERANCE
        return bool(close and k == 0), bool(close and k != 0), k, alias_period, error

    def measure_absorbed_fraction(self, before_planet_fit, best_input, moon_t0s):
        """Measure the fraction of the best moon signal that the planet fit absorbs (audit item B3).

        A moon that stays inside the planet transit looks like extra planet depth, so the planet
        fit takes part of it. The best hypothesis, with the fitted transit geometry, is injected into
        the data before the planet fit and the fit is repeated. The absorbed fraction is one minus
        the depth gained in the moon window after the fit, over the depth gained when the moon is
        added after the fit, over the epochs both fits kept. It assumes the absorption is linear.
        When the data already contain this moon, it measures the absorption of a second copy, which
        is a lower bound once the first copy has pushed the planet radius to its bound.

        Parameters
        ----------
        before_planet_fit : ExoMoonLeastSquares
            Copy of this search taken just before the planet fit.
        best_input : SearchInput
            Best hypothesis.
        moon_t0s : list of float
            Moon mid-transit times of the best hypothesis, one per epoch.

        Returns
        -------
        float or None
            Absorbed fraction, or None when it cannot be measured.
        """
        radius = self.best_snr_radius if np.isfinite(self.best_snr_radius) and self.best_snr_radius > 0 \
            else self.min_radius
        radius = max(radius, self.min_radius)
        try:
            injected = copy.copy(before_planet_fit)
            logging.info("Absorbed-fraction test: repeating the planet fit with the best moon (%.2f R_earth) "
                         "injected; its own fit messages are not logged", radius)
            previous_disable = logging.root.manager.disable
            logging.disable(logging.WARNING)
            try:
                with tempfile.TemporaryDirectory() as plots_dir:
                    injected.object_dir = plots_dir
                    # The injected and the reference moon both use the fitted geometry, so they are
                    # the same signal; the copy still holds the input geometry.
                    flux = before_planet_fit.flux + self.moon_signal(before_planet_fit.time, moon_t0s, radius) - 1
                    time, flux, flux_err = injected.subtract_planet_transit(before_planet_fit.time, flux,
                                                                            before_planet_fit.flux_err,
                                                                            mode=self.subtract_mode)
            finally:
                logging.disable(previous_disable)
            logging.info("Absorbed-fraction test: fold fit converged=%s, at bound: %s; %d of %d transits kept",
                         injected.fold_fit_status["converged"] if injected.fold_fit_status else None,
                         (", ".join(injected.fold_fit_status["at_bound"]) or "none") if injected.fold_fit_status
                         else "-",
                         len(injected.bary_t0s), len(before_planet_fit.bary_t0s))
            # All three depths cover the epochs both fits kept.
            epochs = [t0 for t0 in self.bary_t0s if t0 in set(injected.bary_t0s)]
            original = self.window_depth(best_input, self.time, self.flux, self.flux_err, epochs)
            added = self.window_depth(best_input, self.time, self.flux + self.moon_signal(self.time, moon_t0s, radius)
                                      - 1, self.flux_err, epochs)
            fitted = self.window_depth(best_input, time, flux, flux_err, epochs)
        except Exception as error:
            logging.warning("Could not measure the moon fraction absorbed by the planet fit: %s", error)
            return None
        if not (np.isfinite(original) and np.isfinite(added) and np.isfinite(fitted)) or added - original <= 0:
            logging.warning("Could not measure the moon fraction absorbed by the planet fit: no depth gained")
            return None
        absorbed = float(1 - (fitted - original) / (added - original))
        logging.info("The planet fit absorbs %.0f%% of the best moon signal (%.2f R_earth, injected at the best "
                     "period and phase); the reported radius and SNR are biased low by that much in depth. If the "
                     "data already hold this moon, the value is a lower bound, because the planet radius may "
                     "already sit at its bound", 100 * absorbed, radius)
        return absorbed
