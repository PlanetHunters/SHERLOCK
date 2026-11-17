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

    WOBBLE_GRID_POINTS = 4001

    def planet_wobble_shifts(self, params, t0s):
        """Return the planet mid-transit offset in days that the injected moon causes at each epoch.

        The planet orbits the planet-moon barycentre, so it crosses the star's centre line (sky x
        = 0) early or late. The crossing is found on the Pandora planet coordinates around each
        epoch, and the same crossing of a massless moon, whose planet sits on the barycentre, is
        subtracted, so the offset is the moon's alone, whatever Pandora's time origin.
        """
        massless = copy.copy(params)
        massless.M_moon = params.M_moon * 1e-9
        half = max(self.planet_duration, 0.1) / 2  # the planet centre is on the star within +-T14 / 2
        grid = np.linspace(-half, half, self.WOBBLE_GRID_POINTS)
        shifts = []
        for t0 in t0s:
            time = t0 + grid
            crossings = []
            for model_params in (params, massless):
                xp = np.asarray(pandora.moon_model(model_params).coordinates(time)[0], dtype=float)
                sign = np.flatnonzero(np.sign(xp[:-1]) != np.sign(xp[1:]))
                if len(sign) == 0:
                    crossings.append(np.nan)
                    continue
                i = sign[np.argmin(np.abs(time[sign] - t0))]
                crossings.append(time[i] - xp[i] * (time[i + 1] - time[i]) / (xp[i + 1] - xp[i]))
            shifts.append(crossings[0] - crossings[1] if np.all(np.isfinite(crossings)) else 0.0)
        return np.asarray(shifts, dtype=float)

    @staticmethod
    def shift_epochs(time, t0s, shifts):
        """Shift each point by the offset of its nearest epoch and sort.

        Returns
        -------
        tuple
            (shifted times sorted, the order that sorts the original arrays the same way).
        """
        time = np.asarray(time, dtype=float)
        moved = time + MoonInjection.nearest_epoch_values(time, t0s, shifts)
        order = np.argsort(moved, kind="stable")
        return moved[order], order

    @staticmethod
    def nearest_epoch_values(time, t0s, values):
        """Return, for each time, the value of its nearest epoch."""
        time = np.asarray(time, dtype=float)
        epoch_order = np.argsort(np.asarray(t0s, dtype=float))
        t0s = np.asarray(t0s, dtype=float)[epoch_order]
        values = np.asarray(values, dtype=float)[epoch_order]
        index = np.clip(np.searchsorted(t0s, time), 1, len(t0s) - 1) if len(t0s) > 1 else np.zeros(len(time), int)
        if len(t0s) > 1:
            nearer_left = np.abs(time - t0s[index - 1]) <= np.abs(t0s[index] - time)
            index = np.where(nearer_left, index - 1, index)
        return values[index]

    def inject_moon(self, time, flux, flux_err, t0s, planet_semimajor_axis, planet_ecc, moon_radius,
                    moon_period, moon_mass, initial_alpha=0, plot=False, inject_planet=False, inject_moon=True,
                    retrograde=False, planet_ttv=False, barycentre_ttv=None):
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
        retrograde : bool, optional
            Inject a retrograde moon, default False: the orbit normal is reversed,
            (i, Omega) -> (180 - i, Omega + 180), which mirrors the sky-plane x of the moon about
            the planet and keeps y.
        plot : bool, optional
            If True, generate injection plots, default False.
        inject_planet : bool, optional
            If True, inject planet transit as well, default False.
        inject_moon : bool, optional
            If True, inject moon transit signal, default True.
        planet_ttv : bool, optional
            If True, the real planet transits are moved as the moon would move them: each epoch of
            the light curve is shifted in time by the planet's offset from the barycentre that the
            Pandora model gives (`planet_wobble_shifts`), and the moon is injected at the shifted
            times. The shifted times are left in injected_time (sorted, as the flux returned) and
            the shifts in injected_planet_ttvs; the caller replaces its time with injected_time.
            Default False: the real planet stays on its own times, without the moon's TTV.
        barycentre_ttv : dict, optional
            A timing variation of the planet-moon barycentre, as another planet causes:
            {'amplitude': days, 'period': days, 'phase': rad}, a shift A sin(2 pi t0 / P + phase)
            of each epoch. The light curve epochs are shifted by it (with the moon's TTV on top
            when planet_ttv) and the moon moves with the barycentre, so it is injected at the
            time minus the shift. Default None.

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
        # The barycentre transits on the planet's linear ephemeris: a non-zero offset moves the moon
        # away from the real planet, which stays on that ephemeris.
        params.t0_bary_offset = 0.0  # [days]
        params.M_planet = LcbuilderHelper.convert_from_to(self.planet_mass, u.M_earth, u.kg)  # [kg]
        params.w_bary = self.planet_arg_periastron  # [deg]
        params.ecc_bary = self.planet_ecc  # [0..1]
        # moon parameters
        params.r_moon = moon_radius / star_radius_rearth  # [R_star]
        params.per_moon = moon_period  # [days]
        params.tau_moon = (params.t0_bary / moon_period - (initial_alpha + np.pi) / (2 * np.pi)) % 1
        params.Omega_moon = 180 if retrograde else 0  # [deg]
        params.i_moon = 180 - self.planet_inc if retrograde else self.planet_inc  # [0..180]
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
        self.injected_planet_ttvs = None
        self.injected_time = None
        exposure = self.exposure_at(time)
        self.injected_barycentre_ttvs = None
        bary_shifts = np.zeros(len(t0s))
        if barycentre_ttv:
            bary_shifts = barycentre_ttv['amplitude'] * np.sin(
                2 * np.pi * np.asarray(t0s, dtype=float) / barycentre_ttv['period'] + barycentre_ttv.get('phase', 0.0))
            self.injected_barycentre_ttvs = [float(shift) for shift in bary_shifts]
        if planet_ttv or barycentre_ttv:
            shifts = (self.planet_wobble_shifts(params, t0s) if planet_ttv else np.zeros(len(t0s))) + bary_shifts
            time, order = self.shift_epochs(time, t0s, shifts)
            flux, flux_err = np.asarray(flux)[order], np.asarray(flux_err)[order] if flux_err is not None else None
            exposure = exposure if exposure is None or np.ndim(exposure) == 0 else np.asarray(exposure)[order]
            if self.exposure_lookup is not None:
                self.exposure_lookup = (time, np.asarray(exposure, dtype=float))
            self.injected_planet_ttvs = [float(shift) for shift in shifts]
            self.injected_time = time
            self.injected_flux_err = flux_err
            logging.info("Planet moved by the injection: TTV amplitude %.2f min (RMS %.2f min over %d epochs), of which "
                         "the barycentre %.2f min", np.max(np.abs(shifts)) * 1440, np.std(shifts) * 1440, len(shifts),
                         np.max(np.abs(bary_shifts)) * 1440)
        # The moon follows the barycentre: Pandora places it on the linear ephemeris, so it is evaluated
        # at the time minus the barycentre shift of its epoch.
        moon_time = time - self.nearest_epoch_values(time, t0s, bary_shifts) if barycentre_ttv else time
        injected_flux_total, injected_flux_planet, injected_flux_moon = \
            self.pandora_light_curve(model, moon_time, exposure)
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
    def find_period_alias(cls, moon_period, planet_period, found_period, start_error, mirror=False):
        """Find the alias of a moon period closest to a found period.

        With one look per planet orbit the moon phase at epoch n is theta_n = alpha + 2 pi n P_p / P
        and the moon offset goes as cos(theta_n). The sampling aliases 1/P' = 1/P + k/P_planet
        repeat those phases. The mirror aliases 1/P' = k/P_planet - 1/P repeat them negated, and
        with alpha' = -alpha give the same offsets, cos(-theta_n) = cos(theta_n), up to the orbit
        size, which scales as P'^(2/3): they are the negative-frequency branch of the same family.
        For the same reason a retrograde moon at (P, alpha) has the offsets of a prograde one at
        (P, -alpha).

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
        mirror : bool, optional
            Search the mirror family instead of the sampling family.

        Returns
        -------
        tuple
            (alias_period, k, relative_error). alias_period and k are None if
            no alias beats start_error, and relative_error is then start_error.
        """
        alias_period, alias_k, min_error = None, None, start_error
        for k, frequency in cls.alias_frequencies(moon_period, planet_period, mirror):
            error = abs(1 / frequency - found_period) / moon_period
            if error < min_error:
                alias_period, alias_k, min_error = 1 / frequency, k, error
        return alias_period, alias_k, min_error

    @classmethod
    def alias_frequencies(cls, moon_period, planet_period, mirror=False):
        """Return (k, frequency) of the positive-frequency aliases of a moon period.

        The sampling family 1/P + k/P_planet takes |k| up to MAX_ALIAS_ORDER. The mirror family
        k/P_planet - 1/P takes k within MAX_ALIAS_ORDER of round(2 P_planet / P), its member nearest
        the moon period, so both families span the same frequency range.
        """
        if mirror:
            centre = int(round(2 * planet_period / moon_period))
            pairs = [(k, k / planet_period - 1 / moon_period)
                     for k in range(centre - cls.MAX_ALIAS_ORDER, centre + cls.MAX_ALIAS_ORDER + 1)]
        else:
            pairs = [(k, 1 / moon_period + k / planet_period)
                     for k in range(-cls.MAX_ALIAS_ORDER, cls.MAX_ALIAS_ORDER + 1)]
        return [(k, frequency) for k, frequency in pairs if frequency > 0]

    @classmethod
    def classify_recovered_period(cls, moon_period, planet_period, found_period):
        """Classify a found period as the injected moon period, one of its aliases, or neither.

        The found period is assigned to the nearest member of the sampling family
        1/P' = 1/P + k/P_planet or of the mirror family 1/P' = k/P_planet - 1/P (see
        find_period_alias), and must lie within RECOVERY_PERIOD_TOLERANCE of it. A fixed tolerance
        around the injected period alone counts a neighbouring alias as the period whenever the
        alias spacing, about P^2 / P_planet, is below the tolerance: a 10 d moon of a 218 d planet
        has its k = 1 alias 4.4 % away.

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
            (period_match, alias_match, k, alias_period, relative_error, mirror): k is the order of
            the nearest alias (0 for the injected period), relative_error its distance to the found
            period over the moon period, and mirror whether it belongs to the mirror family.
            period_match needs the sampling family with k = 0, alias_match any other member.
        """
        alias_period, k, error = cls.find_period_alias(moon_period, planet_period, found_period, np.inf)
        mirror_period, mirror_k, mirror_error = cls.find_period_alias(moon_period, planet_period, found_period,
                                                                      error, mirror=True)
        mirror = mirror_k is not None
        if mirror:
            alias_period, k, error = mirror_period, mirror_k, mirror_error
        close = k is not None and error < cls.RECOVERY_PERIOD_TOLERANCE
        exact = close and k == 0 and not mirror
        return bool(exact), bool(close and not exact), k, alias_period, error, mirror

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
