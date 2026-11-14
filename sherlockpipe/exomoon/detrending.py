"""Detrending of an undetrended light curve with the planet and moon windows masked.
"""
import logging
import batman
import numpy as np
import wotan
from sherlockpipe.exomoon.constants import R_earth_to_R_sun


class Detrending:
    """Detrending of an undetrended light curve with the planet and moon windows masked.
    """

    DETREND_MASKS = ('auto', 'moon_window', 'planet', 'none')
    AUTO_PLANET_MASK_RATIO = 2.0  # 'auto' masks only the planet when the moon window is this many biweight windows
    PLANET_MASK_HALF_WIDTH = 0.75  # in T14: the 'planet' detrend mask is +-0.75 T14 around each epoch

    def detrend(self, time, flux, flux_err):
        """Detrend an undetrended light curve without letting the planet or a moon shape the trend (audit item A1).

        A biweight of detrend_window days (Hippke et al. 2019, AJ 158, 143). What keeps the
        planet and a moon out of the trend depends on detrend_mask:

        - 'moon_window': the points within the widest searched moon window of every epoch are
          masked; where the mask is wider than the biweight window, the trend is the straight line
          between the mask edges, so stellar variability inside the window is not followed.
        - 'planet': only +-PLANET_MASK_HALF_WIDTH T14 around each epoch is masked, the same way.
        - 'none': nothing is masked. The planet model (`detrend_planet_flux`) is divided out first,
          so the biweight follows the star under the transit too, and a moon is kept out only by
          the biweight's robustness and its window being several moon transits long.
        - 'auto': one of the first two, see `resolved_detrend_mask`.

        Every point is then divided by the trend.

        Returns
        -------
        tuple
            (time, flux, flux_err), sorted by time, with the flux and errors divided by the trend.
        """
        order = np.argsort(time)
        time, flux = np.asarray(time, dtype=float)[order], np.asarray(flux, dtype=float)[order]
        flux_err = None if flux_err is None else np.asarray(flux_err, dtype=float)[order]
        scheme = self.resolved_detrend_mask()
        if scheme == 'none':
            star = flux / self.detrend_planet_flux(time)
            _, trend = wotan.flatten(time, star, method='biweight', window_length=self.detrend_window,
                                     return_trend=True)
            finite = np.isfinite(trend) & (trend > 0)
        else:
            half_width = self.moon_window_half_width() if scheme == 'moon_window' \
                else self.PLANET_MASK_HALF_WIDTH * self.planet_duration
            mask = self.epoch_distance(time) < half_width
            _, trend = wotan.flatten(time, flux, method='biweight', window_length=self.detrend_window,
                                     return_trend=True, mask=mask)
            # Inside a mask the biweight only sees one side of the window, which is noisy; the
            # trend there is the straight line between the mask edges instead.
            finite = np.isfinite(trend) & (trend > 0) & ~mask
        if np.count_nonzero(finite) < 2:
            raise ValueError("The detrending left no trend: too few points outside the masks")
        trend = np.interp(time, time[finite], trend[finite])
        return time, flux / trend, None if flux_err is None else flux_err / trend

    def resolved_detrend_mask(self):
        """Return the detrend scheme in use: detrend_mask, or for 'auto' 'planet' or 'moon_window'.

        Masking the whole moon window protects any moon from the biweight, but where the window is
        much wider than the biweight window the trend inside it is one straight line, and stellar
        variability on shorter time scales stays in the data the search sees (on Kepler-1514 b with a
        5.3 M_J planet, a 5.8 d window against a 2.7 d biweight left 170 ppm offsets between epochs).
        Masking only the planet transit lets the biweight follow the star there, at the cost of the
        part of a moon outside the mask that the biweight absorbs: little for a moon whose points
        stand out from the noise (about 20 % on Kepler-1514 b at 30-min cadence), up to half for one
        within the point-to-point noise. 'auto' takes the planet mask only when the moon window is
        more than AUTO_PLANET_MASK_RATIO biweight windows wide.
        """
        if self.detrend_mask != 'auto':
            return self.detrend_mask
        width = 2 * self.moon_window_half_width()
        ratio = width / self.detrend_window
        scheme = 'planet' if ratio > self.AUTO_PLANET_MASK_RATIO else 'moon_window'
        if not self._detrend_mask_logged:
            self._detrend_mask_logged = True
            logging.info("Detrending mask: %s (auto). The moon window is %.2f d, %.1f times the %.2f d biweight "
                         "window, %s", scheme, width, ratio, self.detrend_window,
                         "so a straight line across it would miss the stellar variability; only +-%.2f T14 around "
                         "each epoch is masked, and the biweight can absorb part of a moon outside it"
                         % self.PLANET_MASK_HALF_WIDTH if scheme == 'planet'
                         else "so the whole window is masked and a moon is kept out of the trend")
        return scheme

    def detrend_planet_flux(self, time):
        """Return the planet transit model at every time, for the 'none' detrend mask.

        After the fold fit it is the fitted planet (`null_planet_params`). Before it, a circular
        orbit with the a / R* that crosses the star at `planet_speed`, the input radius and b, so the
        model has the input T14 whatever the input e and w.
        """
        if self.planet_a_over_rstar is not None:
            params = self.null_planet_params()
        else:
            params = batman.TransitParams()
            params.per = self.planet_period
            params.rp = self.planet_radius * R_earth_to_R_sun / self.star_radius
            params.a = self.planet_speed() / self.star_radius * self.planet_period / (2 * np.pi)
            params.inc = float(np.degrees(np.arccos(min((self.planet_impact_param or 0.0) / params.a, 1.0))))
            params.ecc, params.w = 0.0, 90.0
            params.u, params.limb_dark = self.ab, "quadratic"
        model = np.ones(len(time))
        half = self.planet_duration if self.planet_duration else 0.5
        for t0 in self.bary_t0s:
            window = np.abs(time - t0) < half
            if np.any(window):
                params.t0 = t0
                model[window] = self.transit_model(params, time[window], self.exposure_at(time[window])) \
                    .light_curve(params)
        return model
