"""Moon transit templates for the radius grid and the moon signal of a hypothesis.
"""
import numpy as np
from sherlockpipe.exomoon.constants import R_earth_to_R_sun


class MoonTemplates:
    """Moon transit templates for the radius grid and the moon signal of a hypothesis.
    """

    TEMPLATE_MIN_SAMPLES = 1000

    @classmethod
    def build_moon_templates(cls, transit_params, radii, star_radius, min_half_window, exposure_time=None):
        """Build moon transit templates that reach unity inside their support.

        Parameters
        ----------
        transit_params : batman.TransitParams
            Transit geometry. Its rp value is overwritten.
        radii : array_like
            Moon radii in Earth radii, in increasing order.
        star_radius : float
            Stellar radius in solar radii.
        min_half_window : float
            Smallest template half-width in days.
        exposure_time : float, optional
            Exposure time in days. The templates are averaged over it, so a template
            sampled at a data time matches an exposure centred there. Default None.

        Returns
        -------
        list of (float, numpy.ndarray, numpy.ndarray)
            [(radius_Rearth, time_days_from_transit_center, flux), ...] for
            the radii that produce a transit.
        """
        cadence = 2 * min_half_window / (cls.TEMPLATE_MIN_SAMPLES - 1)
        max_half_window = transit_params.per / 2
        half_window = min(min_half_window, max_half_window)
        transit_params.rp = radii[-1] * R_earth_to_R_sun / star_radius
        while True:
            samples = max(cls.TEMPLATE_MIN_SAMPLES, int(np.ceil(2 * half_window / cadence)) + 1)
            template_time = np.linspace(-half_window, half_window, samples)
            time_model = transit_params.t0 + template_time
            widest_model = cls.transit_model(transit_params, time_model, exposure_time).light_curve(
                transit_params)
            if (widest_model[0] >= 1 and widest_model[-1] >= 1) or half_window >= max_half_window:
                break
            half_window = min(half_window * 1.25, max_half_window)
        models = []
        for radius in radii:
            transit_params.rp = radius * R_earth_to_R_sun / star_radius
            model_r = cls.transit_model(transit_params, time_model, exposure_time).light_curve(
                transit_params)
            if np.count_nonzero(model_r < 1) > 1:
                models.append((radius, template_time, model_r))
        return models

    def build_search_templates(self):
        """Build the moon transit templates for the radius grid with the current (fitted) geometry."""
        ma = self.moon_template_params()
        search_half_window = self.compute_transit_duration(self.star_radius, self.transit_semimajor_axis(),
                                                           self.planet_period, self.max_radius * R_earth_to_R_sun,
                                                           self.planet_impact_param) / 2
        self.models = self.build_moon_templates(
            ma, np.linspace(self.min_radius, self.max_radius, self.radius_grid_size), self.star_radius,
            max(self.planet_duration * 0.7, search_half_window), self.exposure_time)
        self.planet_depth_column = self.planet_depth_derivative(self.time)
        self.planet_t0_column, self.planet_epoch_index = self.planet_t0_derivative(self.time)
        self.planet_shape_columns = self.planet_shape_derivatives(self.time)

    def moon_signal(self, time, moon_t0s, radius):
        """Return the flux of a moon of the given radius transiting at moon_t0s, with the fitted geometry."""
        flux = np.ones(len(time))
        params = self.moon_template_params()
        params.rp = radius * R_earth_to_R_sun / self.star_radius
        for moon_t0 in moon_t0s:
            window = np.abs(time - moon_t0) < self.planet_period / 2
            if np.any(window):
                params.t0 = moon_t0
                flux[window] *= self.transit_model(params, time[window],
                                                                  self.exposure_at(time[window])).light_curve(params)
        return flux
