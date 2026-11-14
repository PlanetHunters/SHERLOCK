"""Exposure of every point (one for the curve, or one per point when cadences mix), transit
and Pandora models averaged over it, and the flux errors and weights.
"""
import logging
import warnings
import batman
import numpy as np


class ExposureAveragedModel:
    """batman transit model averaged over each exposure with the midpoint rule.

        batman's own supersample_factor converges slowly at long cadence (about 80 ppm off at 15
        sub-exposures over 30 minutes for a 0.25 % transit), while midpoint averaging is within about
        5 ppm. It exposes the same light_curve(params) interface as batman.TransitModel.
    """

    def __init__(self, transit_params, time, exposure_time, sub_exposures):
        time = np.asarray(time, dtype=float)
        offsets = ((np.arange(sub_exposures) + 0.5) / sub_exposures - 0.5) * exposure_time
        self.shape = (len(time), sub_exposures)
        self.model = batman.TransitModel(transit_params, (time[:, None] + offsets[None, :]).ravel())

    def light_curve(self, transit_params):
        return self.model.light_curve(transit_params).reshape(self.shape).mean(axis=1)


class PerPointExposureModel:
    """Transit model for points with different exposures, as a light curve that mixes cadences.

        The points are grouped by exposure, and each group gets the model `transit_model` builds for
        its exposure. It exposes the same light_curve(params) interface as batman.TransitModel.
    """

    def __init__(self, transit_params, time, exposures):
        time = np.asarray(time, dtype=float)
        exposures = np.asarray(exposures, dtype=float)
        self.size = len(time)
        self.groups = []
        for value in np.unique(exposures):
            index = np.flatnonzero(exposures == value)
            self.groups.append((index, ExposureHandling.transit_model(
                transit_params, time[index], float(value) if value > 0 else None)))

    def light_curve(self, transit_params):
        flux = np.empty(self.size)
        for index, model in self.groups:
            flux[index] = model.light_curve(transit_params)
        return flux


class ExposureHandling:
    """Exposure of every point (one for the curve, or one per point when cadences mix), transit
    and Pandora models averaged over it, and the flux errors and weights.
    """

    SUBEXPOSURE_DAYS = 2 / 1440
    # Point spacings within this fraction of a group's shortest spacing are one cadence.
    CADENCE_GROUP_TOLERANCE = 0.1
    # A light curve mixes cadences when more than one cadence holds over this share of the points.
    MIXED_EXPOSURE_SHARE = 0.05

    @classmethod
    def resolve_exposure_time(cls, exposure_time, time):
        """Return the exposure time in days (None for an instantaneous model) and where it came from.

        "auto" takes the median spacing between consecutive distinct times. A spacing longer
        than a day is not an exposure, so it gives an instantaneous model with a warning.

        Raises
        ------
        ValueError
            If exposure_time is not "auto", None or a positive number of days up to 1.
        """
        time = np.asarray(time, dtype=float)
        spacing = np.diff(np.sort(time[np.isfinite(time)]))
        spacing = spacing[spacing > 0]
        median = float(np.median(spacing)) if len(spacing) else None
        if isinstance(exposure_time, str):
            if exposure_time != "auto":
                raise ValueError(f"exposure_time must be a number of days, 'auto' or None, got {exposure_time!r}")
            if median is None or median > 1:
                logging.warning("Exposure time cannot be measured from the point spacing (%s): the planet and moon "
                                "models are instantaneous. Pass exposure_time to set it",
                                "fewer than two distinct times" if median is None else f"median {median:.2f} d")
                return None, "none (instantaneous model)"
            logging.info("Exposure time %.2f min from the median point spacing", median * 1440)
            return median, "median point spacing"
        if exposure_time is None:
            if median is not None and cls.SUBEXPOSURE_DAYS < median <= 1:
                logging.warning("exposure_time=None: the planet and moon models are instantaneous although the points "
                                "are spaced by %.1f min. A long exposure smears the planet ingress, and the residual "
                                "of an unsmeared planet model can look like a moon; use exposure_time='auto'",
                                median * 1440)
            return None, "none (instantaneous model)"
        if not (np.isfinite(exposure_time) and 0 < exposure_time <= 1):
            raise ValueError(f"exposure_time must be a positive number of days up to 1, got {exposure_time}")
        return float(exposure_time), "given"

    @classmethod
    def infer_point_exposures(cls, time, window=5):
        """Return each point's exposure in days, measured from the spacing to its nearest point.

        The nearest-neighbour spacing ignores the gap on one side of a point, and a running median
        over `window` points ignores an isolated missing cadence. The spacings are then grouped:
        values within CADENCE_GROUP_TOLERANCE of a group's shortest spacing form one cadence, and
        each point gets the median of its group. A spacing longer than a day is not an exposure and
        gives 0, an instantaneous model. A small group (at most MIXED_EXPOSURE_SHARE of the points
        and of the time) whose spacing is a whole multiple of a larger group's comes from missing
        points of that cadence and takes its exposure. Nothing about the mission is assumed.

        Parameters
        ----------
        time : array_like
            Mid-exposure times in days, in any order.
        window : int, optional
            Points in the running median, default 5.

        Returns
        -------
        numpy.ndarray
            Exposure in days of each point, in the order of `time`.
        """
        time = np.asarray(time, dtype=float)
        exposures = np.zeros(len(time))
        if len(time) < 2:
            return exposures
        order = np.argsort(time, kind="stable")
        spacing = np.diff(time[order])
        spacing = np.where(spacing > 0, spacing, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nearest = np.fmin(np.r_[np.nan, spacing], np.r_[spacing, np.nan])
            half = window // 2
            padded = np.pad(nearest, half, mode="edge")
            smoothed = np.nanmedian(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
        if np.all(~np.isfinite(smoothed)):
            return exposures
        smoothed = np.where(np.isfinite(smoothed), smoothed, np.nanmedian(smoothed))
        grouped = np.empty(len(smoothed))
        rank = np.argsort(smoothed, kind="stable")
        start = 0
        while start < len(rank):
            end = np.searchsorted(smoothed[rank], smoothed[rank[start]] * (1 + cls.CADENCE_GROUP_TOLERANCE),
                                  side="right")
            grouped[rank[start:end]] = np.median(smoothed[rank[start:end]])
            start = end
        groups = cls.exposure_groups(grouped)
        tolerance = cls.CADENCE_GROUP_TOLERANCE
        for value, share in groups:
            if share > cls.MIXED_EXPOSURE_SHARE:
                continue
            for larger_value, larger_share in groups:
                multiple = value / larger_value if larger_value > 0 else 0
                if larger_share > share and multiple >= 2 - tolerance and \
                        abs(multiple - np.round(multiple)) <= tolerance * np.round(multiple):
                    grouped[grouped == value] = larger_value
                    break
        grouped[grouped > 1] = 0.0
        exposures[order] = grouped
        return exposures

    @staticmethod
    def exposure_groups(exposures):
        """Return [(exposure, share)] of the distinct exposures, the largest share first.

        The share is the larger of the share of the points and the share of the observed time
        (the sum of the exposures), so a long cadence with few points still counts.
        """
        exposures = np.asarray(exposures, dtype=float)
        values, inverse, counts = np.unique(exposures, return_inverse=True, return_counts=True)
        time_spent = np.bincount(inverse, weights=exposures)
        total_time = np.sum(time_spent)
        shares = np.maximum(counts / np.sum(counts), time_spent / total_time if total_time > 0 else 0)
        order = np.argsort(-shares, kind="stable")
        return [(float(values[i]), float(shares[i])) for i in order]

    def setup_point_exposures(self, exposure_time, time):
        """Set the exposure of every point, per point when the light curve mixes cadences.

        A light curve with one cadence keeps one exposure (see `resolve_exposure_time`), and
        exposure_lookup stays None. When "auto" finds more than one cadence holding over
        MIXED_EXPOSURE_SHARE of the points, or when an array is given, every point keeps its own
        exposure (see `infer_point_exposures`): `exposure_at` returns it for any time of the curve,
        and exposure_time becomes the exposure that carries most of the inverse-variance weight,
        used only where one exposure is needed (the moon templates and the reports).
        """
        self.exposure_lookup = None
        self.exposure_groups_found = None
        if exposure_time is not None and not isinstance(exposure_time, str) and np.ndim(exposure_time) > 0:
            exposures = np.asarray(exposure_time, dtype=float)
            if exposures.shape != np.shape(time):
                raise ValueError(f"exposure_time per point must have one value per time ({len(time)}), got "
                                 f"{exposures.shape}")
            if not np.all(np.isfinite(exposures) & (exposures >= 0) & (exposures <= 1)):
                raise ValueError("exposure_time per point must be between 0 (instantaneous) and 1 day")
            source = "given per point"
        elif isinstance(exposure_time, str) and exposure_time == "auto":
            exposures = self.infer_point_exposures(time)
            groups = self.exposure_groups(exposures)
            if sum(share > self.MIXED_EXPOSURE_SHARE for _, share in groups) < 2:
                self.exposure_time, self.exposure_source = self.resolve_exposure_time(exposure_time, time)
                return
            source = "per point, from the spacing to the nearest point"
        else:
            self.exposure_time, self.exposure_source = self.resolve_exposure_time(exposure_time, time)
            return
        groups = self.exposure_groups(exposures)
        if len(groups) == 1:
            value = groups[0][0]
            self.exposure_time = value if value > 0 else None
            self.exposure_source = source
            return
        order = np.argsort(np.asarray(time, dtype=float), kind="stable")
        self.exposure_lookup = (np.asarray(time, dtype=float)[order], exposures[order])
        self.exposure_groups_found = groups
        self.calibrate_errors_per_exposure()
        weights = self.point_weights(self.flux_err, len(exposures))
        carried = {value: float(np.sum(weights[exposures == value])) for value, _ in groups}
        dominant = max(carried, key=carried.get)
        self.exposure_time = dominant if dominant > 0 else None
        self.exposure_source = source
        logging.info("Light curve with %d exposures, each point modelled with its own: %s. The moon templates use "
                     "%.2f min, which carries %.0f%% of the weight", len(groups),
                     ", ".join(f"{value * 1440:.2f} min ({100 * share:.0f}% of the points or of the time)"
                               for value, share in groups),
                     dominant * 1440, 100 * carried[dominant] / max(sum(carried.values()), 1e-300))

    def calibrate_errors_per_exposure(self):
        """Scale the flux errors of each exposure group to its own point-to-point scatter.

        The weights of a light curve that mixes cadences compare points of different cadences,
        so their errors must be on one scale. Each group's errors are multiplied by the ratio of
        its point-to-point scatter (1.4826 MAD of consecutive differences over sqrt 2) to its
        median error, relative to the same ratio of the group with most points, which keeps the
        overall scale of the input errors. Groups with fewer than 20 consecutive pairs are left as given.
        """
        self.exposure_error_scales = {}
        if self.flux_err is None or self.exposure_lookup is None:
            return
        time = np.asarray(self.time, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        flux_err = np.array(self.flux_err, dtype=float)
        exposures = self.exposure_at(time)
        order = np.argsort(time, kind="stable")
        ratios = {}
        for value, _ in self.exposure_groups_found:
            members = order[exposures[order] == value]
            # Consecutive points of the group in time order.
            group_time, group_flux = time[members], flux[members]
            steps = np.diff(group_time)
            close = steps <= 3 * max(value, np.median(steps) if len(steps) else 0)
            differences = np.diff(group_flux)[close]
            median_err = np.median(flux_err[members]) if len(members) else np.nan
            if len(differences) < 20 or not median_err > 0:
                continue
            scatter = 1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2)
            if scatter > 0:
                ratios[value] = scatter / median_err
        if not ratios:
            return
        reference = next((value for value, _ in self.exposure_groups_found if value in ratios), None)
        for value, ratio in ratios.items():
            scale = ratio / ratios[reference]
            flux_err[exposures == value] *= scale
            self.exposure_error_scales[value] = float(scale)
        logging.info("Flux errors scaled per exposure to the point-to-point scatter: %s",
                     ", ".join(f"{value * 1440:.2f} min x{scale:.3f}" for value, scale in
                               self.exposure_error_scales.items()))
        self.flux_err = flux_err

    @staticmethod
    def usable_errors(flux_err, size):
        """Return whether flux_err holds one positive, finite error per point."""
        if flux_err is None:
            return False
        flux_err = np.asarray(flux_err, dtype=float)
        return flux_err.shape == (size,) and bool(np.all(np.isfinite(flux_err) & (flux_err > 0)))

    @classmethod
    def point_weights(cls, flux_err, size):
        """Return inverse-variance weights, or ones when the errors are not usable (see `usable_errors`)."""
        if not cls.usable_errors(flux_err, size):
            return np.ones(size)
        return 1 / np.asarray(flux_err, dtype=float) ** 2

    def exposure_at(self, time):
        """Return the exposure in days of the points at these times of the light curve.

        With one exposure for the whole curve this is exposure_time (a float or None). With
        exposures per point it is an array with the exposure of the nearest point of the curve to
        each time, so it must be called with times of the curve's own points (any subset, in any
        order), before they are shifted by a fold or a null offset.
        """
        if self.exposure_lookup is None:
            return self.exposure_time
        reference_time, reference_exposure = self.exposure_lookup
        time = np.atleast_1d(np.asarray(time, dtype=float))
        index = np.clip(np.searchsorted(reference_time, time), 1, len(reference_time) - 1)
        nearer_left = np.abs(time - reference_time[index - 1]) <= np.abs(reference_time[index] - time)
        return reference_exposure[np.where(nearer_left, index - 1, index)]

    @staticmethod
    def exposure_subset(exposure, mask):
        """Return the exposures of the masked points: the same float for one exposure, the masked array per point."""
        if exposure is None or np.ndim(exposure) == 0:
            return exposure
        return np.asarray(exposure)[mask]

    @classmethod
    def transit_model(cls, transit_params, time, exposure_time=None):
        """Build a batman model, averaged over the exposure when one is given.

        The exposure is split into sub-exposures of at most SUBEXPOSURE_DAYS and averaged at their
        midpoints, so 2-minute data stay instantaneous and 30-minute data use 15 sub-exposures
        (Kipping 2010, MNRAS 408, 1758).

        Parameters
        ----------
        transit_params : batman.TransitParams
            Transit parameters.
        time : array_like
            Mid-exposure times in days.
        exposure_time : float or array_like, optional
            Exposure time in days. None or a value up to SUBEXPOSURE_DAYS gives an instantaneous model.
            An array gives each point its own exposure (see `PerPointExposureModel`).

        Returns
        -------
        batman.TransitModel, ExposureAveragedModel or PerPointExposureModel
            An object with a light_curve(params) method.
        """
        if exposure_time is not None and np.ndim(exposure_time) > 0:
            exposures = np.asarray(exposure_time, dtype=float)
            if exposures.shape != np.shape(time):
                raise ValueError(f"{exposures.shape} exposures for {np.shape(time)} times")
            values = np.unique(exposures)
            if len(values) > 1:
                return PerPointExposureModel(transit_params, time, exposures)
            exposure_time = float(values[0]) if len(values) and values[0] > 0 else None
        if exposure_time is None or exposure_time <= cls.SUBEXPOSURE_DAYS:
            return batman.TransitModel(transit_params, time)
        sub_exposures = int(np.ceil(exposure_time / cls.SUBEXPOSURE_DAYS))
        return ExposureAveragedModel(transit_params, time, exposure_time, sub_exposures)

    @classmethod
    def pandora_light_curve(cls, model, time, exposure_time=None):
        """Evaluate a Pandora moon model, averaged over the exposure like `transit_model`.

        Pandora's own supersampling expects supersampled input times, so the exposure is
        averaged here at the midpoints of sub-exposures of at most SUBEXPOSURE_DAYS.

        Parameters
        ----------
        model : pandoramoon.moon_model
            Pandora model.
        time : array_like
            Mid-exposure times in days.
        exposure_time : float or array_like, optional
            Exposure time in days. None or a value up to SUBEXPOSURE_DAYS gives instantaneous fluxes.
            An array gives each point its own exposure.

        Returns
        -------
        tuple of numpy.ndarray
            (total, planet, moon) fluxes at the given times.
        """
        time = np.asarray(time, dtype=float)
        if exposure_time is not None and np.ndim(exposure_time) > 0:
            exposures = np.asarray(exposure_time, dtype=float)
            values = np.unique(exposures)
            if len(values) > 1:
                fluxes = None
                for value in values:
                    index = np.flatnonzero(exposures == value)
                    part = cls.pandora_light_curve(model, time[index], value if value > 0 else None)
                    if fluxes is None:
                        fluxes = tuple(np.empty(len(time)) for _ in part)
                    for whole, piece in zip(fluxes, part):
                        whole[index] = piece
                return fluxes
            exposure_time = float(values[0]) if len(values) and values[0] > 0 else None
        if exposure_time is None or exposure_time <= cls.SUBEXPOSURE_DAYS:
            return tuple(np.asarray(flux, dtype=float) for flux in model.light_curve(time))
        sub_exposures = int(np.ceil(exposure_time / cls.SUBEXPOSURE_DAYS))
        offsets = ((np.arange(sub_exposures) + 0.5) / sub_exposures - 0.5) * exposure_time
        fine = (time[:, None] + offsets[None, :]).ravel()
        order = np.argsort(fine, kind='stable')
        fluxes = model.light_curve(fine[order])
        averaged = []
        for flux in fluxes:
            unsorted = np.empty(len(fine))
            unsorted[order] = np.asarray(flux, dtype=float)
            averaged.append(unsorted.reshape(len(time), sub_exposures).mean(axis=1))
        return tuple(averaged)
