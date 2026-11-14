"""The joint fit of each moon hypothesis: the moon window with the planet depth, mid-time
and shape corrections, on the stacked epochs.
"""
from numba import njit
import numpy as np


@njit(cache=True)
def _point_terms(point, epoch, inside, depth_column, t0_column, shapes, shape_columns, depth_lookup, depth_shared,
                 t0_lookup, columns, values):
    """Fill the non-zero (column, value) pairs of one point of the `planet_fit` design; return how many."""
    columns[0], values[0] = 0, 1.0
    columns[1], values[1] = 1, 1.0 if inside[point] else 0.0
    count = 2
    index = epoch[point]
    column = depth_shared if depth_shared >= 0 else (depth_lookup[index] if index >= 0 else -1)
    if column >= 0:
        columns[count], values[count] = column, depth_column[point]
        count += 1
    for term in range(shape_columns.shape[0]):
        if shape_columns[term] >= 0:
            columns[count], values[count] = shape_columns[term], shapes[point, term]
            count += 1
    column = t0_lookup[index] if index >= 0 else -1
    if column >= 0:
        columns[count], values[count] = column, t0_column[point]
        count += 1
    return count


@njit(cache=True)
def _planet_normal_equations(epoch, inside, depth_column, t0_column, shapes, shape_columns, depth_lookup,
                             depth_shared, t0_lookup, weights, flux, n_columns):
    """Return (D^T W D, D^T W f) of the `planet_fit` design, summed over each point's non-zero columns."""
    normal = np.zeros((n_columns, n_columns))
    right = np.zeros(n_columns)
    columns = np.empty(4 + shape_columns.shape[0], dtype=np.int64)
    values = np.empty(4 + shape_columns.shape[0])
    for point in range(epoch.shape[0]):
        count = _point_terms(point, epoch, inside, depth_column, t0_column, shapes, shape_columns, depth_lookup,
                             depth_shared, t0_lookup, columns, values)
        weight = weights[point]
        for first in range(count):
            weighted = values[first] * weight
            right[columns[first]] += weighted * flux[point]
            for second in range(count):
                normal[columns[first], columns[second]] += weighted * values[second]
    return normal, right


@njit(cache=True)
def _window_cross_terms(selection, epoch, inside, depth_column, t0_column, shapes, shape_columns, depth_lookup,
                        depth_shared, t0_lookup, weights, flux, n_columns):
    """Return the moon-window row and column (index 1) of D^T W D and D^T W f for the selected points.

    Added to the normal equations of the same points without the window, they give the
    equations with these points inside it.
    """
    normal = np.zeros((n_columns, n_columns))
    right = np.zeros(n_columns)
    columns = np.empty(4 + shape_columns.shape[0], dtype=np.int64)
    values = np.empty(4 + shape_columns.shape[0])
    for position in range(selection.shape[0]):
        point = selection[position]
        count = _point_terms(point, epoch, inside, depth_column, t0_column, shapes, shape_columns, depth_lookup,
                             depth_shared, t0_lookup, columns, values)
        weight = weights[point]
        normal[1, 1] += weight
        right[1] += weight * flux[point]
        for term in range(count):
            if columns[term] != 1:
                normal[1, columns[term]] += weight * values[term]
                normal[columns[term], 1] += weight * values[term]
    return normal, right


@njit(cache=True)
def _planet_model(epoch, inside, depth_column, t0_column, shapes, shape_columns, depth_lookup, depth_shared,
                  t0_lookup, coefficients):
    """Return the planet terms of the `planet_fit` design times their coefficients (columns 2 and up)."""
    model = np.zeros(epoch.shape[0])
    columns = np.empty(4 + shape_columns.shape[0], dtype=np.int64)
    values = np.empty(4 + shape_columns.shape[0])
    for point in range(epoch.shape[0]):
        count = _point_terms(point, epoch, inside, depth_column, t0_column, shapes, shape_columns, depth_lookup,
                             depth_shared, t0_lookup, columns, values)
        for term in range(2, count):
            model[point] += values[term] * coefficients[columns[term]]
    return model


class JointPlanetFit:
    """The joint fit of each moon hypothesis: the moon window with the planet depth, mid-time
    and shape corrections, on the stacked epochs.
    """

    # Planet shape terms fitted with each moon, one coefficient shared by all epochs (see planet_fit).
    PLANET_SHAPE_TERMS = ("impact_squared", "u1", "u2")
    # Largest change of b^2 the first-order geometry column covers (a change of b of 0.1 at b = 0.5).
    MAX_LINEAR_IMPACT_SQUARED_CORRECTION = 0.1

    def search_planet_depth_column(self):
        """Return the planet columns that `split_scenarios` stacks with the flux, or None.

        It is one array per point: the depth derivative, the mid-time derivative and the epoch
        index (the last two zero and -1 without the mid-time fit), when they match the light curve
        being searched.
        """
        column = self.planet_depth_column
        if column is None or self.time is None or len(column) != len(self.time):
            return None
        t0_column, epoch_index = self.planet_t0_column, self.planet_epoch_index
        if t0_column is None or len(t0_column) != len(self.time):
            t0_column, epoch_index = np.zeros(len(column)), np.full(len(column), -1)
        elif not self.fit_planet_t0:
            # The epoch index still places the per-epoch depth corrections.
            t0_column = np.zeros(len(column))
        shape = self.planet_shape_columns
        if not self.fit_planet_shape or shape is None or len(shape) != len(self.time):
            shape = np.zeros((len(column), len(self.PLANET_SHAPE_TERMS)))
        # Every hypothesis searches the same curve, so the stacked columns are built once per set
        # of inputs (the arrays are replaced, never changed in place, when the curve or the
        # planet model changes).
        key = (id(self.time), len(self.time), id(column), id(self.planet_t0_column), id(self.planet_epoch_index),
               id(self.planet_shape_columns), self.fit_planet_t0, self.fit_planet_shape)
        cached = getattr(self, "_search_columns_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        columns = np.column_stack((column, t0_column, epoch_index, shape))
        self._search_columns_cache = (key, columns)
        return columns

    @classmethod
    def planet_fit(cls, flux, inside, planet_columns, epoch_depth=False, flux_err=None):
        """Fit the moon window together with the planet corrections, on the stacked epochs.

        The stacked points are fitted with a constant, the moon window (1 inside, 0 outside), one
        planet depth correction shared by all epochs and one mid-time shift per epoch. The moon
        window is one template in the frame of the moon, the stacking of the search. The planet
        terms are the derivatives of the subtracted planet model, in the frame of the planet: the
        depth with respect to all epochs at once, the mid-time with respect to the points of its
        own epoch only. They separate from the moon because it sits at a different place of the
        planet transit in each epoch. A moon inside the transit deepens the fold fit by its mean
        depth and pulls each epoch's mid-time fit towards it; the subtraction then leaves both
        errors behind, and these terms return them. With epoch_depth the depth correction is one
        per epoch instead of one shared by all, for transits whose depth changes between epochs;
        a moon still separates from it because it covers a different part of each transit.

        Parameters
        ----------
        flux : numpy.ndarray
            Stacked flux.
        inside : numpy.ndarray of bool
            Points inside the moon window.
        planet_columns : numpy.ndarray, shape (n, 3)
            Depth derivative, mid-time derivative and epoch index of each point (see
            `search_planet_depth_column`).
        epoch_depth : bool, optional
            Fit one depth correction per epoch instead of one shared by all, default False.
        flux_err : numpy.ndarray, optional
            Flux errors. The fit is then weighted by their inverse variance, and the inflation
            compares with the weighted difference of means; equal errors give the unweighted fit.

        Returns
        -------
        dict
            "model": the planet correction at each point, to subtract from the flux; "depth": the
            depth correction in flux units (negative when the subtracted planet was too deep);
            "t0_shifts": {epoch index: mid-time correction in days}; "epoch_depths": {epoch index:
            depth correction}, empty without epoch_depth, where "depth" is their mean; "inflation":
            the factor by which the planet terms widen the moon-depth uncertainty against the plain
            difference of means.
        """
        flux = np.asarray(flux, dtype=float)
        n_points, n_inside = len(flux), int(np.sum(inside))
        unchanged = {"model": np.zeros(n_points), "depth": 0.0, "t0_shifts": {}, "epoch_depths": {}, "shape": {},
                     "inflation": 1.0}
        if n_inside == 0 or n_points - n_inside < 2:
            return unchanged
        names, n_columns, terms = cls.planet_design(planet_columns, epoch_depth)
        if n_columns == 2:
            return unchanged
        weights = cls.point_weights(flux_err, n_points)
        inside = np.ascontiguousarray(inside, dtype=np.bool_)
        normal, right = _planet_normal_equations(terms[0], inside, *terms[1:], np.ascontiguousarray(weights),
                                                 flux, n_columns)
        solved = cls.solve_planet_normal(normal, right, names, n_columns,
                                                         float(np.sum(weights[inside])),
                                                         float(np.sum(weights[~inside])))
        if solved is None:
            return unchanged
        fit, planet_coefficients = solved
        fit["model"] = _planet_model(terms[0], inside, *terms[1:], planet_coefficients)
        return fit

    @classmethod
    def planet_design(cls, planet_columns, epoch_depth=False):
        """Return (names, n_columns, terms) of the `planet_fit` design for these planet columns.

        The design has a constant, the moon window and the planet terms. Each per-epoch term is
        zero outside its epoch, so each point has at most 4 + len(PLANET_SHAPE_TERMS) non-zero
        columns, and the normal equations are summed over those (`_planet_normal_equations`):
        the same sums as the dense product at a fraction of its cost. terms holds, in the order
        the numba kernels take them around the moon window: the epoch index, then (depth column,
        t0 column, shape columns, shape column indices, per-epoch depth column indices, shared
        depth column index, per-epoch t0 column indices).
        """
        n_points = len(planet_columns)
        depth_column, t0_column, epoch = planet_columns[:, 0], planet_columns[:, 1], planet_columns[:, 2].astype(int)
        names, n_columns = [], 2
        lookup_size = int(max(np.max(epoch), 0)) + 1 if n_points else 1
        depth_lookup = np.full(lookup_size, -1, dtype=np.int64)
        depth_shared = -1
        depth_epochs = np.unique(epoch[(epoch >= 0) & (depth_column != 0)]) if epoch_depth else []
        if len(depth_epochs) > 0:
            for index in depth_epochs:
                depth_lookup[index] = n_columns
                n_columns += 1
                names.append(("depth", int(index)))
        elif np.any(depth_column != 0):
            depth_shared = n_columns
            n_columns += 1
            names.append("depth")
        n_shapes = len(cls.PLANET_SHAPE_TERMS)
        shapes = np.zeros((n_points, n_shapes))
        shape_columns = np.full(n_shapes, -1, dtype=np.int64)
        for offset, shape_name in enumerate(cls.PLANET_SHAPE_TERMS):
            if planet_columns.shape[1] > 3 + offset and np.any(planet_columns[:, 3 + offset] != 0):
                shapes[:, offset] = planet_columns[:, 3 + offset]
                shape_columns[offset] = n_columns
                n_columns += 1
                names.append(shape_name)
        t0_lookup = np.full(lookup_size, -1, dtype=np.int64)
        for index in np.unique(epoch[(epoch >= 0) & (t0_column != 0)]):
            t0_lookup[index] = n_columns
            n_columns += 1
            names.append(int(index))
        terms = (np.ascontiguousarray(epoch, dtype=np.int64), np.ascontiguousarray(depth_column, dtype=float),
                 np.ascontiguousarray(t0_column, dtype=float), shapes, shape_columns, depth_lookup, depth_shared,
                 t0_lookup)
        return names, n_columns, terms

    @classmethod
    def solve_planet_normal(cls, normal, right, names, n_columns, inside_weight, outside_weight):
        """Solve the `planet_fit` normal equations; return (fit without "model", coefficients) or None.

        A term the stacked points cannot tell from the others (an epoch with only a few points,
        for example) is dropped rather than fitted; None when only the constant and the moon
        window are left. coefficients holds the planet term of every design column (zero for
        the constant, the window and the dropped terms).
        """
        keep = list(range(n_columns))
        if np.linalg.matrix_rank(normal) < n_columns:
            keep = [0, 1]
            for column_index in range(2, n_columns):
                trial = keep + [column_index]
                if np.linalg.matrix_rank(normal[np.ix_(trial, trial)]) == len(keep) + 1:
                    keep.append(column_index)
            names = [names[i - 2] for i in keep[2:]]
            if len(keep) == 2:
                return None
            normal, right = normal[np.ix_(keep, keep)], right[keep]
        normal_inverse = np.linalg.inv(normal)
        coefficients = normal_inverse @ right
        planet_coefficients = np.zeros(n_columns)
        planet_coefficients[np.asarray(keep[2:], dtype=np.int64)] = coefficients[2:]
        fitted = dict(zip(names, coefficients[2:]))
        inflation = float(np.sqrt(normal_inverse[1, 1] / (1 / inside_weight + 1 / outside_weight)))
        shape = {name: float(fitted.pop(name)) for name in cls.PLANET_SHAPE_TERMS if name in fitted}
        epoch_depths = {name[1]: float(fitted.pop(name)) for name in list(fitted) if isinstance(name, tuple)}
        depth = float(np.mean(list(epoch_depths.values()))) if epoch_depths else float(fitted.pop("depth", 0.0))
        return ({"depth": depth, "t0_shifts": {name: float(value) for name, value in fitted.items()},
                 "epoch_depths": epoch_depths, "shape": shape, "inflation": inflation}, planet_coefficients)

    @classmethod
    def planet_correction(cls, fit, columns):
        """Return the planet correction of a `planet_fit` result on the columns of `search_planet_depth_column`."""
        if fit["epoch_depths"]:
            correction = np.zeros(len(columns))
            for index, value in fit["epoch_depths"].items():
                correction = correction + np.where(columns[:, 2] == index, value * columns[:, 0], 0.0)
        else:
            correction = fit["depth"] * columns[:, 0]
        for index, shift in fit["t0_shifts"].items():
            correction = correction + np.where(columns[:, 2] == index, shift * columns[:, 1], 0.0)
        for offset, name in enumerate(cls.PLANET_SHAPE_TERMS):
            if columns.shape[1] > 3 + offset:
                correction = correction + fit["shape"].get(name, 0.0) * columns[:, 3 + offset]
        return correction

    @classmethod
    def planet_depth_fit(cls, it_flux, oot_flux, it_column, oot_column, it_err=None, oot_err=None):
        """Fit the moon-window depth together with one planet depth correction, on the stacked epochs.

        The stacked points are fitted with a constant, the moon window (1 inside, 0 outside) and
        the planet depth column, the derivative of the subtracted planet model with respect to its
        depth placed at each epoch's own mid-time. The moon window is one template in the frame of
        the moon, the stacking of the search; the planet column is one depth change shared by all
        epochs, in the frame of the planet. They separate because the moon sits at a different
        place of the planet transit in each epoch. A moon inside the transit in most epochs deepens
        the fold fit by its mean depth, and the subtraction then leaves that depth back where the
        moon is absent; the shared correction returns it, so the moon is measured whole.

        Returns
        -------
        tuple
            (it_flux, oot_flux, correction, inflation): the fluxes with the planet correction
            removed, so their window depth is the fitted moon depth; the depth correction in flux
            units (negative when the subtracted planet was too deep); and the factor by which the
            planet column widens the moon-depth uncertainty against the plain difference of means.
            Without a usable column the fluxes come back unchanged, with correction 0 and
            inflation 1.
        """
        n_it = len(it_flux)
        column = np.concatenate((it_column, oot_column))
        columns = np.column_stack((column, np.zeros(len(column)), np.full(len(column), -1)))
        inside = np.concatenate((np.ones(n_it, dtype=bool), np.zeros(len(oot_flux), dtype=bool)))
        errors = None if it_err is None or oot_err is None else np.concatenate((it_err, oot_err))
        fit = cls.planet_fit(np.concatenate((it_flux, oot_flux)), inside, columns, flux_err=errors)
        return (it_flux - fit["model"][:n_it], oot_flux - fit["model"][n_it:], fit["depth"], fit["inflation"])
