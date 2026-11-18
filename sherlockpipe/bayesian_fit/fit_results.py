"""The results of a Bayesian fit, read from the files of allesfitter into plain arrays and tables.

The fit report is drawn from a :class:`FitResults`, so it does not depend on how allesfitter stores its output, and
it can be built (and tested) without running a fit. Everything that needs allesfitter itself (the model, the Gaussian
process baseline and the residuals of the posterior median) goes through one object that can be replaced.
"""
import gzip
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SOLAR_DENSITY_CGS = 1.41
# the same star, twice the period (odd and even fits): a/R*, 1/rsuma and 1/cos i grow by 2^(2/3)
ODD_EVEN_ORBIT_SCALE = 0.5 ** (2 / 3)
RESAMPLED_SAMPLES = 20000


@dataclass
class Prior:
    """A prior of allesfitter: ``uniform low high``, ``normal mean sigma`` or ``trunc_normal low high mean sigma``."""
    kind: str
    args: tuple

    @property
    def bounds(self):
        """The hard limits of the prior, or None when it has none (a plain normal)."""
        if self.kind == "uniform":
            return self.args[0], self.args[1]
        if self.kind == "trunc_normal":
            return self.args[0], self.args[1]
        return None


def parse_priors(params_file):
    """The prior of every fitted parameter of an allesfitter ``params.csv``."""
    priors = {}
    if not params_file or not os.path.exists(params_file):
        return priors
    with open(params_file) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 4 or parts[0].startswith("#") or not parts[3].strip():
                continue
            words = parts[3].split()
            try:
                priors[parts[0]] = Prior(words[0], tuple(float(w) for w in words[1:]))
            except ValueError:
                continue
    return priors


@dataclass
class VariantFit:
    """The fit of the odd or the even transits."""
    kind: str
    table: pd.DataFrame
    priors: Dict[str, Prior]
    derived: Optional[pd.DataFrame] = None


@dataclass
class FitResults:
    """Everything the fit report draws."""
    object_id: str
    ra: Optional[float]
    dec: Optional[float]
    magnitudes: Dict[str, Optional[float]]
    planets: List[str]
    masked: List[str]
    samples: pd.DataFrame
    derived: Dict[str, np.ndarray]
    table: pd.DataFrame
    derived_table: pd.DataFrame
    priors: Dict[str, Prior]
    time: np.ndarray
    flux: np.ndarray
    baseline: np.ndarray
    model: np.ndarray
    residuals: np.ndarray
    logz: float
    logzerr: float
    niter: int
    ncall: int
    efficiency: float
    information: float
    ess: float
    logz_curve: np.ndarray
    logl: np.ndarray
    logvol: np.ndarray
    weights: np.ndarray
    star_density: Optional[tuple] = None
    noise: Optional[pd.Series] = None
    settings: dict = field(default_factory=dict)
    variants: Dict[str, VariantFit] = field(default_factory=dict)

    @property
    def detrended(self):
        """The flux without the baseline (the Gaussian process or the spline)."""
        return self.flux - self.baseline

    def median(self, name):
        return float(np.median(self.samples[name]))

    def derived_of(self, planet, key):
        """The posterior samples of a derived quantity of a planet (``b_tra``, ``T_tra_tot``...), or None."""
        samples = self.derived.get(f"{planet}_{key}")
        return None if samples is None or np.ndim(samples) != 1 else np.asarray(samples)

    @property
    def combined_density(self):
        samples = self.derived.get("combined_host_density")
        return None if samples is None or np.ndim(samples) != 1 else np.asarray(samples)


def star_density_prior(star_row):
    """The stellar density (cgs) and its error from the mass and the radius of the star, or None."""
    try:
        mass, radius = float(star_row["M_star"]), float(star_row["R_star"])
        mass_err = float(np.nanmax([star_row.get("M_star_lerr", np.nan), star_row.get("M_star_uerr", np.nan)]))
        radius_err = float(np.nanmax([star_row.get("R_star_lerr", np.nan), star_row.get("R_star_uerr", np.nan)]))
    except (TypeError, ValueError, KeyError):
        return None
    if not (np.isfinite(mass) and np.isfinite(radius) and radius > 0):
        return None
    density = mass / radius ** 3 * SOLAR_DENSITY_CGS
    error = density * np.sqrt((mass_err / mass) ** 2 + (3 * radius_err / radius) ** 2) if np.isfinite(mass_err + radius_err) else np.nan
    return density, error


def _get(result, key):
    """Attribute or item of a dynesty results object (a dict with attribute access) or of a plain namespace."""
    try:
        return result[key]
    except (TypeError, KeyError, IndexError):
        return getattr(result, key)


def _fitted_rows(table):
    names = table["#name"].astype(str)
    fitted = table[~names.str.startswith("#") & (table["lower_error"].astype(str) != "(fixed)")]
    return fitted


def read_variants(fit_dir):
    """The odd and the even fits that live next to the main one (``fit_[n]/odd`` and ``fit_[n]/even``)."""
    variants = {}
    for kind in ("odd", "even"):
        folder = Path(fit_dir).parent / kind
        table_file = folder / "results" / "ns_table.csv"
        if table_file.exists():
            derived_file = folder / "results" / "ns_derived_table.csv"
            derived = pd.read_csv(derived_file) if derived_file.exists() else None
            variants[kind] = VariantFit(kind, pd.read_csv(table_file), parse_priors(str(folder / "params.csv")), derived)
    return variants


def load_fit_results(fit_dir, object_id=None, star_row=None, candidates=None, alles=None, seed=1):
    """
    Reads the output of a fit.

    :param fit_dir: the directory of the main fit (the one with ``results/`` and ``params.csv``)
    :param object_id: the id of the target
    :param star_row: the row of ``params_star.csv`` (a dict-like), for the coordinates, magnitudes and density prior
    :param candidates: the candidates dataframe of the fit (with ``name`` and ``number``), to name the masked ones
    :param alles: an allesfitter ``allesclass`` of the fit; it is created from the directory when it is None
    :param seed: the seed of the resampling of the posterior
    :return: the :class:`FitResults`
    """
    results_dir = os.path.join(fit_dir, "results")
    with gzip.open(os.path.join(results_dir, "save_ns.pickle.gz"), "rb") as handle:
        sampling = pickle.load(handle)
    table = pd.read_csv(os.path.join(results_dir, "ns_table.csv"))
    derived_table = pd.read_csv(os.path.join(results_dir, "ns_derived_table.csv"))
    with open(os.path.join(results_dir, "ns_derived_samples.pickle"), "rb") as handle:
        derived_all = pickle.load(handle)
    names = _fitted_rows(table)["#name"].tolist()
    raw = np.asarray(_get(sampling, "samples"))
    if raw.shape[1] != len(names):
        raise ValueError(f"The nested sampling has {raw.shape[1]} parameters but ns_table.csv lists {len(names)} fitted ones")
    logwt, logz_all = np.asarray(_get(sampling, "logwt")), np.asarray(_get(sampling, "logz"))
    weights = np.exp(logwt - logz_all[-1])
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    index = rng.choice(len(weights), min(RESAMPLED_SAMPLES, max(len(weights), 1)), p=weights)
    samples = pd.DataFrame(raw[index], columns=names)
    derived = {key: np.asarray(value)[index] if np.ndim(value) == 1 and len(value) == len(weights) else np.asarray(value)
               for key, value in derived_all.items()}
    if alles is None:
        import contextlib
        import io
        import alexfitter
        with contextlib.redirect_stdout(io.StringIO()):
            alles = alexfitter.allesclass(fit_dir)
    time = np.asarray(alles.data["lc"]["time"], dtype=float)
    flux = np.asarray(alles.data["lc"]["flux"], dtype=float)
    try:
        baseline = np.asarray(alles.get_posterior_median_baseline("lc", "flux", xx=time), dtype=float)
    except Exception:
        logging.exception("The baseline of the fit could not be computed: it is taken as zero")
        baseline = np.zeros_like(flux)
    model = np.asarray(alles.get_posterior_median_model("lc", "flux", xx=time), dtype=float)
    residuals = flux - model - baseline
    fitted = [str(c) for c in alles.settings.get("companions_phot", [])]
    masked = []
    if candidates is not None and "name" in candidates:
        masked = [str(n) for n in candidates["name"] if str(n) and str(n) not in fitted]
    noise = None
    noise_file = os.path.join(fit_dir, "priors", "summary_phot.csv")
    if os.path.exists(noise_file):
        noise = pd.read_csv(noise_file).iloc[0]
    star = {} if star_row is None else star_row
    settings = {key: alles.settings.get(key) for key in ("baseline_flux_lc", "ns_tol", "ns_nlive", "ns_modus", "fast_fit_width",
                                                         "use_stellar_density_prior", "secondary_eclipse", "host_ld_law_lc")}
    settings["circular"] = bool(table[table["#name"].astype(str).str.endswith("_f_c")]["lower_error"].astype(str).eq("(fixed)").all())
    return FitResults(
        object_id=object_id or "", ra=_number(star, "ra"), dec=_number(star, "dec"),
        magnitudes={k: _number(star, k) for k in ("v", "j", "h", "k")},
        planets=fitted, masked=masked, samples=samples, derived=derived, table=table, derived_table=derived_table,
        priors=parse_priors(os.path.join(fit_dir, "params.csv")), time=time, flux=flux, baseline=baseline, model=model,
        residuals=residuals, logz=float(logz_all[-1]), logzerr=float(np.asarray(_get(sampling, "logzerr"))[-1]),
        niter=int(_get(sampling, "niter")), ncall=int(np.sum(_get(sampling, "ncall"))), efficiency=float(_get(sampling, "eff")),
        information=float(np.asarray(_get(sampling, "information"))[-1]), ess=float(1 / np.sum(weights ** 2)),
        logz_curve=logz_all, logl=np.asarray(_get(sampling, "logl")), logvol=np.asarray(_get(sampling, "logvol")),
        weights=weights, star_density=star_density_prior(star) if star is not None and len(star) else None, noise=noise,
        settings=settings, variants=read_variants(fit_dir))


def _number(row, key):
    try:
        value = float(row[key])
    except (TypeError, ValueError, KeyError):
        return None
    return value if np.isfinite(value) else None


def model_curve(time, rr, rsuma, cosi, epoch, period, q1, q2):
    """A quadratic limb darkened transit (batman) from the parameters of allesfitter (circular orbit)."""
    import batman
    params = batman.TransitParams()
    params.t0, params.per, params.rp = epoch, period, rr
    params.a = (1 + rr) / rsuma
    params.inc = float(np.degrees(np.arccos(np.clip(cosi, 0, 1))))
    params.ecc, params.w = 0.0, 90.0
    params.limb_dark = "quadratic"
    params.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
    return batman.TransitModel(params, time).light_curve(params)
