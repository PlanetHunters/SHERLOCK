"""Shared helpers of the Bayesian fit tests."""
import pandas as pd

from sherlockpipe.bayesian_fit.fitter import Fitter


def star_df():
    return pd.DataFrame([{"R_star": 0.48, "R_star_lerr": 0.014, "R_star_uerr": 0.014, "M_star": 0.48,
                          "M_star_lerr": 0.02, "M_star_uerr": 0.02, "ld_a": 0.17, "ld_b": 0.42}])


def candidates_df(*specs):
    rows = []
    for number, period, t0, depth in specs:
        rows.append({"name": f"SOI_{number}", "number": number, "period": period, "per_err": 4.8e-5, "t0": t0,
                     "duration": 70.0, "depth": depth, "rp_rs": "-"})
    return pd.DataFrame(rows)


def main_params_text(candidates):
    """The params.csv of a main fit, as the Fitter writes it (without the instrument block)."""
    return "".join(Fitter.fill_candidate_params(row, star_df(), False) for _, row in candidates.iterrows())


def prior(text, name):
    """(value, low, high) of the uniform prior of a parameter in a params.csv text."""
    for line in text.splitlines():
        parts = line.split(",")
        if parts[0] == name:
            low, high = parts[3].split()[1:3]
            return float(parts[1]), float(low), float(high)
    raise KeyError(name)


# ---------------------------------------------------------------------------------------------- fit results
import gzip
import os
import pickle
import types

import numpy as np

from sherlockpipe.bayesian_fit.fit_results import FitResults, Prior, VariantFit, model_curve


def make_fit_results(planets=("SOI_1",), variants=True, pegged=False, n_samples=3000, n_time=6000, seed=3, masked=(),
                     rr_prior=(0.015, 0.0245), ess_scale=1.0):
    """A synthetic FitResults with one transit per planet period: no allesfitter, no sampling."""
    rng = np.random.default_rng(seed)
    columns, derived, priors = {}, {}, {}
    for i, planet in enumerate(planets):
        period = 1.4923 * (1 + i * 0.9)
        columns[f"{planet}_rr"] = np.clip(rng.normal(0.023, 0.0007, n_samples), *rr_prior)
        columns[f"{planet}_rsuma"] = rng.normal(0.113, 0.003, n_samples)
        columns[f"{planet}_cosi"] = np.abs(rng.normal(0.04, 0.012, n_samples))
        columns[f"{planet}_epoch"] = rng.normal(2540.2 + i * 0.3, 7e-4, n_samples)
        columns[f"{planet}_period"] = rng.normal(period, 2e-6, n_samples)
        columns[f"{planet}_sbratio_lc"] = np.abs(rng.normal(0.0, 0.05, n_samples))
        derived[f"{planet}_b_tra"] = columns[f"{planet}_cosi"] / (columns[f"{planet}_rsuma"] / 1.0) / 8.8
        derived[f"{planet}_T_tra_tot"] = rng.normal(1.21, 0.04, n_samples)
        derived[f"{planet}_depth_tr_undil_lc"] = columns[f"{planet}_rr"] ** 2 * 1e3 * 1.2
        derived[f"{planet}_R_companion_(R_earth)"] = columns[f"{planet}_rr"] * 0.48 * 109.1
        derived[f"{planet}_a/R_star"] = (1 + columns[f"{planet}_rr"]) / columns[f"{planet}_rsuma"]
        derived[f"{planet}_host_density"] = rng.normal(6.2, 0.6, n_samples)
        priors[f"{planet}_rr"] = Prior("uniform", rr_prior)
        priors[f"{planet}_rsuma"] = Prior("uniform", (0.09, 0.138))
        priors[f"{planet}_cosi"] = Prior("uniform", (0.0, 0.06))
        priors[f"{planet}_epoch"] = Prior("uniform", (1713.43, 1713.47))
        priors[f"{planet}_period"] = Prior("uniform", (period - 5e-5, period + 5e-5))
        priors[f"{planet}_sbratio_lc"] = Prior("uniform", (0.0, 1.0))
    columns["host_ldc_q1_lc"] = rng.uniform(0.1, 0.9, n_samples)
    columns["host_ldc_q2_lc"] = rng.uniform(0.1, 0.9, n_samples)
    columns["ln_err_flux_lc"] = rng.normal(-6.36, 0.003, n_samples)
    samples = pd.DataFrame(columns)
    derived["combined_host_density"] = rng.normal(6.2, 0.6, n_samples)
    time = np.linspace(2539.0, 2543.0, n_time)
    own = np.ones_like(time)
    for planet in planets:
        median = {k: float(np.median(samples[f"{planet}_{k}"])) for k in ("rr", "rsuma", "cosi", "epoch", "period")}
        own = own * model_curve(time, median["rr"], median["rsuma"], median["cosi"], median["epoch"], median["period"], .5, .4)
    noise = rng.normal(0, 1.7e-3, n_time)
    baseline = 0.002 * np.sin(time * 2)
    flux = own + baseline + noise
    table_rows = [{"#name": c, "median": float(np.median(samples[c])), "lower_error": float(np.std(samples[c])),
                   "upper_error": float(np.std(samples[c])), "label": c, "unit": ""} for c in samples.columns]
    table_rows.append({"#name": "SOI_1_f_c", "median": 0.0, "lower_error": "(fixed)", "upper_error": "(fixed)", "label": "", "unit": ""})
    variant_fits = {}
    if variants:
        for kind, shift in (("odd", 0.0), ("even", 0.0003)):
            rows = []
            for planet in planets:
                rr = float(np.median(samples[f"{planet}_rr"])) + shift
                rsuma = 0.0903 if pegged else 0.0714
                cosi = 0.0593 if pegged else 0.0258
                for parameter, value, err in (("rr", rr, 0.0008), ("rsuma", rsuma, 0.0003), ("cosi", cosi, 0.0008)):
                    rows.append({"#name": f"{planet}_{parameter}", "median": value, "lower_error": err, "upper_error": err, "label": "", "unit": ""})
            vpriors = {f"{p}_rr": Prior("uniform", rr_prior) for p in planets}
            vpriors.update({f"{p}_rsuma": Prior("uniform", (0.09, 0.138) if pegged else (0.0567, 0.0875)) for p in planets})
            vpriors.update({f"{p}_cosi": Prior("uniform", (0.0, 0.06) if pegged else (0.0, 0.0378)) for p in planets})
            variant_fits[kind] = VariantFit(kind, pd.DataFrame(rows), vpriors)
    weights = np.full(n_samples, 1.0 / n_samples)
    return FitResults(
        object_id="TIC 1", ra=185.3, dec=57.7, magnitudes={"v": 12.8}, planets=list(planets), masked=list(masked),
        samples=samples, derived=derived, table=pd.DataFrame(table_rows),
        derived_table=pd.DataFrame({"#property": [], "value": []}), priors=priors, time=time, flux=flux, baseline=baseline,
        model=own, residuals=noise, logz=96896.45, logzerr=0.12, niter=21034, ncall=562256, efficiency=3.5, information=13.7,
        ess=n_samples * ess_scale, logz_curve=np.linspace(96830, 96896, 3000), logl=np.linspace(-80, 0, 3000),
        logvol=-np.linspace(0, 33, 3000), weights=weights, star_density=(6.08, 0.59), noise=None,
        settings={"baseline_flux_lc": "sample_GP_Matern32", "circular": True}, variants=variant_fits)


class FakeAlles:
    """The part of allesfitter's allesclass the loader reads."""

    def __init__(self, results: FitResults):
        self.data = {"lc": {"time": results.time, "flux": results.flux}}
        self.settings = {"companions_phot": results.planets, "baseline_flux_lc": "sample_GP_Matern32", "ns_tol": "0.05",
                         "ns_nlive": "500", "ns_modus": "dynamic", "use_stellar_density_prior": "True"}
        self._baseline, self._model = results.baseline, results.model

    def get_posterior_median_baseline(self, inst, key, xx=None):
        return self._baseline

    def get_posterior_median_model(self, inst, key, xx=None):
        return self._model


def write_fit_directory(directory, results: FitResults, variants=True):
    """Writes the files of allesfitter that the loader reads, from a synthetic FitResults."""
    os.makedirs(f"{directory}/results", exist_ok=True)
    os.makedirs(f"{directory}/priors", exist_ok=True)
    names = list(results.samples.columns)
    n = len(results.samples)
    sampling = types.SimpleNamespace(samples=results.samples.to_numpy(), logwt=np.log(results.weights) + results.logz,
                                     logz=np.full(n, results.logz), logzerr=np.full(n, results.logzerr), niter=results.niter,
                                     ncall=np.full(n, results.ncall // n), eff=results.efficiency, information=np.full(n, results.information),
                                     logl=np.linspace(-80, 0, n), logvol=-np.linspace(0, 33, n))
    with gzip.open(f"{directory}/results/save_ns.pickle.gz", "wb") as handle:
        pickle.dump(sampling, handle)
    results.table.to_csv(f"{directory}/results/ns_table.csv", index=False)
    pd.DataFrame({"#property": ["x"], "value": [1.0]}).to_csv(f"{directory}/results/ns_derived_table.csv", index=False)
    with open(f"{directory}/results/ns_derived_samples.pickle", "wb") as handle:
        pickle.dump(results.derived, handle)
    lines = ["#name,value,fit,bounds,label,unit"]
    for name, prior in results.priors.items():
        lines.append(f"{name},0,1,{prior.kind} {' '.join(str(a) for a in prior.args)},{name},")
    with open(f"{directory}/params.csv", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    pd.DataFrame([{"ln_yerr_median": -6.3}]).to_csv(f"{directory}/priors/summary_phot.csv", index=False)
    if variants:
        parent = os.path.dirname(directory)
        for kind, variant in results.variants.items():
            os.makedirs(f"{parent}/{kind}/results", exist_ok=True)
            variant.table.to_csv(f"{parent}/{kind}/results/ns_table.csv", index=False)
            with open(f"{parent}/{kind}/params.csv", "w") as handle:
                handle.write("#name,value,fit,bounds,label,unit\n" + "\n".join(
                    f"{n},0,1,{p.kind} {' '.join(str(a) for a in p.args)},{n}," for n, p in variant.priors.items()) + "\n")
