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
