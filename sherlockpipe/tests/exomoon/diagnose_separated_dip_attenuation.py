"""Ablate local timing and baseline effects on paired artificial dips."""

import copy
import json

import batman
import numpy as np

from sherlockpipe.exomoon import ExoMoonLeastSquares
from compare_original_vs_worktree import _case, measure


def _no_baseline(time, flux, flux_err, center, duration):
    return np.ones_like(flux), {"status": "ablation_disabled"}


def _fixed_timing(time, flux, flux_err, t0_guess, duration_window, transit_params,
                  exposure_times_seconds=None):
    params = copy.deepcopy(transit_params)
    params.t0 = t0_guess
    model = ExoMoonLeastSquares._transit_model(params, time, exposure_times_seconds)
    return np.asarray([t0_guess]), model


def _measure(case, baseline=True, timing=True):
    original_baseline = ExoMoonLeastSquares._profile_local_baseline
    original_timing = ExoMoonLeastSquares.fit_single_t0
    try:
        if not baseline:
            ExoMoonLeastSquares._profile_local_baseline = staticmethod(_no_baseline)
        if not timing:
            ExoMoonLeastSquares.fit_single_t0 = staticmethod(_fixed_timing)
        return measure(case)
    finally:
        ExoMoonLeastSquares._profile_local_baseline = staticmethod(original_baseline)
        ExoMoonLeastSquares.fit_single_t0 = staticmethod(original_timing)


if __name__ == "__main__":
    results = {}
    for case_name, offset in (("separated", 0.42), ("overlapping", 0.03)):
        case = _case(case_name, offset, False)
        results[case_name] = {
            "normal": _measure(case),
            "baseline_disabled": _measure(case, baseline=False),
            "timing_fixed": _measure(case, timing=False),
            "baseline_and_timing_disabled": _measure(case, baseline=False, timing=False),
        }
    print(json.dumps(results, indent=2))
