"""Paired synthetic EMLS comparison: invoke once per source tree.

Run the same script in separate processes with PYTHONPATH pointed at either
the 1.6.0 baseline worktree or the modified worktree. Outputs JSON for pairing.
"""

import json
import os
import time

import batman
import numpy as np

from sherlockpipe.exomoon import ExoMoonLeastSquares


EPOCHS = np.array([0.0, 5.0, 10.0])
R_EARTH_PER_R_SUN = 109.076


def _params(t0):
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = t0, 5.0, 0.08, 12.0, 89.0
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.2, 0.1], "quadratic"
    return p


def _case(name, moon_offset=None, gapped=False, ttv=None):
    local = np.arange(-0.65, 0.650001, 0.005)
    time = np.concatenate([epoch + local for epoch in EPOCHS])
    if gapped:
        time = time[~(np.abs(time - 5.0 + 0.10) < 0.035)]
    error = np.where(np.arange(len(time)) % 3 == 0, 8e-4, 2e-4) if gapped else np.full(len(time), 2e-4)
    centers = EPOCHS + (np.zeros(3) if ttv is None else np.asarray(ttv))
    planet = np.ones_like(time)
    for center, reference in zip(centers, EPOCHS):
        mask = np.abs(time - reference) < 0.8
        planet[mask] = batman.TransitModel(_params(center), time[mask]).light_curve(_params(center))
    moon = np.zeros_like(time)
    if moon_offset is not None:
        for center in centers:
            moon += 8e-4 * np.exp(-0.5 * ((time - center - moon_offset) / 0.025) ** 2)
    noise = np.random.default_rng(117).normal(0.0, error)
    return {"name": name, "time": time, "error": error,
            "planet": planet + noise, "injected": planet - moon + noise,
            "signal": -moon, "centers": centers}


def _emls(case, flux):
    return ExoMoonLeastSquares(
        "", 1, 1.0, 1.0, [0.2, 0.1], 0.08 * R_EARTH_PER_R_SUN, 0.8,
        100.0, 5.0, 0.0, 0.30, 12.0 / 215.032, 89.0, 0.3, 0.0, 0.1,
        90.0, 0.0, 0.3, 0.6, EPOCHS, case["time"], flux, case["error"],
        5.5, min_cadences_per_t0=8, fit_rp_per_transit=True,
        min_transit_completeness=0.1, max_subtraction_rms_ratio=100.0)


def _fit(case, flux):
    emls = _emls(case, flux)
    started = time.perf_counter()
    out_t, residual, out_err = emls.subtract_planet_transit(
        case["time"], flux, case["error"], plot=False)
    elapsed = time.perf_counter() - started
    # Pair by exact observation time; both algorithms must preserve the values
    # and error bars for each retained cadence.
    return emls, {float(t): (float(r), float(e)) for t, r, e in zip(out_t, residual, out_err)}, elapsed


def measure(case):
    emls, planet_rows, elapsed = _fit(case, case["planet"])
    _, moon_rows, _ = _fit(case, case["injected"])
    common = np.array(sorted(set(planet_rows) & set(moon_rows)))
    idx = np.searchsorted(case["time"], common)
    errors = np.array([planet_rows[t][1] for t in common])
    p_resid = np.array([planet_rows[t][0] for t in common])
    m_resid = np.array([moon_rows[t][0] for t in common])
    flux = case["planet"][idx]
    signal = case["signal"][idx]
    inside = np.abs(common - np.round(common / 5.0) * 5.0) < 0.35
    weight = 1 / errors**2
    transfer = np.nan
    distortion = np.nan
    if np.any(signal):
        difference = m_resid - p_resid
        denom = np.sum(weight * signal**2)
        transfer = np.sum(weight * difference * signal) / denom
        distortion = np.sqrt(np.sum(weight * (difference - transfer * signal)**2) / denom)
    local_t0 = getattr(emls, "local_t0s", None)
    if local_t0 is None:
        local_t0 = case["centers"] - np.asarray(getattr(emls, "ttvs", []))
    local_t0 = np.asarray(local_t0)
    timing_error = (float(np.max(np.abs(local_t0 - case["centers"])))
                    if len(local_t0) == len(case["centers"]) else np.nan)
    return {
        "case": case["name"],
        "status": getattr(getattr(emls, "subtraction_status", {}), "get", lambda *_: None)("status") or "legacy_returned",
        "seconds": elapsed,
        "retained_cadences": len(common),
        "retained_epochs": len(getattr(emls, "bary_t0s", [])),
        "timing_max_abs_error_days": timing_error,
        "planet_only_intransit_rms": float(np.sqrt(np.mean((p_resid[inside] - 1.0)**2))),
        "planet_only_weighted_rms": float(np.sqrt(np.average((p_resid - 1.0)**2, weights=weight))),
        "moon_transfer": float(transfer),
        "moon_orthogonal_distortion": float(distortion),
    }


if __name__ == "__main__":
    output = {
        "source": os.path.realpath(__import__("sherlockpipe.exomoon", fromlist=["x"]).__file__),
        "cases": [
            measure(_case("planet_regular_no_ttv")),
            *[measure(_case(f"moon_offset_{offset:.3f}d_regular_no_ttv", offset))
              for offset in (0.00, 0.03, 0.06, 0.09, 0.12, 0.20, 0.42)],
            measure(_case("overlap_gapped_known_ttv", 0.03, True, [-0.035, 0.0, 0.028])),
            measure(_case("separated_gapped_known_ttv", 0.42, True, [-0.035, 0.0, 0.028])),
        ],
    }
    print(json.dumps(output, indent=2))
