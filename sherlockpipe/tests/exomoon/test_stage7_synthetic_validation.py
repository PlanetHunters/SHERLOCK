"""Compact, deterministic whole-subtraction validation for EMLS Stage 7."""

import csv
import sys
import time

IMPORT_START = time.perf_counter()

import batman
import numpy as np

from sherlockpipe.exomoon import ExoMoonLeastSquares

IMPORT_SECONDS = time.perf_counter() - IMPORT_START


EPOCHS = np.array([0.0, 5.0, 10.0])
R_EARTH_PER_R_SUN = 109.076


def _params(center):
    params = batman.TransitParams()
    params.t0 = center
    params.per = 5.0
    params.rp = 0.08
    params.a = 12.0
    params.inc = 89.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.2, 0.1]
    params.limb_dark = "quadratic"
    return params


def _planet_truth(time, centers, exposures):
    flux = np.ones_like(time)
    exposure_values = np.broadcast_to(exposures, len(time))
    for center, reference in zip(centers, EPOCHS):
        local = np.abs(time - reference) < 0.8
        flux[local] = ExoMoonLeastSquares._transit_model(
            _params(center), time[local], exposure_values[local])
    return flux


def _case(name, moon_offset, gapped, exposures, ttv):
    step = exposures / 86400.0 if exposures else 0.005
    local = np.arange(-0.65, 0.65001, step)
    time = np.concatenate([epoch + local for epoch in EPOCHS])
    if gapped:
        # Remove an ingress segment from the middle epoch but preserve two-sided OOT.
        gap = (np.abs(time - EPOCHS[1] + 0.10) < 0.035)
        time = time[~gap]
    errors = np.where(np.arange(len(time)) % 3 == 0, 8e-4, 2e-4) if gapped else np.full(len(time), 2e-4)
    centers = EPOCHS + ttv
    exposure_array = np.full(len(time), exposures) if exposures else None
    planet = _planet_truth(time, centers, 0.0 if exposure_array is None else exposure_array)
    moon = np.zeros_like(time)
    if moon_offset is not None:
        for center in centers:
            moon += 8e-4 * np.exp(-0.5 * ((time - (center + moon_offset)) / 0.025) ** 2)
    rng = np.random.default_rng(117)
    noise = rng.normal(0.0, errors)
    return {
        "name": name,
        "time": time,
        "errors": errors,
        "planet": planet + noise,
        "injected": planet - moon + noise,
        "signal": -moon,
        "centers": centers,
        "exposures": exposure_array,
    }


def _emls(case):
    return ExoMoonLeastSquares(
        "", 1, 1.0, 1.0, [0.2, 0.1], 0.08 * R_EARTH_PER_R_SUN, 0.8,
        100.0, 5.0, 0.0, 0.30, 12.0 / 215.032, 89.0, 0.3, 0.0, 0.1,
        90.0, 0.0, 0.3, 0.6, EPOCHS, case["time"], case["planet"], case["errors"],
        5.5, min_cadences_per_t0=8, fit_rp_per_transit=True,
        min_transit_completeness=0.1, max_subtraction_rms_ratio=100.0,
        exposure_times_seconds=case["exposures"],
    )


def _measure(case):
    planet_emls = _emls(case)
    start = time.perf_counter()
    planet_time, planet_residual, planet_errors = planet_emls.subtract_planet_transit(
        case["time"], case["planet"], case["errors"], exposure_times_seconds=case["exposures"])
    elapsed = time.perf_counter() - start
    indices = planet_emls.retained_cadence_indices
    row = {
        "case": case["name"],
        "status": planet_emls.subtraction_status["status"],
        "seconds": elapsed,
        "epochs_retained": len(planet_emls.bary_t0s),
        "epochs_rejected": len(EPOCHS) - len(planet_emls.bary_t0s),
        "cadences_retained": len(indices),
        "cadences_rejected": len(case["time"]) - len(indices),
        "shared_nfev": planet_emls.shared_fit_status.nfev,
        "timing_bias_days": np.nan,
        "rp_bias": np.nan,
        "contact_bias_days": np.nan,
        "weighted_residual_rms": np.sqrt(np.average((planet_residual - 1.0) ** 2,
                                                       weights=1.0 / planet_errors ** 2)),
        "transfer": np.nan,
        "orthogonal_distortion": np.nan,
    }
    if len(planet_emls.local_t0s):
        row["timing_bias_days"] = float(np.max(np.abs(np.asarray(planet_emls.local_t0s) - case["centers"])))
    if hasattr(planet_emls, "shared_planet_params"):
        row["rp_bias"] = float(planet_emls.shared_planet_params.rp - 0.08)
        true_duration = ExoMoonLeastSquares.contact_duration(_params(0.0))
        row["contact_bias_days"] = float(planet_emls.planet_duration - true_duration)
    if np.any(case["signal"]):
        injected_emls = _emls(case)
        _, injected_residual, injected_errors = injected_emls.subtract_planet_transit(
            case["time"], case["injected"], case["errors"], exposure_times_seconds=case["exposures"])
        assert np.array_equal(indices, injected_emls.retained_cadence_indices)
        metrics = ExoMoonLeastSquares.signal_transfer_metrics(
            case["signal"][indices], injected_residual - planet_residual, injected_errors)
        row["transfer"] = metrics["weighted_transfer_coefficient"]
        row["orthogonal_distortion"] = metrics["weighted_orthogonal_distortion"]
    return row


def _matrix():
    return [
        _case("planet_regular_instant_no_ttv", None, False, 0.0, np.zeros(3)),
        _case("separate_regular_instant_no_ttv", 0.42, False, 0.0, np.zeros(3)),
        _case("overlap_gapped_hetero_120s_ttv", 0.03, True, 120.0, np.array([-0.035, 0.0, 0.028])),
        _case("separate_gapped_hetero_1800s_ttv", 0.42, True, 1800.0, np.array([-0.035, 0.0, 0.028])),
    ]


def _run_matrix():
    return [_measure(case) for case in _matrix()]


def test_stage7_matrix_is_deterministic_and_exercises_actual_subtraction():
    first = _run_matrix()
    second = _run_matrix()
    for left, right in zip(first, second):
        assert left["status"] == right["status"] == "ok"
        assert left["epochs_retained"] == right["epochs_retained"] == 3
        assert left["cadences_retained"] == right["cadences_retained"]
        assert left["shared_nfev"] == right["shared_nfev"]
        for key in ("timing_bias_days", "rp_bias", "contact_bias_days", "weighted_residual_rms",
                    "transfer", "orthogonal_distortion"):
            if np.isnan(left[key]):
                assert np.isnan(right[key])
            else:
                np.testing.assert_approx_equal(left[key], right[key], significant=12)
        assert left["timing_bias_days"] < 0.01
        assert left["weighted_residual_rms"] < 0.002
    assert first[1]["transfer"] > 0.70
    assert first[2]["transfer"] > 0.20
    assert first[3]["transfer"] > 0.50


def test_stage7_small_medium_subtraction_profile_has_no_startup_claim():
    # These are warm timings only: module import and first BATMAN construction
    # are deliberately excluded and reported separately by the invoking command.
    durations = []
    for samples in (300, 2400):
        sample_time = np.linspace(-0.4, 0.4, samples)
        ExoMoonLeastSquares._transit_model(_params(0.0), sample_time, 120.0)
        start = time.perf_counter()
        for _ in range(3):
            ExoMoonLeastSquares._transit_model(_params(0.0), sample_time, 120.0)
        durations.append((time.perf_counter() - start) / 3)
    assert all(np.isfinite(duration) and duration > 0 for duration in durations)


if __name__ == "__main__":
    if "--profile" in sys.argv:
        print(f"cold_validation_import_seconds={IMPORT_SECONDS:.6f}")
        for samples in (300, 2400):
            sample_time = np.linspace(-0.4, 0.4, samples)
            ExoMoonLeastSquares._transit_model(_params(0.0), sample_time, 120.0)
            start = time.perf_counter()
            for _ in range(3):
                ExoMoonLeastSquares._transit_model(_params(0.0), sample_time, 120.0)
            print(f"warm_model_{samples}_cadences_seconds={(time.perf_counter() - start) / 3:.6f}")
    else:
        rows = _run_matrix()
        writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
