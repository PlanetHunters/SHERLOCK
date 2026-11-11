"""False-alarm test of EMLS on synthetic planets without a moon.

Each case takes a real light curve, removes its real transits, and injects a planet with the
target's parameters at a shifted ephemeris t0 + offset, so every offset is an independent noise
realization of the target's own noise. No moon is injected: any detection is a false alarm.
Run it before merging a change to how EMLS models or subtracts the planet (see RESULTS.md).

Conditions (how the injected planet differs from the model EMLS subtracts):
    match  same limb darkening, no exposure smearing: EMLS can model it exactly
    ld     limb darkening u1 + 0.15, u2 - 0.15 (EMLS starts from the given values)
    smear  integrated over the exposure (--exposure-seconds, default Kepler long cadence);
           EMLS models the exposure it measures from the point spacing (exposure_time="auto").
           --instantaneous runs EMLS without it, as the 2026-11-11 results in RESULTS.md did

Inputs:
    --curve   light curve as .npz or .csv with time, flux, flux_err columns. Mask other planets
              of the system first; EMLS detrends it itself (biweight, 3 transit durations).
    --params  emls_results.json written by `python -m sherlockpipe.emls`, or a JSON with
              "planet" (planet_* keys of the catalog driver) and "star" [mass, radius, [u1, u2]].

The FAP of every case uses the null SNRs of one search on the real curve, computed once per
target and code version with the nulls command (any JSON with "null_snrs" works, for example an
emls_results.json run with --null_runs).

Workflow:
    python synthetic_planet.py nulls --curve lc.csv --params emls_results.json --out OUT --cpus 8
    python synthetic_planet.py plan --curve lc.csv --params emls_results.json --nulls OUT/nulls.json \\
        --out OUT --cpus 2 | xargs -P 4 -I{} sh -c "{}"
    python synthetic_planet.py summary OUT

Each case is one search (minutes); the nulls are one search plus --null-runs (hours).
"""
import argparse
import glob
import json
import os
import shlex
import sys
import time as clock

import numpy as np

KEPLER_LONG_CADENCE_SECONDS = 1765.5
R_SUN_IN_R_EARTH = 109.076
AU_IN_R_SUN = 215.032
CONDITIONS = ("match", "ld", "smear")
LD_SHIFT = 0.15


def load_params(path):
    """Return (planet, star) from an emls_results.json or a catalog driver result JSON.

    planet uses the catalog driver keys (planet_period, planet_t0, ...); star is
    {"mass", "radius", "ld", "exposure_seconds"}.
    """
    source = json.load(open(path))
    planet, star = source["planet"], source["star"]
    if "planet_period" in planet:
        mass, radius, ld = star
        return dict(planet), {"mass": float(mass), "radius": float(radius), "ld": [float(v) for v in ld],
                              "exposure_seconds": None}
    moon = source.get("moon_radius_range") or {}
    converted = {
        "planet_radius": planet["radius"], "planet_radius_err": planet.get("radius_err", 0.1 * planet["radius"]),
        "planet_period": planet["period"], "planet_t0": planet["t0"], "planet_duration": planet["duration"],
        "planet_inc": planet["inc"], "planet_inc_err": planet.get("inc_err", 0.5),
        "planet_ecc": planet.get("ecc", 0.0), "planet_ecc_err": 0.0,
        "planet_arg_periastron": planet.get("arg_periastron", 0.0), "planet_mass": planet["mass"],
        "planet_semimajor_axis": planet["a"], "planet_impact_param": planet["impact"],
        "min_radius": moon.get("min_radius", planet["radius"] / 20),
        "max_radius": moon.get("max_radius", planet["radius"] / 2)}
    return {key: float(value) for key, value in converted.items()}, \
        {"mass": float(star["mass"]), "radius": float(star["radius"]), "ld": [float(v) for v in star["ld"]],
         "exposure_seconds": source.get("exposure_time_s")}


def load_curve(path):
    """Return finite time, flux, flux_err from an .npz or a .csv light curve."""
    if path.endswith(".npz"):
        data = np.load(path)
        columns = [np.asarray(data[key], dtype=float) for key in ("time", "flux", "flux_err")]
    else:
        import pandas as pd
        frame = pd.read_csv(path)
        frame.columns = [name.lstrip("#").strip() for name in frame.columns]
        columns = [frame[key].to_numpy(float) for key in ("time", "flux", "flux_err")]
    finite = np.all([np.isfinite(column) for column in columns], axis=0)
    return tuple(column[finite] for column in columns)


def remove_real_transits(time, flux, flux_err, period, t0, duration):
    """Drop the points within one transit duration of the real mid-times."""
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    keep = np.abs(phase) > duration
    return time[keep], flux[keep], flux_err[keep]


def synthetic_offsets(period, duration, count):
    """Offsets of the injected ephemeris, kept 2 durations plus 1 day away from the real transits."""
    margin = 2 * duration + 1.0
    if period - margin <= margin:
        raise ValueError(f"Period {period} d is too short for offsets {margin} d away from the real transits")
    return np.linspace(margin, period - margin, count)


def epochs_in_span(t0, period, time):
    first = t0 - period * np.floor((t0 - time.min()) / period)
    return list(np.arange(first, time.max(), period))


def inject_planet(time, flux, planet, star, t0, condition, exposure_seconds=KEPLER_LONG_CADENCE_SECONDS):
    import batman
    params = batman.TransitParams()
    params.t0, params.per = t0, planet["planet_period"]
    params.rp = planet["planet_radius"] / (star["radius"] * R_SUN_IN_R_EARTH)
    params.a = planet["planet_semimajor_axis"] * AU_IN_R_SUN / star["radius"]
    params.inc, params.ecc, params.w = planet["planet_inc"], planet["planet_ecc"], planet["planet_arg_periastron"]
    params.limb_dark = "quadratic"
    params.u = list(star["ld"])
    kwargs = {}
    if condition == "ld":
        params.u = [star["ld"][0] + LD_SHIFT, star["ld"][1] - LD_SHIFT]
    elif condition == "smear":
        kwargs = {"supersample_factor": 15, "exp_time": exposure_seconds / 86400}
    elif condition != "match":
        raise ValueError(f"Unknown condition {condition}")
    return flux * batman.TransitModel(params, time, **kwargs).light_curve(params)


def build_emls(out, cpus, planet, star, time, flux, flux_err, t0, null_runs, exposure_time="auto"):
    from sherlockpipe.exomoon import ExoMoonLeastSquares
    os.makedirs(out, exist_ok=True)
    return ExoMoonLeastSquares(
        out, cpus, star["mass"], star["radius"], star["ld"], planet["planet_radius"], planet["planet_radius_err"],
        planet["planet_mass"], planet["planet_period"], t0, planet["planet_duration"],
        planet["planet_semimajor_axis"], planet["planet_inc"], planet["planet_inc_err"], planet["planet_ecc"],
        planet["planet_ecc_err"], planet["planet_arg_periastron"], planet["planet_impact_param"],
        planet["min_radius"], planet["max_radius"], epochs_in_span(t0, planet["planet_period"], time),
        time, flux, flux_err, 5.5, period_grid_size=10000, radius_grid_size=20, subtract_mode="subtract",
        fit_rp_per_transit=False, min_transit_completeness=0.75, max_subtraction_rms_ratio=1.5,
        null_runs=null_runs, detrend_window=3 * planet["planet_duration"], detrend_mask="auto",
        exposure_time=exposure_time)


def _json(value):
    return value.item() if hasattr(value, "item") else (value.tolist() if hasattr(value, "tolist") else str(value))


def code_version():
    try:
        from sherlockpipe import __version__
        return __version__
    except ImportError:
        return None


def nulls(args):
    """One search on the real curve with --null-runs nulls; writes OUT/nulls.json."""
    planet, star = load_params(args.params)
    time, flux, flux_err = load_curve(args.curve)
    emls = build_emls(os.path.join(args.out, "nulls_search"), args.cpus, planet, star, time, flux, flux_err,
                      planet["planet_t0"], args.null_runs, None if args.instantaneous else "auto")
    start = clock.time()
    emls.run(plot=False)
    result = {"curve": os.path.abspath(args.curve), "params": os.path.abspath(args.params),
              "sherlockpipe_version": code_version(), "null_runs": args.null_runs, "seconds": clock.time() - start,
              "max_snr": emls.max_snr, "fap": emls.fap, "fap_upper": emls.fap_upper, "detected": emls.detected,
              "null_snrs": emls.null_snrs}
    path = os.path.join(args.out, "nulls.json")
    json.dump(result, open(path, "w"), indent=1, default=_json)
    print("NULLS", path, len(emls.null_snrs or []), "nulls, real-curve SNR", emls.max_snr, "FAP", emls.fap)


def run(args):
    planet, star = load_params(args.params)
    exposure = args.exposure_seconds or star["exposure_seconds"] or KEPLER_LONG_CADENCE_SECONDS
    null_snrs = json.load(open(args.nulls))["null_snrs"]
    if not null_snrs:
        raise ValueError(f"{args.nulls} has no null_snrs")
    time, flux, flux_err = remove_real_transits(*load_curve(args.curve), planet["planet_period"],
                                                planet["planet_t0"], planet["planet_duration"])
    offset = synthetic_offsets(planet["planet_period"], planet["planet_duration"], args.offsets)[args.offset_index]
    t0 = planet["planet_t0"] + offset
    flux = inject_planet(time, flux, planet, star, t0, args.condition, exposure)
    planet["planet_t0"] = t0
    case_dir = os.path.join(args.out, f"{args.condition}_{args.offset_index}")
    emls = build_emls(case_dir, args.cpus, planet, star, time, flux, flux_err, t0, 0,
                      None if args.instantaneous else "auto")
    start = clock.time()
    emls.run(plot=False, null_snrs=null_snrs)
    result = {"condition": args.condition, "offset_index": args.offset_index, "offset": float(offset),
              "t0": float(t0), "epochs": len(emls.bary_t0s), "seconds": clock.time() - start,
              "sherlockpipe_version": code_version(), "exposure_time": emls.exposure_time}
    for key in ("max_snr", "fap", "fap_upper", "detected", "recovered_period", "best_snr_radius",
                "planet_shape_check", "red_noise_factor", "planet_depth_correction", "planet_shape_corrections"):
        result[key] = getattr(emls, key, None)
    json.dump(result, open(os.path.join(case_dir, "result.json"), "w"), indent=1, default=_json)
    print("DONE", json.dumps({k: result[k] for k in ("condition", "offset_index", "max_snr", "fap_upper", "detected")},
                             default=str))


def plan(args):
    """Print one run command per offset and condition, for xargs or a job queue."""
    script = os.path.abspath(__file__)
    for condition in args.conditions:
        for index in range(args.offsets):
            command = [sys.executable, script, "run", "--curve", args.curve, "--params", args.params,
                       "--nulls", args.nulls, "--offset-index", str(index), "--offsets", str(args.offsets),
                       "--condition", condition, "--out", args.out, "--cpus", str(args.cpus)]
            if args.exposure_seconds:
                command += ["--exposure-seconds", str(args.exposure_seconds)]
            if args.instantaneous:
                command += ["--instantaneous"]
            print(" ".join(shlex.quote(part) for part in command))


def summarize(rows):
    """Per-condition counts: cases, detected, FAP < 0.01, FAP < 0.05, SNR median and max, FAP median."""
    lines = {}
    for condition in CONDITIONS:
        cases = [row for row in rows if row["condition"] == condition]
        if not cases:
            continue
        snr = np.array([row["max_snr"] for row in cases], dtype=float)
        fap = np.array([row["fap"] for row in cases], dtype=float)
        lines[condition] = {"cases": len(cases), "detected": sum(bool(row["detected"]) for row in cases),
                            "fap_below_0.01": int(np.sum(fap < 0.01)), "fap_below_0.05": int(np.sum(fap < 0.05)),
                            "snr_median": float(np.median(snr)), "snr_max": float(snr.max()),
                            "fap_median": float(np.median(fap))}
    return lines


def summary(args):
    for directory in args.dirs:
        rows = [json.load(open(path)) for path in glob.glob(os.path.join(directory, "*", "result.json"))]
        for condition, line in summarize(rows).items():
            print(f"{directory} {condition:5}: {line['cases']} cases, detected {line['detected']}, "
                  f"fap<0.01 {line['fap_below_0.01']}, fap<0.05 {line['fap_below_0.05']}, SNR median "
                  f"{line['snr_median']:.2f} max {line['snr_max']:.2f}, FAP median {line['fap_median']:.3f}")
    print("With a calibrated FAP, about 1 case in 20 falls below FAP 0.05 and none should be detected.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def inputs(command):
        command.add_argument("--curve", required=True)
        command.add_argument("--params", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--cpus", type=int, default=4)
        command.add_argument("--instantaneous", action="store_true",
                             help="run EMLS with an instantaneous model (exposure_time=None)")

    null = commands.add_parser("nulls", help="null SNRs of the real curve, once per target and code version")
    inputs(null)
    null.add_argument("--null-runs", type=int, default=100)
    for name in ("run", "plan"):
        command = commands.add_parser(name)
        inputs(command)
        command.add_argument("--nulls", required=True, help="JSON with null_snrs (from the nulls command)")
        command.add_argument("--offsets", type=int, default=20)
        command.add_argument("--exposure-seconds", type=float, default=None,
                             help="smear exposure; default: the params exposure, else Kepler long cadence")
        if name == "run":
            command.add_argument("--offset-index", type=int, required=True)
            command.add_argument("--condition", choices=CONDITIONS, required=True)
        else:
            command.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    many = commands.add_parser("summary")
    many.add_argument("dirs", nargs="+")
    args = parser.parse_args(argv)
    {"nulls": nulls, "run": run, "plan": plan, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
