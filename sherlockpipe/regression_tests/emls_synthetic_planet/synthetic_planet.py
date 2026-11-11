"""False-alarm test of EMLS on synthetic planets without a moon.

Each case takes a real light curve, removes its real transits, and injects a planet with the
catalog parameters at a shifted ephemeris t0 + offset, so every offset is an independent noise
realization of the target's own noise. No moon is injected: any detection is a false alarm.

Conditions (how the injected planet differs from the model EMLS subtracts):
    match  same limb darkening, no exposure smearing: EMLS can model it exactly
    ld     limb darkening u1 + 0.15, u2 - 0.15 (EMLS keeps the catalog values fixed)
    smear  integrated over the Kepler long cadence (29.4 min); EMLS runs without exposure_time,
           as the real-data runs did

The FAP uses the null SNRs of the control search of the same target and code version (--nulls),
so each case costs one search.

Usage:
    python synthetic_planet.py run --curve curve.npz --params control_result.json --nulls control_result.json
                                   --offset-index 3 --offsets 20 --condition ld --out OUT --cpus 4
    python synthetic_planet.py summary RESULTS_DIR [RESULTS_DIR ...]
"""
import argparse
import glob
import json
import os
import time as clock

import numpy as np

KEPLER_LONG_CADENCE = 0.020434
R_SUN_IN_R_EARTH = 109.076
AU_IN_R_SUN = 215.032


def remove_real_transits(time, flux, flux_err, period, t0, duration):
    """Drop the points within one transit duration of the real mid-times."""
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    keep = np.abs(phase) > duration
    return time[keep], flux[keep], flux_err[keep]


def synthetic_offsets(period, duration, count):
    """Offsets of the injected ephemeris, kept 2 durations plus 1 day away from the real transits."""
    margin = 2 * duration + 1.0
    return np.linspace(margin, period - margin, count)


def inject_planet(time, flux, planet, star_radius, ld, t0, condition):
    import batman
    params = batman.TransitParams()
    params.t0, params.per = t0, planet["planet_period"]
    params.rp = planet["planet_radius"] / (star_radius * R_SUN_IN_R_EARTH)
    params.a = planet["planet_semimajor_axis"] * AU_IN_R_SUN / star_radius
    params.inc, params.ecc, params.w = planet["planet_inc"], planet["planet_ecc"], planet["planet_arg_periastron"]
    params.limb_dark = "quadratic"
    params.u = list(ld)
    kwargs = {}
    if condition == "ld":
        params.u = [ld[0] + 0.15, ld[1] - 0.15]
    elif condition == "smear":
        kwargs = {"supersample_factor": 15, "exp_time": KEPLER_LONG_CADENCE}
    elif condition != "match":
        raise ValueError(f"Unknown condition {condition}")
    return flux * batman.TransitModel(params, time, **kwargs).light_curve(params)


def run(args):
    from sherlockpipe.exomoon import ExoMoonLeastSquares
    source = json.load(open(args.params))
    planet = dict(source["planet"])
    star_mass, star_radius, ld = source["star"]
    null_snrs = json.load(open(args.nulls))["null_snrs"]
    data = np.load(args.curve)
    time, flux, flux_err = remove_real_transits(np.asarray(data["time"]), np.asarray(data["flux"]),
                                                np.asarray(data["flux_err"]), planet["planet_period"],
                                                planet["planet_t0"], planet["planet_duration"])
    offset = synthetic_offsets(planet["planet_period"], planet["planet_duration"], args.offsets)[args.offset_index]
    t0 = planet["planet_t0"] + offset
    flux = inject_planet(time, flux, planet, star_radius, ld, t0, args.condition)
    planet["planet_t0"] = t0
    t0s = list(np.arange(t0 - planet["planet_period"] * np.floor((t0 - time.min()) / planet["planet_period"]),
                         time.max(), planet["planet_period"]))
    os.makedirs(args.out, exist_ok=True)
    emls = ExoMoonLeastSquares(
        args.out, args.cpus, star_mass, star_radius, ld, planet["planet_radius"], planet["planet_radius_err"],
        planet["planet_mass"], planet["planet_period"], t0, planet["planet_duration"],
        planet["planet_semimajor_axis"], planet["planet_inc"], planet["planet_inc_err"], planet["planet_ecc"],
        planet["planet_ecc_err"], planet["planet_arg_periastron"], planet["planet_impact_param"],
        planet["min_radius"], planet["max_radius"], t0s, time, flux, flux_err, 5.5,
        period_grid_size=10000, radius_grid_size=20, subtract_mode="subtract", fit_rp_per_transit=False,
        min_transit_completeness=0.75, max_subtraction_rms_ratio=1.5, null_runs=0,
        detrend_window=3 * planet["planet_duration"], detrend_mask="auto", exposure_time=None)
    start = clock.time()
    emls.run(plot=False, null_snrs=null_snrs)
    result = {"condition": args.condition, "offset_index": args.offset_index, "offset": float(offset),
              "t0": float(t0), "epochs": len(emls.bary_t0s), "seconds": clock.time() - start}
    for key in ("max_snr", "fap", "fap_upper", "detected", "recovered_period", "best_snr_radius",
                "planet_shape_check", "red_noise_factor"):
        result[key] = getattr(emls, key, None)
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    print("DONE", json.dumps({k: result[k] for k in ("condition", "offset_index", "max_snr", "fap_upper", "detected")},
                             default=str))


def summary(args):
    for directory in args.dirs:
        rows = [json.load(open(path)) for path in glob.glob(os.path.join(directory, "*", "result.json"))]
        for condition in ("match", "ld", "smear"):
            cases = [row for row in rows if row["condition"] == condition]
            if not cases:
                continue
            snr = np.array([row["max_snr"] for row in cases])
            fap = np.array([row["fap"] for row in cases])
            detected = sum(bool(row["detected"]) for row in cases)
            print(f"{directory} {condition:5}: {len(cases)} cases, detected {detected}, fap<0.05 "
                  f"{int(np.sum(fap < 0.05))}, SNR median {np.median(snr):.2f} max {snr.max():.2f}, "
                  f"FAP median {np.median(fap):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("--curve", required=True)
    one.add_argument("--params", required=True)
    one.add_argument("--nulls", required=True)
    one.add_argument("--offset-index", type=int, required=True)
    one.add_argument("--offsets", type=int, default=20)
    one.add_argument("--condition", choices=("match", "ld", "smear"), required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--cpus", type=int, default=4)
    many = commands.add_parser("summary")
    many.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
