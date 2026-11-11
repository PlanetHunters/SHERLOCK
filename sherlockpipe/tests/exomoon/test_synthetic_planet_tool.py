"""Unit tests of the synthetic planet false-alarm tool (inputs, injection, job plan; no search)."""
import importlib.util
import json
import os

import numpy as np
import pytest

TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "regression_tests", "emls_synthetic_planet",
                    "synthetic_planet.py")


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("synthetic_planet", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER_PLANET = {"planet_radius": 10.99, "planet_radius_err": 0.9, "planet_period": 30.36, "planet_t0": 148.09,
                 "planet_duration": 0.25, "planet_inc": 89.58, "planet_inc_err": 0.5, "planet_ecc": 0.0,
                 "planet_ecc_err": 0.0, "planet_arg_periastron": 0.0, "planet_mass": 36.4,
                 "planet_semimajor_axis": 0.172, "planet_impact_param": 0.359, "min_radius": 0.55, "max_radius": 5.5}


def test_both_parameter_formats_give_the_same_planet(tool, tmp_path):
    driver = tmp_path / "driver.json"
    driver.write_text(json.dumps({"planet": DRIVER_PLANET, "star": [0.74, 0.765, [0.61, 0.11]]}))
    emls = tmp_path / "emls_results.json"
    emls.write_text(json.dumps({
        "planet": {"radius": 10.99, "radius_err": 0.9, "period": 30.36, "t0": 148.09, "duration": 0.25,
                   "inc": 89.58, "inc_err": 0.5, "ecc": 0.0, "arg_periastron": 0.0, "mass": 36.4, "a": 0.172,
                   "impact": 0.359},
        "star": {"mass": 0.74, "radius": 0.765, "ld": [0.61, 0.11]}, "exposure_time_s": 1800.0,
        "moon_radius_range": {"min_radius": 0.55, "max_radius": 5.5}}))
    planet_driver, star_driver = tool.load_params(str(driver))
    planet_emls, star_emls = tool.load_params(str(emls))
    assert planet_driver == pytest.approx(planet_emls)
    assert star_driver["ld"] == star_emls["ld"] and star_driver["radius"] == star_emls["radius"]
    assert star_driver["exposure_seconds"] is None and star_emls["exposure_seconds"] == 1800.0


def test_csv_and_npz_curves_load_the_same_finite_points(tool, tmp_path):
    time = np.linspace(0, 10, 50)
    flux = np.ones(50)
    flux[7] = np.nan
    np.savez(tmp_path / "lc.npz", time=time, flux=flux, flux_err=np.full(50, 1e-3))
    (tmp_path / "lc.csv").write_text("#time,flux,flux_err\n" + "\n".join(
        f"{t},{f},0.001" for t, f in zip(time, flux)))
    from_npz = tool.load_curve(str(tmp_path / "lc.npz"))
    from_csv = tool.load_curve(str(tmp_path / "lc.csv"))
    assert len(from_npz[0]) == len(from_csv[0]) == 49
    np.testing.assert_allclose(from_npz, from_csv)


def test_injection_conditions_change_the_transit_shape(tool):
    star = {"mass": 0.74, "radius": 0.765, "ld": [0.61, 0.11], "exposure_seconds": None}
    time = np.linspace(-0.3, 0.3, 601)
    flux = np.ones_like(time)
    match, ld, smear = (tool.inject_planet(time, flux, DRIVER_PLANET, star, 0.0, condition)
                        for condition in tool.CONDITIONS)
    centre = np.argmin(np.abs(time))
    assert 0.97 < match[centre] < 0.99
    assert ld[centre] < match[centre]
    assert smear[centre] > match[centre]
    assert match[0] == ld[0] == 1.0
    with pytest.raises(ValueError):
        tool.inject_planet(time, flux, DRIVER_PLANET, star, 0.0, "other")


def test_offsets_avoid_the_real_transits(tool):
    offsets = tool.synthetic_offsets(30.36, 0.25, 20)
    assert len(offsets) == 20 and offsets[0] == pytest.approx(1.5) and offsets[-1] == pytest.approx(28.86)
    with pytest.raises(ValueError):
        tool.synthetic_offsets(2.5, 0.25, 20)


def test_plan_prints_one_command_per_case(tool, capsys):
    tool.main(["plan", "--curve", "lc.csv", "--params", "p.json", "--nulls", "n.json", "--out", "OUT",
               "--offsets", "5", "--conditions", "match", "ld"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 10
    assert all(" run " in line and "--offsets 5" in line for line in lines)


def test_summary_counts(tool):
    rows = [{"condition": "ld", "max_snr": 4.0 + i, "fap": fap, "detected": fap < 0.005}
            for i, fap in enumerate((0.001, 0.02, 0.3))]
    line = tool.summarize(rows)["ld"]
    assert (line["cases"], line["detected"], line["fap_below_0.01"], line["fap_below_0.05"]) == (3, 1, 1, 2)
    assert line["snr_median"] == 5.0
