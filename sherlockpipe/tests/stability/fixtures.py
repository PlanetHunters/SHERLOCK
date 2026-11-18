"""Shared helpers for the stability tests: small planets, recording calculators and properties files."""
import numpy as np

from sherlockpipe.system_stability.stability_calculator import PlanetInput, SimulationInput, StabilityCalculator


def make_planet(**kw):
    """A planet with every parameter fixed; override what the test sweeps."""
    base = dict(period=4.2345, period_low_err=0, period_up_err=0, radius=None, radius_low_err=None, radius_up_err=None,
                mass=12.0, mass_low_err=0, mass_up_err=0, eccentricity=0.1, ecc_low_err=0, ecc_up_err=0,
                inclination=0, inc_low_err=0, inc_up_err=0, omega=60, omega_low_err=0, omega_up_err=0,
                omega_big=0, omega_big_low_err=0, omega_big_up_err=0, period_bins=1, mass_bins=1, ecc_bins=1,
                inc_bins=1, omega_bins=1, omega_big_bins=1)
    base.update(kw)
    return PlanetInput(**base)


class RecordingCalculator(StabilityCalculator):
    """Does not integrate anything: it records the scenarios the grid produced."""

    def run_simulation(self, simulation_input: SimulationInput) -> dict:
        return {"index": simulation_input.index, "star_mass": simulation_input.star_mass,
                "periods": list(simulation_input.planet_periods), "masses": list(simulation_input.mass_arr),
                "ecc": list(simulation_input.ecc_arr), "inc": list(simulation_input.inc_arr),
                "omega": list(simulation_input.omega_arr), "omega_big": list(simulation_input.omega_big_arr)}

    def store_simulation_results(self, simulation_results, results_dir):
        self.collected = simulation_results


def column(results, key, planet):
    """The values of one parameter of one planet over every scenario."""
    return [r[key][planet] for r in results]


def distinct(results, key, planet):
    return sorted(set(column(results, key, planet)), key=str)


def write_yaml(path, text):
    with open(path, "w") as handle:
        handle.write(text)
    return path


TWO_PLANETS_YAML = """
STAR:
  M: 0.82
  M_LOW: 0.0
  M_UP: 0.0
  M_BINS: 1
BODIES:
  - P: 4.2345
    P_LOW: 0
    P_UP: 0
    P_BINS: 1
    M: 12.0
    M_LOW: 0
    M_UP: 0
    M_BINS: 1
    E: 0.10
    E_LOW: 0.10
    E_UP: 0.30
    E_BINS: 3
    I: 0
    I_LOW: 0
    I_UP: 0
    I_BINS: 1
    O: 60
    O_LOW: 0
    O_UP: 0
    O_BINS: 1
    OM: 0
    OM_LOW: 0
    OM_UP: 0
    OM_BINS: 1
  - P: 6.35
    P_LOW: 0
    P_UP: 0
    P_BINS: 1
    M: 20.0
    M_LOW: 8
    M_UP: 20
    M_BINS: 2
    E: 0.10
    E_LOW: 0.10
    E_UP: 0.30
    E_BINS: 3
    I: 0
    I_LOW: 0
    I_UP: 0
    I_BINS: 1
    O: 200
    O_LOW: 0
    O_UP: 0
    O_BINS: 1
    OM: 0
    OM_LOW: 0
    OM_UP: 0
    OM_BINS: 1
"""
