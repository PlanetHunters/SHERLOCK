"""Moon orbit model: audit items G3, G4, G5, G6, G11, G12, G14, G17 and G19.

G5/G6: the outer moon orbit is the Domingos et al. (2006) prograde limit with the planet
eccentricity. G11: moon durations and offsets scale with the eccentric planet speed at
conjunction. G12: the dead moon inclination, eccentricity and periastron fields are gone; the
model is coplanar and circular. G3/G4/G14: the run logs where that screening model stops holding.
G17: the best window's overlap with the planet transit and whether its period is constrained.
G19: the default period grid is 10000 periods.
"""
import logging

import batman
import numpy as np
import pytest

from sherlockpipe.exomoon import (AU_TO_RSUN, ExoMoonLeastSquares, M_earth_to_M_sun, R_earth_to_R_sun,
                                  SearchInput)

SUN = 1.0
JUPITER, NEPTUNE = 317.8, 17.15


def system(tmp_path, planet_mass, planet_radius, period, ecc=0.0, w=90.0, max_radius=None):
    """An EMLS instance around the Sun with data only to satisfy the constructor."""
    a_au = ExoMoonLeastSquares.compute_semimajor_axis(SUN, period)
    a_over_rstar = a_au * AU_TO_RSUN
    duration = period / np.pi * np.arcsin((1 + planet_radius * R_earth_to_R_sun) / a_over_rstar)
    t0s = np.arange(4) * period
    time = np.concatenate([t0 + np.arange(-2.0, 2.0, 10 / 1440) for t0 in t0s])
    flux = np.ones_like(time)
    max_radius = planet_radius / 2 if max_radius is None else max_radius
    return ExoMoonLeastSquares(str(tmp_path), 1, SUN, 1.0, [0.4, 0.2], planet_radius, 0.1 * planet_radius,
                               planet_mass, period, 0.0, duration, a_au, 90.0, 0.5, ecc, 0.0, w, 0.0,
                               planet_radius / 20, max_radius, t0s, time, flux, np.full(len(time), 1e-4), 5.5,
                               period_grid_size=40, radius_grid_size=4)


# --- G5 / G6 -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('ecc, ratio', [(0.0, 0.4895), (0.15, 0.4138), (0.3, 0.3382)])
def test_outer_moon_orbit_is_the_domingos_prograde_limit(ecc, ratio):
    mass = JUPITER * M_earth_to_M_sun
    hill = ExoMoonLeastSquares.compute_hill_radius(SUN, mass, 1.0)

    a_max = ExoMoonLeastSquares.max_moon_semimajor_axis(SUN, mass, 1.0, ecc)

    assert a_max / hill == pytest.approx(ratio, abs=1e-3)


def test_outer_limit_is_logged(tmp_path, caplog):
    emls = system(tmp_path, JUPITER, 11.2, 300.0)
    emls.models = [(1.0, np.array([-1.0, 1.0]), np.array([0.999, 0.999]))]
    emls.cpus = 1

    with caplog.at_level(logging.INFO):
        period_grid, _, _, _ = emls.grid_search(progress=False)

    assert period_grid[-1] == pytest.approx(emls.max_moon_period(JUPITER), rel=1e-9)
    assert any('Outer moon limit: a = 0.489 r_H (Domingos et al. 2006, prograde' in r.getMessage()
               for r in caplog.records)


def test_no_stable_orbit_for_a_very_eccentric_planet():
    with pytest.raises(ValueError, match='No stable prograde moon orbit'):
        ExoMoonLeastSquares.max_moon_semimajor_axis(SUN, JUPITER * M_earth_to_M_sun, 1.0, 0.98)


# --- G11 ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('ecc, w', [(0.3, 90.0), (0.3, 270.0), (0.15, 90.0)])
def test_moon_window_follows_the_eccentric_planet_speed(tmp_path, ecc, w):
    """Acceptance (0.3, 90): the window is within 2 % of the batman T14 with the same orbit."""
    emls = system(tmp_path, NEPTUNE, 3.9, 30.0, ecc=ecc, w=w)
    period = emls.max_moon_period(NEPTUNE)
    hypothesis = emls.search_input(period, 0.0, NEPTUNE, period, 0.0)
    moon = batman.TransitParams()
    moon.t0, moon.per, moon.rp = 0.0, 30.0, emls.max_radius * R_earth_to_R_sun
    moon.a, moon.inc, moon.ecc, moon.w = emls.planet_semimajor_axis * AU_TO_RSUN, 90.0, ecc, w
    moon.u, moon.limb_dark = [0.4, 0.2], 'quadratic'
    true_duration = ExoMoonLeastSquares.model_transit_duration(moon, 1.0)

    assert hypothesis.moon_transit_duration == pytest.approx(true_duration, rel=0.02)
    # The offset is a_m over the planet speed at conjunction.
    a_moon = ExoMoonLeastSquares.compute_semimajor_axis(NEPTUNE * M_earth_to_M_sun, period)
    speed = 2 * np.pi * emls.planet_semimajor_axis / 30.0 * (1 + ecc * np.sin(np.deg2rad(w))) / np.sqrt(1 - ecc ** 2)
    tau = hypothesis.half_moon_orbit_transit_duration - hypothesis.half_moon_transit_duration
    assert tau == pytest.approx(a_moon / speed, rel=1e-6)


# --- G12 ---------------------------------------------------------------------------------------------------------

def test_dead_moon_orbit_fields_are_gone(tmp_path):
    emls = system(tmp_path, NEPTUNE, 3.9, 30.0)
    hypothesis = emls.search_input(5.0, 0.0, NEPTUNE, 10.0, 0.0)

    assert isinstance(hypothesis, SearchInput)
    for name in ('moon_inc', 'moon_ecc', 'moon_arg_periastron'):
        assert not hasattr(hypothesis, name)
        assert not hasattr(emls, name + '_grid')


# --- G3 / G4 / G14 -----------------------------------------------------------------------------------------------

def test_screening_model_warns_for_wide_moons_of_a_giant(tmp_path, caplog):
    jupiter = system(tmp_path, JUPITER, 11.2, 300.0)
    neptune = system(tmp_path, NEPTUNE, 3.9, 100.0, max_radius=0.4)

    with caplog.at_level(logging.INFO):
        jupiter.check_model_validity()
        neptune.check_model_validity()

    assert jupiter.model_validity['epsilon'] > 0.1
    assert neptune.model_validity['epsilon'] < 0.1
    # A 5.6 R_earth moon of Jupiter is about 28 M_earth (Chen & Kipping), q ~ 0.09.
    assert jupiter.model_validity['q'] == pytest.approx(0.087, abs=0.01)
    assert neptune.model_validity['q'] < 0.01
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum('Moon timing model' in message for message in warnings) == 1
    assert sum('q = M_m / M_p' in message for message in warnings) == 1


# --- G17 ---------------------------------------------------------------------------------------------------------

def test_neptune_at_30_days_always_overlaps_the_planet_transit(tmp_path):
    """Acceptance: the overlap fraction is 1.0 for Neptune at 30 d, where tau_max / T is about 0.3."""
    emls = system(tmp_path, NEPTUNE, 3.9, 30.0)
    widest = emls.max_moon_period(NEPTUNE)
    for period in (widest / 10, widest):
        for alpha in np.linspace(0, 2 * np.pi, 7, endpoint=False):
            hypothesis = emls.search_input(period, alpha, NEPTUNE, widest, 0.0)
            moon_t0s = [ExoMoonLeastSquares.compute_moon_t0(
                emls.planet_t0, t0, period, alpha,
                ExoMoonLeastSquares.compute_moon_ts(t0, hypothesis.half_moon_orbit_transit_duration,
                                                    hypothesis.half_moon_transit_duration))[0]
                for t0 in emls.bary_t0s]
            emls.assess_planet_overlap(moon_t0s, hypothesis.half_moon_orbit_transit_duration,
                                       hypothesis.half_moon_transit_duration)
            assert emls.planet_overlap_fraction == 1.0
        assert emls.period_constrained == bool(emls.best_tau >= emls.planet_duration / 4)
    emls.assess_planet_overlap(moon_t0s, hypothesis.half_moon_transit_duration * 1.05,
                               hypothesis.half_moon_transit_duration)
    assert emls.period_constrained is False


# --- G19 ---------------------------------------------------------------------------------------------------------

def test_default_period_grid_is_10000(tmp_path):
    emls = system(tmp_path, NEPTUNE, 3.9, 30.0)
    default = ExoMoonLeastSquares(str(tmp_path), 1, SUN, 1.0, [0.4, 0.2], 3.9, 0.39, NEPTUNE, 30.0, 0.0,
                                  emls.planet_duration, emls.planet_semimajor_axis, 90.0, 0.5, 0.0, 0.0, 90.0, 0.0,
                                  0.2, 2.0, emls.bary_t0s, emls.time, emls.flux, emls.flux_err, 5.5)

    assert default.period_grid_size == 10000
