import numpy as np
import pytest
import batman

from sherlockpipe.exomoon import ExoMoonLeastSquares


def make_params(a=15.0, inc=90.0, rp=0.1, period=10.0, ecc=0.0, w=90.0):
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = period
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.w = w
    params.u = []
    params.limb_dark = "uniform"
    return params


def dense_batman_contact_duration(params, samples=400001):
    """Independently locate BATMAN's sampled first/fourth contact interval."""
    time = np.linspace(-params.per / 4, params.per / 4, samples)
    model = batman.TransitModel(params, time).light_curve(params)
    in_transit = np.flatnonzero(model < 1 - 1e-14)
    if not len(in_transit):
        return 0.0, time[1] - time[0]
    return time[in_transit[-1]] - time[in_transit[0]], time[1] - time[0]


def test_circular_central_contact_duration_matches_analytic_and_batman():
    params = make_params(a=20, inc=90, rp=0.12, period=12)
    expected = params.per / np.pi * np.arcsin((1 + params.rp) / params.a)
    duration = ExoMoonLeastSquares.contact_duration(params)
    dense_duration, step = dense_batman_contact_duration(params)

    assert duration == pytest.approx(expected, abs=1e-12)
    assert duration == pytest.approx(dense_duration, abs=2 * step)


def test_nonzero_impact_and_grazing_contacts_match_batman():
    for impact_parameter in (0.55, 1.05):
        params = make_params(a=20, rp=0.1, period=12)
        params.inc = ExoMoonLeastSquares.inclination_from_impact_parameter(
            params.a, impact_parameter)
        expected = params.per / np.pi * np.arcsin(np.sqrt(
            ((1 + params.rp) ** 2 - impact_parameter ** 2)
            / (params.a ** 2 - impact_parameter ** 2)))
        duration = ExoMoonLeastSquares.contact_duration(params)
        dense_duration, step = dense_batman_contact_duration(params)

        assert duration == pytest.approx(expected, abs=1e-12)
        assert duration == pytest.approx(dense_duration, abs=2 * step)


@pytest.mark.parametrize("omega", (0.0, 90.0, 270.0))
def test_eccentric_contacts_follow_batman_conjunction_convention(omega):
    params = make_params(a=25, rp=0.08, period=15, ecc=0.35, w=omega)
    params.inc = ExoMoonLeastSquares.inclination_from_impact_parameter(
        params.a, 0.45, params.ecc, params.w)

    duration = ExoMoonLeastSquares.contact_duration(params)
    dense_duration, step = dense_batman_contact_duration(params)

    assert duration > 0
    assert duration == pytest.approx(dense_duration, abs=2 * step)


def test_impact_parameter_conversion_is_eccentricity_consistent():
    inclination = ExoMoonLeastSquares.inclination_from_impact_parameter(18, 0.7, 0.3, 270)
    assert ExoMoonLeastSquares.impact_parameter(18, inclination, 0.3, 270) == pytest.approx(0.7)


def test_contact_duration_is_independent_of_missing_ingress_samples():
    params = make_params(a=18, inc=87.5, rp=0.1, period=9)
    complete_time = np.linspace(-0.2, 0.2, 401)
    missing_ingress_time = complete_time[complete_time > -0.02]
    complete_span = np.ptp(complete_time[batman.TransitModel(params, complete_time).light_curve(params) < 1])
    missing_span = np.ptp(missing_ingress_time[
        batman.TransitModel(params, missing_ingress_time).light_curve(params) < 1])

    assert missing_span < complete_span
    assert ExoMoonLeastSquares.contact_duration(params) > complete_span
    assert ExoMoonLeastSquares.contact_duration(params) == pytest.approx(
        ExoMoonLeastSquares.contact_duration(params))


def test_no_transit_and_invalid_geometry_are_explicit():
    no_transit = make_params(a=15, inc=80, rp=0.1)
    assert ExoMoonLeastSquares.contact_duration(no_transit) == 0

    invalid = make_params()
    invalid.ecc = 1.0
    with pytest.raises(ValueError, match="valid elliptic orbit"):
        ExoMoonLeastSquares.contact_duration(invalid)


@pytest.mark.parametrize("star_radius", (0.3, 1.8))
def test_circular_orbit_window_inverse_preserves_units(star_radius):
    a_over_rstar = 22.0
    period = 11.0
    impact_parameter = 0.6
    moon_offset_rsun = 0.45 * star_radius
    target_duration = ExoMoonLeastSquares.compute_transit_duration(
        star_radius, a_over_rstar * star_radius, period, moon_offset_rsun,
        impact_parameter)
    min_period = ExoMoonLeastSquares.min_period_for_orbit_duration(
        star_radius, a_over_rstar, period, 20 * 3.003e-6, target_duration,
        impact_parameter)
    recovered_offset_rsun = ExoMoonLeastSquares.compute_semimajor_axis(
        20 * 3.003e-6, min_period) * 215.032
    recovered_duration = ExoMoonLeastSquares.compute_transit_duration(
        star_radius, a_over_rstar * star_radius, period, recovered_offset_rsun,
        impact_parameter)

    assert min_period > 0
    assert recovered_duration == pytest.approx(target_duration, rel=2e-3)
