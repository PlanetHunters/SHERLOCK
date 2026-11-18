"""The t0 prior measured from the planet mid-times (timing_prior.py, 2026-11-17).

The prior used to be only the TTV of the largest searched moon, a fraction of a minute for small
moon radii, so a 30-min TTV from another planet was pulled back to the linear ephemeris and left a
residual at ingress and egress. It is now centred on a timing model of the mid-times measured
without a prior and widened by their scatter beyond the timing precision.
"""
import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares
from sherlockpipe.tests.exomoon.test_audit_fixes import PERIOD
from sherlockpipe.tests.exomoon.test_planet_fit import CADENCE, curve, emls_for, planet_params, stacked_residual_peak, t14

MINUTE = 1 / 1440
TINY_MOON_TTV = 0.1 * MINUTE  # the moon floor of a search with small moon radii


def sinusoid(epochs, amplitude, period):
    return amplitude * np.sin(2 * np.pi * np.arange(epochs) * PERIOD / period)


# --- the statistics ------------------------------------------------------------------------------------------------

def test_no_excess_scatter_when_the_times_match_their_precision():
    rng = np.random.default_rng(1)
    precisions = np.full(400, 0.5 * MINUTE)
    assert ExoMoonLeastSquares.excess_scatter(rng.normal(0, 1, 400) * precisions * 0.9, precisions, 400) == 0.0


def test_excess_scatter_recovers_the_extra_variation():
    rng = np.random.default_rng(2)
    precisions = np.full(2000, 0.5 * MINUTE)
    oc = rng.normal(0, 1, 2000) * np.hypot(precisions, 2 * MINUTE)
    assert ExoMoonLeastSquares.excess_scatter(oc, precisions, 2000) == pytest.approx(2 * MINUTE, rel=0.05)


@pytest.mark.parametrize('kind', ['none', 'linear', 'sinusoid'])
def test_the_timing_model_is_chosen_by_bic(kind):
    epochs = 30
    rng = np.random.default_rng(3)
    numbers = np.arange(epochs, dtype=float)
    times = numbers * PERIOD
    precisions = np.full(epochs, 0.5 * MINUTE)
    signal = {"none": np.zeros(epochs), "linear": (5 + 0.8 * numbers) * MINUTE,
              "sinusoid": sinusoid(epochs, 20 * MINUTE, 7.3 * PERIOD)}[kind]
    oc = signal + rng.normal(0, 1, epochs) * precisions

    model = ExoMoonLeastSquares.fit_timing_model(times, numbers, oc, precisions, times, numbers)

    assert model["name"] == kind
    assert np.max(np.abs(model["centres"] - signal)) < 1.5 * MINUTE
    if kind == "sinusoid":
        assert model["period"] == pytest.approx(7.3 * PERIOD, rel=0.03)


def test_an_epoch_far_from_the_others_is_refitted_out():
    epochs = 20
    rng = np.random.default_rng(4)
    numbers = np.arange(epochs, dtype=float)
    precisions = np.full(epochs, 0.5 * MINUTE)
    oc = rng.normal(0, 1, epochs) * precisions
    oc[7] = 60 * MINUTE  # an epoch whose fit ran away, such as one with partial coverage

    model = ExoMoonLeastSquares.fit_timing_model(numbers * PERIOD, numbers, oc, precisions, numbers * PERIOD, numbers)

    assert not model["used"][7] and np.sum(model["used"]) == epochs - 1
    assert model["name"] == "none" and model["excess"] < 0.5 * MINUTE


# --- the planet stage ----------------------------------------------------------------------------------------------

def strong_ttv_fixture(noise_snr=30, epochs=20, seed=5):
    truth = planet_params()
    noise = np.sqrt(t14(truth) / CADENCE) * truth.rp ** 2 / noise_snr
    shifts = sinusoid(epochs, 30 * MINUTE, 7.3 * PERIOD)
    return truth, shifts, curve(truth, epochs=epochs, noise=noise, shifts=shifts, seed=seed)


def test_a_30_minute_ttv_is_followed_with_a_tiny_moon_floor(tmp_path):
    truth, shifts, data = strong_ttv_fixture()
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)

    result = emls.subtract_planet_transit(*data[1:])

    precision = np.median(emls.ttv_precisions)
    # The fitted times are as good as one epoch's own precision (about 2 min here), not 30 min off.
    assert np.sqrt(np.mean(np.square(np.array(emls.ttvs) - shifts))) < 1.5 * precision
    assert emls.ttv_model["name"] == "sinusoid"
    assert emls.ttv_model["period"] == pytest.approx(7.3 * PERIOD, rel=0.05)
    moon_only = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV,
                         ttv_prior_from_data=False)
    moon_only_result = moon_only.subtract_planet_transit(*data[1:])
    assert stacked_residual_peak(emls, result, data[0]) < 0.5 * stacked_residual_peak(moon_only, moon_only_result,
                                                                                      data[0])
    assert len(emls.ttv_precisions) == len(emls.ttv_measured) == len(emls.ttv_prior_sigmas) == len(emls.bary_t0s)


def test_the_moon_only_prior_pulls_the_same_ttv_back(tmp_path):
    """The old behaviour, kept with ttv_prior_from_data=False: the fit stays near the ephemeris."""
    truth, shifts, data = strong_ttv_fixture()
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV,
                    ttv_prior_from_data=False)

    result = emls.subtract_planet_transit(*data[1:])

    assert np.median(np.abs(np.array(emls.ttvs) - shifts)) > 5 * MINUTE
    assert stacked_residual_peak(emls, result, data[0]) > 0.05 * truth.rp ** 2


def test_low_snr_epochs_follow_the_timing_model(tmp_path):
    """At SNR 8 per epoch the prior still matters: centred on the measured trend, it gives better times
    than no prior (the epochs follow the noise) and than the moon-only prior (they stay on the ephemeris)."""
    truth, shifts, data = strong_ttv_fixture(noise_snr=8, epochs=30, seed=6)

    def timing_error(**kwargs):
        emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, min_transit_completeness=0, **kwargs)
        emls.subtract_planet_transit(*data[1:])
        assert len(emls.ttvs) == len(shifts)
        return np.sqrt(np.mean(np.square(np.array(emls.ttvs) - shifts))), emls

    measured, emls = timing_error(ttv_prior_width=TINY_MOON_TTV)
    free, _ = timing_error(ttv_prior_width=None)
    moon_only, _ = timing_error(ttv_prior_width=TINY_MOON_TTV, ttv_prior_from_data=False)

    assert emls.ttv_model["name"] == "sinusoid"
    assert measured < 0.7 * free
    assert measured < 0.7 * moon_only


def test_no_timing_variation_keeps_the_prior_on_the_ephemeris(tmp_path):
    truth = planet_params()
    noise = np.sqrt(t14(truth) / CADENCE) * truth.rp ** 2 / 10
    data = curve(truth, epochs=20, noise=noise, seed=7)
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)

    emls.subtract_planet_transit(*data[1:])

    precision = np.median([p for p in emls.ttv_precisions if p is not None])
    assert emls.ttv_model["name"] == "none"
    # The free fits scatter somewhat more than the Carter et al. precision; the excess absorbs it.
    assert emls.ttv_excess_used < 2 * precision
    assert np.max(np.abs(emls.ttv_prior_centres)) == 0.0


def test_unmeasurable_epochs_keep_the_ephemeris_prior(tmp_path):
    """At SNR 3 per epoch the timing precision is about the ingress, so no timing model is fitted and
    the prior stays on the linear ephemeris with the moon floor (test_low_snr_epochs_stay_near_the_ephemeris)."""
    truth, shifts, data = strong_ttv_fixture(noise_snr=3, epochs=20, seed=8)
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, min_transit_completeness=0,
                    ttv_prior_width=TINY_MOON_TTV)

    emls.subtract_planet_transit(*data[1:])

    assert emls.ttv_model is None and emls.ttv_excess_used == 0.0
    assert np.all(np.array(emls.ttv_prior_centres) == 0.0)


def test_a_drifting_ephemeris_is_followed(tmp_path):
    """A catalog ephemeris off by 1.5 min per epoch: the linear timing model recentres the prior."""
    truth = planet_params()
    noise = np.sqrt(t14(truth) / CADENCE) * truth.rp ** 2 / 30
    shifts = 1.5 * MINUTE * np.arange(20)
    data = curve(truth, epochs=20, noise=noise, shifts=shifts, seed=9)
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)

    emls.subtract_planet_transit(*data[1:])

    assert emls.ttv_model["name"] == "linear"
    assert emls.ttv_model["coefficients_min"][1] == pytest.approx(1.5, abs=0.15)
    assert np.sqrt(np.mean(np.square(np.array(emls.ttvs) - shifts))) < 1.5 * np.median(emls.ttv_precisions)


# --- the moon windows follow the timing model (move_to_barycentre_frame) ------------------------------------------

def test_the_subtracted_epochs_move_to_the_frame_of_the_linear_ephemeris(tmp_path):
    """A 30-min sinusoidal TTV moves the whole system: after the planet stage each epoch is shifted back by its
    fitted planet mid-time, so the planet of the moon stage sits on the linear ephemeris."""
    truth, shifts, data = strong_ttv_fixture()
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)

    result = emls.subtract_planet_transit(*data[1:])

    # The fitted planet times, which the search's moon offsets are measured from (close to the model here).
    assert np.allclose(emls.barycentre_shifts, emls.ttvs)
    assert np.allclose(emls.barycentre_shifts, emls.ttv_prior_centres, atol=1 * MINUTE)
    mid_times = np.array(emls.planet_mid_times()) - np.array(emls.bary_t0s)
    assert np.max(np.abs(mid_times)) < 1e-9  # was up to 30 min
    assert len(result[0]) == len(data[1])  # subtract mode keeps every point, in order
    for t0, shift in zip(emls.bary_t0s, emls.barycentre_shifts):
        near = np.abs(data[1] - t0) < 1
        assert np.allclose(result[0][near] - data[1][near], -shift, atol=1e-9)


def test_without_a_timing_model_the_times_stay(tmp_path):
    truth = planet_params()
    noise = np.sqrt(t14(truth) / CADENCE) * truth.rp ** 2 / 10
    data = curve(truth, epochs=20, noise=noise, seed=7)
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)

    result = emls.subtract_planet_transit(*data[1:])

    assert emls.ttv_model["name"] == "none"
    assert not np.any(emls.barycentre_shifts)
    assert np.array_equal(result[0], data[1])


def test_the_shift_can_be_switched_off(tmp_path):
    truth, shifts, data = strong_ttv_fixture()
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV,
                    moon_windows_follow_ttv=False)

    result = emls.subtract_planet_transit(*data[1:])

    assert not np.any(emls.barycentre_shifts)
    assert np.array_equal(result[0], data[1])
    assert np.max(np.abs(np.array(emls.planet_mid_times()) - np.array(emls.bary_t0s))) > 20 * MINUTE


def test_the_planet_columns_follow_the_shifted_epochs(tmp_path):
    """The depth, mid-time and shape columns the joint fit uses are centred on the planet in the shifted
    frame (the mid-time column was once left on the unshifted times, and the joint fit then took 16 % of
    a moon on KOI-94.01 with a 60-min TTV)."""
    truth, shifts, data = strong_ttv_fixture()
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)
    time = emls.subtract_planet_transit(*data[1:])[0]

    depth = np.abs(emls.planet_depth_derivative(time))
    t0_column, epoch_index = emls.planet_t0_derivative(time)
    shape = np.abs(emls.planet_shape_derivatives(time)).sum(axis=1)
    for index, t0 in enumerate(emls.bary_t0s):
        near = np.abs(time - t0) < 1
        for column in (depth[near], np.abs(t0_column[near]), shape[near]):
            centre = np.sum(time[near] * column) / np.sum(column)
            assert abs(centre - t0) < 5 * MINUTE  # the bug put them up to 30 min off
        assert np.all(epoch_index[near & (np.abs(t0_column) > 0)] == index)


def test_the_pre_pass_keeps_the_timing_of_the_planet_itself(tmp_path):
    """ttv_timing_model is fitted after the windows follow the pre-pass offsets, so it only holds the
    residual timing; the pre-pass summary keeps the measured variation of the planet (2026-11-18)."""
    truth, shifts, data = strong_ttv_fixture()
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)

    emls.measure_epoch_offsets(*data[1:])

    timing = emls.prepass_timing
    assert timing["model"]["name"] == "sinusoid"
    assert timing["model"]["period"] == pytest.approx(7.3 * PERIOD, rel=0.05)
    assert len(timing["t0s"]) == len(timing["measured_minutes"]) == len(timing["precision_minutes"])
    measured = np.array(timing["measured_minutes"], dtype=float)
    assert np.nanmax(np.abs(measured)) > 20  # the 30-min variation, in minutes
    assert len(timing["offsets_minutes"]) == len(timing["offsets_t0s"]) == len(emls.bary_t0s)
    assert np.allclose(np.array(timing["offsets_minutes"]) / 1440, [emls.epoch_offsets[n] for n in
                       sorted(emls.epoch_offsets)])
    assert timing["offsets_clipped"] is False


def test_the_pre_pass_reports_no_timing_model(tmp_path):
    truth = planet_params()
    noise = np.sqrt(t14(truth) / CADENCE) * truth.rp ** 2 / 10
    data = curve(truth, epochs=20, noise=noise, seed=7)
    emls = emls_for(tmp_path, data, t14(truth), skip_fold_fit=True, ttv_prior_width=TINY_MOON_TTV)

    emls.measure_epoch_offsets(*data[1:])

    assert emls.epoch_offsets is None
    assert emls.prepass_timing["model"]["name"] == "none"
    assert emls.prepass_timing["offsets_minutes"] is None
