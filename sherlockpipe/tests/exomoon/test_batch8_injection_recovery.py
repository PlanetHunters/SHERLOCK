import numpy as np

from sherlockpipe.exomoon import ExoMoonLeastSquares, InjectionTruthEvent


class _FakePandora:
    class model_params:
        pass

    class moon_model:
        def __init__(self, params):
            self.params = params

        def light_curve(self, time):
            moon = np.ones_like(time, dtype=float)
            # The middle injected epoch has no sampled occultation.  The last
            # event has one cadence only, so its coverage is explicitly partial.
            moon[np.isclose(time, 0.0)] = 0.999
            moon[np.isclose(time, 20.0)] = 0.998
            return moon, np.ones_like(time), moon


def _emls(time, flux=None):
    if flux is None:
        flux = np.ones_like(time)
    return ExoMoonLeastSquares(
        '', 1, 1.0, 1.0, [0.2, 0.1], 8.72, 0.8, 100.0, 10.0,
        0.0, 0.30, 12.0 / 215.032, 89.0, 0.5, 0.0, 0.3, 90.0, 0.0,
        0.3, 0.6, [0.0, 10.0, 20.0], time, flux, np.full(len(time), 2e-4),
        5.5, min_cadences_per_t0=1, skip_fold_fit=True,
    )


def _inject(emls, time, flux, cadence_ids=(40, 41, 42), epoch_ids=(100, 101, 102)):
    return emls._inject_moon_with_truth(
        time, flux, np.full(len(time), 2e-4), [0.0, 10.0, 20.0],
        emls.planet_semimajor_axis, emls.planet_ecc, 0.4, 2.0, 0.1,
        cadence_ids=cadence_ids, epoch_ids=epoch_ids)


def test_injection_is_pure_and_fresh_objects_have_independent_truth(monkeypatch):
    import sherlockpipe.exomoon as exomoon

    monkeypatch.setattr(exomoon, 'pandora', _FakePandora)
    time = np.array([0.0, 9.8, 20.0])
    flux = np.ones(3)
    first, second = _emls(time), _emls(time)
    original_time, original_flux = time.copy(), flux.copy()
    duration, epochs = first.planet_duration, list(first.bary_t0s)

    injected_one, truth_one = _inject(first, time, flux)
    injected_two, truth_two = _inject(second, time, flux)
    legacy_flux = first.inject_moon(
        time, flux, np.full(3, 2e-4), [0.0, 10.0, 20.0],
        first.planet_semimajor_axis, first.planet_ecc, 0.4, 2.0, 0.1)

    np.testing.assert_array_equal(time, original_time)
    np.testing.assert_array_equal(flux, original_flux)
    assert first.planet_duration == duration
    assert first.bary_t0s == epochs
    assert not hasattr(first, 'injected_flux_moon_pandora')
    np.testing.assert_array_equal(injected_one, injected_two)
    np.testing.assert_array_equal(injected_one, legacy_flux)
    assert truth_one['events'] == truth_two['events']
    assert truth_one is not truth_two


def test_truth_is_keyed_by_original_epoch_and_cadence_with_missing_and_partial_coverage(monkeypatch):
    import sherlockpipe.exomoon as exomoon

    monkeypatch.setattr(exomoon, 'pandora', _FakePandora)
    time = np.array([0.0, 9.8, 20.0])
    _, truth = _inject(_emls(time), time, np.ones(3))

    assert set(truth['events']) == {100, 101, 102}
    assert truth['events'][100].moon_t0_cadence_id == 40
    assert truth['events'][101].coverage == 'missing_coverage'
    assert truth['events'][101].cadence_ids == (41,)
    assert truth['events'][102].coverage == 'partial_coverage'
    assert truth['events'][102].moon_t0_cadence_id == 42


def test_recovery_matching_uses_epoch_ids_when_first_is_rejected():
    truth = {
        100: InjectionTruthEvent(100, 0.0, (40,), 0.0, 40, 'covered'),
        101: InjectionTruthEvent(101, 10.0, (41,), None, None, 'missing_coverage'),
        102: InjectionTruthEvent(102, 20.0, (42,), 20.0, 42, 'partial_coverage'),
    }
    errors, statuses = ExoMoonLeastSquares.match_recovered_events(
        truth, {102: 20.03}, {100: 'rejected', 101: 'missing_coverage', 102: 'partial_coverage'})

    assert set(errors) == {102}
    assert np.isclose(errors[102], 0.03)
    assert statuses == {100: 'rejected', 101: 'missing_coverage', 102: 'partial_coverage'}


def test_alias_units_and_phase_reference_transform_are_invariant():
    result = ExoMoonLeastSquares.assess_period_alias(2.0, 2.0, 10.0)
    assert result['comparison'] == 'absolute_period_difference_days / injected_period_days'
    assert result['relative_error'] == 0.0

    phase = 0.4
    reference_one, reference_two = 0.0, 5.0
    period, epoch, max_tau = 3.0, 20.0, 0.2
    shifted_phase = phase + 2 * np.pi * (reference_two - reference_one) / period
    moon_one, _ = ExoMoonLeastSquares.compute_moon_t0(reference_one, epoch, period, phase, max_tau)
    moon_two, _ = ExoMoonLeastSquares.compute_moon_t0(reference_two, epoch, period, shifted_phase, max_tau)
    assert moon_one == moon_two
