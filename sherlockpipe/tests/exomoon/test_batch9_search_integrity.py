import numpy as np
import pytest

from sherlockpipe.exomoon import ExoMoonLeastSquares


def _residuals(models, flux=(1.0, 0.96, 1.0)):
    return ExoMoonLeastSquares.calculate_residuals(
        np.array([-0.1, 0.0, 0.1]), np.asarray(flux), np.full(3, 0.01),
        np.array([]), np.array([]), np.array([]), 1.0, 3, models,
        0.2, 0.5, 1.0, 3,
    )


def test_wrapped_phase_minimum_is_refined_only_with_a_convex_bracket(monkeypatch):
    alpha = np.arange(4) * np.pi / 2
    # Around phase zero, the unwrapped samples are at -pi/2, 0, +pi/2.
    residuals = np.array([[0.01], [(np.pi / 2 + 0.1) ** 2], [4.0],
                          [(np.pi / 2 - 0.1) ** 2]])
    refined = ExoMoonLeastSquares.refine_alpha(alpha, residuals)

    assert refined[0, 0] == pytest.approx(2 * np.pi - 0.1)
    assert refined[0, 1] == pytest.approx(0.01)

    flat = ExoMoonLeastSquares.refine_alpha(alpha, np.ones((4, 1)))
    assert flat[0, 0] == pytest.approx(0.0)
    assert flat[0, 1] == pytest.approx(1.0)

    # The local-minimum check alone is not a curvature guarantee.
    monkeypatch.setattr(np, 'polyfit', lambda *_: np.array([-1.0, 0.0, 0.0]))
    concave = ExoMoonLeastSquares.refine_alpha(alpha, residuals)
    assert concave[0, 0] == pytest.approx(0.0)


def test_phase_refinement_rejects_a_worse_evaluated_proposal():
    calls = []

    def objective(alpha):
        calls.append(alpha)
        return 3.0

    alpha, score = ExoMoonLeastSquares.accept_refined_alpha(0.0, 2.0, 2 * np.pi - 0.1, objective)
    assert calls == [pytest.approx(2 * np.pi - 0.1)]
    assert alpha == 0.0
    assert score == 2.0


def test_radius_template_selection_is_bounded_and_does_not_turn_mean_deficit_into_radius():
    template_time = np.array([-0.1, 0.0, 0.1])
    lower = (0.2, template_time, np.array([1.0, 0.98, 1.0]))
    upper = (0.5, template_time, np.array([1.0, 0.96, 1.0]))
    outside = (0.7, template_time, upper[2])
    alternative, null, radius, model = _residuals([outside, lower, upper])

    assert alternative < null
    assert radius == 0.5
    np.testing.assert_allclose(model, upper[2])

    alternative, null, radius, _ = _residuals([lower], flux=lower[2])
    assert alternative < null
    assert radius == 0.2


def test_no_dip_and_nonfinite_template_objectives_fall_back_to_the_null():
    flat = (0.3, np.array([-0.1, 0.0, 0.1]), np.ones(3))
    alternative, null, radius, model = _residuals([flat])
    assert alternative == null
    assert radius == 0.2
    np.testing.assert_allclose(model, 1.0)

    nonfinite = (0.3, np.array([-0.1, 0.0, 0.1]), np.array([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        _residuals([nonfinite])


def test_invalid_radius_and_period_grid_sizes_are_rejected():
    args = [
        '', 1, 1.0, 1.0, [0.2, 0.1], 8.0, 0.2, 100.0, 10.0,
        0.0, 0.2, 12.0 / 215.032, 89.0, 0.5, 0.0, 0.3, 90.0, 0.0,
        0.2, 0.5, [0.0], np.array([0.0]), np.array([1.0]), np.array([0.01]), 5.0,
    ]
    with pytest.raises(ValueError, match="period grid"):
        ExoMoonLeastSquares(*args, period_grid_size=1)
    with pytest.raises(ValueError, match="Moon radii"):
        ExoMoonLeastSquares(*args, radius_grid_size=0)


def test_tiny_flat_and_invalid_spectra_are_safe_or_rejected_explicitly():
    sr, _, power, raw_sde, sde = ExoMoonLeastSquares.spectra([4.0])
    np.testing.assert_allclose(sr, [1.0])
    np.testing.assert_allclose(power, [0.0])
    assert raw_sde == sde == 0.0

    _, _, power, raw_sde, sde = ExoMoonLeastSquares.spectra([4.0, 4.0])
    np.testing.assert_allclose(power, 0.0)
    assert raw_sde == sde == 0.0
    with pytest.raises(ValueError, match="finite"):
        ExoMoonLeastSquares.spectra([1.0, np.nan], kernel_size=3)


def test_period_indexed_labels_preserve_evaluated_candidate_identity():
    candidates = ExoMoonLeastSquares.spectrum_candidates(
        [2.0, 5.0], [[0.1, -4.0], [0.2, -1.0]], [0.25, 0.45])

    assert candidates[1] == {
        'period_days': 5.0, 'moon_alpha_radians': 0.2,
        'score': -1.0, 'radius_earth': 0.45,
    }
    with pytest.raises(ValueError, match="aligned"):
        ExoMoonLeastSquares.spectrum_candidates([2.0], [[0.1, -1.0], [0.2, -2.0]], [0.3])
