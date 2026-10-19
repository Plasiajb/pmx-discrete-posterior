import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics.derivatives import _EvaluationError
from pmx_discrete_posterior.numerics.laplace import _curvature_terms


@pytest.mark.parametrize("matrix", [np.eye(1), [[2., .3], [.3, 4.]], np.diag([1e300, 1e-300])])
def test_logdet_is_stable_without_determinant_or_inverse(matrix):
    matrix = np.asarray(matrix)
    logdet, estimate, mode_error = _curvature_terms(matrix, np.zeros_like(matrix), 0.)
    assert logdet == pytest.approx(np.linalg.slogdet(matrix)[1], abs=1e-12, rel=0)
    assert 0 <= estimate < 1e-3
    assert mode_error == 0.


@pytest.mark.parametrize("matrix", [np.diag([-1., -1.]), np.diag([1., 0.]), [[1., 2.], [2., 1.]],
                                   [[1., 1.], [1., 1.]], [[1., .5], [.2, 1.]], [[1., math.nan], [0., 1.]]])
def test_nonpositive_singular_asymmetric_and_nonfinite_curvature_is_rejected(matrix):
    original = np.array(matrix)
    with pytest.raises((InputError, _EvaluationError, np.linalg.LinAlgError)):
        _curvature_terms(matrix, np.zeros_like(matrix), 0.)
    np.testing.assert_equal(matrix, original)


@pytest.mark.parametrize("error", [None, [[1e-6]], [[-1., 0], [0, 0]], [[0, 1e-8], [0, 0]], [[0, math.inf], [math.inf, 0]]])
def test_error_must_be_present_compatible_symmetric_and_nonnegative(error):
    with pytest.raises((InputError, _EvaluationError)):
        _curvature_terms(np.eye(2), error, 0.)


def test_positive_definiteness_alone_cannot_accept_uncertain_log_integral():
    matrix = np.array([[1., 1 - 1e-7], [1 - 1e-7, 1.]])
    assert np.linalg.eigvalsh(matrix)[0] > 0
    with pytest.raises(_EvaluationError, match="exceeds 1e-3"):
        _curvature_terms(matrix, np.full((2, 2), 1e-8), 0.)
    with pytest.raises(_EvaluationError, match="does not resolve"):
        _curvature_terms(matrix, np.full((2, 2), 1e-6), 0.)


def test_error_screen_matches_scalar_logdet_sensitivity():
    _, estimate, _ = _curvature_terms([[2.]], [[1e-4]], 0.)
    exact_loss = .5 * abs(math.log((2 - 1e-4) / 2))
    assert estimate >= exact_loss - 1e-16
    assert estimate < 1e-3


def test_componentwise_error_envelope_covers_deterministic_matrix_perturbations():
    matrix = np.array([[3., .2, -.1], [.2, 1.5, .4], [-.1, .4, 2.]])
    envelope = np.full((3, 3), 1e-5)
    logdet, estimate, _ = _curvature_terms(matrix, envelope, 0.)
    rng = np.random.default_rng(14)
    for _ in range(20):
        delta = rng.uniform(-1, 1, (3, 3)) * envelope
        delta = (delta + delta.T) / 2
        actual = .5 * abs(np.linalg.slogdet(matrix + delta)[1] - logdet)
        assert actual <= estimate


@pytest.mark.parametrize("dimension", [1, 3, 10])
def test_diagonal_envelope_gate_agrees_with_exact_worst_case(dimension):
    threshold = -math.expm1(-.002/dimension)
    matrix = np.eye(dimension)
    small = .99*threshold
    _, estimate, _ = _curvature_terms(matrix, small*matrix, 0.)
    exact = -.5*dimension*math.log1p(-small)
    assert exact <= estimate < 1e-3
    with pytest.raises(_EvaluationError, match="exceeds 1e-3"):
        _curvature_terms(matrix, 1.01*threshold*matrix, 0.)


def test_error_envelope_uses_matrix_norm_not_largest_entry():
    with pytest.raises(_EvaluationError, match="exceeds 1e-3"):
        _curvature_terms(np.eye(10), np.full((10, 10), 1e-4), 0.)


def test_entrywise_nonnegative_error_envelope_need_not_be_positive_definite():
    error = np.array([[0., 1e-5], [1e-5, 0.]])
    assert np.linalg.eigvalsh(error)[0] < 0
    _, estimate, _ = _curvature_terms(np.eye(2), error, 0.)
    assert 0 < estimate < 1e-3


def test_diagonal_scaling_handles_large_scale_range_with_relative_uncertainty():
    matrix = np.diag([1e-300, 1e300])
    logdet, estimate, mode_error = _curvature_terms(matrix, 1e-5*matrix, 0.)
    assert logdet == pytest.approx(0., abs=1e-12, rel=0)
    assert 0 < estimate < 1e-3
    assert mode_error == 0.


def test_positive_diagonal_and_positive_determinant_do_not_prove_positive_curvature():
    matrix = 2*np.ones((3, 3))-np.eye(3)
    assert np.linalg.det(matrix) > 0 and np.all(np.diag(matrix) > 0)
    with pytest.raises(np.linalg.LinAlgError):
        _curvature_terms(matrix, np.zeros_like(matrix), 0.)


@pytest.mark.parametrize("gradient", [None, True, -1., math.nan, math.inf])
def test_mode_error_requires_a_real_finite_nonnegative_gradient_envelope(gradient):
    with pytest.raises(InputError):
        _curvature_terms(np.eye(1), np.zeros((1, 1)), gradient)


def test_mode_error_accounts_for_curvature_uncertainty():
    _, _, estimate = _curvature_terms([[2.]], [[1e-4]], 1e-3)
    assert estimate >= .5*1e-6/(2-1e-4)


def test_nonfinite_mode_error_is_rejected():
    with pytest.raises((OverflowError, FloatingPointError, _EvaluationError)):
        _curvature_terms([[1e-300]], [[0.]], 1e300)


def test_mode_error_does_not_square_tiny_gradient_before_scaling():
    _, _, estimate = _curvature_terms([[np.nextafter(0., 1.)]], [[0.]], 1e-163)
    assert .001 < estimate < .0011
