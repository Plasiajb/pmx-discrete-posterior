import numpy as np
import pytest
from scipy.optimize import rosen, rosen_hess

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics.derivatives import hessian


@pytest.mark.parametrize("dimension", [1, 2, 10])
def test_quadratic_hessian(dimension):
    matrix = np.diag(np.arange(dimension) + 1.) + np.full((dimension, dimension), 0.2)
    point = np.arange(dimension) / 10 + 0.3
    result = hessian(lambda x: x @ matrix @ x / 2, point)
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["hessian"], matrix, atol=1e-7)
    assert result["estimated_error"].shape == (dimension, dimension)


def test_nonlinear_hessian():
    point = np.array([0.2, 0.4, 0.6])
    result = hessian(rosen, point)
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["hessian"], rosen_hess(point), atol=1e-6)


def test_indefinite_hessian_remains_indefinite():
    result = hessian(lambda x: x[0] ** 2 - x[1] ** 2, [0, 0])
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["hessian"], np.diag([2, -2]), atol=1e-8)


@pytest.mark.parametrize("offset", [1e16, 1e20])
def test_rounded_flat_hessian_is_rejected(offset):
    result = hessian(lambda x: offset + x[0] ** 2, [0])
    assert result["status"] == "failed", result
    assert result["hessian"] is None


def test_zero_function_has_zero_hessian():
    result = hessian(lambda x: 0., [0])
    assert result["status"] == "success", result
    np.testing.assert_array_equal(result["hessian"], [[0]])


def test_tiny_step_cannot_hide_positive_curvature_loss():
    result = hessian(lambda x: 5e7 + 1.5*x[0]**2, [0], {"backend_options": {"initial_step": 1e-4}})
    assert result["status"] == "failed", result
    assert result["hessian"] is None


@pytest.mark.parametrize("objective", [lambda x: np.nan, lambda x: [1], lambda x: np.inf])
def test_failure_propagates(objective):
    result = hessian(objective, [0])
    assert result["status"] == "failed"
    assert result["hessian"] is None


def test_nested_rtol_validation():
    with pytest.raises(InputError, match="rtol"):
        hessian(lambda x: x @ x, [0], {"rtol": 0})
