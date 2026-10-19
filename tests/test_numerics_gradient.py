import numpy as np
import pytest
from scipy.optimize import rosen, rosen_der

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics.derivatives import gradient


@pytest.mark.parametrize("dimension", [1, 2, 10])
def test_quadratic_gradient(dimension):
    matrix = np.eye(dimension) + np.full((dimension, dimension), 0.2)
    point = np.arange(dimension) / 10 + 0.3
    result = gradient(lambda x: x @ matrix @ x / 2, point)
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["gradient"], matrix @ point, atol=1e-8)
    assert result["estimated_error"].shape == (dimension,)


def test_nonlinear_gradient():
    point = np.array([0.2, 0.4, 0.6])
    result = gradient(rosen, point)
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["gradient"], rosen_der(point), atol=1e-7)


def test_common_offset_and_old_boundary_gradient():
    result = gradient(lambda x: 5e7 + (x[0] - 6) ** 2, [5])
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["gradient"], [-2], atol=1e-6)


def test_unresolvable_gradient_is_not_accepted_as_zero():
    result = gradient(lambda x: 1e20 + x[0] ** 2, [0])
    assert result["status"] == "failed"
    assert result["gradient"] is None


@pytest.mark.parametrize("options", [
    {"backend": "unknown"}, {"wat": 1}, {"atol": -1}, {"rtol": True},
    {"atol": 0, "rtol": 0}, {"backend_options": {"initial_step": 0}},
    {"backend_options": {"maxiter": 1}}, {"backend_options": {"order": 3}},
    {"backend_options": {"tolerances": {}}}, {"backend_options": {"step_factor": 1}},
])
def test_bad_options(options):
    with pytest.raises(InputError):
        gradient(lambda x: x @ x, [0], options)


@pytest.mark.parametrize("point", [[], [np.nan], [True], [[0]], ["0"]])
def test_bad_point(point):
    with pytest.raises(InputError):
        gradient(lambda x: x @ x, point)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, [1], True, 1j])
def test_failed_objective_not_hidden(bad_value):
    result = gradient(lambda x: bad_value, [0])
    assert result["status"] == "failed"
    assert result["gradient"] is None
    assert "objective failed" in result["failure_reason"]


def test_mutation_and_nondeterminism_rejected():
    def mutate(x):
        x[0] += 1
        return x[0]
    assert gradient(mutate, [0])["status"] == "failed"
    values = iter([1., 2.])
    assert gradient(lambda x: next(values), [0])["status"] == "failed"
