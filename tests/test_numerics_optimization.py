import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics.optimization import minimize


@pytest.mark.parametrize("dimension", [1, 2, 10])
@pytest.mark.parametrize("analytic", [False, True])
def test_correlated_quadratic(dimension, analytic):
    matrix = np.eye(dimension) + np.full((dimension, dimension), 0.2)
    mode = np.arange(dimension)/10 + 1.
    result = minimize(lambda x: (x-mode) @ matrix @ (x-mode) / 2 + 3,
                      np.zeros(dimension), gradient=(lambda x: matrix @ (x-mode)) if analytic else None)
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["minimizer"], mode, atol=1e-6)
    assert abs(result["objective_value"] - 3) < 1e-10
    assert "hess_inv" not in result


@pytest.mark.parametrize("sign", [-1, 1])
def test_nc01_no_artificial_five_eta_boundary(sign):
    # Exact archived objective, whose unconstrained mode lies outside [-5, 5].
    objective = lambda x: .5*((sign*12-x[0])**2+x[0]**2) + np.log(2*np.pi)
    result = minimize(objective, [0])
    assert result["status"] == "success", result
    np.testing.assert_allclose(result["minimizer"], [sign*6], atol=1e-6)


def test_false_success_from_bad_gradient_rejected():
    result = minimize(lambda x: (x[0]-6)**2, [5], gradient=lambda x: np.zeros(1))
    assert result["method_details"]["backend_success"] is True
    assert result["status"] == "failed"
    assert "stationarity" in result["failure_reason"]
    assert result["minimizer"] is None


def test_flat_rounding_cannot_certify_stationarity():
    result = minimize(lambda x: 1e20 + (x[0]-1)**2, [0])
    assert result["status"] == "failed"
    assert result["minimizer"] is None


def test_iteration_limit_not_success():
    result = minimize(lambda x: (x[0]-6)**2, [0], {"backend_options": {"maxiter": 1}})
    assert result["status"] == "failed"


@pytest.mark.parametrize("options", [
    {"method": "bounded"}, {"bounds": [-5, 5]}, {"initial_point": [0]},
    {"backend_options": {"unknown": 3}}, {"backend_options": {"c2": 1}},
    {"backend_options": {"maxiter": True}}, {"stationarity_tolerance": 0},
    {"backend_options": {"gtol": np.nan}},
])
def test_bad_options(options):
    with pytest.raises(InputError):
        minimize(lambda x: x @ x, [1], options)


def test_ignored_gradient_option_rejected():
    with pytest.raises(InputError, match="ignored"):
        minimize(lambda x: x @ x, [1], {"backend_options": {"finite_diff_rel_step": 1e-3}}, gradient=lambda x: 2*x)


@pytest.mark.parametrize("objective", [lambda x: np.nan, lambda x: np.inf, lambda x: [1]])
def test_bad_objective(objective):
    result = minimize(objective, [0])
    assert result["status"] == "failed"
    assert result["objective_value"] is None


def test_gradient_wrong_dimension_and_failed_node():
    result = minimize(lambda x: x @ x, [1, 2], gradient=lambda x: [0])
    assert result["status"] == "failed"
    result = minimize(lambda x: 0. if x[0] == 0 else np.nan, [0])
    assert result["status"] == "failed"
