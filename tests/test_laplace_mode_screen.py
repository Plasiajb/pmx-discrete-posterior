import numpy as np
import pytest
from scipy.stats import norm

from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics.laplace import integrate


def flat_gaussian(initial_point=None):
    options = {"derivatives": {"backend_options": {"initial_step": 1e5}}}
    if initial_point is not None:
        options["optimization"] = {"initial_point": [initial_point]}
    return integrate(
        conditional_log_likelihood=lambda u: norm.logpdf(5e6, loc=u[0], scale=1e7),
        random_effect_distribution=RandomEffectDistribution([("iiv", None, "eta")], [[1e14]]),
        integration_method="laplace", numerical_options=options,
    )


def test_accurate_hessian_cannot_validate_a_displaced_mode_in_a_flat_gaussian():
    result = flat_gaussian()
    assert result["method_details"]["optimization"]["backend_success"]
    assert result["method_details"]["conditional_mode"] == [0.]
    assert result["status"] == "failed", result
    assert result["log_marginal_likelihood"] is None
    assert result["method_details"]["stage"] == "stationarity"
    assert result["method_details"]["estimated_log_integral_error_from_stationarity"] == pytest.approx(.0625, rel=1e-3)
    assert result["method_details"]["estimated_log_integral_error_from_hessian"] < 1e-3


def test_flat_gaussian_with_explicit_exact_mode_matches_analytic_integral():
    result = flat_gaussian(2.5e6)
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(norm.logpdf(5e6, scale=np.sqrt(2e14)), abs=1e-7, rel=0)
    assert result["method_details"]["estimated_log_integral_error_from_stationarity"] < 1e-3
