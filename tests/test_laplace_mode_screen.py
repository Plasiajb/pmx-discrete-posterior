import numpy as np
import pytest
from scipy.stats import norm

from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics.laplace import integrate
from pmx_discrete_posterior.numerics import laplace


def flat_gaussian(initial_point=None):
    # 步长现在属于标准化坐标，使用默认值 / Steps now use standardized coordinates; use defaults.
    options = {}
    if initial_point is not None:
        options["optimization"] = {"initial_point": [initial_point]}
    return integrate(
        conditional_log_likelihood=lambda u: norm.logpdf(5e6, loc=u[0], scale=1e7),
        random_effect_distribution=RandomEffectDistribution([("iiv", None, "eta")], [[1e14]]),
        integration_method="laplace", numerical_options=options,
    )


def test_standardization_resolves_the_former_flat_gaussian_early_stop():
    result = flat_gaussian()
    assert result["method_details"]["optimization"]["backend_success"]
    assert result["status"] == "success", result
    assert result["method_details"]["conditional_mode"] == pytest.approx([2.5e6], abs=1e-3, rel=0)
    assert result["log_marginal_likelihood"] == pytest.approx(norm.logpdf(5e6, scale=np.sqrt(2e14)), abs=1e-7, rel=0)


def test_accurate_hessian_cannot_validate_a_displaced_mode_in_a_flat_gaussian(monkeypatch):
    def displaced(objective, point, options):
        # z=0对应eta=0；标准化梯度绝对值为0.5 / z=0 maps to eta=0; standardized gradient magnitude is 0.5.
        return {"status": "success", "minimizer": np.zeros(1), "objective_value": objective(np.zeros(1)),
                "method_details": {"backend_success": True, "gradient_inf_with_error": .5}}
    monkeypatch.setattr(laplace, "minimize", displaced)
    result = flat_gaussian()
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
