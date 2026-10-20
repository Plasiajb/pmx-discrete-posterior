"""Precision-limited endpoint recovery, with analytic and rejection controls."""

from copy import deepcopy
import math

import numpy as np
import pytest

from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics import laplace


def derivative_record(gradient, error=None, status="failed"):
    g = np.asarray(gradient, dtype=float)
    e = np.full_like(g, 1e-8) if error is None else np.asarray(error, dtype=float)
    return {"status": status, "gradient": g if status == "success" else None,
            "estimated_error": e, "failure_reason": "precision limited" if status == "failed" else None,
            "method_details": {"precision_limited_candidate": {"gradient": g.tolist(), "estimated_error": e.tolist()}}}


def failed_endpoint(objective, point):
    record = {"point": point.tolist(), "value": objective(point)}
    return {"status": "failed", "minimizer": None, "objective_value": None,
            "failure_reason": "optimization gradient failed: precision limited",
            "method_details": {
                "last_objective_evaluation": deepcopy(record), "best_objective_evaluation": deepcopy(record),
                "gradient_failure": {"point": point.tolist(), "phase": "search",
                                     "result": derivative_record(np.zeros_like(point))}}}


def run_gaussian(monkeypatch, matrix, point, *, options=None, modify=None, shift=None):
    matrix = np.atleast_2d(matrix)
    point = np.atleast_1d(point).astype(float)
    shift = np.zeros_like(point) if shift is None else np.asarray(shift)
    prior = RandomEffectDistribution([("iiv", None, f"eta_{i}") for i in range(len(point))], np.eye(len(point)))

    def endpoint(objective, start, optimization_options):
        result = failed_endpoint(objective, point)
        if modify:
            modify(result)
        return result

    monkeypatch.setattr(laplace, "minimize", endpoint)
    return laplace.integrate(
        conditional_log_likelihood=lambda x: -.5 * (x-shift) @ matrix @ (x-shift) - prior.log_density(x),
        random_effect_distribution=prior, integration_method="laplace", numerical_options=options,
    )


@pytest.mark.parametrize("matrix,point", [
    ([[1e12]], [1e-10]),
    (np.array([[2., .3], [.3, 1.]]) * 1e8, [1e-7, -2e-7]),
])
def test_stiff_gaussian_recovery_matches_analytic_integral(monkeypatch, matrix, point):
    result = run_gaussian(monkeypatch, matrix, point)
    assert result["status"] == "success", result
    exact = .5 * len(point) * math.log(2*math.pi) - .5 * np.linalg.slogdet(matrix)[1]
    assert result["log_marginal_likelihood"] == pytest.approx(exact, abs=1e-5, rel=0)
    recovery = result["method_details"]["precision_recovery"]
    assert recovery["status"] == "accepted" and recovery["optimization_status"] == "failed"
    assert recovery["warning"] and len(recovery["gradient_checks"]) == 2
    assert len(recovery["local_probes"]) == 2 * len(point)


@pytest.mark.parametrize("tolerance", [1e-5, 1e-8])
def test_explicit_absolute_stationarity_limit_is_not_bypassed(monkeypatch, tolerance):
    result = run_gaussian(monkeypatch, [[1e12]], [1e-10],
                          options={"optimization": {"stationarity_tolerance": tolerance}})
    assert result["status"] == "failed"
    assert "explicit" in result["failure_reason"]


def test_curvature_relative_check_rejects_displaced_soft_mode(monkeypatch):
    result = run_gaussian(monkeypatch, [[1.]], [.1])
    assert result["status"] == "failed"
    assert result["method_details"]["stage"] == "stationarity"


@pytest.mark.parametrize("change", ["wrong_point", "not_best", "nonfinite", "callback", "final_check", "missing_evidence"])
def test_ineligible_optimizer_failures_remain_failed(monkeypatch, change):
    def modify(result):
        details = result["method_details"]
        failure = details["gradient_failure"]
        if change == "wrong_point":
            failure["point"] = [100.]
        elif change == "not_best":
            details["best_objective_evaluation"]["point"] = [100.]
        elif change == "nonfinite":
            failure["result"]["method_details"]["precision_limited_candidate"]["gradient"] = [float("nan")]
        elif change == "callback":
            failure.pop("result")
            failure["source"] = "callback"
        elif change == "final_check":
            failure["phase"] = "stationarity"
        else:
            failure["result"]["method_details"] = {}
    result = run_gaussian(monkeypatch, [[1e12]], [1e-10], modify=modify)
    assert result["status"] == "failed" and result["log_marginal_likelihood"] is None


def test_local_probes_reject_false_gradient_even_with_positive_hessian(monkeypatch):
    monkeypatch.setattr(laplace, "gradient", lambda *args: derivative_record([0.]))
    result = run_gaussian(monkeypatch, [[1.]], [0.], shift=[.1])
    assert result["status"] == "failed"
    assert "probe" in result["failure_reason"]


def test_independent_step_schedules_must_agree(monkeypatch):
    records = iter([derivative_record([0.]), derivative_record([1.])])
    monkeypatch.setattr(laplace, "gradient", lambda *args: next(records))
    result = run_gaussian(monkeypatch, [[1e12]], [0.])
    assert result["status"] == "failed"
    assert "gradient" in result["failure_reason"]


@pytest.mark.parametrize("record", [
    {"status": "failed", "gradient": None, "estimated_error": None, "method_details": {}, "failure_reason": "callback"},
    derivative_record([0.], [-1.]),
    derivative_record([float("nan")]),
])
def test_bad_fresh_derivative_cannot_recover(monkeypatch, record):
    monkeypatch.setattr(laplace, "gradient", lambda *args: record)
    result = run_gaussian(monkeypatch, [[1e12]], [0.])
    assert result["status"] == "failed"


def test_ordinary_success_does_not_enter_recovery(monkeypatch):
    def accepted(objective, point, options):
        return {"status": "success", "minimizer": np.zeros(1), "objective_value": objective(np.zeros(1)),
                "method_details": {"gradient_inf_with_error": 0.}}
    monkeypatch.setattr(laplace, "minimize", accepted)
    prior = RandomEffectDistribution([("iiv", None, "eta")], [[1.]])
    result = laplace.integrate(conditional_log_likelihood=lambda x: 0.,
                               random_effect_distribution=prior, integration_method="laplace")
    assert result["status"] == "success"
    assert "precision_recovery" not in result["method_details"]


@pytest.mark.parametrize("width", [1e-7, 1e-8])
def test_narrow_smooth_feature_cannot_masquerade_as_stationarity(width):
    # 非模拟反例：粗步长可能漏掉很窄的非零梯度 / Real-function counterexample: coarse steps can miss a narrow nonzero gradient.
    prior = RandomEffectDistribution([("iiv", None, "eta")], [[1.]])

    def objective(x):
        z = float(x[0])
        return .5e12*z*z + 1e6*z*math.exp(-(z/width)**2)

    result = laplace.integrate(conditional_log_likelihood=lambda x: -objective(x)-prior.log_density(x),
                               random_effect_distribution=prior, integration_method="laplace")
    assert result["status"] == "failed", result
    assert result["log_marginal_likelihood"] is None
