"""功能 / Function: 核验标准化后的完整Laplace目标 / Verify the complete Laplace target after standardization.
输入 / Inputs: 解析正态、非正态及回调反例 / Analytic normal, nonnormal and callback counterexamples.
输出 / Outputs: 积分、坐标和失败边界断言 / Integral, coordinate and failure assertions.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm, t

from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics import laplace


def integrate(density, likelihood, options=None):
    """功能 / Function: 调用被测积分 / Invoke the tested integrator.
    输入 / Inputs: 分布、似然、设置 / Distribution, likelihood, options.
    输出 / Outputs: 原积分结果 / Original integration result.
    """
    return laplace.integrate(conditional_log_likelihood=likelihood, random_effect_distribution=density,
                             integration_method="laplace", numerical_options=options)


@pytest.mark.parametrize("explicit_gtol", [1e-7, None])
def test_shifted_correlated_gaussian_with_original_start_and_diagnostics(explicit_gtol):
    """功能 / Function: 验证Jacobian、原始坐标诊断及显式起点 / Verify Jacobian, original diagnostics and explicit start.
    输入 / Inputs: 多维相关解析模型 / Multivariate analytic model. 输出 / Outputs: 精确积分与模式 / Exact integral and mode.
    """
    mean = np.array([1.2, -.8])
    covariance = np.array([[.2, .1], [.1, 1.5]])
    density = SimpleNamespace(mean=mean, covariance=covariance, effect_keys=("a", "b"),
                              log_density=lambda eta: multivariate_normal.logpdf(eta, mean, covariance))
    a, y, sd = np.array([1.5, -.4]), .3, .8
    options = {"optimization": {"initial_point": [1.1, -.6]}}
    if explicit_gtol is not None:
        options["optimization"]["backend_options"] = {"gtol": explicit_gtol}
    result = integrate(density, lambda eta: norm.logpdf(y, a @ eta, sd), options)
    assert result["status"] == "success", result
    details = result["method_details"]
    expected = norm.logpdf(y, a @ mean, math.sqrt(sd**2 + a @ covariance @ a))
    mode = mean + covariance @ a * (y-a@mean)/(sd**2 + a@covariance@a)
    assert result["log_marginal_likelihood"] == pytest.approx(expected, abs=1e-7)
    factor = np.linalg.cholesky(covariance)
    returned_mode = np.asarray(details["conditional_mode"])
    raw_hessian = np.linalg.inv(covariance)+np.outer(a, a)/sd**2
    exact_gradient = raw_hessian @ (returned_mode-mode)
    if explicit_gtol is not None:
        # 严格坐标精度仍使用原搜索设置 / Retain original search settings for strict coordinate accuracy.
        np.testing.assert_allclose(returned_mode, mode, atol=1e-7)
        np.testing.assert_allclose(details["standardized_mode"], np.linalg.solve(factor, mode-mean), atol=1e-7)
    else:
        # 默认设置独立用解析梯度核验，不放宽积分精度 / Check default stopping with the analytic gradient, not a looser integral test.
        assert details["optimization"]["backend_options"]["gtol"] == 1e-6
        assert np.max(np.abs(factor.T @ exact_gradient)) <= 1e-6
        np.testing.assert_allclose(details["standardized_mode"], np.linalg.solve(factor, returned_mode-mean),
                                   atol=1e-12, rtol=0)
    np.testing.assert_allclose(details["standardized_initial_point"], np.linalg.solve(factor, [1.1, -.6]-mean))
    assert details["initial_point"] == [1.1, -.6]
    assert options["optimization"]["initial_point"] == [1.1, -.6]
    assert options["optimization"].get("backend_options") == (
        None if explicit_gtol is None else {"gtol": explicit_gtol})
    assert details["optimization"]["coordinate_system"] == "prior_standardized"
    assert details["hessian"]["coordinate_system"] == "prior_standardized"
    raw_objective = -norm.logpdf(y, a@mode, sd)-density.log_density(mode)
    assert details["objective_value"] == pytest.approx(raw_objective, abs=1e-8)
    assert details["log_determinant"] == pytest.approx(np.linalg.slogdet(raw_hessian)[1], abs=1e-7)
    reconstructed = -details["objective_value"] + math.log(2*math.pi) - .5*details["log_determinant"]
    assert reconstructed == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("variance", [1e-8, 1e-100, 1e-300, 1e14])
def test_scaled_normal_density_integrates_to_constant(variance):
    """功能 / Function: 验证小方差仍参与积分且归一化不变 / Verify active tiny variances and normalization.
    输入 / Inputs: 多尺度方差 / Multiple variance scales. 输出 / Outputs: 常数似然原值 / Unchanged constant likelihood.
    """
    density = RandomEffectDistribution([("iiv", None, "eta")], [[variance]])
    result = integrate(density, lambda eta: -2.3)
    assert result["status"] == "success", result
    assert result["method_details"]["coordinate_system"] == "prior_standardized"
    assert result["log_marginal_likelihood"] == pytest.approx(-2.3, abs=1e-7)


def test_generic_nonnormal_distribution_density_is_not_replaced():
    """功能 / Function: 协方差不等于正态密度 / Covariance does not imply normal density.
    输入 / Inputs: 已规范化t分布 / Normalized t distribution. 输出 / Outputs: t分布的局部Laplace值 / Local t-density Laplace value.
    """
    density = SimpleNamespace(mean=np.array([2.]), covariance=np.array([[9.]]), effect_keys=("eta",),
                              log_density=lambda eta: t.logpdf(eta[0], df=5, loc=2., scale=math.sqrt(5.4)))
    result = integrate(density, lambda eta: 0.)
    expected = density.log_density([2.]) + .5*math.log(2*math.pi) - .5*math.log(2/9)
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(expected, abs=1e-6)
    assert abs(expected) > .05
    assert result["method_details"]["standardized_mode"] == pytest.approx([0.], abs=1e-8)


def test_overridden_builtin_density_is_preserved():
    """功能 / Function: 保留子类密度回调 / Preserve subclass density callback.
    输入 / Inputs: 带覆盖密度的内置分布子类 / Built-in subclass with overridden density.
    输出 / Outputs: 非正态Laplace值 / Nonnormal Laplace value.
    """
    class StudentDensity(RandomEffectDistribution):
        def log_density(self, eta):
            """功能 / Function: 规范化t密度 / Normalized t density.
            输入 / Inputs: eta. 输出 / Outputs: log density.
            """
            return t.logpdf(eta[0], df=5, scale=math.sqrt(.6))
    density = StudentDensity([("iiv", None, "eta")], [[1.]])
    result = integrate(density, lambda eta: 0.)
    expected = density.log_density([0.]) + .5*math.log(2*math.pi) - .5*math.log(2)
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("callback_kind", ["mutating", "nondeterministic"])
def test_transform_does_not_hide_bad_likelihood_callbacks(callback_kind):
    """功能 / Function: 变换不掩盖回调问题 / Transformation does not hide callback faults.
    输入 / Inputs: 修改参数或非确定回调 / Mutating or nondeterministic callback. 输出 / Outputs: 明确失败 / Explicit failure.
    """
    count = 0
    def likelihood(eta):
        nonlocal count
        count += 1
        if callback_kind == "mutating":
            eta[0] += 1
            return 0.
        return float(count)
    result = integrate(RandomEffectDistribution([("iiv", None, "eta")], [[.1]]), likelihood)
    assert result["status"] == "failed"
    assert result["log_marginal_likelihood"] is None
    assert "mutated" in result["failure_reason"] or "deterministic" in result["failure_reason"]


def test_zero_dimension_bypasses_coordinate_transform(monkeypatch):
    """功能 / Function: 零维不构造标准化 / Zero dimension bypasses standardization.
    输入 / Inputs: 零维分布 / Empty distribution. 输出 / Outputs: 直接似然 / Direct likelihood.
    """
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-dimensional transform must not be constructed")
    monkeypatch.setattr(laplace, "PriorCoordinates", forbidden)
    result = integrate(RandomEffectDistribution([], np.zeros((0, 0))), lambda eta: -3.)
    assert result["status"] == "success"
    assert result["log_marginal_likelihood"] == -3.


@pytest.mark.parametrize("variance", [1e-8, 1e-100, 1e-300])
@pytest.mark.parametrize("fixed_zero", [False, True])
def test_tiny_active_effect_retains_nonconstant_likelihood_information(variance, fixed_zero):
    """功能 / Function: 核验极小方差下仍利用观测信息 / Verify observation information is retained at tiny variance.
    输入 / Inputs: 可选严格零坐标及微小活动坐标 / Optional fixed zero and tiny active effect.
    输出 / Outputs: 解析积分与非零标准化众数 / Analytic integral and nonzero standardized mode.
    """
    keys = [("iiv", None, "eta")]
    covariance = [[variance]]
    if fixed_zero:
        keys.insert(0, ("iiv", None, "fixed"))
        covariance = [[0., 0.], [0., variance]]
    density = RandomEffectDistribution(keys, covariance)
    sd = math.sqrt(variance)
    result = integrate(density, lambda eta: norm.logpdf(.6, 1.5*eta[0]/sd, .8))
    assert result["status"] == "success", result
    expected_mode = 1.5*.6/(.8**2+1.5**2)
    assert result["log_marginal_likelihood"] == pytest.approx(
        norm.logpdf(.6, scale=math.sqrt(.8**2+1.5**2)), abs=1e-7, rel=0)
    details = result["method_details"]
    assert details["dimension"] == 1
    assert details["conditional_mode"][0]/sd == pytest.approx(expected_mode, abs=1e-7)
    if fixed_zero:
        assert density.expand(details["conditional_mode"])["iiv_values"]["fixed"] == 0.


@pytest.mark.parametrize("kind", ["mutating", "nondeterministic"])
def test_transformed_density_callback_failures_are_not_hidden(kind):
    """功能 / Function: 保留密度回调的错误检查 / Preserve density callback checks.
    输入 / Inputs: 有副作用的密度 / Density with side effects. 输出 / Outputs: failed/None.
    """
    count = 0
    def log_density(eta):
        nonlocal count
        count += 1
        if kind == "mutating":
            eta[0] += 1.
            return 0.
        return float(count)
    density = SimpleNamespace(mean=np.array([0.]), covariance=np.array([[.01]]),
                              effect_keys=("eta",), log_density=log_density)
    result = integrate(density, lambda eta: 0.)
    assert result["status"] == "failed" and result["log_marginal_likelihood"] is None
    assert "mutated" in result["failure_reason"] or "deterministic" in result["failure_reason"]


def test_transformed_hessian_failure_keeps_coordinate_labels(monkeypatch):
    """功能 / Function: 曲率失败仍保留坐标说明 / Preserve coordinate labels on curvature failure.
    输入 / Inputs: 注入Hessian失败 / Injected Hessian failure.
    输出 / Outputs: 原始众数、标准化曲率标签及明确失败 / Raw mode, standardized curvature label and explicit failure.
    """
    def rejected(objective, mode, options):
        return {"status": "failed", "hessian": None, "failure_reason": "injected unresolved curvature",
                "method_details": {"backend_status": [[-2]]}}
    monkeypatch.setattr(laplace, "hessian", rejected)
    result = integrate(RandomEffectDistribution([("iiv", None, "eta")], [[.01]]), lambda eta: -2.3)
    details = result["method_details"]
    assert result["status"] == "failed" and result["log_marginal_likelihood"] is None
    assert details["stage"] == "hessian" and details["conditional_mode"] == [0.]
    assert details["hessian"]["coordinate_system"] == "prior_standardized"
