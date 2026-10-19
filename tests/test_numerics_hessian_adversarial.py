"""功能 / Function: 独立核验曲率和共同偏移风险 / Independently audit curvature and common offsets.
输入 / Inputs: 解析目标及公开求导接口 / Analytic objectives and public derivative functions.
输出 / Outputs: 回归断言，不写文件 / Regression assertions without file writes.

Provenance: accepted v0.2 numerical-core audit counterexamples and frozen
manuscript gate predicates, retained as self-contained analytic regressions.
The historical manuscript implementation is preserved at tag manuscript-20261018.
These tests reproduce formulas only; they do not import the old package, execute
formal adapters, read formal results, or certify the complete integration layer.
"""

import math

import mpmath as mp
import numpy as np
import pytest

from pmx_discrete_posterior.numerics.derivatives import gradient, hessian


_LOG_2PI = math.log(2 * math.pi)
_ATOL = _RTOL = 1e-6
_POSTERIOR_ATOL = 1e-6


def _options(initial_step=0.5):
    """功能 / Function: 显式给出约定设置 / Supply the agreed explicit settings.
    输入 / Inputs: 初始步长 / Initial step. 输出 / Outputs: 新设置字典 / Fresh options dict.
    """
    return {
        "backend": "scipy", "atol": _ATOL, "rtol": _RTOL,
        "backend_options": {
            "initial_step": initial_step, "step_factor": 2, "order": 8, "maxiter": 10,
        },
    }


def _check_derivative(result, key, expected, *, allow_failure):
    """功能 / Function: 拒绝假成功并核验失败契约 / Reject false success and check failure contracts.
    输入 / Inputs: 结果、解析导数及失败许可 / Result, exact derivative and rejection allowance.
    输出 / Outputs: 已核验数组或None / Checked array or None.
    """
    assert isinstance(result, dict)
    assert {key, "estimated_error", "status", "failure_reason", "method_details"} <= result.keys()
    assert result["status"] in {"success", "failed"}, result
    assert isinstance(result["method_details"], dict), result
    if result["status"] == "failed":
        assert allow_failure, result
        assert result[key] is None, result
        assert isinstance(result["failure_reason"], str), result
        assert result["failure_reason"].strip(), result
        return None

    assert result["failure_reason"] is None, result
    expected = np.asarray(expected, dtype=float)
    value = np.asarray(result[key], dtype=float)
    assert value.shape == expected.shape, result
    assert np.isfinite(value).all(), result
    assert result["estimated_error"] is not None, result
    error = np.asarray(result["estimated_error"], dtype=float)
    assert error.shape == expected.shape, result
    assert np.isfinite(error).all() and np.all(error >= 0), result
    assert np.all(error <= _ATOL + _RTOL * np.abs(value)), result
    # 经验误差不当作严格误差界 / Empirical estimates are not rigorous error bounds.
    np.testing.assert_allclose(value, expected, atol=_ATOL, rtol=_RTOL, err_msg=str(result))
    return value


def _nc02_objective(state, common):
    """功能 / Function: 重建原NC-02规范化联合目标 / Rebuild the normalized NC-02 joint objective.
    输入 / Inputs: A/B状态、共同观测开关 / A/B state and common-observation flag.
    输出 / Outputs: 单ETA标量目标 / Scalar objective on a length-one ETA vector.
    """
    assert state in ("A", "B")

    def objective(eta):
        """功能 / Function: 评价联合负对数密度 / Evaluate negative log joint density.
        输入 / Inputs: ETA向量 / ETA vector. 输出 / Outputs: 含常数的标量 / Scalar including constants.
        """
        informative = math.sqrt(2) * float(eta[0]) if state == "A" else math.sqrt(math.log(3))
        predictions = np.array([10000.0, informative] if common else [informative])
        log_likelihood = float(np.sum(-0.5 * _LOG_2PI - 0.5 * predictions**2))
        prior_negative_log = 0.5 * _LOG_2PI + 0.5 * float(np.asarray(eta) @ eta)
        return -log_likelihood + prior_negative_log

    return objective


def _legacy_second_difference(objective, step):
    """功能 / Function: 仅重现旧三点差分 / Reproduce only the old three-point stencil.
    输入 / Inputs: 一维目标和步长 / One-dimensional objective and step.
    输出 / Outputs: 零点旧曲率 / Legacy curvature at zero, never an acceptance oracle.
    """
    center = objective(np.array([0.0]))
    return (objective(np.array([step])) - 2 * center + objective(np.array([-step]))) / step**2


def _equal_prior_posterior(objectives, curvatures):
    """功能 / Function: 独立计算零众数高斯的后验 / Independently normalize zero-mode Gaussian weights.
    输入 / Inputs: 两个目标和正曲率 / Two objectives and positive curvatures.
    输出 / Outputs: 等先验两状态概率 / Two-state equal-prior probabilities.
    """
    curvatures = np.asarray(curvatures, dtype=float)
    assert curvatures.shape == (2,) and np.all(curvatures > 0)
    values = np.array([objective(np.array([0.0])) for objective in objectives])
    # 共同Laplace常数和等先验抵消 / Common Laplace constants and equal priors cancel.
    logs = -(values - values.min()) - 0.5 * np.log(curvatures)
    weights = np.exp(logs - logs.max())
    return weights / weights.sum()


@pytest.mark.parametrize("common", [False, True], ids=["baseline", "common-observation"])
@pytest.mark.parametrize("explicit_options", [False, True], ids=["defaults", "explicit-defaults"])
def test_nc02_common_factor_is_accurate_or_explicitly_rejected(common, explicit_options):
    """功能 / Function: 保留NC-02原验收标准 / Preserve the original NC-02 acceptance criterion.
    输入 / Inputs: 原共同因子反例及设置 / Original common-factor case and options.
    输出 / Outputs: 准确曲率和1e-6后验或显式失败，并打印诊断 / Accurate curvature and 1e-6 posterior or explicit failure, with printed diagnostics.
    """
    objectives = [_nc02_objective(state, common) for state in ("A", "B")]
    options = _options() if explicit_options else None
    results = [hessian(objective, [0.0], options) for objective in objectives]
    checked = [
        _check_derivative(result, "hessian", [[exact]], allow_failure=common)
        for result, exact in zip(results, (3.0, 1.0))
    ]
    label = f"NC02 common={common} options={'explicit-defaults' if explicit_options else 'defaults'} states=A,B"
    curvatures = [None if value is None else float(value[0, 0]) for value in checked]
    if any(value is None for value in checked):
        reasons = [result["failure_reason"] for result in results]
        print(f"{label} status=rejected H={curvatures!r} posterior=None reasons={reasons!r}")
        return  # 不将失败状态当作零权重 / Never normalize a failed state as zero weight.
    posterior = _equal_prior_posterior(objectives, curvatures)
    np.testing.assert_allclose(posterior, [0.5, 0.5], atol=_POSTERIOR_ATOL, rtol=0)
    print(f"{label} status=accepted H={curvatures!r} posterior={posterior.tolist()!r}")


@pytest.mark.parametrize("common", [False, True], ids=["passing-control", "archived-failure"])
def test_old_nc02_signature_and_frozen_formal_numerical_gates(common):
    """功能 / Function: 核对旧失真及冻结门槛 / Check the old distortion and frozen numerical gates.
    输入 / Inputs: 原NC-02公式 / Original NC-02 formulas.
    输出 / Outputs: 解析门槛断言，不执行正式适配器 / Analytic gate assertions, not formal adapter runs.
    """
    objectives = [_nc02_objective(state, common) for state in ("A", "B")]
    exact = np.array([3.0, 1.0])
    coarse = np.array([_legacy_second_difference(f, 1e-4) for f in objectives])
    fine = np.array([_legacy_second_difference(f, 5e-5) for f in objectives])
    if common:
        np.testing.assert_allclose(coarse, [2.9802322387695312, 1.4901161193847656], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(fine, [0.0, 0.0])
        np.testing.assert_allclose(
            _equal_prior_posterior(objectives, coarse),
            [0.5505102591565486, 0.44948974368636974], atol=1e-8, rtol=0,
        )
    else:
        np.testing.assert_allclose(coarse, exact, atol=1e-6, rtol=0)
        np.testing.assert_allclose(fine, exact, atol=1e-6, rtol=0)

    for independent, old, half_step in zip(exact, coarse, fine):
        # 零点解析梯度及目标差均为零 / Exact mode gradient and objective discrepancy are zero.
        scaled_gradient = 0.0 / math.sqrt(independent)
        objective_delta = 0.0
        oxc_passed = bool(
            old > 0 and independent > 0 and scaled_gradient <= 1e-4
            and abs(math.log(independent) - math.log(old)) <= 1e-3
            and objective_delta <= 1e-3
        )
        tac_passed = bool(
            0.0 <= 1e-3 and old > 0 and half_step > 0
            and abs(math.log(half_step) - math.log(old)) <= 1e-3
        )
        assert oxc_passed == (not common)
        assert tac_passed == (not common)


def _false_stable_objectives():
    """功能 / Function: 构造半步长也假稳定的高斯对 / Construct a falsely half-step-stable Gaussian pair.
    输入 / Inputs: 无；冻结C和a / None; fixed C and a.
    输出 / Outputs: 等积分目标和解析曲率 / Equal-integral objectives and exact curvatures.
    """
    offset, curvature = 5e7, 0.997
    return (
        lambda x: offset + 0.5 * curvature * x[0]**2,
        lambda x: (offset + 0.5 * math.log(curvature)) + 0.5 * x[0]**2,
    ), (curvature, 1.0)


def test_false_stable_stencil_has_zero_step_disagreement_but_wrong_posterior():
    """功能 / Function: 证明步长一致不等于准确 / Demonstrate that step agreement is not accuracy.
    输入 / Inputs: 固定假稳定目标 / Fixed false-stability objectives.
    输出 / Outputs: 正曲率、零步长差及超限后验的断言 / Positive curvature, zero step delta, wrong posterior assertions.
    """
    objectives, exact = _false_stable_objectives()
    coarse = np.array([_legacy_second_difference(f, 0.002) for f in objectives])
    fine = np.array([_legacy_second_difference(f, 0.001) for f in objectives])
    np.testing.assert_array_equal(coarse, [0.998377799987793, 0.998377799987793])
    np.testing.assert_array_equal(coarse, fine)
    assert np.all(coarse > 0)
    np.testing.assert_array_equal(np.abs(np.log(coarse) - np.log(fine)), [0.0, 0.0])
    assert np.all(np.abs(coarse - exact) > _ATOL + _RTOL * np.abs(exact))
    posterior = _equal_prior_posterior(objectives, coarse)
    assert posterior[0] == pytest.approx(0.49962443679183555, abs=1e-12, rel=0)
    assert np.all(np.abs(posterior - 0.5) > _POSTERIOR_ATOL)
    np.testing.assert_allclose(_equal_prior_posterior(objectives, exact), [0.5, 0.5], atol=1e-8, rtol=0)


@pytest.mark.parametrize("initial_step", [0.5, 0.002, 0.001])
def test_false_stable_hessians_are_correct_or_explicitly_rejected(initial_step):
    """功能 / Function: 阻止正定且假稳定的错误成功 / Prevent positive, falsely stable successful results.
    输入 / Inputs: 假稳定目标和步长 / False-stability objectives and step.
    输出 / Outputs: 正确曲率和后验或显式失败 / Correct curvature and posterior or explicit rejection.
    """
    objectives, exact = _false_stable_objectives()
    checked = [
        _check_derivative(hessian(f, [0.0], _options(initial_step)), "hessian", [[value]], allow_failure=True)
        for f, value in zip(objectives, exact)
    ]
    if all(value is not None for value in checked):
        posterior = _equal_prior_posterior(objectives, [value[0, 0] for value in checked])
        np.testing.assert_allclose(posterior, [0.5, 0.5], atol=_POSTERIOR_ATOL, rtol=0)


@pytest.mark.parametrize("offset", [0.0, 5e7, 1e16])
@pytest.mark.parametrize("curvature", [1.0, 1e-4], ids=["unit", "near-flat"])
def test_offset_cannot_turn_nonzero_derivatives_into_successful_zero(offset, curvature):
    """功能 / Function: 防止舍入平台伪装成驻点 / Prevent a rounded plateau from masquerading as stationarity.
    输入 / Inputs: 偏移二次函数、非众数零点 / Offset quadratic evaluated at the non-mode zero.
    输出 / Outputs: 非零解析导数或明确失败 / Nonzero analytic derivatives or explicit failure.
    """
    objective = lambda x: offset + 0.5 * curvature * (x[0] - 1.0)**2
    _check_derivative(gradient(objective, [0.0]), "gradient", [-curvature], allow_failure=offset != 0)
    _check_derivative(hessian(objective, [0.0]), "hessian", [[curvature]], allow_failure=offset != 0)


@pytest.mark.parametrize("offset", [0.0, 5e7])
def test_correlated_gaussian_matches_independent_matrix_derivatives(offset):
    """功能 / Function: 核验非对角高斯曲率 / Check non-diagonal Gaussian curvature.
    输入 / Inputs: 相关Omega、线性模型及偏移 / Correlated Omega, linear model and offset.
    输出 / Outputs: 解析梯度及完整Hessian断言 / Analytic gradient and full-Hessian assertions.
    """
    omega = np.array([[0.4, 0.15], [0.15, 0.25]])
    matrix = np.array([[1.0, 0.2], [0.3, -0.4]])
    observations, bias = np.array([1.4, -0.2]), np.array([0.2, 0.1])
    point, residual_variance = np.array([0.2, -0.1]), 0.7**2
    precision = np.linalg.solve(omega, np.eye(2))
    expected_hessian = precision + matrix.T @ matrix / residual_variance
    expected_gradient = precision @ point + matrix.T @ (matrix @ point + bias - observations) / residual_variance

    def objective(x):
        """功能 / Function: 评价相关高斯联合目标 / Evaluate the correlated Gaussian joint objective.
        输入 / Inputs: 二维ETA / Two-dimensional ETA. 输出 / Outputs: 规范化负对数密度 / Normalized negative log density.
        """
        residual = observations - matrix @ x - bias
        likelihood = 0.5 * (2 * _LOG_2PI + 2 * math.log(residual_variance) + residual @ residual / residual_variance)
        prior = 0.5 * (2 * _LOG_2PI + np.linalg.slogdet(omega)[1] + x @ precision @ x)
        return float(offset + likelihood + prior)

    _check_derivative(gradient(objective, point), "gradient", expected_gradient, allow_failure=offset != 0)
    checked = _check_derivative(hessian(objective, point), "hessian", expected_hessian, allow_failure=offset != 0)
    if checked is not None:
        np.testing.assert_allclose(checked, checked.T, atol=_ATOL, rtol=0)
        assert np.linalg.eigvalsh(checked).min() > 0


@pytest.mark.parametrize("offset", [0.0, 5e7])
def test_proportional_residual_curvature_matches_independent_high_precision(offset):
    """功能 / Function: 核验预测相关方差的全部导数 / Check all prediction-dependent variance derivatives.
    输入 / Inputs: 非线性预测、比例残差及偏移 / Nonlinear prediction, proportional residual and offset.
    输出 / Outputs: 独立80位梯度及曲率对照 / Independent 80-digit gradient and curvature comparisons.
    """
    def objective(x):
        """功能 / Function: 保留log(SD)项 / Retain the log-SD term.
        输入 / Inputs: 单ETA向量 / One-ETA vector. 输出 / Outputs: 完整联合目标 / Full joint objective.
        """
        prediction = 2 + math.exp(0.7 * float(x[0]))
        sd = 0.2 * prediction
        likelihood = 0.5 * _LOG_2PI + math.log(sd) + 0.5 * ((3.2 - prediction) / sd)**2
        prior = 0.5 * math.log(2 * math.pi * 0.3) + float(x[0])**2 / 0.6
        return offset + likelihood + prior

    with mp.workdps(80):
        def reference(eta):
            """功能 / Function: 独立高精度参考 / Independent high-precision reference.
            输入 / Inputs: 任意精度ETA / Multiprecision ETA. 输出 / Outputs: 完整联合目标 / Full joint objective.
            """
            prediction = 2 + mp.exp(mp.mpf("0.7") * eta)
            sd = mp.mpf("0.2") * prediction
            likelihood = mp.log(2 * mp.pi) / 2 + mp.log(sd) + (mp.mpf("3.2") - prediction)**2 / (2 * sd**2)
            prior = mp.log(2 * mp.pi * mp.mpf("0.3")) / 2 + eta**2 / mp.mpf("0.6")
            return mp.mpf(offset) + likelihood + prior

        point = mp.mpf("0.2")
        exact_gradient = float(mp.diff(reference, point))
        exact_hessian = float(mp.diff(reference, point, 2))
    assert exact_hessian == pytest.approx(5.112752454004908, abs=1e-12, rel=0)
    _check_derivative(gradient(objective, [0.2]), "gradient", [exact_gradient], allow_failure=offset != 0)
    _check_derivative(hessian(objective, [0.2]), "hessian", [[exact_hessian]], allow_failure=offset != 0)
