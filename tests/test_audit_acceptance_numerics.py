"""功能 / Function: 验收NC-01/NC-02的完整Laplace链 / Accept NC-01/NC-02 through the full Laplace chain.
输入 / Inputs: 原审计解析高斯反例及受控优化故障 / Archived Gaussian counterexamples and controlled optimizer faults.
输出 / Outputs: 断言与stdout证据；不改源码或正式输入 / Assertions and stdout evidence; no source or formal-input changes.

Provenance: accepted v0.2 NC-01/NC-02 numerical-core audit, preserved here as
self-contained analytic counterexamples without developer-workspace dependencies.
Tests use the current unified integrator and compute_posterior, never the old
scalar API or a standalone Hessian as a substitute for integration acceptance.
All objectives below are exactly Gaussian in ETA; no global multimode guarantee
or general nonlinear Laplace approximation accuracy is requested.
"""

import json
import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.stats import norm

from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics import integrate, optimization
from pmx_discrete_posterior.posterior import compute_posterior


POSTERIOR_ATOL = 1e-6
LOG_ATOL = 1e-6


def _model_inputs(state_predictions, *, dimension=1, variance=1.0, sd=1.0, common_prediction=None):
    """功能 / Function: 构建真实公开入口输入 / Build actual public-entry inputs.
    输入 / Inputs: 状态预测函数、ETA维度/方差、SD及共同预测 / State functions, ETA dimension/variance, SD and common prediction.
    输出 / Outputs: 七组输入；浓度mg/L、时间h、ETA无量纲 / Seven input groups; mg/L, hours and dimensionless ETA.
    """
    names = [f"u{i}" for i in range(dimension)]

    def relations(*, parameters, covariates, iiv_values, iov_values, occasions, evaluation_interval):
        """功能 / Function: 将具名效应接入个体参数 / Map named effects into individual parameters.
        输入 / Inputs: 解析参数及评价区间 / Resolved inputs and interval.
        输出 / Outputs: 一段无量纲效应参数 / One segment of dimensionless effect parameters.
        """
        return {"units": {name: "1" for name in names}, "segments": [{
            "start": evaluation_interval[0], "end": evaluation_interval[1],
            "occasion_id": None, "values": {name: iiv_values[name] for name in names},
        }]}

    def predict(*, individual_parameters, candidate_state, prediction_records, individual_data):
        """功能 / Function: 提供合法非负测试预测 / Supply valid nonnegative test predictions.
        输入 / Inputs: 已解析效应、候选及无实测值记录 / Resolved effects, candidate and value-free records.
        输出 / Outputs: 按记录排列的mg/L浓度；不读取实测值或先验 / Ordered mg/L predictions; no observations or priors read.
        """
        eta = individual_parameters["segments"][0]["values"]["u0"]
        informative = state_predictions[candidate_state["state_id"]](eta)
        return {
            "record_ids": [record["record_id"] for record in prediction_records],
            "units": ["mg/L"] * len(prediction_records),
            "values": [common_prediction if record["record_id"] == "common" else informative
                       for record in prediction_records],
        }

    records = [{"record_id": "informative", "time": 2.0, "value": 0.0,
                "time_unit": "h", "value_unit": "mg/L"}]
    if common_prediction is not None:
        records.insert(0, {"record_id": "common", "time": 1.0, "value": 0.0,
                           "time_unit": "h", "value_unit": "mg/L"})
    states = [{"state_id": state, "events": []} for state in state_predictions]
    return {
        "model": {
            "parameters": {"values": {"omega": variance, "zero_cov": 0.0, "sigma": sd},
                           "units": {"omega": "1", "zero_cov": "1", "sigma": "mg/L"}},
            "parameter_model": {"covariate_names": [], "relations": relations}, "predict": predict,
            "iiv": {"names": names, "distribution": "normal", "covariance_parameters": [
                ["omega" if i == j else "zero_cov" for j in range(dimension)] for i in range(dimension)]},
            "iov": None, "random_effects_dependence": "independent_blocks",
            "residual_model": {"distribution": "normal", "error_type": "additive",
                               "parameter_names": {"sd": "sigma"}, "dependence": "conditionally_independent"},
        },
        "individual_data": {"covariates": {}, "occasions": [], "dosing_prehistory": {"events": []},
                            "initialization": {"mode": "zero", "start_time": 0.0, "event_phase": "before_events"}},
        "observations": records, "candidate_states": states,
        "priors": {state: 1.0 / len(states) for state in state_predictions}, "integration_method": "laplace",
    }


def _emit(label, result):
    """功能 / Function: 输出可归档的实际诊断 / Print archivable actual diagnostics.
    输入 / Inputs: 标签和真实结果 / Label and actual result.
    输出 / Outputs: stdout JSON；不另写文件 / JSON on stdout; no extra file writes.
    """
    rows = result.get("state_results", [result])
    evidence = []
    for row in rows:
        details = row["method_details"]
        evidence.append({
            "state": row.get("state_id"), "status": row["status"], "stage": details["stage"],
            "mode": details.get("conditional_mode"), "log_integral": row["log_marginal_likelihood"],
            "posterior": row.get("posterior_probability"), "reason": row["failure_reason"],
            "logdet": details.get("log_determinant"),
            "hessian_error_screen": details.get("estimated_log_integral_error_from_hessian"),
            "mode_error_screen": details.get("estimated_log_integral_error_from_stationarity"),
            "optimization": details.get("optimization"), "hessian": details.get("hessian"),
        })
    print(json.dumps({"case": label, "status": result["status"], "rows": evidence}, ensure_ascii=True))


def _assert_posterior_success(result, expected_log, expected_modes):
    """功能 / Function: 分别核验积分、众数与后验 / Check integrals, modes and posterior separately.
    输入 / Inputs: 结果、相同解析log积分及各状态众数 / Result, equal exact log integral and state modes.
    输出 / Outputs: 必须成功且绝对误差达标的断言 / Mandatory success and absolute-error assertions.
    """
    assert result["status"] == "success", result
    rows = result["state_results"]
    assert [row["state_id"] for row in rows] == list(expected_modes), result
    for row in rows:
        assert row["status"] == "success" and row["failure_reason"] is None, row
        assert row["log_marginal_likelihood"] == pytest.approx(expected_log, abs=LOG_ATOL, rel=0)
        assert row["posterior_probability"] == pytest.approx(1.0 / len(rows), abs=POSTERIOR_ATOL, rel=0)
        np.testing.assert_allclose(row["method_details"]["conditional_mode"], expected_modes[row["state_id"]],
                                   atol=1e-6, rtol=0)


@pytest.mark.parametrize("dimension", [1, 2], ids=["one-ETA", "two-ETA"])
@pytest.mark.parametrize("target", [-12.0, 12.0, -2.0, 2.0], ids=["lower-outside", "upper-outside", "lower-interior", "upper-interior"])
def test_nc01_true_gaussian_modes_outside_old_bounds_and_interior(dimension, target):
    """功能 / Function: 实际求出旧界外和内点高斯众数 / Actually find Gaussian modes outside the old bounds and inside.
    输入 / Inputs: ETA维度与带符号目标；众数target/2 / ETA dimension and signed target; mode target/2.
    输出 / Outputs: 统一积分及完整后验均准确，不准用拒绝代替正向通过 / Accurate unified integration and posterior; rejection cannot pass this control.
    """
    effects = RandomEffectDistribution([("iiv", None, f"u{i}") for i in range(dimension)], np.eye(dimension))
    direct = integrate(conditional_log_likelihood=lambda eta: norm.logpdf(target, loc=eta[0], scale=1.0),
                       random_effect_distribution=effects, integration_method="laplace")
    _emit(f"NC01 direct target={target} d={dimension}", direct)
    expected_log = float(norm.logpdf(target, scale=math.sqrt(2.0)))
    mode = [target / 2.0] + [0.0] * (dimension - 1)
    assert direct["status"] == "success", direct
    assert direct["log_marginal_likelihood"] == pytest.approx(expected_log, abs=LOG_ATOL, rel=0)
    np.testing.assert_allclose(direct["method_details"]["conditional_mode"], mode, atol=1e-6, rtol=0)
    if abs(target) == 12.0:
        assert abs(direct["method_details"]["conditional_mode"][0]) > 5.0

    # 非负光滑预测仅给两状态增加同一-1/2对数因子 / Smooth nonnegative predictions add the same -1/2 log factor.
    predictions = {"A": lambda eta: math.hypot(1.0, target - eta),
                   "B": lambda eta: math.sqrt(1.0 + target * target / 2.0 + math.log(2.0))}
    result = compute_posterior(**_model_inputs(predictions, dimension=dimension))
    _emit(f"NC01 posterior target={target} d={dimension}", result)
    _assert_posterior_success(result, expected_log - 0.5, {"A": mode, "B": [0.0] * dimension})


@pytest.mark.parametrize("sign", [-1.0, 1.0], ids=["lower", "upper"])
def test_nc01_false_solver_success_near_old_boundary_is_rejected(monkeypatch, sign):
    """功能 / Function: 验证错误优化点不会获后验认证 / Ensure a false optimizer point cannot certify a posterior.
    输入 / Inputs: 仅注入SciPy成功标志及旧边界附近点 / Inject only SciPy success and a point near the old boundary.
    输出 / Outputs: 独立驻点拒绝、保留另一成功积分、全部后验None / Independent stationarity rejection, other integral retained, all posteriors None.
    """
    original = optimization.scipy_minimize
    calls = []

    def false_first_success(fun, x0, *args, **kwargs):
        """功能 / Function: 只伪造第一状态的优化器回执 / Fake only the first state's optimizer receipt.
        输入 / Inputs: 真实目标及SciPy参数 / Real objective and SciPy arguments.
        输出 / Outputs: 非驻点success或真实第二状态结果 / Nonstationary success or real second-state result.
        """
        calls.append(1)
        if len(calls) == 1:
            point = np.array([sign * (5.0 - 1.1092637652865278e-7)])
            return OptimizeResult(x=point, fun=fun(point), success=True, message="audit injected success", nit=1)
        return original(fun, x0, *args, **kwargs)

    monkeypatch.setattr(optimization, "scipy_minimize", false_first_success)
    target = sign * 12.0
    result = compute_posterior(**_model_inputs({
        "A": lambda eta: math.hypot(1.0, target - eta),
        "B": lambda eta: math.sqrt(73.0 + math.log(2.0)),
    }))
    _emit(f"NC01 false optimizer sign={sign}", result)
    assert result["status"] == "failed" and len(calls) == 2, result
    a, b = result["state_results"]
    assert a["status"] == "failed" and a["log_marginal_likelihood"] is None, a
    assert a["method_details"]["optimization"]["backend_success"] is True
    assert "independent stationarity check" in a["failure_reason"], a
    assert a["method_details"]["optimization"]["gradient_inf_with_error"] > 2.0
    assert b["status"] == "success" and math.isfinite(b["log_marginal_likelihood"]), b
    assert all(row["posterior_probability"] is None for row in (a, b))


@pytest.mark.parametrize("exact_start", [False, True], ids=["standardized-default", "accept-exact-mode"])
def test_nc01_flat_gaussian_wrong_stationarity_and_exact_mode(exact_start):
    """功能 / Function: 标准化后真实BFGS须找到解析众数 / Standardized real BFGS must reach the analytic mode.
    输入 / Inputs: 方差1e14、SD1e7、目标5e6及默认标准化步长 / Variance 1e14, SD 1e7, target 5e6 and default standardized steps.
    输出 / Outputs: 两种起点均满足解析log积分标准 / Both starts meet the analytic log-integral criterion.
    """
    args = _model_inputs({"A": lambda eta: math.hypot(1.0, 5e6 - eta)}, variance=1e14, sd=1e7)
    args["numerical_options"] = {}
    if exact_start:
        args["numerical_options"]["optimization"] = {"initial_point": [2.5e6]}
    result = compute_posterior(**args)
    _emit(f"NC01 flat exact_start={exact_start}", result)
    row = result["state_results"][0]
    _assert_posterior_success(result, float(norm.logpdf(5e6, scale=math.sqrt(2e14))) - 0.5e-14,
                              {"A": [2.5e6]})
    assert row["method_details"]["optimization"]["backend_success"] is True


@pytest.mark.parametrize("dimension", [1, 2], ids=["one-ETA", "two-ETA"])
@pytest.mark.parametrize("common_prediction", [0.0, 10.0, 10000.0], ids=["normalization-only", "moderate-common", "archived-common"])
def test_nc02_common_observation_full_posterior_is_accurate_or_numerically_rejected(dimension, common_prediction):
    """功能 / Function: 验收共同似然因子的全流程不变性 / Accept full-pipeline common-factor invariance.
    输入 / Inputs: 原NC-02的平方等价非负预测，额外ETA独立 / Squared-equivalent nonnegative NC-02 predictions; extra ETA independent.
    输出 / Outputs: 后验误差<=1e-6或明确数值拒绝；基线必须算准 / Posterior error <=1e-6 or explicit numerical rejection; baseline must be accurate.
    """
    # abs只消除负浓度；y=0时平方和似然与原脚本严格相同 / abs removes negative concentrations; at y=0 squared residuals equal the archived model.
    predictions = {"A": lambda eta: math.sqrt(2.0) * abs(eta),
                   "B": lambda eta: math.sqrt(math.log(3.0))}
    baseline = compute_posterior(**_model_inputs(predictions, dimension=dimension))
    expected_log = -0.5 * math.log(2.0 * math.pi * 3.0)
    modes = {"A": [0.0] * dimension, "B": [0.0] * dimension}
    _emit(f"NC02 baseline d={dimension} common={common_prediction}", baseline)
    _assert_posterior_success(baseline, expected_log, modes)
    result = compute_posterior(**_model_inputs(predictions, dimension=dimension, common_prediction=common_prediction))
    _emit(f"NC02 augmented d={dimension} common={common_prediction}", result)
    if result["status"] == "success":
        common_log = float(norm.logpdf(0.0, loc=common_prediction, scale=1.0))
        _assert_posterior_success(result, expected_log + common_log, modes)
        np.testing.assert_allclose([r["posterior_probability"] for r in result["state_results"]],
                                   [r["posterior_probability"] for r in baseline["state_results"]],
                                   atol=POSTERIOR_ATOL, rtol=0)
        return

    # 小共同因子必须成功，巨大因子才允许有原因的隔离 / Small factors must succeed; only the stress case permits explained isolation.
    assert common_prediction == 10000.0, result
    assert result["status"] == "failed" and "posterior not normalized" in result["failure_reason"], result
    assert [row["state_id"] for row in result["state_results"]] == ["A", "B"]
    failed = [row for row in result["state_results"] if row["status"] == "failed"]
    assert failed, result
    for row in result["state_results"]:
        assert row["posterior_probability"] is None, row
    for row in failed:
        assert row["log_marginal_likelihood"] is None and row["failure_reason"], row
        assert row["method_details"]["stage"] in {"optimization", "hessian", "curvature"}, row
        reason = row["failure_reason"].lower()
        assert any(word in reason for word in ("resolution", "step agreement", "curvature", "hessian", "derivative backend")), row
        assert "modelexecutionerror" not in reason and "predict failed" not in reason, row


def test_nc02_unresolved_hessian_blocks_full_posterior_after_real_optimization():
    """功能 / Function: 直接验证Hessian失效向完整后验传播 / Verify Hessian failure propagation through the full posterior.
    输入 / Inputs: 中等共同观测及旧1e-4步长，其他容差默认 / Moderate common observation and old 1e-4 step; default tolerances.
    输出 / Outputs: 真实优化成功后明确拒绝Hessian，无部分归一化 / Explicit Hessian rejection after real optimizer success, without partial normalization.
    """
    args = _model_inputs({"A": lambda eta: math.sqrt(2.0) * abs(eta),
                          "B": lambda eta: math.sqrt(math.log(3.0))}, common_prediction=10.0)
    args["numerical_options"] = {"derivatives": {"backend_options": {"initial_step": 1e-4}}}
    result = compute_posterior(**args)
    _emit("NC02 unresolved Hessian after real optimizer", result)
    assert result["status"] == "failed" and "posterior not normalized" in result["failure_reason"], result
    assert [row["state_id"] for row in result["state_results"]] == ["A", "B"]
    for row in result["state_results"]:
        assert row["status"] == "failed", row
        assert row["method_details"]["optimization"]["backend_success"] is True
        assert row["method_details"]["conditional_mode"] == [0.0]
        assert row["method_details"]["stage"] == "hessian", row
        assert row["log_marginal_likelihood"] is row["posterior_probability"] is None
        assert "derivative" in row["failure_reason"].lower(), row
