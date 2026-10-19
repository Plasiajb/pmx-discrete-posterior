"""功能 / Function: 验收原API-01至04，不修改源码 / Accept original API-01 through 04 without source edits.
输入 / Inputs: 显式事件、归一化先验及零维解析模型 / Explicit events, normalized priors and a zero-dimensional analytic model.
输出 / Outputs: 顶层真实调用断言；API-03仅证明路径隔离 / Real public-call assertions; API-03 establishes route isolation only.
"""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import pytest

import pmx_discrete_posterior as pmx
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.prediction.linear import LinearPredictor


def _event(amount=2.0, time=0.0):
    """功能 / Function: 创建显式静脉剂量 / Create an explicit IV dose.
    输入 / Inputs: mg药量与h时刻 / Amount in mg and time in h.
    输出 / Outputs: 单个事件字典 / One event dictionary.
    """
    return {"time": time, "amount": amount, "amount_unit": "mg", "route": "iv", "compartment": "central"}


@pytest.fixture(params=["laplace", "ghq"])
def case(request):
    """功能 / Function: 创建真实零维房室流程 / Create a real zero-dimensional compartment workflow.
    输入 / Inputs: 积分路线；两者均走零维直接似然 / Integration route; both use direct zero-dimensional likelihood.
    输出 / Outputs: 七组输入及调用记录；不模拟积分器 / Seven input groups and call records; no integrator mocks.
    """
    calls = {"relations": [], "predict": []}

    def relations(*, parameters, covariates, iiv_values, iov_values, occasions, evaluation_interval):
        """功能 / Function: 解析固定一房室参数 / Resolve fixed one-compartment parameters.
        输入 / Inputs: 参数与评价区间；无随机效应 / Parameters and evaluation interval; no random effects.
        输出 / Outputs: 覆盖区间的参数段，并记录调用 / Parameter segment covering the interval and a call record.
        """
        assert covariates == {} and iiv_values == {} and iov_values == {} and occasions == []
        calls["relations"].append(evaluation_interval)
        return {"units": {"k": "1/h", "V": "L"},
                "segments": [{"start": evaluation_interval[0], "end": evaluation_interval[1],
                              "occasion_id": None, "values": {key: parameters["values"][key] for key in ("k", "V")}}]}

    def system(values, units):
        """功能 / Function: 定义一房室消除系统 / Define one-compartment elimination.
        输入 / Inputs: k单位1/h、V单位L / k in 1/h and V in L.
        输出 / Outputs: 药量单位mg、浓度单位mg/L的线性系统 / Linear system in mg and mg/L.
        """
        assert units == {"k": "1/h", "V": "L"}
        return {"state_names": ["central"], "amount_unit": "mg", "time_unit": "h", "concentration_unit": "mg/L",
                "matrix": [[-values["k"]]], "observation_vector": [1.0 / values["V"]],
                "inputs": {("iv", "central"): {"fractions": [1.0], "lag": 0.0}}}

    linear = LinearPredictor(system)

    def predict(*, individual_parameters, candidate_state, prediction_records, individual_data):
        """功能 / Function: 记录并调用现有房室预测器 / Record and call the existing compartment predictor.
        输入 / Inputs: 标准四组预测输入 / The four standard predictor inputs.
        输出 / Outputs: 原始预测及事件记录，不解码ID / Original predictions and event records; no ID decoding.
        """
        result = linear(individual_parameters=individual_parameters, candidate_state=candidate_state,
                        prediction_records=prediction_records, individual_data=individual_data)
        calls["predict"].append({"state": deepcopy(candidate_state), "values": result["values"].tolist()})
        return result

    args = {
        "model": {"parameters": {"values": {"k": math.log(2.0), "V": 1.0, "sigma": 1.0},
                                  "units": {"k": "1/h", "V": "L", "sigma": "mg/L"}},
                  "parameter_model": {"covariate_names": [], "relations": relations},
                  "predict": predict, "iiv": None, "iov": None,
                  "random_effects_dependence": "independent_blocks",
                  "residual_model": {"distribution": "normal", "error_type": "additive",
                                     "parameter_names": {"sd": "sigma"}, "dependence": "conditionally_independent"}},
        "individual_data": {"covariates": {}, "occasions": [], "dosing_prehistory": {"events": []},
                            "initialization": {"mode": "zero", "start_time": 0.0, "event_phase": "before_events"}},
        "observations": [{"record_id": "r1", "time": 1.0, "value": 1.0, "time_unit": "h", "value_unit": "mg/L"}],
        "candidate_states": [{"state_id": sid, "events": [_event()]} for sid in ("a", "b")],
        "priors": {"a": 0.5, "b": 0.5}, "integration_method": request.param,
    }
    if request.param == "ghq":
        args["numerical_options"] = {"ghq": {"variant": "prior_centered", "orders": [3, 5, 7],
                                             "log_integral_atol": 1e-8, "posterior_atol": 1e-8, "max_nodes": 32}}
    return args, calls


def _assert_success(result, args, concentrations):
    """功能 / Function: 用独立正态公式核对完整结果 / Check complete results against an independent normal formula.
    输入 / Inputs: 结果、调用输入、每状态解析浓度mg/L / Result, inputs and analytic concentrations in mg/L per state.
    输出 / Outputs: 逐状态结果；断言对数似然及后验绝对误差不超过1e-12 / State rows; log-likelihood and posterior absolute errors at most 1e-12.
    """
    assert result["status"] == "success", result
    assert result["integration_method"] == args["integration_method"]
    assert result.get("failure_reason") is None
    assert [row["state_id"] for row in result["state_results"]] == [state["state_id"] for state in args["candidate_states"]]
    logs = {sid: -0.5 * math.log(2.0 * math.pi) - 0.5 * (1.0 - c) ** 2 for sid, c in concentrations.items()}
    weights = {sid: args["priors"][sid] * math.exp(logs[sid]) for sid in logs}
    denominator = math.fsum(weights.values())
    rows = {row["state_id"]: row for row in result["state_results"]}
    for sid, row in rows.items():
        assert row["prior_probability"] == args["priors"][sid]
        assert row["posterior_probability"] == pytest.approx(weights[sid] / denominator, abs=1e-12, rel=0)
        if args["priors"][sid] == 0:
            assert row == {"state_id": sid, "prior_probability": 0.0, "posterior_probability": 0.0,
                           "status": "skipped_zero_prior", "log_marginal_likelihood": None, "failure_reason": None}
        else:
            assert row["status"] == "success" and row.get("failure_reason") is None
            assert row["log_marginal_likelihood"] == pytest.approx(logs[sid], abs=1e-12, rel=0)
            assert row["method_details"]["dimension"] == 0
            assert row["method_details"]["scope"] == "direct"
    assert math.fsum(row["posterior_probability"] for row in rows.values()) == pytest.approx(1.0, abs=1e-12, rel=0)
    return rows


@pytest.mark.parametrize("priors", [
    pytest.param({"a": -1.0, "b": 1.0}, id="negative-original"),
    pytest.param({"a": -0.1, "b": 1.1}, id="negative-sums-one"),
    pytest.param({"a": math.nan, "b": 1.0}, id="nan"),
    pytest.param({"a": math.inf, "b": 1.0}, id="positive-infinity"),
    pytest.param({"a": -math.inf, "b": 1.0}, id="negative-infinity"),
    pytest.param({"a": 0.0, "b": 0.0}, id="all-zero"),
    pytest.param({"a": 2.0, "b": 1.0}, id="unnormalized-weights-now-unsupported"),
    pytest.param({"a": 0.2, "b": 0.3}, id="sum-below-one"),
    pytest.param({"a": 1.0}, id="missing-b"),
    pytest.param({"b": 1.0}, id="missing-a"),
    pytest.param({"a": 0.5, "b": 0.5, "unknown": 0.0}, id="unknown-zero-key"),
    pytest.param({"unknown": 1.0}, id="unknown-only"),
    pytest.param({"a": True, "b": 0.0}, id="boolean"),
    pytest.param({"a": "0.5", "b": 0.5}, id="string"),
])
def test_api01_invalid_priors_rejected_before_model(case, priors):
    """功能 / Function: 从公开入口拒绝非法先验 / Reject invalid priors at the public entry.
    输入 / Inputs: 非法数值、总和或状态键 / Invalid values, sums or state keys.
    输出 / Outputs: InputError且模型未调用 / InputError with no model calls.
    """
    args, calls = case
    args["priors"] = priors
    with pytest.raises(InputError, match="prior"):
        pmx.compute_posterior(**args)
    assert calls == {"relations": [], "predict": []}


@pytest.mark.parametrize("prior_a", [0.0, 1.0, 0.3, 0.5, 1e-300])
def test_api01_normalized_and_zero_priors_complete_flow(case, prior_a):
    """功能 / Function: 核对合法零先验、偏斜先验及极小正概率 / Check legal zero, unequal and tiny positive priors.
    输入 / Inputs: 等似然及归一化先验 / Equal likelihoods and normalized priors.
    输出 / Outputs: 后验等于先验，零项保留且不执行 / Posterior equals prior; zero rows remain without execution.
    """
    args, calls = case
    args["priors"] = {"b": 1.0 - prior_a, "a": prior_a}
    before = deepcopy(args)
    rows = _assert_success(pmx.compute_posterior(**args), args, {"a": 1.0, "b": 1.0})
    assert args == before
    assert {item["state"]["state_id"] for item in calls["predict"]} == {sid for sid, prior in args["priors"].items() if prior > 0}
    for sid, prior in args["priors"].items():
        if prior > 0:
            assert rows[sid]["posterior_probability"] > 0
            assert rows[sid]["posterior_probability"] == pytest.approx(prior, abs=0, rel=1e-12)
    print("API-01 equal-likelihood probabilities:", {sid: row["posterior_probability"] for sid, row in rows.items()})


def test_api01_zero_prior_can_be_restored_by_fresh_public_call(case):
    """功能 / Function: 核对零先验恢复不会复用空积分 / Check restoring a zero prior recomputes its likelihood.
    输入 / Inputs: 先验先为0/1，再改为0.3/0.7 / Priors first 0/1, then 0.3/0.7.
    输出 / Outputs: 两轮完整结果，恢复后真实评价该候选 / Two complete results with real evaluation after restoration.
    """
    args, calls = case
    args["candidate_states"][1]["events"] = [_event(4.0)]
    args["priors"] = {"a": 0.0, "b": 1.0}
    _assert_success(pmx.compute_posterior(**args), args, {"a": 1.0, "b": 2.0})
    assert {item["state"]["state_id"] for item in calls["predict"]} == {"b"}
    calls["predict"].clear()
    args["priors"] = {"a": 0.3, "b": 0.7}
    rows = _assert_success(pmx.compute_posterior(**args), args, {"a": 1.0, "b": 2.0})
    assert {item["state"]["state_id"] for item in calls["predict"]} == {"a", "b"}
    assert rows["a"]["posterior_probability"] > 0.3


@pytest.mark.parametrize("state_id", ["omega11", "omega_11", "omega2", "omega0", "custom-prefix:01", "arbitrary/id-x"])
def test_api02_identical_events_ignore_id_spelling(case, state_id):
    """功能 / Function: 旧样式或任意ID不得改变事件含义 / Legacy-looking or arbitrary IDs cannot change event meaning.
    输入 / Inputs: 同一显式事件和不同ID / Identical explicit events and different IDs.
    输出 / Outputs: 原样事件、相同浓度和相同对数似然 / Unchanged events, equal concentrations and log likelihoods.
    """
    args, calls = case
    args["candidate_states"][0]["state_id"] = state_id
    args["priors"] = {state_id: 0.3, "b": 0.7}
    before = deepcopy(args)
    rows = _assert_success(pmx.compute_posterior(**args), args, {state_id: 1.0, "b": 1.0})
    assert rows[state_id]["log_marginal_likelihood"] == rows["b"]["log_marginal_likelihood"]
    assert calls["predict"]
    for item in calls["predict"]:
        assert item["state"]["events"] == [_event()]
        assert item["values"] == pytest.approx([1.0], abs=1e-12, rel=0)
    assert args == before


@pytest.mark.parametrize("events,concentration", [
    pytest.param([], 0.0, id="empty-events-not-bits"),
    pytest.param([_event(4.0)], 2.0, id="amount-controls-prediction"),
    pytest.param([_event(2.0, 0.5)], math.sqrt(2.0), id="time-controls-prediction"),
    pytest.param([_event(1.0), _event(1.0, 0.5)], 0.5 + 1.0 / math.sqrt(2.0), id="two-explicit-events"),
])
def test_api02_changed_events_change_result_with_fixed_ids(case, events, concentration):
    """功能 / Function: 反对照排除忽略事件的常数实现 / Negative control excludes an implementation ignoring events.
    输入 / Inputs: 固定ID，但改变给药量、时间或事件数 / Fixed IDs with changed amounts, times or event counts.
    输出 / Outputs: 浓度与后验按解析公式变化；重排仍按ID对应 / Analytically changed concentrations/posteriors; reordering preserves ID correspondence.
    """
    args, calls = case
    args["candidate_states"] = [{"state_id": "omega11", "events": events}, {"state_id": "omega00", "events": [_event()]}]
    args["priors"] = {"omega00": 0.5, "omega11": 0.5}
    rows = _assert_success(pmx.compute_posterior(**args), args, {"omega11": concentration, "omega00": 1.0})
    assert rows["omega11"]["log_marginal_likelihood"] != rows["omega00"]["log_marginal_likelihood"]
    assert rows["omega11"]["posterior_probability"] < 0.5
    for item in calls["predict"]:
        expected = concentration if item["state"]["state_id"] == "omega11" else 1.0
        assert item["values"] == pytest.approx([expected], abs=1e-12, rel=0)
    args["candidate_states"].reverse()
    reordered = _assert_success(pmx.compute_posterior(**args), args, {"omega11": concentration, "omega00": 1.0})
    assert rows == reordered


@pytest.mark.parametrize("candidate", [
    {"state_id": "omega11"}, {"state_id": "omega2", "events": "11"}, {"state_id": "omega_11", "events": None},
])
def test_api02_label_without_explicit_events_is_rejected(case, candidate):
    """功能 / Function: 拒绝依赖ID补全给药历史 / Reject requests relying on IDs to fill dose histories.
    输入 / Inputs: 缺失或非序列events / Missing or non-sequence events.
    输出 / Outputs: InputError且无模型调用 / InputError without model calls.
    """
    args, calls = case
    args["candidate_states"][0] = candidate
    args["priors"] = {candidate["state_id"]: 0.5, "b": 0.5}
    with pytest.raises(InputError):
        pmx.compute_posterior(**args)
    assert calls == {"relations": [], "predict": []}


def test_api03_legacy_reporting_and_oxc_paths_are_not_exposed():
    """功能 / Function: 核对旧报告及字符串解码路径隔离 / Check isolation of old reporting and string-decoding paths.
    输入 / Inputs: 当前包命名空间与模块定位 / Current package namespace and module discovery.
    输出 / Outputs: 旧模块/导出不存在；不声称旧实现已修复 / Old modules/exports absent; no claim of repairing old implementations.
    """
    assert pmx.__all__ == ["compute_posterior"]
    for module in ("reporting", "oxc"):
        assert importlib.util.find_spec(f"pmx_discrete_posterior.{module}") is None
    for name in ("NoCallRule", "apply_no_call_rule", "summarize_posterior_evidence", "build_case_card",
                 "threshold_sensitivity_panel", "reweight_prior_sensitivity",
                 "OxcOneEtaDoseHistoryModel", "build_oxc_dose_history_states"):
        assert not hasattr(pmx, name), name


@pytest.mark.parametrize("keyword", ["no_call_rule", "margin_threshold", "entropy_threshold"])
def test_api03_legacy_nan_reporting_arguments_are_rejected(case, keyword):
    """功能 / Function: 旧NaN报告参数不能被静默接收 / Legacy NaN reporting arguments cannot be silently accepted.
    输入 / Inputs: 公开入口额外报告参数 / Extra reporting argument at the public entry.
    输出 / Outputs: TypeError且不执行模型；不是阈值校验修复 / TypeError without model execution; not a threshold-validation repair.
    """
    args, calls = case
    with pytest.raises(TypeError, match="unexpected keyword"):
        pmx.compute_posterior(**args, **{keyword: math.nan})
    assert calls == {"relations": [], "predict": []}


def test_api03_reporting_not_smuggled_into_numerical_options(case):
    """功能 / Function: 排除数值设置中的隐式报告路径 / Exclude an implicit reporting route in numerical options.
    输入 / Inputs: 未支持reporting组及NaN阈值 / Unsupported reporting group with NaN thresholds.
    输出 / Outputs: InputError且不执行模型 / InputError without model execution.
    """
    args, calls = case
    args.setdefault("numerical_options", {})["reporting"] = {"margin_threshold": math.nan, "entropy_threshold": math.nan}
    with pytest.raises(InputError, match="unsupported"):
        pmx.compute_posterior(**args)
    assert calls == {"relations": [], "predict": []}


def test_api03_equal_posterior_has_no_reporting_decision(case):
    """功能 / Function: 概率相等不生成报告或强制决策 / Equal probabilities do not generate a report or forced decision.
    输入 / Inputs: 两个等似然候选与等先验 / Two equal-likelihood candidates and equal priors.
    输出 / Outputs: 成功的0.5/0.5概率，无no-call或判读字段 / Successful 0.5/0.5 probabilities without no-call or interpretation fields.
    """
    args, _ = case
    result = pmx.compute_posterior(**args)
    _assert_success(result, args, {"a": 1.0, "b": 1.0})
    forbidden = {"no_call", "map_state_id", "interpretation", "margin", "entropy", "report", "no_call_reasons"}
    assert forbidden.isdisjoint(result)
    assert all(forbidden.isdisjoint(row) for row in result["state_results"])


def _place_dose(args, location, amount):
    """功能 / Function: 在指定公开剂量载体放置事件 / Put an event in a selected public dose carrier.
    输入 / Inputs: 顶层输入、载体和mg剂量 / Public inputs, carrier and dose in mg.
    输出 / Outputs: 原位修改测试输入；不改生产文件 / Mutated test inputs only; no production files changed.
    """
    if location in ("candidate", "zero_prior_candidate"):
        args["candidate_states"][0]["events"] = [_event(amount)]
        if location == "zero_prior_candidate":
            args["priors"] = {"a": 0.0, "b": 1.0}
    elif location == "common":
        args["individual_data"]["dosing_prehistory"]["events"] = [_event(amount, 0.25)]
    else:
        assert location == "periodic"
        periodic = _event(amount)
        periodic["offset"] = periodic.pop("time")
        args["individual_data"]["initialization"] = {
            "mode": "steady_state", "start_time": 0.0, "event_phase": "before_events",
            "steady_state": {"parameter_rule": "fixed_reference", "reference_occasion_id": None},
        }
        args["individual_data"]["dosing_prehistory"]["regimen"] = {
            "period": 2.0, "anchor_time": 0.0, "time_unit": "h", "events": [periodic],
        }


@pytest.mark.parametrize("location", ["candidate", "zero_prior_candidate", "common", "periodic"])
@pytest.mark.parametrize("amount", [-300.0, math.nan, math.inf, -math.inf], ids=["negative", "nan", "positive-infinity", "negative-infinity"])
def test_api04_invalid_doses_rejected_before_model(case, location, amount):
    """功能 / Function: 非法剂量不得截成零或藏于零先验 / Invalid doses cannot be clipped to zero or hidden behind a zero prior.
    输入 / Inputs: 负值/非有限剂量及四种载体 / Negative/nonfinite doses in four carriers.
    输出 / Outputs: 顶层InputError，未进入模型 / Public InputError before model evaluation.
    """
    args, calls = case
    _place_dose(args, location, amount)
    with pytest.raises(InputError, match="dose"):
        pmx.compute_posterior(**args)
    assert calls == {"relations": [], "predict": []}


@pytest.mark.parametrize("location", ["candidate", "common", "periodic"])
@pytest.mark.parametrize("amount", [0.0, 2.0], ids=["legal-zero", "positive-control"])
def test_api04_legal_zero_and_positive_doses_complete_flow(case, location, amount):
    """功能 / Function: 零剂量和正剂量均能完成真实模型链 / Zero and positive doses complete the real model chain.
    输入 / Inputs: 三种显式载体中的0或2mg / Zero or 2 mg in three explicit carriers.
    输出 / Outputs: 解析浓度、正态似然与后验一致，无零剂量误拒绝 / Analytic concentration, normal likelihood and posterior agreement, without rejecting zero.
    """
    args, calls = case
    _place_dose(args, location, amount)
    if location == "candidate":
        concentrations = {"a": amount / 2.0, "b": 1.0}
    elif location == "common":
        concentrations = {sid: 1.0 + amount * 2.0 ** (-0.75) for sid in ("a", "b")}
    else:
        # 功能 / Function: 计算边界前周期残留 / Compute periodic carryover before the boundary.
        # 输入 / Inputs: 周期2h、半衰期1h / Period 2 h, half-life 1 h.
        # 输出 / Outputs: 1h观测残留amount/6，不自动续给药 / Carryover amount/6 at 1 h, without automatic future dosing.
        concentrations = {sid: 1.0 + amount / 6.0 for sid in ("a", "b")}
    before = deepcopy(args)
    _assert_success(pmx.compute_posterior(**args), args, concentrations)
    assert args == before
    assert calls["predict"]
    for item in calls["predict"]:
        assert item["values"] == pytest.approx([concentrations[item["state"]["state_id"]]], abs=1e-12, rel=0)
    print("API-04 analytic concentrations:", location, amount, concentrations)
