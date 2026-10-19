"""功能 / Function: 定义并检查积分及后验结果，不重新计算 / Define and validate integration/posterior results without recalculating them.
输入 / Inputs: 计算层产生的字典 / Dictionaries produced by calculation layers.
输出 / Outputs: 已校验的独立副本；非法组合抛InputError / Validated copies; inconsistent combinations raise InputError.
模型层已有返回对象和校验保持原位；TypedDict本身不执行运行时检查 / Existing model results/validators stay in place; TypedDict does not validate at runtime.
"""

from collections.abc import Mapping
from copy import deepcopy
from math import inf
from numbers import Real
from typing import Any, Literal, NotRequired, TypedDict

from .inputs.integration_method import validate_integration_method
from .inputs.priors import validate_priors
from .validation import InputError, fields, identifier, probability_scalar, scalar


class IntegrationResult(TypedDict):
    """功能 / Function: 表示一个候选的积分结果 / Represent one candidate's integration result.
    输入 / Inputs: 积分层的状态、数值与失败原因 / Integration status, value and failure reason.
    输出 / Outputs: status、log_marginal_likelihood、failure_reason及可选method_details / A plain dictionary with the declared fields.
    success为有限值或真零似然的-inf；failed为None加非空原因 / Success has a finite value or -inf for true zero likelihood; failure has None and a nonempty reason.
    """

    status: Literal["success", "failed"]
    log_marginal_likelihood: float | None
    failure_reason: str | None
    method_details: NotRequired[dict[str, Any]]


def validate_integration_result(result: Mapping[str, Any]) -> IntegrationResult:
    """功能 / Function: 检查积分结果的字段及状态一致性 / Check integration-result fields and status consistency.
    输入 / Inputs: 一个结果字典；正对数密度也合法 / One result dictionary; positive log densities are valid.
    输出 / Outputs: 独立副本；不提升优化/Hessian结果为积分成功 / Independent copy; does not promote optimizer/Hessian success to integration success.
    -inf仅表示积分层已确认的真零似然；此处不证明其来源 / -inf denotes true zero likelihood established by the integration layer; this check cannot verify its origin.
    下溢、溢出或有限求积节点全零均不能自动当成真零似然 / Underflow, overflow or zero values at all finite quadrature nodes do not establish true zero likelihood.
    """
    fields(result, {"status", "log_marginal_likelihood", "failure_reason"}, {"method_details"}, name="integration_result")
    status, reason = result["status"], result["failure_reason"]
    if not isinstance(status, str) or status not in ("success", "failed"):
        raise InputError("integration status must be success or failed")
    if status == "success":
        if reason is not None:
            raise InputError("successful integration must have failure_reason=None")
        raw_value = result["log_marginal_likelihood"]
        # 仅接纳显式负无穷，不把有限数转换溢出当成真零 / Accept explicit negative infinity, not overflow converting a finite value.
        value = -inf if isinstance(raw_value, Real) and raw_value == -inf else scalar(raw_value, name="log_marginal_likelihood")
    else:
        if result["log_marginal_likelihood"] is not None:
            raise InputError("failed integration must have log_marginal_likelihood=None")
        if not isinstance(reason, str) or not reason.strip():
            raise InputError("failed integration requires a nonempty failure_reason")
        value = None
    checked: IntegrationResult = {"status": status, "log_marginal_likelihood": value, "failure_reason": reason}
    if "method_details" in result:
        if not isinstance(result["method_details"], Mapping):
            raise InputError("method_details must be a mapping when provided")
        checked["method_details"] = deepcopy(dict(result["method_details"]))
    return checked


class StateResult(TypedDict):
    """功能 / Function: 在积分结果上标识候选及其概率 / Identify a candidate and its probabilities alongside integration fields.
    输入 / Inputs: 候选标识、先验、积分字段和后验值或None / State ID, prior, integration fields and posterior value or None.
    输出 / Outputs: 单层字典；不重复嵌套另一份积分结果 / Flat dictionary without a duplicate nested integration result.
    status描述逐候选积分或零先验跳过，不等于完整后验已经成功 / Status describes candidate integration or a zero-prior skip, not overall posterior availability.
    """

    state_id: str
    prior_probability: float
    posterior_probability: float | None
    status: Literal["success", "failed", "skipped_zero_prior"]
    log_marginal_likelihood: float | None
    failure_reason: str | None
    method_details: NotRequired[dict[str, Any]]


def validate_state_result(result: Mapping[str, Any]) -> StateResult:
    """功能 / Function: 检查逐候选结果，不计算或修正概率 / Validate one state result without calculating or correcting probabilities.
    输入 / Inputs: 完整逐候选字典 / Complete per-state dictionary.
    输出 / Outputs: 独立副本；失败后验为None，零先验的可用后验必须为0 / Independent copy; failed posteriors are None, available zero-prior posteriors must be zero.
    成功积分可暂没有后验值，由完整结果层检查是否可用 / Successful integration may lack a posterior until the overall result is validated.
    skipped_zero_prior仅表示未算积分，要求先验/后验为0，积分值及失败原因为None / skipped_zero_prior means no integration: prior/posterior are zero, integral and failure reason are None.
    """
    required = {"state_id", "prior_probability", "posterior_probability", "status", "log_marginal_likelihood", "failure_reason"}
    fields(result, required, {"method_details"}, name="state_result")
    state_id = identifier(result["state_id"], name="state_id")
    prior = probability_scalar(result["prior_probability"], name="prior_probability")
    probability = result["posterior_probability"]
    if probability is not None:
        probability = probability_scalar(probability, name="posterior_probability")
    # 跳过只属于逐候选结果，不伪装为积分成功 / A skip belongs to a state result, never to a successful integral.
    if isinstance(result["status"], str) and result["status"] == "skipped_zero_prior":
        if prior != 0 or probability != 0:
            raise InputError("skipped_zero_prior requires exactly zero prior and posterior probabilities")
        if result["log_marginal_likelihood"] is not None or result["failure_reason"] is not None:
            raise InputError("skipped_zero_prior requires log_marginal_likelihood=None and failure_reason=None")
        checked = {"status": "skipped_zero_prior", "log_marginal_likelihood": None, "failure_reason": None}
        if "method_details" in result:
            if not isinstance(result["method_details"], Mapping):
                raise InputError("method_details must be a mapping when provided")
            checked["method_details"] = deepcopy(dict(result["method_details"]))
    else:
        checked = validate_integration_result({key: result[key] for key in
                                              ("status", "log_marginal_likelihood", "failure_reason", "method_details") if key in result})
    if probability is not None:
        if checked["status"] == "failed":
            raise InputError("failed state must have posterior_probability=None")
        if prior == 0 and probability != 0:
            raise InputError("a zero-prior state cannot have positive posterior probability")
        if checked["log_marginal_likelihood"] == -inf and probability != 0:
            raise InputError("a true-zero-likelihood state cannot have positive posterior probability")
    return {**checked, "state_id": state_id, "prior_probability": prior, "posterior_probability": probability}


class PosteriorResult(TypedDict):
    """功能 / Function: 表示完整候选集合的后验输出 / Represent posterior output for the complete candidate set.
    输入 / Inputs: 整体状态、方法和所有逐候选结果 / Overall status, method and all state results.
    输出 / Outputs: status、integration_method、state_results、failure_reason及可选method_details / Plain dictionary with optional method details.
    整体失败仍保留候选及已完成积分；后验为None，只有零先验跳过项保留0 / Failure preserves candidates/integrals; posteriors are None except skipped zero-prior entries retain zero.
    """

    status: Literal["success", "failed"]
    integration_method: Literal["laplace", "ghq"]
    state_results: list[StateResult]
    failure_reason: str | None
    method_details: NotRequired[dict[str, Any]]


def validate_posterior_result(
    result: Mapping[str, Any], *, candidate_states, priors: Mapping[str, float],
) -> PosteriorResult:
    """功能 / Function: 核对完整性、先验对应及后验可用状态 / Check completeness, prior correspondence and posterior availability.
    输入 / Inputs: 输出字典和本次计算的候选/先验输入 / Output dictionary and the calculation's original candidates/priors.
    输出 / Outputs: 独立副本，保留结果顺序；拒绝漏候选和失败后的部分归一化 / Copy preserving result order; reject omissions and partial normalization after failure.
    复用概率输入校验，不重算贝叶斯权重或证明积分精度 / Reuse probability validation; do not recompute Bayesian weights or certify integral accuracy.
    """
    fields(result, {"status", "integration_method", "state_results", "failure_reason"}, {"method_details"}, name="posterior_result")
    status, reason = result["status"], result["failure_reason"]
    if not isinstance(status, str) or status not in ("success", "failed"):
        raise InputError("posterior status must be success or failed")
    if status == "success" and reason is not None:
        raise InputError("successful posterior must have failure_reason=None")
    if status == "failed" and (not isinstance(reason, str) or not reason.strip()):
        raise InputError("failed posterior requires a nonempty failure_reason")
    method = validate_integration_method(result["integration_method"])
    expected_priors = validate_priors(priors, candidate_states=candidate_states)
    rows = result["state_results"]
    if not isinstance(rows, (list, tuple)) or not rows:
        raise InputError("state_results must be a nonempty list or tuple")
    states = [validate_state_result(row) for row in rows]
    ids = [row["state_id"] for row in states]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_priors):
        raise InputError("state_results must contain every candidate exactly once, with no extra states")
    if any(row["prior_probability"] != expected_priors[row["state_id"]] for row in states):
        raise InputError("reported priors differ from the supplied priors")
    if status == "success":
        if any(row["status"] not in ("success", "skipped_zero_prior") or row["posterior_probability"] is None for row in states):
            raise InputError("successful posterior requires a complete successfully evaluated candidate set")
        if not any(row["prior_probability"] > 0 and row["log_marginal_likelihood"] != -inf for row in states):
            raise InputError("successful posterior requires positive evidence from at least one positive-prior state")
        # 仅复用概率范围与总和检查，不加权或归一化 / Reuse probability bounds/sum checks without weighting or normalization.
        try:
            validate_priors({row["state_id"]: row["posterior_probability"] for row in states}, candidate_states=candidate_states)
        except InputError as exc:
            raise InputError(f"invalid posterior probability vector: {exc}") from exc
    elif any(row["posterior_probability"] is not None for row in states if row["status"] != "skipped_zero_prior"):
        raise InputError("failed posterior must leave all posterior probabilities unavailable (None), except skipped zero-prior entries")
    checked = {"status": status, "integration_method": method, "state_results": states, "failure_reason": reason}
    if "method_details" in result:
        if not isinstance(result["method_details"], Mapping):
            raise InputError("method_details must be a mapping when provided")
        checked["method_details"] = deepcopy(dict(result["method_details"]))
    return checked
