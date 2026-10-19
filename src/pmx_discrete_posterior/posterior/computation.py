"""功能 / Function: 连接模型评价、逐候选积分和后验归一化 / Connect model evaluation, per-candidate integration and posterior normalization.
输入 / Inputs: 已约定的七组用户输入 / The seven agreed user-input groups.
输出 / Outputs: 保留全部候选的后验结果 / Posterior result preserving every candidate.
"""

import math

from ..interfaces.inputs import resolve_hessian_options, validate_inputs
from ..interfaces.protocols import ConditionalLogLikelihood
from ..interfaces.results import PosteriorResult, validate_integration_result, validate_posterior_result
from ..interfaces.validation import InputError, scalar
from ..models.likelihood import conditional_log_likelihood
from ..models.parameter_resolution import resolve_parameters
from ..models.prediction import evaluate_prediction, prepare_prediction_records
from ..models.random_effects import resolve_random_effects
from ..models.residuals import resolve_residuals
from ..numerics import integrate
from .normalization import normalize


def _bind_likelihood(*, model, individual_data, observations, candidate_state,
                     random_effect_distribution, prediction_request) -> ConditionalLogLikelihood:
    """功能 / Function: 为一个候选绑定已有模型链 / Bind the existing model chain to one candidate.
    输入 / Inputs: 已校验数据、单个候选及内部生成的效应分布/预测请求 / Validated data, one candidate and internally resolved distribution/requests.
    输出 / Outputs: 接收活动效应向量的条件似然回调；不含效应密度或状态先验 / Conditional likelihood callback excluding effect density and state priors.
    """
    def likelihood(random_effect_values):
        """功能 / Function: 在同一效应向量下联合评价全部观测 / Jointly evaluate all observations under the same effect vector.
        输入 / Inputs: 当前活动随机效应 / Current active random effects.
        输出 / Outputs: 标量条件对数似然；每点评价重新解析参数和预测内初始化 / Scalar conditional log-likelihood; resolve parameters and predictor initialization at every point.
        """
        parameters = resolve_parameters(
            parameter_model=model["parameter_model"], parameters=model["parameters"],
            covariates=individual_data["covariates"], **random_effect_distribution.expand(random_effect_values),
            occasions=individual_data["occasions"], evaluation_interval=prediction_request["evaluation_interval"],
        )
        predictions = evaluate_prediction(
            predict=model["predict"], individual_parameters=parameters, candidate_state=candidate_state,
            prediction_records=prediction_request["prediction_records"], individual_data=individual_data,
        )
        residual = resolve_residuals(
            residual_model=model["residual_model"], parameters=model["parameters"], predictions=predictions,
            prediction_records=prediction_request["prediction_records"], dependence=model["residual_model"]["dependence"],
        )
        return conditional_log_likelihood(observations=observations, predictions=predictions, residual_distribution=residual)
    return likelihood


def _ghq_posterior(rows, priors, states, settings, details):
    """功能 / Function: 用同阶完整积分检查后验稳定性 / Check posterior stability using complete matched-order integrals.
    输入 / Inputs: 成功逐状态结果、先验、候选、GHQ设置及检查记录 / Successful state results, priors, candidates, GHQ settings and diagnostics.
    输出 / Outputs: 最后一阶的概率；失败抛异常并保留已做检查 / Final-order probabilities; failure raises while retaining performed checks.
    复用normalize；不重算积分，不混用不同阶数 / Reuse normalize without reintegration or mixing orders.
    """
    active = [row for row in rows if row["prior_probability"] > 0]
    if all(row.get("method_details", {}).get("scope") == "direct" for row in active):
        return normalize({row["state_id"]: row["log_marginal_likelihood"] for row in rows}, priors, candidate_states=states)
    histories = {}
    for row in active:
        history = row.get("method_details", {}).get("order_results", [])
        if not isinstance(history, list) or [item.get("order") for item in history] != settings["orders"]:
            raise FloatingPointError("GHQ posterior requires complete matching order histories for all positive-prior states")
        if history[-1]["log_marginal_likelihood"] != row["log_marginal_likelihood"]:
            raise FloatingPointError("GHQ final integral differs from its last order estimate")
        histories[row["state_id"]] = history
    previous = None
    for index, order in enumerate(settings["orders"]):
        logs = {row["state_id"]: (histories[row["state_id"]][index]["log_marginal_likelihood"]
                                  if row["prior_probability"] > 0 else None) for row in rows}
        probabilities = normalize(logs, priors, candidate_states=states)
        delta = max(abs(probabilities[key] - previous[key]) for key in priors) if previous is not None else None
        log_roundoff = max(scalar(history[index]["estimated_log_integral_roundoff"], minimum=0,
                                 name="estimated_log_integral_roundoff") for history in histories.values())
        # softmax在log输入上的无穷范数灵敏度不超过1/2；输入误差仍只是经验估计 / Softmax has infinity-norm sensitivity at most 1/2; input errors remain empirical estimates.
        posterior_roundoff = min(1., .5 * log_roundoff + 8 * math.ulp(1.))
        details["posterior_order_results"].append({"order": order, "probabilities": probabilities, "max_change": delta,
                                                    "estimated_posterior_roundoff": posterior_roundoff})
        if posterior_roundoff > settings["posterior_atol"] / 4:
            details["posterior_stability_passed"] = False
            raise FloatingPointError("GHQ log-estimate resolution/roundoff cannot support posterior_atol")
        previous = probabilities
    details["posterior_stability_passed"] = all(item["max_change"] <= settings["posterior_atol"]
                                                for item in details["posterior_order_results"][-2:])
    if not details["posterior_stability_passed"]:
        raise FloatingPointError("GHQ final two posterior changes exceed posterior_atol")
    return probabilities


def compute_posterior(*, model, individual_data, observations, candidate_states,
                      priors, integration_method, numerical_options=None) -> PosteriorResult:
    """功能 / Function: 调度全部候选、积分并计算后验概率 / Integrate candidates and calculate posterior probabilities.
    输入 / Inputs: model模型与显式独立性声明；individual_data个体信息；observations观测历史；candidate_states候选历史；priors归一化先验；integration_method积分方法；numerical_options可选数值设置 / Model with explicit independence declarations, individual data, observations, candidate histories, normalized priors, integration method and optional numerical options.
    输出 / Outputs: 按候选顺序的PosteriorResult；非法输入抛InputError，计算失败保留逐状态积分且不作局部归一化 / Ordered PosteriorResult; invalid input raises InputError, execution failure preserves per-state integrals without partial normalization.
    支持Laplace与先验中心GHQ；零先验跳过积分并保留为零后验 / Supports Laplace and prior-centered GHQ; zero-prior states skip integration and retain zero posterior.
    """
    checked = validate_inputs(model=model, individual_data=individual_data, observations=observations,
                              candidate_states=candidate_states, priors=priors,
                              integration_method=integration_method, numerical_options=numerical_options)
    model, data = checked["model"], checked["individual_data"]
    states, priors, options = checked["candidate_states"], checked["priors"], checked["numerical_options"]
    if "random_effects_dependence" not in model or "dependence" not in model["residual_model"]:
        raise InputError("compute_posterior requires explicit model.random_effects_dependence and model.residual_model.dependence")
    effects = resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
                                     occasions=data["occasions"], dependence=model["random_effects_dependence"])
    request = prepare_prediction_records(checked["observations"], data)
    # 公共配置错误在任何候选执行前报出 / Report shared configuration errors before executing any candidate.
    if checked["integration_method"] == "laplace":
        point = options["optimization"].get("initial_point", effects.mean)
        if point.shape != effects.mean.shape:
            raise InputError("initial_point must match the active random-effect dimension")
        resolve_hessian_options(options["derivatives"])
    rows = []
    for state in states:
        state_id = state["state_id"]
        row = {"state_id": state_id, "prior_probability": priors[state_id], "posterior_probability": None}
        if priors[state_id] == 0:
            row.update(status="skipped_zero_prior", log_marginal_likelihood=None,
                       failure_reason=None, posterior_probability=0.0)
        else:
            likelihood = _bind_likelihood(model=model, individual_data=data, observations=checked["observations"],
                                          candidate_state=state, random_effect_distribution=effects, prediction_request=request)
            try:
                result = integrate(conditional_log_likelihood=likelihood, random_effect_distribution=effects,
                                   integration_method=checked["integration_method"], numerical_options=options)
                row.update(validate_integration_result(result))
            except Exception as exc:
                row.update(status="failed", log_marginal_likelihood=None,
                           failure_reason=f"integration: {type(exc).__name__}: {exc}")
        rows.append(row)
    # 不丢弃失败候选；仍完成其余候选以保留诊断 / Do not drop failed candidates; finish the remaining candidates to retain diagnostics.
    failed_ids = [row["state_id"] for row in rows if row["status"] == "failed"]
    output = {"status": "failed", "integration_method": checked["integration_method"],
              "state_results": rows, "failure_reason": None}
    if checked["integration_method"] == "ghq":
        output["method_details"] = {"posterior_atol": options["ghq"]["posterior_atol"],
                                     "posterior_stability_passed": None, "posterior_order_results": []}
    if failed_ids:
        output["failure_reason"] = f"integration failed for positive-prior candidates {failed_ids}; posterior not normalized"
    else:
        try:
            if checked["integration_method"] == "ghq":
                probabilities = _ghq_posterior(rows, priors, states, options["ghq"], output["method_details"])
            else:
                probabilities = normalize({row["state_id"]: row["log_marginal_likelihood"] for row in rows},
                                          priors, candidate_states=states)
        except (FloatingPointError, InputError, KeyError, TypeError) as exc:
            output["failure_reason"] = f"normalization or posterior stability: {exc}"
        else:
            for row in rows:
                row["posterior_probability"] = probabilities[row["state_id"]]
            output["status"] = "success"
    return validate_posterior_result(output, candidate_states=states, priors=priors)
