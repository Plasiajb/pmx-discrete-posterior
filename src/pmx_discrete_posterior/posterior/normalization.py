"""功能 / Function: 以离散先验加权并归一化 / Weight marginal likelihoods by discrete priors and normalize.
输入 / Inputs: 按候选标识对应的对数边际似然及先验 / Marginal log-likelihoods and priors keyed by candidate ID.
输出 / Outputs: 后验映射；缺失或数值失败不作为零概率 / Posterior mapping; missing values or numerical failure are not zero probabilities.
"""

from collections.abc import Mapping
import math

import numpy as np
from scipy.special import logsumexp

from ..interfaces.inputs.priors import validate_priors
from ..interfaces.results import validate_integration_result
from ..interfaces.validation import InputError


def normalize(log_marginal_likelihoods, priors, *, candidate_states):
    """功能 / Function: 在完整候选集合上计算贝叶斯权重 / Calculate Bayesian weights over the complete candidate set.
    输入 / Inputs: 同标识映射及候选；仅零先验允许积分值None / Aligned mappings and candidates; only zero-prior states may have a None integral.
    输出 / Outputs: 按候选顺序的概率字典；非法输入抛InputError，全零证据或数值下溢抛FloatingPointError / Ordered probabilities; invalid input raises InputError, zero evidence or numerical underflow raises FloatingPointError.
    内部函数，不删除失败候选、不自动修正输入先验 / Internal function; never remove failed candidates or renormalize input priors.
    """
    checked_priors = validate_priors(priors, candidate_states=candidate_states)
    if not isinstance(log_marginal_likelihoods, Mapping) or set(log_marginal_likelihoods) != set(checked_priors):
        raise InputError("log_marginal_likelihoods must contain every candidate exactly once")
    active_ids, active_logs = [], []
    for key, prior in checked_priors.items():
        value = log_marginal_likelihoods[key]
        if value is None and prior == 0:
            continue
        checked = validate_integration_result({"status": "success", "log_marginal_likelihood": value, "failure_reason": None})
        if prior > 0 and checked["log_marginal_likelihood"] != -math.inf:
            active_ids.append(key)
            active_logs.append(checked["log_marginal_likelihood"])
    if not active_ids:
        raise FloatingPointError("all positive-prior candidates have zero marginal likelihood; posterior is undefined")
    # 先去掉共同似然常数，再加log先验；避免巨大常数吞掉先验差异 / Center likelihoods before adding log priors so large offsets do not erase prior differences.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        logs = np.asarray(active_logs)
        weights = (logs - np.max(logs)) + np.array([math.log(checked_priors[key]) for key in active_ids])
        centered = weights - np.max(weights)
        probabilities = np.exp(centered - logsumexp(centered))
    if np.any(probabilities == 0):
        lost = [key for key, probability in zip(active_ids, probabilities) if probability == 0]
        raise FloatingPointError(f"nonzero posterior probability underflow for candidates {lost}")
    output = dict.fromkeys(checked_priors, 0.0)
    output.update((key, float(probability)) for key, probability in zip(active_ids, probabilities))
    try:
        return validate_priors(output, candidate_states=candidate_states)
    except InputError as exc:
        raise FloatingPointError(f"posterior normalization produced invalid probabilities: {exc}") from exc
