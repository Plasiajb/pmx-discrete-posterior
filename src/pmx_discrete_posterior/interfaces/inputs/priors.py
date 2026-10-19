"""功能 / Function: 离散状态先验输入契约 / Discrete-state prior contracts.
输入 / Inputs: 按state_id对应的归一化概率 / Normalized probabilities keyed by state_id.
输出 / Outputs: 独立有序副本，保留零先验状态 / Independent ordered copy preserving zero-prior states.
"""

from collections.abc import Mapping
import math

from ..validation import InputError, probability_scalar
from .candidate_states import validate_candidate_states


def validate_priors(priors, *, candidate_states):
    """功能 / Function: 检查先验概率与候选一一对应 / Check one prior probability per candidate.
    输入 / Inputs: priors映射和candidate_states；每项在[0,1] / Prior mapping and states; each probability in [0,1].
    输出 / Outputs: 按候选顺序复制原概率，不归一化、不删除零项 / Copy in state order, no normalization or zero removal.
    总和仅容许8个float64在1处的ULP舍入差 / Sum tolerance is eight float64 ULPs at one.
    单项概率转换不能将非零值变为零或掩盖越界 / Conversion must not turn nonzero probabilities into zero or hide invalid bounds.
    """
    states = validate_candidate_states(candidate_states)
    ids = [state["state_id"] for state in states]
    if not isinstance(priors, Mapping) or set(priors) != set(ids):
        raise InputError("priors must contain exactly the candidate state identifiers")
    result = {key: probability_scalar(priors[key], name=f"prior for {key}") for key in ids}
    total = math.fsum(result.values())
    if abs(total - 1.0) > 8 * math.ulp(1.0):
        raise InputError("priors must sum to one; unnormalized weights are not accepted")
    return result
