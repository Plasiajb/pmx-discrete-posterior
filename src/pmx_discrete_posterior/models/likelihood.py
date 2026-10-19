"""功能 / Function: 评价已解析的条件观测分布 / Evaluate a resolved observation distribution.
输入 / Inputs: 观测、预测、残差分布 / Observations, predictions and residual distribution.
输出 / Outputs: 含归一化常数的条件对数似然 / Conditional log-likelihood including normalizing constants.
"""

import math

import numpy as np
from scipy.stats import norm

from ..interfaces.validation import InputError, ModelExecutionError, fields, names, numeric_array
from .prediction import validate_predictions, validate_records


def conditional_log_likelihood(*, observations, predictions, residual_distribution):
    """功能 / Function: 累加明确条件独立的高斯对数密度 / Sum explicitly conditionally independent normal log-densities.
    输入 / Inputs: 同一记录顺序和单位的三组数据 / Three inputs with matching identities, order and units.
    输出 / Outputs: 有限标量；不包含任何先验，不重算残差尺度 / Finite scalar without priors or residual-scale reconstruction.
    """
    observations = validate_records(observations, observed=True)
    requests = [{k: v for k, v in record.items() if k != "value"} for record in observations]
    predictions = validate_predictions(predictions, requests)
    fields(residual_distribution, {"distribution", "dependence", "record_ids", "means", "sd", "units"}, name="residual_distribution")
    if residual_distribution["distribution"] != "normal" or residual_distribution["dependence"] != "conditionally_independent":
        raise InputError("only explicitly conditionally independent normal residuals are supported")
    if names(residual_distribution["record_ids"]) != predictions["record_ids"]:
        raise InputError("residual record IDs differ from predictions")
    units = residual_distribution["units"]
    if not isinstance(units, (list, tuple)) or tuple(units) != predictions["units"]:
        raise InputError("residual units differ from predictions")
    means = numeric_array(residual_distribution["means"], name="residual means")
    sd = numeric_array(residual_distribution["sd"], name="residual SDs")
    if not np.array_equal(means, predictions["values"]):
        raise InputError("residual distribution was resolved for different predictions")
    if sd.shape != means.shape or (sd <= 0).any():
        raise InputError("one finite positive residual SD is required for each record")
    values = np.array([record["value"] for record in observations])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        terms = norm.logpdf(values, loc=means, scale=sd)
    if not np.isfinite(terms).all():
        raise ModelExecutionError("normal log-density overflow; do not substitute zero likelihood")
    try:
        result = math.fsum(float(term) for term in terms)
    except OverflowError as exc:
        raise ModelExecutionError("log-likelihood accumulation overflow") from exc
    if not math.isfinite(result):
        raise ModelExecutionError("nonfinite conditional log-likelihood")
    return result
