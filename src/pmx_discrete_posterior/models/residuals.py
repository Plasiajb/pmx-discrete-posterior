"""功能 / Function: 解析条件残差尺度 / Resolve conditional residual scales.
输入 / Inputs: 显式高斯残差规则、参数引用及预测 / Gaussian residual rules, parameter references and predictions.
输出 / Outputs: 已解析分布，不读取观测值 / Resolved distribution without reading observed values.
"""

import numpy as np

from ..interfaces.inputs.model_inputs import validate_residual_model
from ..interfaces.validation import InputError, validate_parameters
from .prediction import validate_predictions


def resolve_residuals(*, residual_model, parameters, predictions, prediction_records, dependence):
    """功能 / Function: 构造正态加性或比例误差 / Resolve normal additive or proportional error.
    输入 / Inputs: sd（浓度单位）或coefficient_sd（无量纲）；明确条件独立 / SD or dimensionless coefficient SD, explicit conditional independence.
    输出 / Outputs: means、sd、record_ids、units及分布声明 / Resolved means, SDs, identities, units and declaration.
    """
    parameters = validate_parameters(parameters)
    residual_model = validate_residual_model(residual_model, parameters=parameters)
    if dependence != "conditionally_independent":
        raise InputError("declare conditionally_independent residuals; correlated residuals are unsupported")
    error_type = residual_model["error_type"]
    role = "sd" if error_type == "additive" else "coefficient_sd"
    ref = residual_model["parameter_names"][role]
    coefficient = parameters["values"][ref]
    predictions = validate_predictions(predictions, prediction_records)
    unit = predictions["units"][0] if error_type == "additive" else "1"
    if parameters["units"][ref] != unit:
        raise InputError("residual SD unit does not match its declared role")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        sd = (np.full(len(predictions["values"]), coefficient) if error_type == "additive"
              else coefficient * predictions["values"])
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise InputError("residual scales must be finite and positive; no floor or zero-variance substitution")
    return {"distribution": "normal", "dependence": dependence, "record_ids": predictions["record_ids"],
            "means": predictions["values"].copy(), "sd": sd, "units": predictions["units"]}
