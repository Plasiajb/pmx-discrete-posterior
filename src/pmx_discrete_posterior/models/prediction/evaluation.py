"""功能 / Function: 准备预测请求、调用预测器并校验结果 / Prepare requests, invoke a predictor and validate results.
输入 / Inputs: 已解析参数、候选事件和预测上下文 / Resolved parameters, candidate events and context.
输出 / Outputs: 无残差预测；不向回调传入实测值或原始效应 / Noise-free predictions without observed values or raw effects.
"""

from copy import deepcopy

from ...interfaces.inputs.observations import validate_records
from ...interfaces.inputs.candidate_states import validate_candidate_history, validate_event
from ...interfaces.inputs.individual_data import validate_prediction_context
from ...interfaces.validation import (
    InputError, ModelExecutionError, fields, identifier, names, numeric_array, scalar,
)
from ..parameter_resolution import validate_individual_parameters


def prepare_prediction_records(observations, individual_data):
    """功能 / Function: 去掉实测值并确定评价范围 / Remove observations' values and establish interval.
    输入 / Inputs: 实际观测、初始化上下文 / Observations and initialization context.
    输出 / Outputs: prediction_records、evaluation_interval / Prediction requests and evaluation interval.
    """
    records = validate_records(observations, observed=True)
    context = validate_prediction_context(individual_data, records[0]["time_unit"])
    start = context["initialization"]["start_time"]
    if any(r["time"] < start for r in records):
        raise InputError("observations cannot precede initialization")
    return {"prediction_records": [{k: v for k, v in r.items() if k != "value"} for r in records],
            "evaluation_interval": (start, max(r["time"] for r in records))}


def validate_predictions(predictions, prediction_records):
    """功能 / Function: 防止预测与观测错位 / Prevent prediction-observation misalignment.
    输入 / Inputs: predictions与请求记录 / Predictions and requests.
    输出 / Outputs: 有限、非负、按序向量 / Finite nonnegative ordered vector; no clipping.
    """
    records = validate_records(prediction_records, observed=False)
    fields(predictions, {"record_ids", "values", "units"}, name="predictions")
    ids = names(predictions["record_ids"], name="prediction IDs")
    if ids != tuple(r["record_id"] for r in records):
        raise InputError("prediction identifiers must match request order exactly")
    units = predictions["units"]
    if not isinstance(units, (list, tuple)) or list(units) != [r["value_unit"] for r in records]:
        raise InputError("prediction units must match each requested record")
    values = numeric_array(predictions["values"], name="concentration predictions")
    if values.shape != (len(records),) or (values < 0).any():
        raise InputError("one nonnegative prediction per record is required; no clipping")
    return {"record_ids": ids, "values": values, "units": tuple(units)}


def evaluate_prediction(*, predict, individual_parameters, candidate_state, prediction_records, individual_data):
    """功能 / Function: 在隔离输入下调用预测器 / Invoke predictor with isolated inputs.
    输入 / Inputs: predict、个体参数、候选事件、预测请求及上下文 / Predictor and explicit context.
    输出 / Outputs: 已校验predictions；输入错误和回调失败分开 / Validated predictions; distinct input and execution errors.
    """
    if not callable(predict):
        raise InputError("predict must be callable")
    records = validate_records(prediction_records, observed=False)
    context = validate_prediction_context(individual_data, records[0]["time_unit"])
    start, end = context["initialization"]["start_time"], max(r["time"] for r in records)
    if min(r["time"] for r in records) < start:
        raise InputError("prediction request precedes initialization")
    ip = validate_individual_parameters(individual_parameters, (start, end))
    state = validate_candidate_history(candidate_state, initialization_start=start,
                                       common_events=context["dosing_prehistory"]["events"])
    try:
        result = predict(individual_parameters=deepcopy(ip), candidate_state=deepcopy(state),
                         prediction_records=deepcopy(records), individual_data=deepcopy(context))
        return validate_predictions(result, records)
    except Exception as exc:
        raise ModelExecutionError(f"predict failed: {exc}") from exc
