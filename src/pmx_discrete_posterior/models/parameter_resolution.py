"""功能 / Function: 接入个体参数关系 / Adapt individual parameter relationships.
输入 / Inputs: 参数、协变量、当前IIV/IOV / Parameters, covariates, current IIV/IOV.
输出 / Outputs: 覆盖评价区间的参数分段 / Parameter segments covering the evaluation interval.
"""

from collections.abc import Mapping
from copy import deepcopy

from ..interfaces.inputs.model_inputs import validate_parameter_model
from ..interfaces.inputs.individual_data import validate_occasions
from ..interfaces.validation import (
    InputError, ModelExecutionError, fields, identifier, names, numeric_array, scalar, validate_parameters,
)




def validate_individual_parameters(individual_parameters, evaluation_interval):
    """功能 / Function: 校验连续参数分段 / Validate contiguous parameter segments.
    输入 / Inputs: units、segments；有限评价区间 / Units, segments and finite interval.
    输出 / Outputs: 独立副本 / Independent copy, or InputError.
    """
    fields(individual_parameters, {"units", "segments"}, name="individual_parameters")
    evaluation_interval = numeric_array(evaluation_interval, name="evaluation_interval")
    if evaluation_interval.shape != (2,):
        raise InputError("evaluation_interval must have two endpoints")
    lower, upper = (scalar(x, name="evaluation endpoint") for x in evaluation_interval)
    if lower > upper:
        raise InputError("evaluation_interval must be ordered")
    segments = individual_parameters["segments"]
    if not isinstance(segments, (list, tuple)) or not segments:
        raise InputError("segments must be a nonempty sequence")
    result = []
    for segment in segments:
        fields(segment, {"start", "end", "occasion_id", "values"}, name="parameter segment")
        start, end = scalar(segment["start"]), scalar(segment["end"])
        if start >= end or (result and start != result[-1]["end"]):
            raise InputError("parameter segments must have positive width and be contiguous")
        occ = segment["occasion_id"]
        if occ is not None:
            identifier(occ, name="segment occasion_id")
        validated = validate_parameters({"values": segment["values"], "units": individual_parameters["units"]})
        if not validated["values"]:
            raise InputError("resolved individual parameters cannot be empty")
        result.append({"start": start, "end": end, "occasion_id": occ, "values": validated["values"]})
    if result[0]["start"] > lower or result[-1]["end"] < upper:
        raise InputError("resolved parameters do not cover evaluation/initialization interval")
    return {"units": dict(validated["units"]), "segments": result}


def parameters_at(individual_parameters, time):
    """功能 / Function: 选择当前参数段 / Select the current parameter segment.
    输入 / Inputs: 已校验参数及时间 / Validated parameters and time.
    输出 / Outputs: 该段；边界采用新段，末端闭合 / Segment; new segment at boundaries, final endpoint included.
    """
    time = scalar(time, name="parameter time")
    segments = individual_parameters["segments"]
    for i, segment in enumerate(segments):
        if segment["start"] <= time < segment["end"] or (i == len(segments) - 1 and time == segment["end"]):
            return deepcopy(segment)
    raise InputError("time outside resolved parameter segments; no extrapolation")


def resolve_parameters(*, parameter_model, parameters, covariates, iiv_values,
                       iov_values, occasions, evaluation_interval):
    """功能 / Function: 调用一次用户参数关系 / Call the user's parameter relationship once.
    输入 / Inputs: 显式参数、具名效应、协变量和时间上下文 / Explicit parameters, named effects, covariates and time context.
    输出 / Outputs: individual_parameters；回调错误分类保留 / Resolved parameters, with explicit execution errors.
    """
    parameter_model = validate_parameter_model(parameter_model)
    required = parameter_model["covariate_names"]
    if not isinstance(covariates, Mapping) or set(covariates) != set(required):
        raise InputError("covariates must match covariate_names exactly")
    p = validate_parameters(parameters)
    covariates = {k: scalar(covariates[k], name=k) for k in required}
    if not isinstance(iiv_values, Mapping) or not isinstance(iov_values, Mapping):
        raise InputError("named IIV and IOV mappings are required")
    iiv = {identifier(k): scalar(v, name=k) for k, v in iiv_values.items()}
    occasion_list = validate_occasions(occasions)
    ids = {o["occasion_id"] for o in occasion_list}
    if iov_values and set(iov_values) != ids:
        raise InputError("IOV values must match all declared occasion identifiers")
    iov = {}
    for occ, effects in iov_values.items():
        if not isinstance(effects, Mapping):
            raise InputError("each occasion requires named IOV values")
        iov[occ] = {identifier(k): scalar(v, name=k) for k, v in effects.items()}
    if iov and len({tuple(sorted(values)) for values in iov.values()}) != 1:
        raise InputError("IOV effect names must match between occasions")
    evaluation_interval = numeric_array(evaluation_interval, name="evaluation_interval")
    if evaluation_interval.shape != (2,):
        raise InputError("evaluation_interval must have two endpoints")
    interval = tuple(scalar(x) for x in evaluation_interval)
    if interval[0] > interval[1]:
        raise InputError("evaluation_interval must be ordered")
    # 功能 / Function: 隔离回调输入 / Isolate callback inputs.
    # 输入 / Inputs: 已校验副本 / Validated copies. 输出 / Outputs: 独立回调结果 / Independent callback result.
    try:
        result = parameter_model["relations"](
            parameters=deepcopy(p), covariates=deepcopy(covariates),
            iiv_values=deepcopy(iiv), iov_values=deepcopy(iov),
            occasions=deepcopy(occasion_list), evaluation_interval=interval)
        validated = validate_individual_parameters(result, interval)
        for segment in validated["segments"]:
            if segment["occasion_id"] is not None:
                matching = [o for o in occasion_list if o["occasion_id"] == segment["occasion_id"]]
                if not matching or segment["start"] < matching[0]["start"] or segment["end"] > matching[0]["end"]:
                    raise InputError("parameter segment inconsistent with its occasion")
            elif iov:
                raise InputError("IOV parameter segments must identify their occasion")
        return validated
    except Exception as exc:
        raise ModelExecutionError(f"parameter_model.relations failed: {exc}") from exc
