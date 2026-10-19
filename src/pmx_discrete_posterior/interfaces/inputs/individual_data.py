"""功能 / Function: 个体数据及初始化上下文契约 / Individual-data and initialization-context contracts.
输入 / Inputs: 协变量、occasion、共同前史和初始化 / Covariates, occasions, common history and initialization.
输出 / Outputs: 校验副本，不执行参数或预测回调 / Validated copies without parameter or prediction calls.
"""

from collections.abc import Mapping
from copy import deepcopy

from ..validation import InputError, fields, identifier, names, scalar
from .candidate_states import validate_event
from .model_inputs import validate_model
from .observations import validate_observations


def validate_occasions(occasions):
    """功能 / Function: 检查有序occasion及半开区间 / Check ordered occasions and half-open intervals.
    输入 / Inputs: occasion_id、start、end记录 / Occasion records in the common time unit.
    输出 / Outputs: 校验后的列表 / Validated list; no inferred occasions.
    """
    if not isinstance(occasions, (list, tuple)):
        raise InputError("occasions must be a list or tuple; use [] when absent")
    result = []
    for occasion in occasions:
        fields(occasion, {"occasion_id", "start", "end"}, name="occasion")
        item = {"occasion_id": identifier(occasion["occasion_id"]),
                "start": scalar(occasion["start"]), "end": scalar(occasion["end"])}
        if item["start"] >= item["end"] or (result and item["start"] < result[-1]["end"]):
            raise InputError("occasions must have positive width, be ordered, and not overlap")
        result.append(item)
    names([o["occasion_id"] for o in result], allow_empty=True)
    return result


def validate_prediction_context(individual_data, time_unit):
    """功能 / Function: 校验并限制预测上下文 / Validate and restrict predictor context.
    输入 / Inputs: individual_data及共同时间单位 / Individual data and common time unit.
    输出 / Outputs: 仅occasions、dosing_prehistory、initialization / Only prediction-relevant context.
    """
    fields(individual_data, {"occasions", "dosing_prehistory", "initialization"}, {"covariates"}, name="individual_data")
    if "covariates" in individual_data:
        if not isinstance(individual_data["covariates"], Mapping):
            raise InputError("covariates must be a mapping")
        for key, value in individual_data["covariates"].items():
            identifier(key)
            scalar(value, name=key)
    occasions = validate_occasions(individual_data["occasions"])
    init = individual_data["initialization"]
    fields(init, {"mode", "start_time", "event_phase"}, {"provided_state", "steady_state"}, name="initialization")
    if identifier(init["event_phase"], name="event_phase") != "before_events":
        raise InputError("initial state must use event_phase='before_events'")
    start = scalar(init["start_time"])
    mode = identifier(init["mode"], name="initialization mode")
    expected = {"zero": set(), "provided": {"provided_state"}, "steady_state": {"steady_state"}}
    if mode not in expected or set(init) - {"mode", "start_time", "event_phase"} != expected[mode]:
        raise InputError("initialization mode and active fields disagree")
    init = deepcopy(init)
    init["start_time"] = start
    if mode == "provided":
        provided = init["provided_state"]
        fields(provided, {"values", "units", "pending_inputs"}, name="provided_state")
        if not isinstance(provided["values"], Mapping) or not isinstance(provided["units"], Mapping):
            raise InputError("provided values/units must be named mappings")
        state_names = names(provided["values"])
        if set(state_names) != set(provided["units"]):
            raise InputError("provided state values and units differ")
        provided["values"] = {k: scalar(provided["values"][k], name=k, minimum=0) for k in state_names}
        provided["units"] = {k: identifier(provided["units"][k]) for k in state_names}
        if not isinstance(provided["pending_inputs"], (list, tuple)):
            raise InputError("pending_inputs must be a sequence")
        pending = []
        for record in provided["pending_inputs"]:
            fields(record, {"time", "values", "units"}, name="pending input")
            if not isinstance(record["values"], Mapping) or not isinstance(record["units"], Mapping):
                raise InputError("pending input values/units must be named mappings")
            if set(record["values"]) != set(state_names) or set(record["units"]) != set(state_names):
                raise InputError("pending input must use all state names and units")
            pending_units = {k: identifier(record["units"][k], name=f"pending unit of {k}") for k in state_names}
            if pending_units != provided["units"]:
                raise InputError("pending input units disagree with state units")
            pending.append({"time": scalar(record["time"], minimum=start),
                            "values": {k: scalar(record["values"][k], minimum=0) for k in state_names},
                            "units": pending_units})
        provided["pending_inputs"] = pending
    elif mode == "steady_state":
        steady = init["steady_state"]
        fields(steady, {"parameter_rule", "reference_occasion_id"}, name="steady_state")
        if identifier(steady["parameter_rule"], name="parameter_rule") != "fixed_reference":
            raise InputError("only explicit fixed_reference conditional steady state is supported")
        ref = steady["reference_occasion_id"]
        if ref is not None:
            identifier(ref, name="reference_occasion_id")
        if occasions:
            matching = [o for o in occasions if o["occasion_id"] == ref
                        and (o["start"] <= start < o["end"] or o is occasions[-1] and start == o["end"])]
            if len(matching) != 1:
                raise InputError("reference occasion must explicitly contain start_time")
        elif ref is not None:
            raise InputError("reference_occasion_id must be None when no occasions exist")
    prehistory = individual_data["dosing_prehistory"]
    fields(prehistory, {"events"}, {"regimen"}, name="dosing_prehistory")
    if not isinstance(prehistory["events"], (list, tuple)):
        raise InputError("common events must be a sequence")
    events = [validate_event(e) for e in prehistory["events"]]
    if any(e["time"] < start for e in events):
        raise InputError("explicit common events must not precede the initialization boundary")
    result_history = {"events": events}
    if mode == "steady_state":
        if "regimen" not in prehistory:
            raise InputError("steady_state requires a periodic regimen")
        regimen = prehistory["regimen"]
        fields(regimen, {"period", "anchor_time", "time_unit", "events"}, name="regimen")
        period = scalar(regimen["period"], positive=True)
        if identifier(regimen["time_unit"], name="regimen time_unit") != time_unit:
            raise InputError("regimen and observation time units must match")
        if not isinstance(regimen["events"], (list, tuple)) or not regimen["events"]:
            raise InputError("periodic regimen needs explicit events")
        periodic = [validate_event(e, periodic=True) for e in regimen["events"]]
        if any(not 0 <= e["offset"] < period for e in periodic):
            raise InputError("periodic offsets must be in [0, period)")
        result_history["regimen"] = {"period": period, "anchor_time": scalar(regimen["anchor_time"]),
                                     "time_unit": time_unit, "events": periodic}
    elif "regimen" in prehistory:
        raise InputError("regimen is only applicable to steady_state initialization")
    return {"occasions": occasions, "dosing_prehistory": result_history, "initialization": init}


def validate_individual_data(individual_data, *, model, observations):
    """功能 / Function: 检查个体输入及其与模型/观测的关系 / Check individual input against model and observations.
    输入 / Inputs: 四组个体数据、model、实际observations / Four individual groups, model and observations.
    输出 / Outputs: 独立个体数据副本；不声明随机效应或残差独立性 / Independent copy; no implicit independence.
    """
    fields(individual_data, {"covariates", "occasions", "dosing_prehistory", "initialization"}, name="individual_data")
    model = validate_model(model)
    records = validate_observations(observations)
    context = validate_prediction_context(individual_data, records[0]["time_unit"])
    required = model["parameter_model"]["covariate_names"]
    if set(individual_data["covariates"]) != set(required):
        raise InputError("covariates must match covariate_names exactly")
    start = context["initialization"]["start_time"]
    if any(record["time"] < start for record in records):
        raise InputError("observations cannot precede initialization")
    if model["iov"] is not None:
        occasions = context["occasions"]
        if not occasions:
            raise InputError("IOV requires explicit occasions")
        end = max(record["time"] for record in records)
        # 仅整个声明范围的末端允许闭合；中间空隙不能借前段末端跨过 / Only the final declared endpoint is closed.
        times = [start, *(record["time"] for record in records)]
        if any(not any(o["start"] <= time < o["end"] for o in occasions)
               and time != occasions[-1]["end"] for time in times):
            raise InputError("initialization and observations must belong to a declared IOV occasion")
        relevant = [o for o in occasions if (o["end"] > start or o is occasions[-1] and o["end"] == start)
                    and o["start"] <= end]
        if (not relevant or relevant[0]["start"] > start or relevant[-1]["end"] < end
                or any(a["end"] != b["start"] for a, b in zip(relevant, relevant[1:]))):
            raise InputError("IOV occasions must cover the evaluation interval without gaps")
    return {"covariates": {key: scalar(individual_data["covariates"][key], name=key) for key in required}, **context}
