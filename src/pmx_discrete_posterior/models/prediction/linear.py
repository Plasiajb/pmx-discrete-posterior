"""功能 / Function: 可选线性房室预测适配器 / Optional linear-compartment prediction adapter.
输入 / Inputs: 用户的参数到速率系统回调 / User-supplied parameter-to-system callback.
输出 / Outputs: 满足predict协议的可调用对象 / Callable implementing the predict protocol.
"""

from copy import deepcopy

import numpy as np

from ...interfaces.validation import InputError, ModelExecutionError, fields, identifier, names, numeric_array, scalar
from ..parameter_resolution import parameters_at, validate_individual_parameters
from . import validate_event, validate_prediction_context, validate_records
from .initialization import advance, clock_add, periodic_state, validate_matrix


def _validate_system(system):
    """功能 / Function: 校验已解析的线性系统 / Validate a resolved linear system.
    输入 / Inputs: 速率矩阵、观测映射、给药映射及单位 / Rates, output and input maps, units.
    输出 / Outputs: 独立系统副本 / Independent system copy.
    """
    fields(system, {"state_names", "amount_unit", "time_unit", "concentration_unit", "matrix", "observation_vector", "inputs"}, name="linear system")
    state_names = names(system["state_names"])
    matrix = validate_matrix(system["matrix"])
    output = numeric_array(system["observation_vector"], name="observation_vector")
    n = len(state_names)
    if matrix.shape != (n, n) or output.shape != (n,) or (output < 0).any():
        raise InputError("linear system dimensions or concentration mapping invalid")
    if not isinstance(system["inputs"], dict):
        raise InputError("inputs must map (route, compartment) to fractions and lag")
    inputs = {}
    for key, entry in system["inputs"].items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise InputError("input keys must be (route, compartment) tuples")
        key = tuple(identifier(x) for x in key)
        fields(entry, {"fractions", "lag"}, name="input mapping")
        fractions = numeric_array(entry["fractions"], name="input fractions")
        if fractions.shape != (n,) or (fractions < 0).any():
            raise InputError("input fractions must match compartments and be nonnegative")
        inputs[key] = {"fractions": fractions, "lag": scalar(entry["lag"], minimum=0)}
    return {"state_names": state_names, "matrix": matrix, "observation_vector": output, "inputs": inputs,
            **{k: identifier(system[k]) for k in ("amount_unit", "time_unit", "concentration_unit")}}


def _dose_input(event, system):
    """功能 / Function: 将剂量映射为已解析输入药量 / Map a dose to resolved input amounts.
    输入 / Inputs: 显式给药与当时系统 / Explicit dose and contemporaneous system.
    输出 / Outputs: 药量向量和滞后 / Amount vector and lag.
    """
    if event["amount_unit"] != system["amount_unit"]:
        raise InputError("dose amount unit differs from model amount unit")
    key = (event["route"], event["compartment"])
    if key not in system["inputs"]:
        raise InputError(f"unsupported route/compartment: {key}")
    spec = system["inputs"][key]
    with np.errstate(over="raise", invalid="raise"):
        vector = spec["fractions"] * event["amount"]
    return vector, spec["lag"]


class LinearPredictor:
    """功能 / Function: 以当前个体参数运行线性房室模型 / Run linear compartments at current individual parameters.
    输入 / Inputs: system(values, units)返回明确矩阵/映射 / Callback returns explicit matrices and mappings.
    输出 / Outputs: predict协议的浓度向量 / Concentration vector using the predict protocol.
    """

    def __init__(self, system):
        """功能 / Function: 保存模型特异回调 / Store the model-specific callback.
        输入 / Inputs: 可调用system / Callable system. 输出 / Outputs: 适配器 / Adapter.
        """
        if not callable(system):
            raise InputError("system must be callable")
        self._system = system

    def __call__(self, *, individual_parameters, candidate_state, prediction_records, individual_data):
        """功能 / Function: 初始化、传播、给药及读出浓度 / Initialize, propagate, dose and read concentrations.
        输入 / Inputs: 已解析参数及predict标准四组输入 / Four standard predict inputs with resolved parameters.
        输出 / Outputs: record_ids、values、units；不自动续给药 / Predictions without automatic future doses.
        """
        records = validate_records(prediction_records, observed=False)
        context = validate_prediction_context(individual_data, records[0]["time_unit"])
        init = context["initialization"]
        start, end = init["start_time"], max(r["time"] for r in records)
        if min(r["time"] for r in records) < start:
            raise InputError("prediction time precedes initialization")
        ip = validate_individual_parameters(individual_parameters, (start, end))
        fields(candidate_state, {"state_id", "events"}, name="candidate_state")
        identifier(candidate_state["state_id"])
        if not isinstance(candidate_state["events"], (list, tuple)):
            raise InputError("candidate events must be a sequence")
        candidate_events = [validate_event(e) for e in candidate_state["events"]]
        common_events = context["dosing_prehistory"]["events"]
        if any(e["time"] < start for e in candidate_events):
            raise InputError("candidate input precedes initialization")
        if any(e == shared for e in candidate_events for shared in common_events):
            raise InputError("same dose occurs in common and candidate history")
        segments = ip["segments"]
        systems = [_validate_system(self._system(values=deepcopy(s["values"]), units=deepcopy(ip["units"]))) for s in segments]
        for system in systems:
            if any(system[k] != systems[0][k] for k in ("state_names", "amount_unit", "time_unit", "concentration_unit")):
                raise InputError("state structure and units must stay constant between segments")
            if system["time_unit"] != records[0]["time_unit"] or system["concentration_unit"] != records[0]["value_unit"]:
                raise InputError("linear system and observation units differ")
        current = next(i for i, s in enumerate(segments) if s == parameters_at(ip, start))
        reference = systems[current]
        state_names = reference["state_names"]
        state = np.zeros(len(state_names))
        pending = []
        if init["mode"] == "provided":
            supplied = init["provided_state"]
            if set(supplied["values"]) != set(state_names) or any(u != reference["amount_unit"] for u in supplied["units"].values()):
                raise InputError("provided state does not match system states/units")
            state = np.array([supplied["values"][k] for k in state_names])
            pending = [{"time": e["time"], "values": np.array([e["values"][k] for k in state_names])}
                       for e in supplied["pending_inputs"]]
        elif init["mode"] == "steady_state":
            if segments[current]["occasion_id"] != init["steady_state"]["reference_occasion_id"]:
                raise InputError("resolved reference parameters do not match initialization occasion")
            regimen = context["dosing_prehistory"]["regimen"]
            resolved = []
            for event in regimen["events"]:
                vector, lag = _dose_input(event, reference)
                resolved.append({"offset": event["offset"], "lag": lag, "values": vector})
            initial = periodic_state(matrix=reference["matrix"], period=regimen["period"],
                                     anchor_time=regimen["anchor_time"], start_time=start, input_events=resolved)
            state, pending = initial["values"], initial["pending_inputs"]
        # 功能 / Function: 明确事件优先级 / Define event priority explicitly.
        # 输入 / Inputs: 参数边界、给药、观测 / Boundaries, doses, observations.
        # 输出 / Outputs: switch(0) -> input(1) -> sample(2)，保留同时间记录 / Stable same-time event order.
        schedule = [(s["start"], 0, i, i) for i, s in enumerate(segments) if start < s["start"] <= end]
        schedule.extend((e["time"], 1, i, e["values"]) for i, e in enumerate(pending))
        for index, event in enumerate([*common_events, *candidate_events], start=len(pending)):
            if event["time"] > end:
                continue  # 功能 / Function: 未来给药不影响过去观测 / Future administrations cannot affect past observations.
            segment = parameters_at(ip, event["time"])
            system = systems[next(i for i, s in enumerate(segments) if s == segment)]
            vector, lag = _dose_input(event, system)
            schedule.append((clock_add(event["time"], lag), 1, index, vector))
        schedule.extend((r["time"], 2, i, i) for i, r in enumerate(records))
        result, last = np.empty(len(records)), start
        for time, priority, _, value in sorted(schedule, key=lambda row: row[:3]):
            if time > end:
                break
            state = advance(systems[current]["matrix"], state, time-last)
            last = time
            if priority == 0:
                current = value
            elif priority == 1:
                state = state + value
            else:
                result[value] = systems[current]["observation_vector"] @ state
            if not np.isfinite(state).all():
                raise ModelExecutionError("drug amount overflow")
        if not np.isfinite(result).all() or (result < 0).any():
            raise ModelExecutionError("invalid concentration prediction")
        return {"record_ids": tuple(r["record_id"] for r in records), "values": result,
                "units": tuple(r["value_unit"] for r in records)}
