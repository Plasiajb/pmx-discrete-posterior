"""功能 / Function: 实际观测和预测记录契约 / Observation and prediction-record contracts.
输入 / Inputs: 有序具名记录 / Ordered named records.
输出 / Outputs: 校验副本，不排序、不转换单位 / Copies without reordering or unit conversion.
"""

from ..validation import InputError, fields, identifier, names, scalar


def validate_records(records, *, observed):
    """功能 / Function: 校验观测或预测请求 / Validate observations or prediction requests.
    输入 / Inputs: record_id、time、time_unit、value_unit；观测另有value，可选event_phase / Records, observed values and optional event phase.
    输出 / Outputs: 保持顺序及显式event_phase的副本；未填写时不注入字段 / Copies preserving order and explicit phase; omitted fields stay omitted.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise InputError("records must be a nonempty list or tuple")
    required = {"record_id", "time", "time_unit", "value_unit"} | ({"value"} if observed else set())
    result = []
    for record in records:
        fields(record, required, {"event_phase"}, name="observation" if observed else "prediction record")
        item = {"record_id": identifier(record["record_id"]), "time": scalar(record["time"]),
                "time_unit": identifier(record["time_unit"]), "value_unit": identifier(record["value_unit"])}
        # 功能 / Function: 保留显式观测顺序 / Preserve explicitly requested readout order.
        # 输入 / Inputs: 可选event_phase / Optional event_phase. 输出 / Outputs: 已校验字段 / Validated field.
        if "event_phase" in record:
            phase = identifier(record["event_phase"], name="event_phase")
            if phase not in ("before_events", "after_events"):
                raise InputError("event_phase must be before_events or after_events")
            item["event_phase"] = phase
        if observed:
            item["value"] = scalar(record["value"], name="observation value")
        result.append(item)
    names([r["record_id"] for r in result], name="record identifiers")
    if len({r["time_unit"] for r in result}) != 1 or len({r["value_unit"] for r in result}) != 1:
        raise InputError("first version requires one time unit and one concentration unit; no conversion")
    return result


def validate_observations(observations):
    """功能 / Function: 校验同一个体的实际观测 / Validate actual observations for one individual.
    输入 / Inputs: 非空记录列表，含record_id、time、value及单位 / Nonempty records with identity, time, value and units.
    输出 / Outputs: 按原顺序的独立列表 / Independent list in original order, or InputError.
    允许重复时间及有限负观测；不推断BLQ或删失 / Allows repeated times and finite negative observations; no BLQ inference.
    """
    return validate_records(observations, observed=True)
