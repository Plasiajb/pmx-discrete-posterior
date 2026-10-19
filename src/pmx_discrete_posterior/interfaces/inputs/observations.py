"""功能 / Function: 实际观测和预测记录契约 / Observation and prediction-record contracts.
输入 / Inputs: 有序具名记录 / Ordered named records.
输出 / Outputs: 校验副本，不排序、不转换单位 / Copies without reordering or unit conversion.
"""

from ..validation import InputError, fields, identifier, names, scalar


def validate_records(records, *, observed):
    """功能 / Function: 校验观测或预测请求 / Validate observations or prediction requests.
    输入 / Inputs: record_id、time、time_unit、value_unit；观测另有value / Records and optional observed values.
    输出 / Outputs: 保持输入顺序的副本 / Copies preserving input order; common time and concentration units required.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise InputError("records must be a nonempty list or tuple")
    required = {"record_id", "time", "time_unit", "value_unit"} | ({"value"} if observed else set())
    result = []
    for record in records:
        fields(record, required, name="observation" if observed else "prediction record")
        item = {"record_id": identifier(record["record_id"]), "time": scalar(record["time"]),
                "time_unit": identifier(record["time_unit"]), "value_unit": identifier(record["value_unit"])}
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
