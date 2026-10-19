"""功能 / Function: 显式候选历史契约 / Explicit candidate-history contracts.
输入 / Inputs: 状态标识和给药事件 / State identifiers and dose events.
输出 / Outputs: 校验副本，不从标签推断给药 / Validated copies without inferring doses from labels.
"""

from ..validation import InputError, fields, identifier, names, scalar


def validate_event(event, *, periodic=False):
    """功能 / Function: 校验显式给药事件 / Validate an explicit dose event.
    输入 / Inputs: time或offset、amount、amount_unit、route、compartment / Dose fields.
    输出 / Outputs: 校验后事件；不解读状态标签 / Validated event, without parsing a state label.
    """
    clock = "offset" if periodic else "time"
    fields(event, {clock, "amount", "amount_unit", "route", "compartment"}, name="dose event")
    return {clock: scalar(event[clock], name=clock), "amount": scalar(event["amount"], name="dose", minimum=0),
            "amount_unit": identifier(event["amount_unit"]), "route": identifier(event["route"]),
            "compartment": identifier(event["compartment"])}


def validate_candidate_state(candidate_state):
    """功能 / Function: 检查单个候选历史 / Validate one candidate history.
    输入 / Inputs: state_id和显式events / State ID and explicit events.
    输出 / Outputs: 独立副本；空事件表示窗口内无剂量 / Copy; empty events mean no doses in the window.
    """
    fields(candidate_state, {"state_id", "events"}, name="candidate_state")
    state_id = identifier(candidate_state["state_id"])
    if not isinstance(candidate_state["events"], (list, tuple)):
        raise InputError("candidate events must be a sequence")
    return {"state_id": state_id, "events": [validate_event(e) for e in candidate_state["events"]]}


def validate_candidate_states(candidate_states):
    """功能 / Function: 检查有序候选集合和唯一标识 / Validate an ordered candidate collection and unique IDs.
    输入 / Inputs: 非空列表或元组；先验另行提供 / Nonempty list or tuple; priors supplied separately.
    输出 / Outputs: 保持原顺序的独立列表 / Independent list in original order, or InputError.
    """
    if not isinstance(candidate_states, (list, tuple)) or not candidate_states:
        raise InputError("candidate_states must be a nonempty list or tuple")
    result = [validate_candidate_state(state) for state in candidate_states]
    names([state["state_id"] for state in result], name="state identifiers")
    return result


def validate_candidate_history(candidate_state, *, initialization_start, common_events):
    """功能 / Function: 检查候选与初始化/共同前史的关系 / Check candidate against initialization and common history.
    输入 / Inputs: 候选、初始时刻、已校验共同事件 / Candidate, initial time and validated common events.
    输出 / Outputs: 独立候选副本；禁止边界前事件及共同剂量重复 / Copy; reject pre-boundary or duplicate common doses.
    """
    result = validate_candidate_state(candidate_state)
    start = scalar(initialization_start, name="initialization_start")
    if any(event["time"] < start for event in result["events"]):
        raise InputError("candidate event precedes initialization")
    if any(event == common for event in result["events"] for common in common_events):
        raise InputError("same dose appears in common and candidate events; specify it once")
    return result
