"""功能 / Function: 观测事件相位契约与解析回归 / Observation phase contract and analytic regression.
输入 / Inputs: 合成单房室数据及非零容积IOV / Synthetic one-compartment data with nonzero volume IOV.
输出 / Outputs: 边界浓度、记录隔离及旧默认断言 / Boundary, isolation and legacy-default assertions.
"""

from copy import deepcopy
import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.inputs.observations import validate_records
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.prediction import evaluate_prediction, prepare_prediction_records
from pmx_discrete_posterior.models.prediction.linear import LinearPredictor

from test_input_observations import observation
from test_linear_prediction import args, dose, system


PHASES = ("before_events", "after_events")


def record(record_id, time, phase=None):
    """功能 / Function: 构造请求 / Build a request.
    输入 / Inputs: 标识、时间、可选相位 / ID, time and optional phase.
    输出 / Outputs: 无实测值；省略相位则不添加字段 / No value or implicit phase field.
    """
    result = {"record_id": record_id, "time": time, "time_unit": "h", "value_unit": "mg/L"}
    if phase is not None:
        result["event_phase"] = phase
    return result


def switched_args(records, *, old_lag=0, new_lag=0):
    """功能 / Function: 构造已解析IOV参数 / Build resolved IOV parameters.
    输入 / Inputs: 记录及两段滞后 / Records and segment lags.
    输出 / Outputs: 6小时处V翻倍、消除率翻倍 / Volume and elimination rate double at hour 6.
    """
    data = args(times=(6,), k=math.log(2) / 6, lag=old_lag)
    data["prediction_records"] = records
    first = data["individual_parameters"]["segments"][0]
    first.update(end=6, occasion_id="old")
    data["individual_parameters"]["segments"].append({
        "start": 6, "end": 100, "occasion_id": "new",
        "values": {"k": math.log(2) / 3, "V": 10 * math.exp(math.log(2)), "lag": new_lag},
    })
    data["individual_data"]["occasions"] = [
        {"occasion_id": "old", "start": 0, "end": 6},
        {"occasion_id": "new", "start": 6, "end": 100},
    ]
    return data


def provide_state(data, amount=100, pending=()):
    """功能 / Function: 指定初始药量 / Supply initial amounts.
    输入 / Inputs: 请求、药量、待输入 / Request, amount and pending inputs.
    输出 / Outputs: 原位设置明确的初始状态 / Explicit initialization in the request.
    """
    init = data["individual_data"]["initialization"]
    init.update(mode="provided", provided_state={
        "values": {"central": amount}, "units": {"central": "mg"},
        "pending_inputs": [{"time": time, "values": {"central": value},
                            "units": {"central": "mg"}} for time, value in pending],
    })


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("phase", PHASES)
def test_record_phase_is_optional_preserved_and_copied(observed, phase):
    records = [record("explicit", 6, phase), record("legacy", 6)]
    if observed:
        records = [dict(r, value=12) for r in records]
    original = deepcopy(records)
    result = validate_records(records, observed=observed)
    assert result == original
    assert "event_phase" not in result[1]
    result[0]["event_phase"] = "changed"
    assert records == original


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("phase", [
    None, True, False, 0, 1, 1.5, [], {}, (), {"before_events"},
    np.array(["before_events"]), b"before_events", "", "before", "after",
    "Before_events", "before_events ", " after_events",
])
def test_invalid_phase_enum_or_type_is_rejected(observed, phase):
    item = dict(record("bad", 6), event_phase=phase)
    if observed:
        item["value"] = 12
    with pytest.raises(InputError):
        validate_records([item], observed=observed)


@pytest.mark.parametrize("observed", [False, True])
def test_phase_does_not_relax_duplicate_ids_or_unknown_fields(observed):
    records = [record("same", 6, phase) for phase in PHASES]
    if observed:
        records = [dict(r, value=12) for r in records]
    with pytest.raises(InputError):
        validate_records(records, observed=observed)
    with pytest.raises(InputError):
        validate_records([dict(records[0], truth="taken")], observed=observed)


def test_phase_does_not_allow_value_in_request_or_missing_observation():
    with pytest.raises(InputError):
        validate_records([observation(event_phase="before_events")], observed=False)
    with pytest.raises(InputError):
        validate_records([record("missing", 6, "before_events")], observed=True)


def test_mixed_phase_passes_to_callback_without_observed_values_or_mutation():
    records = [record("late", 9, "after_events"), record("pre", 6, "before_events"),
               record("default", 6), record("post", 6, "after_events")]
    data = args()
    observations = [dict(r, value=1000 + i) for i, r in enumerate(records)]
    original = deepcopy(observations)
    prepared = prepare_prediction_records(observations, data["individual_data"])
    assert prepared == {"prediction_records": records, "evaluation_interval": (0, 9)}
    altered_values = [dict(r, value=-999) for r in observations]
    assert prepare_prediction_records(altered_values, data["individual_data"]) == prepared
    calls = []

    def predict(**kwargs):
        received = kwargs["prediction_records"]
        calls.append(deepcopy(received))
        assert received == records
        assert all("value" not in r for r in received)
        assert "event_phase" not in received[2]
        received[1]["event_phase"] = "after_events"
        return {"record_ids": tuple(r["record_id"] for r in records),
                "values": [1, 2, 3, 4], "units": ["mg/L"] * 4}

    data["prediction_records"] = prepared["prediction_records"]
    result = evaluate_prediction(predict=predict, **data)
    assert calls == [records]
    assert tuple(result["record_ids"]) == ("late", "pre", "default", "post")
    assert result["values"] == pytest.approx([1, 2, 3, 4])
    assert prepared["prediction_records"] == records
    assert observations == original


def test_omitted_phase_keeps_original_record_shape_and_post_event_default():
    data = switched_args([record("at", 6), record("after", 9), record("initial", 0)])
    data["candidate_state"]["events"] = [dose(0), dose(6, 40)]
    observations = [dict(r, value=12) for r in data["prediction_records"]]
    assert validate_records(observations, observed=True) == observations
    assert prepare_prediction_records(observations, data["individual_data"])["prediction_records"] == data["prediction_records"]
    result = LinearPredictor(system)(**data)
    assert result["values"] == pytest.approx([4.5, 2.25, 10], rel=1e-12)
    assert all("event_phase" not in r for r in data["prediction_records"])


@pytest.mark.parametrize("lag", [0, 2])
@pytest.mark.parametrize("switch_dose", ["taken", "zero", "absent"])
def test_before_at_after_switch_with_volume_iov_lag_and_missed_dose(lag, switch_dose):
    records = [record("after-7", 7, "after_events"), record("pre-6", 6, "before_events"),
               record("post-6", 6, "after_events"), record("pre-5", 5, "before_events"),
               record("default-6", 6), record("post-5", 5, "after_events"),
               record("pre-7", 7, "before_events"), record("pre-8", 8, "before_events"),
               record("post-8", 8, "after_events"), record("pre-6-repeat", 6, "before_events")]
    data = switched_args(records, new_lag=lag)
    data["candidate_state"]["events"] = [dose(0)]
    amount = 40 if switch_dose == "taken" else 0
    if switch_dose != "absent":
        data["candidate_state"]["events"].append(dose(6, amount))
    original = deepcopy(data)
    k_new = math.log(2) / 3
    post6 = 50 + (amount if lag == 0 else 0)
    pre8 = post6 * math.exp(-2 * k_new)
    expected = [post6 * math.exp(-k_new) / 20, 5, post6 / 20,
                10 * math.exp(-math.log(2) * 5 / 6), post6 / 20,
                10 * math.exp(-math.log(2) * 5 / 6), post6 * math.exp(-k_new) / 20,
                pre8 / 20, (pre8 + (amount if lag else 0)) / 20, 5]
    result = LinearPredictor(system)(**data)
    assert result["record_ids"] == tuple(r["record_id"] for r in records)
    assert result["units"] == ("mg/L",) * len(records)
    assert result["values"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result["values"][1] * 10 == pytest.approx(result["values"][2] * 20 - (amount if lag == 0 else 0))
    assert data == original


def test_multiple_same_time_inputs_and_records_preserve_amount_continuity():
    records = [record("post", 6, "after_events"), record("pre", 6, "before_events"),
               record("post-repeat", 6, "after_events"), record("pre-repeat", 6, "before_events"),
               record("later", 9, "before_events")]
    data = switched_args(records)
    provide_state(data, pending=[(6, 30), (6, 40)])
    data["candidate_state"]["events"] = [dose(6, 20)]
    result = LinearPredictor(system)(**data)
    assert result["record_ids"] == tuple(r["record_id"] for r in records)
    assert result["values"] == pytest.approx([7, 5, 7, 5, 3.5], rel=1e-12)
    assert result["values"][0] * 20 - result["values"][1] * 10 == pytest.approx(90)


def test_lagged_input_uses_administration_segment_not_arrival_segment():
    records = [record("pre-6", 6, "before_events"), record("post-6", 6, "after_events"),
               record("pre-9", 9, "before_events"), record("post-9", 9, "after_events")]
    data = switched_args(records, old_lag=2, new_lag=3)
    provide_state(data)
    data["candidate_state"]["events"] = [dose(4, 40), dose(6, 20)]
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([5, 4.5, 2.25, 3.25], rel=1e-12)


@pytest.mark.parametrize("start", [0, 3, 7])
def test_initial_before_events_uses_declared_current_system_without_boundary_ambiguity(start):
    data = switched_args([record("pre", start, "before_events"),
                          record("post", start, "after_events"), record("legacy", start)])
    data["individual_data"]["initialization"]["start_time"] = start
    provide_state(data, pending=[(start, 30)])
    data["candidate_state"]["events"] = [dose(start, 20)]
    volume = 10 if start < 6 else 20
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([100 / volume, 150 / volume, 150 / volume])


def test_initial_zero_predose_sample_without_prior_system_is_valid():
    data = args(times=(0, 0))
    data["prediction_records"] = [record("pre", 0, "before_events"), record("post", 0, "after_events")]
    data["candidate_state"]["events"] = [dose(0)]
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([0, 10])


def test_initial_steady_state_predose_keeps_reference_and_does_not_continue_regimen():
    data = args(times=(0, 0, 12), k=math.log(2) / 12)
    data["prediction_records"] = [record("pre", 0, "before_events"),
                                  record("post", 0, "after_events"), record("later", 12, "before_events")]
    data["individual_data"]["initialization"].update(mode="steady_state", steady_state={
        "parameter_rule": "fixed_reference", "reference_occasion_id": None,
    })
    event = dose(0)
    event["offset"] = event.pop("time")
    data["individual_data"]["dosing_prehistory"]["regimen"] = {
        "period": 12, "anchor_time": 0, "time_unit": "h", "events": [event],
    }
    data["candidate_state"]["events"] = [dose(0)]
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([10, 20, 10], rel=1e-12)


@pytest.mark.parametrize("mode", ["zero", "provided", "steady_state"])
def test_initial_before_events_at_declared_switch_rejects_ambiguous_map(mode):
    data = switched_args([record("initial-pre", 6, "before_events")])
    data["individual_data"]["initialization"]["start_time"] = 6
    if mode == "provided":
        provide_state(data)
    elif mode == "steady_state":
        data["individual_data"]["initialization"].update(mode=mode, steady_state={
            "parameter_rule": "fixed_reference", "reference_occasion_id": "new",
        })
        event = dose(0)
        event["offset"] = event.pop("time")
        data["individual_data"]["dosing_prehistory"]["regimen"] = {
            "period": 12, "anchor_time": 6, "time_unit": "h", "events": [event],
        }
    with pytest.raises(InputError) as error:
        LinearPredictor(system)(**data)
    assert "earlier" in str(error.value).lower()
    assert "initial" in str(error.value).lower() or "start" in str(error.value).lower()


@pytest.mark.parametrize("phase", [None, "after_events"])
def test_initial_after_events_or_omission_at_switch_still_uses_current_system(phase):
    data = switched_args([record("initial", 6, phase), record("later-pre", 7, "before_events")])
    data["individual_data"]["initialization"]["start_time"] = 6
    provide_state(data)
    data["candidate_state"]["events"] = [dose(6, 20)]
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([6, 6 * math.exp(-math.log(2) / 3)], rel=1e-12)


def test_exact_decimal_boundary_does_not_merge_adjacent_float_times():
    boundary = 0.3
    earlier = float(np.nextafter(boundary, -np.inf))
    later = float(np.nextafter(boundary, np.inf))
    records = [record("just-after-pre", later, "before_events"),
               record("at-post", boundary, "after_events"),
               record("just-before-post", earlier, "after_events"),
               record("at-pre", boundary, "before_events"), record("at-default", boundary)]
    data = switched_args(records, old_lag=0.2)
    segments = data["individual_parameters"]["segments"]
    segments[0]["end"] = segments[1]["start"] = boundary
    for segment in segments:
        segment["values"]["k"] = 0
    occasions = data["individual_data"]["occasions"]
    occasions[0]["end"] = occasions[1]["start"] = boundary
    data["candidate_state"]["events"] = [dose(0.1, 100), dose(boundary, 50)]
    result = LinearPredictor(system)(**data)
    assert result["record_ids"] == tuple(r["record_id"] for r in records)
    assert result["values"] == pytest.approx([7.5, 7.5, 0, 0, 7.5], abs=1e-12)


def test_successive_switches_read_current_left_system_not_initial_reference():
    data = switched_args([record("post-second", 12, "after_events"),
                          record("pre-second", 12, "before_events"),
                          record("later", 18, "before_events")])
    segments = data["individual_parameters"]["segments"]
    segments[1]["end"] = 12
    segments.append({"start": 12, "end": 100, "occasion_id": "third",
                     "values": {"k": math.log(2) / 6, "V": 40, "lag": 0}})
    occasions = data["individual_data"]["occasions"]
    occasions[1]["end"] = 12
    occasions.append({"occasion_id": "third", "start": 12, "end": 100})
    provide_state(data)
    data["candidate_state"]["events"] = [dose(6, 50), dose(12, 15)]
    result = LinearPredictor(system)(**data)
    assert result["values"] == pytest.approx([1, 1.25, 0.5], rel=1e-12)
    assert result["values"][0] * 40 - result["values"][1] * 20 == pytest.approx(15)


def test_initial_steady_state_pending_release_is_between_pre_and_post_samples():
    data = args(times=(0, 0, 0, 12), k=math.log(2) / 12, lag=12)
    data["prediction_records"] = [record("pre", 0, "before_events"),
                                  record("post", 0, "after_events"), record("legacy", 0),
                                  record("later", 12, "before_events")]
    data["individual_data"]["initialization"].update(mode="steady_state", steady_state={
        "parameter_rule": "fixed_reference", "reference_occasion_id": None,
    })
    event = dose(0)
    event["offset"] = event.pop("time")
    data["individual_data"]["dosing_prehistory"]["regimen"] = {
        "period": 12, "anchor_time": 0, "time_unit": "h", "events": [event],
    }
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([10, 20, 20, 10], rel=1e-12)


def test_initial_nonzero_first_declared_segment_does_not_infer_predecessor():
    data = switched_args([record("pre", 6, "before_events"), record("post", 6, "after_events")])
    data["individual_parameters"]["segments"] = data["individual_parameters"]["segments"][1:]
    data["individual_data"]["initialization"]["start_time"] = 6
    provide_state(data)
    data["candidate_state"]["events"] = [dose(6, 40)]
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([5, 7], rel=1e-12)
