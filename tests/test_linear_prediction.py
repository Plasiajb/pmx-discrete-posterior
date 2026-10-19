import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.prediction.linear import LinearPredictor


def system(*, values, units):
    assert units == {"k": "1/h", "V": "L", "lag": "h"}
    return {"state_names": ["central"], "amount_unit": "mg", "time_unit": "h", "concentration_unit": "mg/L",
            "matrix": [[-values["k"]]], "observation_vector": [1 / values["V"]],
            "inputs": {("iv", "central"): {"fractions": [1], "lag": values["lag"]}}}


def args(times=(1,), k=.1, lag=0):
    return dict(individual_parameters={"units": {"k": "1/h", "V": "L", "lag": "h"}, "segments": [
        {"start": 0, "end": 100, "occasion_id": None, "values": {"k": k, "V": 10, "lag": lag}}]},
        candidate_state={"state_id": "arbitrary-label", "events": []},
        prediction_records=[{"record_id": str(i), "time": t, "time_unit": "h", "value_unit": "mg/L"} for i, t in enumerate(times)],
        individual_data={"occasions": [], "initialization": {"mode": "zero", "start_time": 0, "event_phase": "before_events"},
                         "dosing_prehistory": {"events": []}})


def dose(time, amount=100):
    return {"time": time, "amount": amount, "amount_unit": "mg", "route": "iv", "compartment": "central"}


def test_single_dose_and_same_time_order_unsorted_records():
    data = args(times=(1, 0, 2), k=3)
    data["candidate_state"]["events"] = [dose(0)]
    result = LinearPredictor(system)(**data)
    assert result["values"] == pytest.approx([10*math.exp(-3), 10, 10*math.exp(-6)], rel=1e-12)


def test_lag_and_provided_pending_are_equivalent():
    explicit = args(times=(2, 4, 5), k=math.log(2), lag=5)
    explicit["individual_parameters"]["segments"][0]["start"] = -2
    explicit["individual_data"]["initialization"]["start_time"] = -2
    explicit["individual_data"]["dosing_prehistory"]["events"] = [dose(-2)]
    expected = LinearPredictor(system)(**explicit)["values"]
    provided = args(times=(2, 4, 5), k=math.log(2), lag=5)
    provided["individual_data"]["initialization"] = {
        "mode": "provided", "start_time": 0, "event_phase": "before_events",
        "provided_state": {"values": {"central": 0}, "units": {"central": "mg"},
                           "pending_inputs": [{"time": 3, "values": {"central": 100}, "units": {"central": "mg"}}]}}
    assert expected == pytest.approx([0, 5, 2.5], abs=1e-12)
    assert LinearPredictor(system)(**provided)["values"] == pytest.approx(expected, abs=1e-12)


def test_occasion_switch_preserves_amount_and_uses_new_volume():
    data = args(times=(6, 9), k=.1)
    segments = data["individual_parameters"]["segments"]
    segments[0]["end"] = 6
    segments.append({"start": 6, "end": 100, "occasion_id": None, "values": {"k": .2, "V": 20, "lag": 0}})
    data["candidate_state"]["events"] = [dose(0), dose(6, 20)]
    expected6 = (100*math.exp(-.6)+20)/20
    assert LinearPredictor(system)(**data)["values"] == pytest.approx([expected6, expected6*math.exp(-.6)], rel=1e-12)


def test_steady_state_is_recomputed_at_each_call_and_not_continued():
    data = args(times=(6, 18), k=math.log(2)/12)
    init = data["individual_data"]["initialization"]
    init.update(mode="steady_state", start_time=6,
                steady_state={"parameter_rule": "fixed_reference", "reference_occasion_id": None})
    event = dose(0)
    event["offset"] = event.pop("time")
    data["individual_data"]["dosing_prehistory"]["regimen"] = {"period": 12, "anchor_time": 0, "time_unit": "h", "events": [event]}
    predictor = LinearPredictor(system)
    assert predictor(**data)["values"] == pytest.approx([10*math.sqrt(2), 5*math.sqrt(2)], rel=1e-12)
    data["individual_parameters"]["segments"][0]["values"]["k"] = math.log(4)/12
    assert predictor(**data)["values"] == pytest.approx([20/3, 5/3], rel=1e-12)


def test_unsupported_route_and_unknown_system_fields():
    data = args()
    data["candidate_state"]["events"] = [dict(dose(0), route="infusion")]
    with pytest.raises(InputError):
        LinearPredictor(system)(**data)
    def bad(**kwargs):
        return dict(system(**kwargs), extra=1)
    with pytest.raises(InputError):
        LinearPredictor(bad)(**args())
