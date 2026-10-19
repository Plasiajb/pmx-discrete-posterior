from copy import deepcopy

import pytest

from pmx_discrete_posterior.interfaces.validation import InputError, ModelExecutionError
from pmx_discrete_posterior.models.prediction import evaluate_prediction, prepare_prediction_records


def context():
    return {"covariates": {"BW": 35}, "occasions": [], "dosing_prehistory": {"events": []},
            "initialization": {"mode": "zero", "start_time": 0, "event_phase": "before_events"}}


def observations():
    return [{"record_id": "x", "time": 12, "value": 99, "time_unit": "h", "value_unit": "mg/L"}]


def parameters():
    return {"units": {"CL": "L/h"}, "segments": [
        {"start": 0, "end": 24, "occasion_id": None, "values": {"CL": 1}}]}


def invoke(predict, **changes):
    args = dict(predict=predict, individual_parameters=parameters(), candidate_state={"state_id": "anything", "events": []},
                prediction_records=prepare_prediction_records(observations(), context())["prediction_records"], individual_data=context())
    args.update(changes)
    return evaluate_prediction(**args)


def test_predictor_has_no_observed_values_covariates_eta_or_priors():
    calls = []
    def predict(**kwargs):
        calls.append(deepcopy(kwargs))
        assert set(kwargs) == {"individual_parameters", "candidate_state", "prediction_records", "individual_data"}
        assert set(kwargs["individual_data"]) == {"occasions", "initialization", "dosing_prehistory"}
        assert "value" not in kwargs["prediction_records"][0]
        return {"record_ids": ["x"], "values": [2.], "units": ["mg/L"]}
    data = context()
    before = deepcopy(data)
    assert invoke(predict, individual_data=data)["values"].tolist() == [2.]
    assert data == before
    obs = observations()
    obs[0]["value"] = -3
    assert prepare_prediction_records(obs, data) == prepare_prediction_records(observations(), data)
    assert len(calls) == 1


@pytest.mark.parametrize("result", [
    {"record_ids": ["wrong"], "values": [2], "units": ["mg/L"]},
    {"record_ids": ["x"], "values": [float("nan")], "units": ["mg/L"]},
    {"record_ids": ["x"], "values": [-2], "units": ["mg/L"]},
    {"record_ids": ["x"], "values": 2, "units": ["mg/L"]},
    {"record_ids": ["x"], "values": [2], "units": ["ng/mL"]},
])
def test_invalid_callback_result_is_not_valid_zero(result):
    with pytest.raises(ModelExecutionError):
        invoke(lambda **kwargs: result)


def test_negative_dose_and_unknown_fields_rejected_before_callback():
    event = {"time": 0, "amount": -1, "amount_unit": "mg", "route": "oral", "compartment": "depot"}
    def never(**kwargs):
        pytest.fail("invalid inputs reached callback")
    with pytest.raises(InputError):
        invoke(never, candidate_state={"state_id": "omega2", "events": [event]})
    event["amount"] = 0
    event["rate"] = 1
    with pytest.raises(InputError):
        invoke(never, candidate_state={"state_id": "x", "events": [event]})
    bad = context()
    bad["truth_metadata"] = {}
    with pytest.raises(InputError):
        invoke(never, individual_data=bad)


def test_inactive_initialization_and_duplicate_common_dose_rejected():
    data = context()
    data["initialization"]["provided_state"] = {}
    with pytest.raises(InputError):
        prepare_prediction_records(observations(), data)
    data = context()
    event = {"time": 0, "amount": 100, "amount_unit": "mg", "route": "oral", "compartment": "depot"}
    data["dosing_prehistory"]["events"] = [event]
    with pytest.raises(InputError):
        invoke(lambda **kwargs: {}, individual_data=data, candidate_state={"state_id": "x", "events": [event]})
