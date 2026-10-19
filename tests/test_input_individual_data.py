from copy import deepcopy

import pytest
import numpy as np

from pmx_discrete_posterior.interfaces.inputs.individual_data import validate_individual_data
from pmx_discrete_posterior.interfaces.validation import InputError
from test_input_model import model_definition
from test_input_observations import observation
from test_input_candidates import event


def individual():
    return {"covariates": {"BW": 25}, "occasions": [], "dosing_prehistory": {"events": []},
            "initialization": {"mode": "zero", "start_time": 0, "event_phase": "before_events"}}


def checked(data):
    return validate_individual_data(data, model=model_definition(), observations=[observation()])


def test_individual_copy_zero_and_explicit_covariates():
    original = individual()
    before = deepcopy(original)
    result = checked(original)
    result["covariates"]["BW"] = 100
    assert original == before
    del original["covariates"]
    with pytest.raises(InputError):
        checked(original)


@pytest.mark.parametrize("covariates", [{}, {"BW": 25, "HEM": 0.4}, {"BW": True}, {"BW": float("inf")}, None])
def test_mismatched_or_invalid_covariates(covariates):
    with pytest.raises(InputError):
        checked({**individual(), "covariates": covariates})


def test_observations_cannot_precede_initialization():
    data = individual()
    data["initialization"]["start_time"] = 25
    with pytest.raises(InputError):
        checked(data)


def test_periodic_history_unit_and_fixed_reference():
    data = individual()
    dose = event()
    dose["offset"] = 0
    del dose["time"]
    data["dosing_prehistory"]["regimen"] = {"period": 12, "anchor_time": 0, "time_unit": "h", "events": [dose]}
    data["initialization"].update(mode="steady_state", steady_state={"parameter_rule": "fixed_reference", "reference_occasion_id": None})
    assert checked(data) == data
    data["dosing_prehistory"]["regimen"]["time_unit"] = "min"
    with pytest.raises(InputError):
        checked(data)


def test_iov_requires_real_occasions_covering_interval():
    model = model_definition()
    model["iov"] = {**model["iiv"], "occasion_key": "occasion_id"}
    data = individual()
    for occasions in ([], [{"occasion_id": "a", "start": 1, "end": 30}],
                      [{"occasion_id": "a", "start": 0, "end": 12}, {"occasion_id": "b", "start": 13, "end": 30}]):
        data["occasions"] = occasions
        with pytest.raises(InputError):
            validate_individual_data(data, model=model, observations=[observation()])
    data["occasions"] = [{"occasion_id": "a", "start": 0, "end": 12}, {"occasion_id": "b", "start": 12, "end": 30}]
    assert validate_individual_data(data, model=model, observations=[observation()]) == data


def test_provided_state_and_pending_inputs_are_copied():
    data = individual()
    data["initialization"].update(mode="provided", provided_state={"values": {"depot": 10}, "units": {"depot": "mg"},
        "pending_inputs": [{"time": 1, "values": {"depot": 5}, "units": {"depot": "mg"}}]})
    result = checked(data)
    result["initialization"]["provided_state"]["pending_inputs"][0]["values"]["depot"] = 100
    assert data["initialization"]["provided_state"]["pending_inputs"][0]["values"]["depot"] == 5


@pytest.mark.parametrize("mode", [[], {}, 1, None])
def test_malformed_mode_is_input_error(mode):
    data = individual()
    data["initialization"]["mode"] = mode
    with pytest.raises(InputError):
        checked(data)


@pytest.mark.parametrize("field,value", [("values", None), ("values", ["depot"]), ("units", None), ("units", ["depot"])])
def test_malformed_pending_mapping_is_input_error(field, value):
    data = individual()
    pending = {"time": 1, "values": {"depot": 1}, "units": {"depot": "mg"}, field: value}
    data["initialization"].update(mode="provided", provided_state={"values": {"depot": 10},
        "units": {"depot": "mg"}, "pending_inputs": [pending]})
    with pytest.raises(InputError):
        checked(data)


def test_observation_at_iov_gap_start_is_not_previous_occasion_endpoint():
    model = model_definition()
    model["iov"] = {**model["iiv"], "occasion_key": "occasion_id"}
    data = individual()
    data["occasions"] = [{"occasion_id": "a", "start": 0, "end": 12}, {"occasion_id": "b", "start": 13, "end": 30}]
    with pytest.raises(InputError):
        validate_individual_data(data, model=model, observations=[observation(time=12)])


@pytest.mark.parametrize("unit", [np.array("mg"), np.array(["mg"]), np.array(["mg", "mg"])])
def test_pending_unit_requires_scalar_string(unit):
    data = individual()
    data["initialization"].update(mode="provided", provided_state={"values": {"depot": 10}, "units": {"depot": "mg"},
        "pending_inputs": [{"time": 1, "values": {"depot": 5}, "units": {"depot": unit}}]})
    with pytest.raises(InputError):
        checked(data)


def test_adjacent_iov_boundary_uses_new_segment_not_previous():
    from pmx_discrete_posterior.models.parameter_resolution import parameters_at
    model = model_definition()
    model["iov"] = {**model["iiv"], "occasion_key": "occasion_id"}
    data = individual()
    data["occasions"] = [{"occasion_id": "a", "start": 0, "end": 12}, {"occasion_id": "b", "start": 12, "end": 30}]
    result = validate_individual_data(data, model=model, observations=[observation(time=12)])
    parameters = {"units": {"CL": "L/h"}, "segments": [
        {**occasion, "values": {"CL": i + 1}} for i, occasion in enumerate(result["occasions"])]}
    assert parameters_at(parameters, 12)["occasion_id"] == "b"
    assert parameters_at(parameters, 12)["values"]["CL"] == 2


@pytest.mark.parametrize("mode", ["zero", "provided", "steady_state"])
def test_final_declared_endpoint_remains_closed_in_zero_duration_evaluation(mode):
    model = model_definition()
    model["iov"] = {**model["iiv"], "occasion_key": "occasion_id"}
    data = individual()
    data["initialization"]["start_time"] = 12
    data["initialization"]["mode"] = mode
    if mode == "provided":
        data["initialization"]["provided_state"] = {"values": {"depot": 0}, "units": {"depot": "mg"}, "pending_inputs": []}
    elif mode == "steady_state":
        data["initialization"]["steady_state"] = {"parameter_rule": "fixed_reference", "reference_occasion_id": "a"}
        dose = event()
        del dose["time"]
        dose["offset"] = 0
        data["dosing_prehistory"]["regimen"] = {"period": 12, "anchor_time": 0, "time_unit": "h", "events": [dose]}
    data["occasions"] = [{"occasion_id": "a", "start": 0, "end": 12}]
    assert validate_individual_data(data, model=model, observations=[observation(time=12)]) == data
