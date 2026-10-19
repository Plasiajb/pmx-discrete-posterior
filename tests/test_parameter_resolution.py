import math
from copy import deepcopy

import pytest

from pmx_discrete_posterior.interfaces.validation import InputError, ModelExecutionError
from pmx_discrete_posterior.models.parameter_resolution import (
    parameters_at, resolve_parameters, validate_individual_parameters,
)


def inputs():
    def relations(**kwargs):
        cl = kwargs["parameters"]["values"]["CL"] * kwargs["covariates"]["BW"] / 70
        cl *= math.exp(kwargs["iiv_values"]["CL"] + kwargs["iov_values"]["o1"]["CL"])
        kwargs["parameters"]["values"]["CL"] = 999
        return {"units": {"CL": "L/h", "V": "L"}, "segments": [
            {"start": 0, "end": 24, "occasion_id": "o1", "values": {"CL": cl, "V": 10}}]}
    return dict(parameter_model={"covariate_names": ["BW"], "relations": relations},
                parameters={"values": {"CL": 10}, "units": {"CL": "L/h"}},
                covariates={"BW": 35}, iiv_values={"CL": math.log(2)},
                iov_values={"o1": {"CL": math.log(3)}},
                occasions=[{"occasion_id": "o1", "start": 0, "end": 24}], evaluation_interval=(0, 24))


def test_relationship_applied_once_and_inputs_unchanged():
    args = inputs()
    before = deepcopy(args)
    ip = resolve_parameters(**args)
    assert ip["segments"][0]["values"]["CL"] == pytest.approx(30)
    assert args == before
    assert resolve_parameters(**args) == ip


@pytest.mark.parametrize("change", [
    {"covariates": {}}, {"covariates": {"BW": 35, "observed_value": 99}},
    {"iiv_values": {"CL": float("nan")}}, {"iov_values": {"bad": {"CL": 0}}},
    {"evaluation_interval": (24, 0)},
])
def test_invalid_input_is_not_model_failure(change):
    args = inputs()
    args.update(change)
    with pytest.raises(InputError):
        resolve_parameters(**args)


def test_bad_callback_output_is_execution_failure():
    args = inputs()
    args["parameter_model"]["relations"] = lambda **kwargs: {"units": {}, "segments": []}
    with pytest.raises(ModelExecutionError):
        resolve_parameters(**args)


def test_segment_boundaries_no_extrapolation():
    ip = {"units": {"k": "1/h"}, "segments": [
        {"start": 0, "end": 12, "occasion_id": None, "values": {"k": 1}},
        {"start": 12, "end": 24, "occasion_id": None, "values": {"k": 2}}]}
    validated = validate_individual_parameters(ip, (0, 24))
    assert parameters_at(validated, 12)["values"]["k"] == 2
    assert parameters_at(validated, 24)["values"]["k"] == 2
    with pytest.raises(InputError):
        parameters_at(validated, -1)
    ip["segments"][1]["start"] = 13
    with pytest.raises(InputError):
        validate_individual_parameters(ip, (0, 24))


def test_unknown_occasion_in_result_is_rejected():
    args = inputs()
    original = args["parameter_model"]["relations"]
    def bad(**kwargs):
        result = original(**kwargs)
        result["segments"][0]["occasion_id"] = "other"
        return result
    args["parameter_model"]["relations"] = bad
    with pytest.raises(ModelExecutionError):
        resolve_parameters(**args)


def test_boolean_interval_endpoint_is_rejected_before_callback():
    args = inputs()
    args["evaluation_interval"] = [False, 24]
    args["parameter_model"]["relations"] = lambda **kwargs: pytest.fail("invalid endpoint reached callback")
    with pytest.raises(InputError):
        resolve_parameters(**args)
