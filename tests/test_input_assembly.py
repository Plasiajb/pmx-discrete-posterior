import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.inputs import validate_inputs
from pmx_discrete_posterior.interfaces.validation import InputError, validate_covariance
from test_input_model import model_definition
from test_input_individual_data import individual
from test_input_observations import observation
from test_input_candidates import event


def inputs():
    return {"model": model_definition(), "individual_data": individual(), "observations": [observation()],
            "candidate_states": [{"state_id": "a", "events": []}, {"state_id": "b", "events": [event()]}],
            "priors": {"a": 1, "b": 0}, "integration_method": "laplace"}


def test_assembly_does_not_execute_model_or_remove_zero_prior_candidate():
    original = inputs()
    result = validate_inputs(**original)
    assert len(result["candidate_states"]) == 2
    assert result["priors"] == {"a": 1, "b": 0}
    assert result["model"]["predict"] is original["model"]["predict"]
    assert result["model"]["parameter_model"]["relations"] is original["model"]["parameter_model"]["relations"]
    result["individual_data"]["covariates"]["BW"] = 100
    result["candidate_states"][1]["events"][0]["amount"] = 1
    assert original["individual_data"]["covariates"]["BW"] == 25
    assert original["candidate_states"][1]["events"][0]["amount"] == 300


@pytest.mark.parametrize("failure", ["additive_unit", "pre_boundary", "duplicate_common"])
def test_cross_input_mismatches_fail_before_callbacks(failure):
    args = inputs()
    if failure == "additive_unit":
        args["model"]["parameters"]["units"]["sd"] = "ng/mL"
    elif failure == "pre_boundary":
        args["candidate_states"][1]["events"][0]["time"] = -1
    else:
        args["individual_data"]["dosing_prehistory"]["events"] = [event()]
    with pytest.raises(InputError):
        validate_inputs(**args)


@pytest.mark.parametrize("name,dimension", [("OXC", 1), ("TAC", 10)])
def test_reference_input_assembly_preserves_predictions_and_likelihood(name, dimension):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("assembled_reference", root / "examples/reference_models.py")
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)
    case = examples.load_reference(name)
    result = validate_inputs(**case, priors={state["state_id"]: .25 for state in case["candidate_states"]}, integration_method="laplace")
    for state, checked_state in zip(case["candidate_states"], result["candidate_states"]):
        before, ll_before = examples.evaluate_reference(case, state, np.zeros(dimension))
        after, ll_after = examples.evaluate_reference(result, checked_state, np.zeros(dimension))
        np.testing.assert_array_equal(before["values"], after["values"])
        assert ll_before == ll_after


def test_shared_covariance_helper_shape_zero_dimension_factor_and_copy():
    matrix = np.array([[1., -.2], [-.2, .5]])
    result = validate_covariance(matrix)
    np.testing.assert_allclose(result["factor"] @ result["factor"].T, matrix, atol=1e-15)
    result["covariance"][0, 0] = 4
    assert matrix[0, 0] == 1
    empty = validate_covariance(np.zeros((0, 0)))
    assert empty["covariance"].shape == empty["factor"].shape == (0, 0)
    for invalid in ([[1, 2]], [[1e-320, 1e308], [1e308, 1e-320]]):
        with pytest.raises(InputError):
            validate_covariance(invalid)


@pytest.mark.parametrize("order", [
    ["interfaces.inputs", "models.parameter_resolution", "models.random_effects", "numerics.derivatives"],
    ["models.random_effects", "models.prediction", "interfaces.inputs", "numerics.optimization"],
    ["numerics.optimization", "interfaces.inputs", "models.residuals"],
])
def test_fresh_process_import_orders_have_no_cycles(order):
    command = "import importlib; " + "; ".join(f"importlib.import_module('pmx_discrete_posterior.{module}')" for module in order)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", command], cwd=root, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": ""})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("rtol", [0, 1e-16])
def test_laplace_rejects_configuration_ineligible_for_hessian(rtol):
    args = inputs()
    args["numerical_options"] = {"derivatives": {"atol": 1e-6, "rtol": rtol}}
    with pytest.raises(InputError):
        validate_inputs(**args)


def test_hessian_tolerance_boundary_and_no_active_effect_exemption():
    args = inputs()
    args["numerical_options"] = {"derivatives": {"atol": 1e-6, "rtol": 100 * np.finfo(float).eps}}
    validate_inputs(**args)
    args["numerical_options"]["derivatives"]["rtol"] = 0
    args["model"]["parameters"]["values"]["var"] = 0
    validate_inputs(**args)
