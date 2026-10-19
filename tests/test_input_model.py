import pytest
import importlib.util
from pathlib import Path
import numpy as np

from pmx_discrete_posterior.interfaces.inputs.model_inputs import validate_parameter_model, validate_random_effect_spec
from pmx_discrete_posterior.interfaces.validation import InputError, validate_parameters
from pmx_discrete_posterior.interfaces.inputs.model_inputs import validate_model, validate_residual_model


class NoExecution:
    def __call__(self, **kwargs):
        raise AssertionError("validation must not execute a model")

    def __deepcopy__(self, memo):
        raise AssertionError("validation must not copy opaque callable state")


def test_parameter_relationship_copy_and_no_execution():
    callback = NoExecution()
    original = {"covariate_names": ["BW"], "relations": callback}
    result = validate_parameter_model(original)
    original["covariate_names"].append("HEM")
    assert result == {"covariate_names": ("BW",), "relations": callback}
    assert result["relations"] is callback
    assert validate_parameter_model({"covariate_names": [], "relations": callback})["covariate_names"] == ()


@pytest.mark.parametrize("change", [
    {"relations": 1}, {"covariate_names": ["BW", "BW"]},
    {"covariate_names": "BW"}, {"covariate_names": [" BW"]}, {"unknown": 1},
])
def test_parameter_relationship_rejects_invalid(change):
    with pytest.raises(InputError):
        validate_parameter_model({"covariate_names": [], "relations": NoExecution(), **change})


def test_parameters_preserve_negative_coefficients_and_do_not_share_data():
    original = {"values": {"theta": -1.01, "variance": 0}, "units": {"theta": "1", "variance": "1"}}
    result = validate_parameters(original)
    result["values"]["theta"] = 2
    assert original["values"]["theta"] == -1.01


@pytest.mark.parametrize("level", ["iiv", "iov"])
def test_random_effect_references_preserve_none_zeros_and_copy(level):
    parameters = {"values": {"v": 0.1}, "units": {"v": "1"}}
    spec = {"names": ["CL", "V"], "distribution": "normal", "covariance_parameters": [["v", None], [None, "v"]]}
    if level == "iov":
        spec["occasion_key"] = "occasion_id"
    result = validate_random_effect_spec(spec, parameters=parameters, level=level)
    result["covariance_parameters"][0][0] = None
    assert spec["covariance_parameters"][0][0] == "v"
    assert result["names"] == ("CL", "V")
    assert validate_random_effect_spec(None, parameters=parameters, level=level) is None


@pytest.mark.parametrize("change", [
    {"distribution": "lognormal"}, {"distribution": ["normal"]},
    {"names": []}, {"names": ["CL", "CL"]},
    {"covariance_parameters": [["missing"]]}, {"covariance_parameters": [[True]]},
    {"covariance_parameters": [[0.1]]}, {"covariance_parameters": [["v", None]]},
    {"dependence": "independent_blocks"},
])
def test_random_effect_invalid_definition(change):
    with pytest.raises(InputError):
        validate_random_effect_spec(
            {"names": ["CL"], "distribution": "normal", "covariance_parameters": [["v"]], **change},
            parameters={"values": {"v": 0.1}, "units": {"v": "1"}}, level="iiv")


def test_random_effect_variance_units_are_not_sd_units():
    with pytest.raises(InputError, match="dimensionless"):
        validate_random_effect_spec({"names": ["CL"], "distribution": "normal", "covariance_parameters": [["v"]]},
                                    parameters={"values": {"v": 0.1}, "units": {"v": "L/h"}}, level="iiv")


def model_definition():
    return {"parameters": {"values": {"sd": 2.992, "var": 0.01}, "units": {"sd": "mg/L", "var": "1"}},
            "parameter_model": {"covariate_names": ["BW"], "relations": NoExecution()},
            "iiv": {"names": ["CL"], "distribution": "normal", "covariance_parameters": [["var"]]},
            "iov": None, "residual_model": {"distribution": "normal", "error_type": "additive", "parameter_names": {"sd": "sd"}},
            "predict": NoExecution()}


def test_complete_model_is_an_isolated_copy_without_calling_or_copying_callbacks():
    model = model_definition()
    result = validate_model(model)
    assert result["predict"] is model["predict"]
    assert result["parameter_model"]["relations"] is model["parameter_model"]["relations"]
    result["parameters"]["values"]["sd"] = 9
    result["iiv"]["covariance_parameters"][0][0] = None
    result["residual_model"]["parameter_names"]["sd"] = "missing"
    assert model["parameters"]["values"]["sd"] == 2.992
    assert model["iiv"]["covariance_parameters"] == [["var"]]
    assert model["residual_model"]["parameter_names"] == {"sd": "sd"}


@pytest.mark.parametrize("field", ["parameters", "parameter_model", "iiv", "iov", "residual_model", "predict"])
def test_all_six_groups_are_explicit(field):
    model = model_definition()
    del model[field]
    with pytest.raises(InputError):
        validate_model(model)


@pytest.mark.parametrize("value", [None, 0, "predict"])
def test_predict_must_be_callable(value):
    model = model_definition()
    model["predict"] = value
    with pytest.raises(InputError):
        validate_model(model)


@pytest.mark.parametrize("sd", [0, -1, float("nan"), True])
def test_invalid_residual_sd(sd):
    model = model_definition()
    model["parameters"]["values"]["sd"] = sd
    with pytest.raises(InputError):
        validate_model(model)


def test_residual_roles_units_and_no_silent_variance_to_sd_conversion():
    model = model_definition()
    residual = model["residual_model"]
    for role in ("variance", "coefficient_sd"):
        with pytest.raises(InputError):
            validate_residual_model({**residual, "parameter_names": {role: "sd"}}, parameters=model["parameters"])
    residual.update(error_type="proportional", parameter_names={"coefficient_sd": "sd"})
    with pytest.raises(InputError, match="unit"):
        validate_model(model)
    model["parameters"]["units"]["sd"] = "1"
    assert validate_model(model)["residual_model"] == residual


def test_independence_not_added_implicitly_and_unknown_fields_rejected():
    model = model_definition()
    assert "dependence" not in validate_model(model)
    for name in ("covariates", "observations", "priors", "dependence"):
        with pytest.raises(InputError):
            validate_model({**model, name: {}})


@pytest.mark.parametrize("matrix", [
    [[-1, 0], [0, 1]], [[1, 0.1], [0, 1]], [[0, 0.1], [0.1, 1]],
    [[1, 2], [2, 1]], [[1, 1], [1, 1]],
])
def test_model_static_covariance_rejects_invalid_geometry(matrix):
    model = model_definition()
    for i in range(2):
        for j in range(2):
            model["parameters"]["values"][f"c{i}{j}"] = matrix[i][j]
            model["parameters"]["units"][f"c{i}{j}"] = "1"
    model["iiv"] = {"names": ["CL", "V"], "distribution": "normal",
                    "covariance_parameters": [["c00", "c01"], ["c10", "c11"]]}
    with pytest.raises(InputError):
        validate_model(model)


@pytest.mark.parametrize("value", [{"CL", "V"}, {"CL": 1}, iter(["CL"])])
def test_model_names_require_a_reusable_ordered_sequence(value):
    model = model_definition()
    model["iiv"]["names"] = value
    with pytest.raises(InputError):
        validate_model(model)


def test_covariate_names_reject_unordered_set():
    with pytest.raises(InputError):
        validate_parameter_model({"covariate_names": {"BW"}, "relations": NoExecution()})


@pytest.mark.parametrize("variance,covariance", [(0, 0), (1e-20, 0), (1, -0.4)])
def test_valid_covariance_not_regularized_or_converted(variance, covariance):
    model = model_definition()
    model["parameters"]["values"].update(v=variance, c=covariance)
    model["parameters"]["units"].update(v="1", c="1")
    model["iiv"] = {"names": ["CL", "V"], "distribution": "normal",
                    "covariance_parameters": [["v", "c"], ["c", "v"]]}
    validated = validate_model(model)
    assert validated["parameters"]["values"]["v"] == variance
    assert validated["iiv"]["covariance_parameters"] == [["v", "c"], ["c", "v"]]


@pytest.mark.parametrize("name,dimension", [("OXC", 1), ("TAC", 10)])
def test_reference_model_static_validation_preserves_all_four_predictions(name, dimension):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("input_reference", root / "examples/reference_models.py")
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)
    case = examples.load_reference(name)
    checked = {**case, "model": validate_model(case["model"])}
    assert checked["model"]["predict"] is case["model"]["predict"]
    for state in case["candidate_states"]:
        before, ll_before = examples.evaluate_reference(case, state, np.zeros(dimension))
        after, ll_after = examples.evaluate_reference(checked, state, np.zeros(dimension))
        np.testing.assert_array_equal(before["values"], after["values"])
        assert ll_before == ll_after
