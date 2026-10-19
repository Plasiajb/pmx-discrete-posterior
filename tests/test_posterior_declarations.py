import pytest

from pmx_discrete_posterior.interfaces.inputs.model_inputs import validate_model, validate_residual_model
from pmx_discrete_posterior.interfaces.validation import InputError
from test_input_model import model_definition


def test_optional_low_level_declarations_are_preserved_without_model_execution():
    model = model_definition()
    model["random_effects_dependence"] = "independent_blocks"
    model["residual_model"]["dependence"] = "conditionally_independent"
    checked = validate_model(model)
    assert checked["random_effects_dependence"] == "independent_blocks"
    assert checked["residual_model"]["dependence"] == "conditionally_independent"
    assert checked["predict"] is model["predict"]
    checked["residual_model"]["dependence"] = "changed"
    assert model["residual_model"]["dependence"] == "conditionally_independent"


def test_low_level_validation_does_not_infer_independence():
    checked = validate_model(model_definition())
    assert "random_effects_dependence" not in checked
    assert "dependence" not in checked["residual_model"]


@pytest.mark.parametrize("value", [None, True, "correlated", {}, ["independent_blocks"]])
def test_random_effect_declaration_rejects_unsupported_or_invalid_values(value):
    model = model_definition()
    model["random_effects_dependence"] = value
    with pytest.raises(InputError, match="random_effects_dependence"):
        validate_model(model)


@pytest.mark.parametrize("value", [None, True, "correlated", {}, ["conditionally_independent"]])
def test_residual_declaration_rejects_unsupported_or_invalid_values(value):
    model = model_definition()
    model["residual_model"]["dependence"] = value
    with pytest.raises(InputError, match="dependence"):
        validate_residual_model(model["residual_model"], parameters=model["parameters"])
