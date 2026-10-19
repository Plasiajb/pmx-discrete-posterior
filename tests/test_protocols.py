import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.protocols import (
    ConditionalLogLikelihood, Integrator, ParameterRelations, Predict, RandomEffectDistributionProtocol, RandomEffectLogDensity,
)
from pmx_discrete_posterior.interfaces.validation import InputError, ModelExecutionError
from pmx_discrete_posterior.models.parameter_resolution import resolve_parameters
from pmx_discrete_posterior.models.prediction import evaluate_prediction, prepare_prediction_records
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution, resolve_random_effects


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protocol_references", ROOT / "examples/reference_models.py")
REFERENCES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCES)


def parameter_arguments(case):
    model, individual = case["model"], case["individual_data"]
    distribution = resolve_random_effects(
        iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
        occasions=individual["occasions"], dependence="independent_blocks",
    )
    request = prepare_prediction_records(case["observations"], individual)
    return dict(parameters=model["parameters"], covariates=individual["covariates"],
                **distribution.expand(distribution.mean), occasions=individual["occasions"],
                evaluation_interval=request["evaluation_interval"])


@pytest.mark.parametrize("name", ["OXC", "TAC"])
def test_parameter_protocol_matches_existing_callbacks_without_wrapper(name):
    case = REFERENCES.load_reference(name)
    callback: ParameterRelations = case["model"]["parameter_model"]["relations"]
    arguments = parameter_arguments(case)
    protocol_signature = inspect.signature(ParameterRelations.__call__)
    protocol_signature.bind(None, **arguments)
    inspect.signature(callback).bind(**arguments)
    assert callback is case["model"]["parameter_model"]["relations"]
    result = resolve_parameters(parameter_model=case["model"]["parameter_model"], **arguments)
    assert set(result) == {"units", "segments"}
    assert len(result["segments"]) == (1 if name == "OXC" else 3)
    assert all(segment["values"]["CL"] > 0 for segment in result["segments"])


def test_protocol_is_not_a_runtime_certificate_or_an_executable_adapter():
    with pytest.raises(TypeError):
        ParameterRelations()
    with pytest.raises(TypeError):
        isinstance(lambda: None, ParameterRelations)


def test_existing_parameter_executor_still_rejects_wrong_callback_signature():
    case = REFERENCES.load_reference("OXC")
    definition = {**case["model"]["parameter_model"], "relations": lambda wrong: wrong}
    with pytest.raises(ModelExecutionError, match="parameter_model.relations failed"):
        resolve_parameters(parameter_model=definition, **parameter_arguments(case))


def test_parameter_protocol_has_no_dosing_observation_or_prior_arguments():
    expected = {"self", "parameters", "covariates", "iiv_values", "iov_values", "occasions", "evaluation_interval"}
    assert set(inspect.signature(ParameterRelations.__call__).parameters) == expected


@pytest.mark.parametrize("name", ["OXC", "TAC"])
def test_predict_protocol_uses_existing_predictor_and_preserves_reference(name):
    case = REFERENCES.load_reference(name)
    callback: Predict = case["model"]["predict"]
    params = resolve_parameters(parameter_model=case["model"]["parameter_model"], **parameter_arguments(case))
    request = prepare_prediction_records(case["observations"], case["individual_data"])
    arguments = dict(individual_parameters=params, candidate_state=case["candidate_states"][0],
                     prediction_records=request["prediction_records"], individual_data=case["individual_data"])
    inspect.signature(Predict.__call__).bind(None, **arguments)
    inspect.signature(callback).bind(**arguments)
    expected, _ = REFERENCES.evaluate_reference(case, arguments["candidate_state"], np.zeros(1 if name == "OXC" else 10))
    actual = evaluate_prediction(predict=callback, **arguments)
    assert callback is case["model"]["predict"]
    np.testing.assert_array_equal(actual["values"], expected["values"])
    assert actual["units"] == expected["units"]
    assert all("value" not in row for row in arguments["prediction_records"])


def test_predict_protocol_excludes_raw_effects_observed_concentrations_and_priors():
    assert set(inspect.signature(Predict.__call__).parameters) == {
        "self", "individual_parameters", "candidate_state", "prediction_records", "individual_data",
    }
    with pytest.raises(TypeError):
        Predict()


@pytest.mark.parametrize("dimension", [0, 1, 3])
def test_distribution_protocol_reuses_normalized_density_and_active_coordinates(dimension):
    actual = RandomEffectDistribution([("iiv", None, str(i)) for i in range(dimension)], np.eye(dimension))
    distribution: RandomEffectDistributionProtocol = actual
    density: RandomEffectLogDensity = actual.log_density
    point = np.zeros(dimension)
    assert distribution is actual
    inspect.signature(RandomEffectLogDensity.__call__).bind(None, point)
    inspect.signature(density).bind(point)
    assert density(point) == pytest.approx(-dimension * np.log(2 * np.pi) / 2)
    assert distribution.mean.shape == (dimension,)
    assert distribution.covariance.shape == (dimension, dimension)
    assert list(distribution.expand(point)["iiv_values"]) == [str(i) for i in range(dimension)]
    with pytest.raises(InputError):
        density(np.zeros(dimension + 1))


def test_distribution_protocol_preserves_iov_order_and_fixed_zero_effects():
    actual = RandomEffectDistribution([("iiv", None, "CL"), ("iov", "a", "V"), ("iov", "b", "V")], np.diag([0, .2, .2]))
    distribution: RandomEffectDistributionProtocol = actual
    assert distribution.effect_keys == (("iov", "a", "V"), ("iov", "b", "V"))
    assert distribution.expand(np.array([.1, -.3])) == {
        "iiv_values": {"CL": 0.0}, "iov_values": {"a": {"V": .1}, "b": {"V": -.3}},
    }


def test_bound_likelihood_protocol_does_not_add_continuous_or_discrete_priors():
    case = REFERENCES.load_reference("OXC")
    state = case["candidate_states"][0]

    def likelihood(random_effect_values):
        return REFERENCES.evaluate_reference(case, state, random_effect_values)[1]

    callback: ConditionalLogLikelihood = likelihood
    point = np.zeros(1)
    inspect.signature(ConditionalLogLikelihood.__call__).bind(None, point)
    assert callback is likelihood
    assert callback(point) == REFERENCES.evaluate_reference(case, state, point)[1]
    assert set(inspect.signature(ConditionalLogLikelihood.__call__).parameters) == {"self", "random_effect_values"}
    with pytest.raises(InputError):
        callback(np.array([0, 0]))


def test_integrator_contract_is_not_an_implemented_integrator_and_excludes_state_priors():
    parameters = inspect.signature(Integrator.__call__).parameters
    assert set(parameters) == {"self", "conditional_log_likelihood", "random_effect_distribution",
                               "integration_method", "numerical_options"}
    assert parameters["integration_method"].default is inspect.Parameter.empty
    assert parameters["numerical_options"].default is None
    assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY
               for name, parameter in parameters.items() if name != "self")
    with pytest.raises(TypeError):
        Integrator()


@pytest.mark.parametrize("protocol", [ParameterRelations, Predict, ConditionalLogLikelihood, RandomEffectLogDensity, Integrator])
def test_protocol_annotations_are_resolvable(protocol):
    hints = get_type_hints(protocol.__call__)
    assert "return" in hints


@pytest.mark.parametrize("order", [("protocols", "results"), ("results", "protocols")])
def test_new_interfaces_import_without_loading_model_or_numerical_execution(order):
    command = "import importlib, sys\n"
    command += "\n".join(f"importlib.import_module('pmx_discrete_posterior.interfaces.{name}')" for name in order)
    command += "\nassert not any(name.startswith(('pmx_discrete_posterior.models', 'pmx_discrete_posterior.numerics')) for name in sys.modules)"
    result = subprocess.run([sys.executable, "-c", command], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": ""})
    assert result.returncode == 0, result.stderr
