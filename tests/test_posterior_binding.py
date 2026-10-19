from copy import deepcopy

import numpy as np
import pytest

from pmx_discrete_posterior.models.prediction import prepare_prediction_records
from pmx_discrete_posterior.models.random_effects import resolve_random_effects
from pmx_discrete_posterior.posterior.computation import _bind_likelihood
from test_numerics_reference_models import examples


def bind(case, state):
    model, data = case["model"], case["individual_data"]
    effects = resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
                                     occasions=data["occasions"], dependence=model["random_effects_dependence"])
    return _bind_likelihood(model=model, individual_data=data, observations=case["observations"],
                            candidate_state=state, random_effect_distribution=effects,
                            prediction_request=prepare_prediction_records(case["observations"], data))


def declared_reference(name):
    case = examples.load_reference(name)
    case["model"]["random_effects_dependence"] = "independent_blocks"
    case["model"]["residual_model"]["dependence"] = "conditionally_independent"
    return case


@pytest.mark.parametrize("name,dimension", [("OXC", 1), ("TAC", 10)])
def test_bound_model_chain_matches_reference_at_multiple_effect_points(name, dimension):
    case = declared_reference(name)
    callbacks = [bind(case, state) for state in case["candidate_states"]]
    for state, callback in zip(case["candidate_states"], callbacks):
        for point in (np.zeros(dimension), np.linspace(-.1, .1, dimension)):
            expected = examples.evaluate_reference(case, state, point)[1]
            assert callback(point) == pytest.approx(expected, abs=1e-12, rel=0)


def test_binding_does_not_pass_observed_values_priors_or_raw_effects_to_predictor():
    case = declared_reference("OXC")
    original = case["model"]["predict"]
    calls = []

    def predict(**kwargs):
        assert set(kwargs) == {"individual_parameters", "candidate_state", "prediction_records", "individual_data"}
        assert all("value" not in record for record in kwargs["prediction_records"])
        assert "priors" not in kwargs["individual_data"]
        calls.append(deepcopy(kwargs))
        return original(**kwargs)

    case["model"]["predict"] = predict
    callback = bind(case, case["candidate_states"][0])
    first, second = callback(np.array([0.])), callback(np.array([.2]))
    assert len(calls) == 2 and first != second
    assert calls[0]["individual_parameters"] != calls[1]["individual_parameters"]
    assert calls[0]["individual_data"]["initialization"]["mode"] == "steady_state"


def test_multiple_observations_are_bound_together_not_integrated_separately():
    case = declared_reference("OXC")
    case["observations"].append({"record_id": "extra", "time": 20., "value": 10., "time_unit": "h", "value_unit": "mg/L"})
    state, eta = case["candidate_states"][2], np.array([.1])
    callback = bind(case, state)
    assert callback(eta) == pytest.approx(examples.evaluate_reference(case, state, eta)[1], abs=1e-12, rel=0)
