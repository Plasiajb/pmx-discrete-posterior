import math

import pytest

from pmx_discrete_posterior.interfaces.inputs.priors import validate_priors
from pmx_discrete_posterior.interfaces.validation import InputError


def states(*ids):
    return [{"state_id": key, "events": []} for key in ids]


def test_priors_follow_candidate_order_keep_zeros_and_copy():
    priors = {"b": 0, "a": 1}
    result = validate_priors(priors, candidate_states=states("a", "b"))
    assert list(result) == ["a", "b"]
    assert result == {"a": 1., "b": 0.}
    result["a"] = 0
    assert priors == {"b": 0, "a": 1}


@pytest.mark.parametrize("priors", [None, [], {}, {"a": 1}, {"a": .5, "b": .5, "c": 0},
    {"a": 1, "b": 1}, {"a": 0, "b": 0}, {"a": -.1, "b": 1.1}, {"a": True, "b": 0},
    {"a": "0.5", "b": .5}, {"a": float("nan"), "b": .5}, {"a": float("inf"), "b": 0},
    {"a": .5, "b": .499999}, {"a": 1 + math.ulp(1.), "b": 0},
])
def test_reject_illegal_or_unnormalized_priors(priors):
    with pytest.raises(InputError):
        validate_priors(priors, candidate_states=states("a", "b"))


def test_tiny_positive_probability_is_not_converted_to_zero():
    priors = {"a": 1., "b": 1e-300}
    assert validate_priors(priors, candidate_states=states("a", "b"))["b"] == 1e-300


def test_only_roundoff_tolerance_no_automatic_normalization():
    priors = {"a": .4, "b": .6 + math.ulp(1.)}
    assert validate_priors(priors, candidate_states=states("a", "b")) == priors
    with pytest.raises(InputError):
        validate_priors({"a": .4, "b": .6 + 1e-12}, candidate_states=states("a", "b"))
