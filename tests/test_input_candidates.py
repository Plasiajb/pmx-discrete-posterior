from copy import deepcopy

import pytest

from pmx_discrete_posterior.interfaces.inputs.candidate_states import validate_candidate_states, validate_candidate_history
from pmx_discrete_posterior.interfaces.validation import InputError


def event(**changes):
    return {"time": 12, "amount": 300, "amount_unit": "mg", "route": "oral", "compartment": "depot", **changes}


def test_labels_do_not_infer_events_or_order():
    states = [{"state_id": "11", "events": []}, {"state_id": "00", "events": [event()]}]
    original = deepcopy(states)
    result = validate_candidate_states(states)
    assert result == original
    result[1]["events"][0]["amount"] = 900
    assert states == original


@pytest.mark.parametrize("states", [[], None, {"state_id": "x", "events": []},
    [{"state_id": "x", "events": []}, {"state_id": "x", "events": []}],
    [{"state_id": "x", "events": [], "prior": 1}],
    [{"state_id": "x", "events": [], "truth": True}],
    [{"state_id": "x", "events": [event(amount=-1)]}],
    [{"state_id": "x", "events": [event(amount=True)]}],
    [{"state_id": "x", "events": [event(rate=1)]}],
])
def test_invalid_candidates(states):
    with pytest.raises(InputError):
        validate_candidate_states(states)


def test_context_prevents_pre_boundary_and_duplicate_common_doses():
    state = {"state_id": "a", "events": [event()]}
    with pytest.raises(InputError):
        validate_candidate_history(state, initialization_start=13, common_events=[])
    with pytest.raises(InputError):
        validate_candidate_history(state, initialization_start=0, common_events=[event()])
    assert validate_candidate_history(state, initialization_start=12, common_events=[]) == state
