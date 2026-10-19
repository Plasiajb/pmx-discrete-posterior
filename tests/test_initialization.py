import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.prediction.initialization import advance, clock_add, periodic_state


@pytest.mark.parametrize("k,expected", [(math.log(2) / 12, 100 * math.sqrt(2)), (math.log(4) / 12, 200 / 3)])
def test_conditional_steady_state_and_no_automatic_future_dose(k, expected):
    result = periodic_state(matrix=[[-k]], period=12, anchor_time=0, start_time=6,
                            input_events=[{"offset": 0, "lag": 0, "values": [100]}])
    assert result["values"][0] == pytest.approx(expected, rel=1e-12)
    assert result["pending_inputs"] == []
    assert advance([[-k]], result["values"], 12)[0] == pytest.approx(expected * math.exp(-12*k), rel=1e-12)


def test_boundary_initial_state_excludes_simultaneous_input():
    k = math.log(2) / 12
    result = periodic_state(matrix=[[-k]], period=12, anchor_time=0, start_time=0,
                            input_events=[{"offset": 0, "lag": 0, "values": [100]}])
    assert result["values"][0] == pytest.approx(100)


def test_lag_over_period_preserves_all_pending_administrations():
    result = periodic_state(matrix=[[-.1]], period=12, anchor_time=0, start_time=0,
                            input_events=[{"offset": 0, "lag": 25, "values": [100]}])
    assert [e["time"] for e in result["pending_inputs"]] == [1, 13]
    assert result["values"][0] == pytest.approx(100 * math.exp(-1.1) / (1-math.exp(-1.2)))


def test_periodic_sum_matches_long_explicit_history_two_compartments():
    matrix = np.array([[-.8, 0], [.8, -.1]])
    event = {"offset": 3, "lag": 2, "values": [100, 0]}
    result = periodic_state(matrix=matrix, period=12, anchor_time=0, start_time=7, input_events=[event])
    expected = sum((advance(matrix, event["values"], 2 + 12*i) for i in range(100)), np.zeros(2))
    assert result["values"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("matrix", [[[0]], [[.1]], [[-.1, -1], [0, -.2]]])
def test_invalid_or_unstable_steady_state(matrix):
    with pytest.raises(InputError):
        periodic_state(matrix=matrix, period=12, anchor_time=0, start_time=0,
                       input_events=[{"offset": 0, "lag": 0, "values": [100]}])


def test_decimal_clock_and_state_continuity():
    assert clock_add(.1, .2) == .3
    state = advance([[-.1]], [100], 6)
    state = advance([[-.2]], state, 3)
    assert state[0] == pytest.approx(100 * math.exp(-1.2), rel=1e-12)
