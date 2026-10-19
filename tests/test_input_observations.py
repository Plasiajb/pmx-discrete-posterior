from copy import deepcopy

import pytest

from pmx_discrete_posterior.interfaces.inputs.observations import validate_observations
from pmx_discrete_posterior.interfaces.validation import InputError


def observation(**changes):
    return {"record_id": "r1", "time": 24, "value": 12, "time_unit": "h", "value_unit": "mg/L", **changes}


def test_preserve_record_order_duplicate_times_and_independent_copy():
    original = [observation(time=24), observation(record_id="r2", time=12, value=-0.1), observation(record_id="r3", time=12)]
    before = deepcopy(original)
    result = validate_observations(original)
    assert result == original
    result[0]["value"] = 100
    assert original == before


@pytest.mark.parametrize("records", [[], None, "records", [observation(), observation()],
    [observation(value=True)], [observation(value=float("nan"))], [observation(time=float("inf"))],
    [observation(truth="taken")], [observation(), observation(record_id="r2", time_unit="min")],
    [observation(), observation(record_id="r2", value_unit="ng/mL")],
])
def test_invalid_observations_are_not_silently_corrected(records):
    with pytest.raises(InputError):
        validate_observations(records)


def test_missing_observation_value_not_a_prediction_request():
    record = observation()
    del record["value"]
    with pytest.raises(InputError):
        validate_observations([record])
