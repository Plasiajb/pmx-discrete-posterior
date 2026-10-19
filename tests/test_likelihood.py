import math

import pytest

from pmx_discrete_posterior.interfaces.validation import InputError, ModelExecutionError
from pmx_discrete_posterior.models.likelihood import conditional_log_likelihood


def args():
    return dict(observations=[{"record_id": k, "time": t, "value": y, "time_unit": "h", "value_unit": "mg/L"}
                              for k, t, y in (("a", 1, 1), ("b", 2, 12))],
                predictions={"record_ids": ["a", "b"], "values": [0, 10], "units": ["mg/L", "mg/L"]},
                residual_distribution={"distribution": "normal", "dependence": "conditionally_independent",
                    "record_ids": ["a", "b"], "means": [0, 10], "sd": [1, 2], "units": ["mg/L", "mg/L"]})


def test_independent_normal_includes_normalizer():
    expected = -math.log(2*math.pi) - math.log(2) - 1
    assert conditional_log_likelihood(**args()) == pytest.approx(expected, abs=1e-12)


def test_negative_observation_allowed_under_normal_error():
    data = args()
    data["observations"][0]["value"] = -1
    assert conditional_log_likelihood(**data) == pytest.approx(conditional_log_likelihood(**args()), abs=1e-12)


def test_joint_record_permutation_does_not_change_likelihood():
    data = args()
    data["observations"].reverse()
    for key in ("record_ids", "values", "units"):
        data["predictions"][key].reverse()
    for key in ("record_ids", "means", "sd", "units"):
        data["residual_distribution"][key].reverse()
    assert conditional_log_likelihood(**data) == pytest.approx(conditional_log_likelihood(**args()), abs=1e-12)


@pytest.mark.parametrize("change", ["ids", "sd", "stale", "prior", "unit"])
def test_mismatch_or_leakage_rejected(change):
    data = args()
    if change == "ids":
        data["predictions"]["record_ids"].reverse()
    elif change == "sd":
        data["residual_distribution"]["sd"] = [1, float("nan")]
    elif change == "stale":
        data["residual_distribution"]["means"] = [0, 11]
    elif change == "prior":
        data["residual_distribution"]["prior"] = .25
    else:
        data["residual_distribution"]["units"] = ["mg", "mg"]
    with pytest.raises(InputError):
        conditional_log_likelihood(**data)


def test_computational_overflow_not_treated_as_mathematical_zero():
    data = args()
    data["observations"][0]["value"] = 1e308
    data["residual_distribution"]["sd"][0] = 1e-300
    with pytest.raises(ModelExecutionError):
        conditional_log_likelihood(**data)
