import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.residuals import resolve_residuals


def args(error_type="additive", value=10, sd=2):
    role = "sd" if error_type == "additive" else "coefficient_sd"
    return dict(residual_model={"distribution": "normal", "error_type": error_type, "parameter_names": {role: "sigma"}},
        parameters={"values": {"sigma": sd}, "units": {"sigma": "mg/L" if error_type == "additive" else "1"}},
        predictions={"record_ids": ["a"], "values": [value], "units": ["mg/L"]},
        prediction_records=[{"record_id": "a", "time": 24, "time_unit": "h", "value_unit": "mg/L"}],
        dependence="conditionally_independent")


def test_additive_and_proportional_sd_are_not_variance_or_percent():
    additive = resolve_residuals(**args())
    assert additive["sd"].tolist() == [2]
    proportional = resolve_residuals(**args("proportional", 10, .183))
    assert proportional["sd"][0] == pytest.approx(1.83)
    assert proportional["means"].tolist() == [10]


@pytest.mark.parametrize("sd", [-1, 0, float("nan"), float("inf")])
def test_invalid_sd_rejected(sd):
    with pytest.raises(InputError):
        resolve_residuals(**args(sd=sd))


def test_proportional_zero_prediction_is_not_regularized():
    with pytest.raises(InputError):
        resolve_residuals(**args("proportional", 0, .183))


def test_distribution_units_dependence_and_unknown_rules():
    for section, key, value in [("residual_model", "distribution", "uniform"),
                                 ("residual_model", "error_type", "combined")]:
        data = args()
        data[section][key] = value
        with pytest.raises(InputError):
            resolve_residuals(**data)
    data = args()
    data["parameters"]["units"]["sigma"] = "1"
    with pytest.raises(InputError):
        resolve_residuals(**data)
    data = args()
    data["dependence"] = None
    with pytest.raises(InputError):
        resolve_residuals(**data)
