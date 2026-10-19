from fractions import Fraction

import pytest

from pmx_discrete_posterior.interfaces.validation import (
    InputError, fields, names, numeric_array, probability_scalar, scalar, validate_parameters,
)


def test_parameters_are_copies_and_coefficients_can_be_negative():
    original = {"values": {"theta": -1.01}, "units": {"theta": "1"}}
    result = validate_parameters(original)
    result["values"]["theta"] = 9
    assert original["values"]["theta"] == -1.01


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), 1j])
def test_invalid_scalar(value):
    with pytest.raises(InputError):
        scalar(value)


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), float("inf"), -.1, 1.1,
                                   Fraction(1, 10**400), Fraction(-1, 10**400), 1 + Fraction(1, 10**400)])
def test_probability_scalar_rejects_invalid_values_and_loss_of_nonzero_probability(value):
    with pytest.raises(InputError):
        probability_scalar(value)


@pytest.mark.parametrize("value", [0, 1, Fraction(1, 4), Fraction(1, 10**300), float.fromhex("0x0.0000000000001p-1022")])
def test_probability_scalar_retains_representable_values(value):
    assert probability_scalar(value) == float(value)


@pytest.mark.parametrize("value", [True, "1", [1, 2], [[1], [2, 3]], [float("nan")]])
def test_numeric_arrays_reject_invalid_shape_or_type(value):
    with pytest.raises(InputError):
        numeric_array(value, ndim=2)


def test_fields_names_and_units_are_strict():
    for value in ({"a": 1, "extra": 2}, {}, None):
        with pytest.raises(InputError):
            fields(value, {"a"})
    with pytest.raises(InputError):
        names(["a", "a"])
    with pytest.raises(InputError):
        validate_parameters({"values": {"CL": 1}, "units": {}})
    assert numeric_array([1, 2]).tolist() == [1.0, 2.0]
    assert numeric_array([[1], [2]], ndim=2).tolist() == [[1.0], [2.0]]


@pytest.mark.parametrize("value,ndim", [([False, 6], 1), ([1., True], 1), ([[1., False], [0, 1]], 2)])
def test_mixed_boolean_numeric_arrays_are_not_silently_promoted(value, ndim):
    with pytest.raises(InputError):
        numeric_array(value, ndim=ndim)
