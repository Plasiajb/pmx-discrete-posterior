import copy

import pytest

from pmx_discrete_posterior.interfaces.inputs.numerical_options import resolve_ghq_options, validate_numerical_options
from pmx_discrete_posterior.interfaces.validation import InputError


def ghq_options(orders=(12, 20, 32), **overrides):
    settings = dict(variant="prior_centered", orders=list(orders), log_integral_atol=1e-6,
                    posterior_atol=1e-6, max_nodes=100000)
    settings.update(overrides)
    return {"ghq": settings}


def test_explicit_configuration_is_copied_and_revalidates():
    options = ghq_options()
    parsed = validate_numerical_options(options)
    assert parsed == options == validate_numerical_options(parsed)
    parsed["ghq"]["orders"][0] = 1
    assert options["ghq"]["orders"][0] == 12


@pytest.mark.parametrize("field,value", [
    ("variant", "adaptive"), ("variant", None), ("orders", [3, 5]),
    ("orders", [3, 3, 7]), ("orders", [3, 7, 5]), ("orders", [True, 3, 5]),
    ("orders", [1., 3, 5]), ("orders", [0, 3, 5]), ("orders", "3,5,7"),
    ("max_nodes", 0), ("max_nodes", True), ("max_nodes", 100.),
    ("log_integral_atol", 0), ("log_integral_atol", float("nan")),
    ("posterior_atol", 0), ("posterior_atol", 1), ("posterior_atol", float("inf")),
    ("unknown", 3),
])
def test_invalid_configuration_rejected(field, value):
    options = ghq_options()
    options["ghq"][field] = value
    with pytest.raises(InputError):
        validate_numerical_options(options)


@pytest.mark.parametrize("field", list(ghq_options()["ghq"]))
def test_every_ghq_setting_is_explicit(field):
    options = copy.deepcopy(ghq_options()["ghq"])
    del options[field]
    with pytest.raises(InputError):
        resolve_ghq_options(options)


@pytest.mark.parametrize("group", ["optimization", "derivatives"])
def test_unused_groups_are_not_silently_ignored(group):
    options = ghq_options()
    options[group] = {}
    with pytest.raises(InputError, match="only the ghq group"):
        validate_numerical_options(options)
