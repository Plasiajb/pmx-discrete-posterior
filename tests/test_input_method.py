import pytest

from pmx_discrete_posterior.interfaces.inputs.integration_method import validate_integration_method
from pmx_discrete_posterior.interfaces.validation import InputError


@pytest.mark.parametrize("method", ["laplace"])
def test_accept_known_integration_names_only(method):
    assert validate_integration_method(method) == method


@pytest.mark.parametrize("method", [None, True, 1, {}, [], "Laplace", "ghq", "GHQ", "aghq", "auto", " laplace", "mcmc"])
def test_no_implicit_method_alias_or_default(method):
    with pytest.raises(InputError):
        validate_integration_method(method)
