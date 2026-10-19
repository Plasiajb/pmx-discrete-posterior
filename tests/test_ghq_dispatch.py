import copy

import pytest

from pmx_discrete_posterior.interfaces.inputs import validate_inputs
from pmx_discrete_posterior.interfaces.results import validate_posterior_result
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics import integrate
from pmx_discrete_posterior.numerics import ghq, laplace
from test_ghq_options import ghq_options
from test_input_assembly import inputs


@pytest.mark.parametrize("method", ["ghq", "laplace"])
def test_dispatches_once_without_repacking_result(monkeypatch, method):
    sentinel = object()
    calls = []
    def chosen(**kwargs):
        calls.append(kwargs)
        return sentinel
    module, other = (ghq, laplace) if method == "ghq" else (laplace, ghq)
    monkeypatch.setattr(module, "integrate", chosen)
    monkeypatch.setattr(other, "integrate", lambda **kw: pytest.fail("wrong integrator"))
    assert integrate(conditional_log_likelihood=None, random_effect_distribution=None,
                     integration_method=method) is sentinel
    assert len(calls) == 1 and calls[0]["integration_method"] == method


def test_inputs_require_matching_explicit_settings():
    args = inputs()
    args["integration_method"] = "ghq"
    with pytest.raises(InputError, match="explicit"):
        validate_inputs(**args)
    args["numerical_options"] = ghq_options()
    assert validate_inputs(**args)["numerical_options"] == ghq_options()
    args["integration_method"] = "laplace"
    with pytest.raises(InputError, match="Laplace"):
        validate_inputs(**args)


def test_posterior_method_details_are_optional_copied_and_not_a_new_status():
    result = dict(status="success", integration_method="ghq", failure_reason=None,
                  state_results=[dict(state_id="a", prior_probability=1., posterior_probability=1.,
                                      status="success", log_marginal_likelihood=0., failure_reason=None)],
                  method_details={"posterior_order_results": [1, 2, 3]})
    args = dict(candidate_states=[dict(state_id="a", events=[])], priors={"a": 1.})
    checked = validate_posterior_result(result, **args)
    checked["method_details"]["posterior_order_results"].append(4)
    assert result["method_details"]["posterior_order_results"] == [1, 2, 3]
    invalid = copy.deepcopy(result)
    invalid["method_details"] = []
    with pytest.raises(InputError):
        validate_posterior_result(invalid, **args)
