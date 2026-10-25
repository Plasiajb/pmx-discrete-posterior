"""Native GHQ removal must reject requests, never route to Laplace or external R."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import get_args, get_type_hints

import pytest
import pmx_discrete_posterior as pmx
from pmx_discrete_posterior.interfaces.inputs import validate_inputs
from pmx_discrete_posterior.interfaces.inputs import numerical_options
from pmx_discrete_posterior.interfaces.inputs.integration_method import validate_integration_method
from pmx_discrete_posterior.interfaces.protocols import Integrator
from pmx_discrete_posterior.interfaces.results import PosteriorResult, validate_posterior_result
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics import integrate, laplace
from pmx_discrete_posterior.posterior import computation
from cli_helpers import cli
from test_posterior_computation import analytic_inputs


def test_removed_module_helpers_and_public_types_are_absent():
    package = Path(pmx.__file__).resolve().parent
    assert not (package / "numerics" / "ghq.py").exists()
    assert importlib.util.find_spec("pmx_discrete_posterior.numerics.ghq") is None
    assert "pmx_discrete_posterior.numerics.ghq" not in sys.modules
    assert not hasattr(numerical_options, "resolve_ghq_options")
    assert not hasattr(computation, "_ghq_posterior")
    assert get_args(get_type_hints(Integrator.__call__)["integration_method"]) == ("laplace",)
    assert get_args(get_type_hints(PosteriorResult)["integration_method"]) == ("laplace",)


@pytest.mark.parametrize("options", [None, {}, {"optimization": {"backend_options": {"maxiter": 10}}}])
def test_laplace_dispatch_once_without_repacking_or_materializing_options(monkeypatch, options):
    sentinel = object()
    calls = []
    def chosen(**kwargs):
        calls.append(kwargs)
        return sentinel
    monkeypatch.setattr(laplace, "integrate", chosen)
    assert integrate(conditional_log_likelihood=None, random_effect_distribution=None,
                     integration_method="laplace", numerical_options=options) is sentinel
    assert len(calls) == 1 and calls[0]["integration_method"] == "laplace"
    assert calls[0]["numerical_options"] is options


@pytest.mark.parametrize("entry", ["validator", "inputs", "dispatch", "direct_laplace", "posterior"])
@pytest.mark.parametrize("options", [None, {"ghq": {"variant": "prior_centered", "orders": [3, 5, 7],
                                          "log_integral_atol": 1e-6, "posterior_atol": 1e-6, "max_nodes": 100}}])
def test_removed_method_rejected_before_any_model_or_backend(monkeypatch, entry, options):
    def forbidden(*args, **kwargs):
        pytest.fail("removed method executed a model, fallback or external process")
    args = analytic_inputs()
    args.update(integration_method="ghq", numerical_options=options)
    args["model"]["predict"] = forbidden
    args["model"]["parameter_model"]["relations"] = forbidden
    monkeypatch.setattr(computation, "integrate", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    direct = laplace.integrate
    monkeypatch.setattr(laplace, "integrate", forbidden)
    with pytest.raises(InputError, match="GHQ has been removed.*external R aghq.*not a package backend.*no fallback"):
        if entry == "validator":
            validate_integration_method("ghq")
        elif entry == "inputs":
            validate_inputs(**args)
        elif entry in ("dispatch", "direct_laplace"):
            (integrate if entry == "dispatch" else direct)(conditional_log_likelihood=forbidden,
                random_effect_distribution=None, integration_method="ghq", numerical_options=options)
        else:
            pmx.compute_posterior(**args)


@pytest.mark.parametrize("method", ["aghq", "r_aghq", "GHQ", "auto"])
def test_external_workflow_is_not_an_alias_or_backend(method):
    with pytest.raises(InputError, match="external R aghq is not a package backend"):
        validate_integration_method(method)


@pytest.mark.parametrize("options", [{"ghq": {}}, {"ghq": {}, "optimization": {}}, {"aghq": {}}])
def test_removed_options_rejected_without_mutation_or_execution(monkeypatch, options):
    before = deepcopy(options)
    with pytest.raises(InputError, match="unsupported"):
        numerical_options.validate_numerical_options(options)
    args = analytic_inputs()
    args["numerical_options"] = options
    monkeypatch.setattr(computation, "integrate", lambda **kw: pytest.fail("invalid options reached integration"))
    with pytest.raises(InputError, match="unsupported"):
        pmx.compute_posterior(**args)
    assert options == before


def test_laplace_default_options_stay_exact():
    assert numerical_options.validate_numerical_options(None) == {
        "optimization": {"method": "BFGS", "stationarity_tolerance": 1e-5,
                         "backend_options": {"maxiter": 1000, "gtol": 1e-6, "xrtol": 0., "c1": 1e-4, "c2": .9}},
        "derivatives": {"backend": "scipy", "atol": 1e-6, "rtol": 1e-6,
                        "backend_options": {"initial_step": .5, "step_factor": 2., "maxiter": 10, "order": 8}},
    }


def test_posterior_method_details_are_optional_copied_and_not_a_new_status():
    result = dict(status="success", integration_method="laplace", failure_reason=None,
                  state_results=[dict(state_id="a", prior_probability=1., posterior_probability=1.,
                                      status="success", log_marginal_likelihood=0., failure_reason=None)],
                  method_details={"audit_checks": [1, 2, 3]})
    args = dict(candidate_states=[dict(state_id="a", events=[])], priors={"a": 1.})
    checked = validate_posterior_result(result, **args)
    checked["method_details"]["audit_checks"].append(4)
    assert result["method_details"]["audit_checks"] == [1, 2, 3]
    invalid = deepcopy(result)
    invalid["method_details"] = []
    with pytest.raises(InputError):
        validate_posterior_result(invalid, **args)
    result["integration_method"] = "ghq"
    with pytest.raises(InputError, match="removed"):
        validate_posterior_result(result, **args)


@pytest.mark.parametrize("arguments", [
    ("--method", "ghq"), ("--method", "aghq"), ("--ghq-orders", "3", "5", "7"),
    ("--log-integral-atol", "1e-6"), ("--posterior-atol", "1e-6"), ("--max-nodes", "100"),
])
def test_reference_cli_rejects_removed_switches(arguments):
    run = cli("OXC", *arguments)
    assert run.returncode == 2 and "usage:" in run.stderr and run.stdout == ""


def test_reference_cli_keeps_compact_laplace_serialization():
    run = cli("OXC", "--method", "laplace")
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "success" and result["integration_method"] == "laplace"
    assert set(result) == {"model", "status", "integration_method", "state_results", "failure_reason"}
    assert len(result["state_results"]) == 4
    for row in result["state_results"]:
        assert set(row) == {"state_id", "prior_probability", "posterior_probability", "status",
                            "log_marginal_likelihood", "failure_reason", "events"}


def test_laplace_failed_state_never_invokes_external_fallback(monkeypatch):
    args = analytic_inputs()
    outputs = iter([{"status": "failed", "log_marginal_likelihood": None, "failure_reason": "curvature failed"},
                    {"status": "success", "log_marginal_likelihood": 0., "failure_reason": None}])
    monkeypatch.setattr(computation, "integrate", lambda **kwargs: next(outputs))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("external fallback"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("external fallback"))
    result = pmx.compute_posterior(**args)
    assert result["status"] == "failed"
    assert [r["state_id"] for r in result["state_results"]] == ["a", "b"]
    assert result["state_results"][1]["log_marginal_likelihood"] == 0.
    assert all(row["posterior_probability"] is None for row in result["state_results"])
