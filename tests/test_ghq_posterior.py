import math

import numpy as np
import pytest

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.posterior import computation
from test_ghq_options import ghq_options
from test_posterior_computation import analytic_inputs, analytic_logs


def arguments():
    args = analytic_inputs()
    args["integration_method"] = "ghq"
    args["numerical_options"] = ghq_options((64, 96, 128))
    return args


def test_complete_analytic_posterior_matches_closed_form_and_laplace():
    args = arguments()
    result = compute_posterior(**args)
    assert result["status"] == "success", result
    expected = analytic_logs()
    logs = [row["log_marginal_likelihood"] for row in result["state_results"]]
    np.testing.assert_allclose(logs, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose([row["posterior_probability"] for row in result["state_results"]],
                               [.4756947064736979, .5243052935263021], atol=1e-10, rtol=0)
    details = result["method_details"]
    assert details["posterior_stability_passed"] is True
    assert [r["order"] for r in details["posterior_order_results"]] == [64, 96, 128]
    assert all(row["method_details"]["dimension"] == 1 for row in result["state_results"])


def test_changed_prior_changes_posterior_not_integrals_or_nodes():
    args = arguments()
    first = compute_posterior(**args)
    keys = list(args["priors"])
    args["priors"] = {keys[0]: .8, keys[1]: .2}
    second = compute_posterior(**args)
    assert first["status"] == second["status"] == "success"
    for a, b in zip(first["state_results"], second["state_results"]):
        assert a["log_marginal_likelihood"] == b["log_marginal_likelihood"]
        assert a["method_details"] == b["method_details"]
        assert a["posterior_probability"] != b["posterior_probability"]


def test_zero_prior_candidate_never_calls_integrator(monkeypatch):
    args = arguments()
    keys = list(args["priors"])
    args["priors"] = {keys[0]: 0., keys[1]: 1.}
    original = computation.integrate
    calls = []
    def counted(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(computation, "integrate", counted)
    result = compute_posterior(**args)
    assert result["status"] == "success", result
    assert len(calls) == 1
    assert result["state_results"][0]["status"] == "skipped_zero_prior"
    assert result["state_results"][0]["posterior_probability"] == 0
    assert result["state_results"][1]["posterior_probability"] == 1


def mocked_result(logs, orders=(64, 96, 128)):
    return dict(status="success", log_marginal_likelihood=logs[-1], failure_reason=None,
                method_details={"scope": "prior_centered_tensor", "order_results": [
                    dict(order=o, log_marginal_likelihood=v, estimated_log_integral_roundoff=0.) for o, v in zip(orders, logs)]})


def test_posterior_gate_is_independent_of_log_integral_gate(monkeypatch):
    outputs = iter([mocked_result([0., 0., 0.]), mocked_result([0., .0006, .0012])])
    monkeypatch.setattr(computation, "integrate", lambda **kwargs: next(outputs))
    args = arguments()
    args["numerical_options"]["ghq"]["log_integral_atol"] = .001
    result = compute_posterior(**args)
    assert result["status"] == "failed"
    assert "posterior_atol" in result["failure_reason"]
    assert result["method_details"]["posterior_stability_passed"] is False
    assert all(r["status"] == "success" and r["posterior_probability"] is None for r in result["state_results"])


def test_unmatched_orders_fail_instead_of_mixing_integrals(monkeypatch):
    outputs = iter([mocked_result([0., 0., 0.]), mocked_result([0., 0., 0.], (32, 64, 128))])
    monkeypatch.setattr(computation, "integrate", lambda **kwargs: next(outputs))
    result = compute_posterior(**arguments())
    assert result["status"] == "failed"
    assert "matching order" in result["failure_reason"]


def test_one_failed_state_blocks_normalization_but_preserves_other_integral(monkeypatch):
    outputs = iter([dict(status="failed", log_marginal_likelihood=None, failure_reason="bad GHQ node"),
                    mocked_result([0., 0., 0.])])
    monkeypatch.setattr(computation, "integrate", lambda **kwargs: next(outputs))
    result = compute_posterior(**arguments())
    assert result["status"] == "failed"
    assert result["method_details"]["posterior_stability_passed"] is None
    assert result["state_results"][1]["log_marginal_likelihood"] == 0.
    assert all(r["posterior_probability"] is None for r in result["state_results"])


def test_posterior_underflow_is_still_failure(monkeypatch):
    outputs = iter([mocked_result([0., 0., 0.]), mocked_result([-1000., -1000., -1000.])])
    monkeypatch.setattr(computation, "integrate", lambda **kwargs: next(outputs))
    result = compute_posterior(**arguments())
    assert result["status"] == "failed"
    assert "underflow" in result["failure_reason"]
    assert result["state_results"][1]["log_marginal_likelihood"] == -1000.


def test_actual_unstable_integrals_block_even_single_positive_prior():
    args = arguments()
    keys = list(args["priors"])
    args["priors"] = {keys[0]: 1., keys[1]: 0.}
    args["numerical_options"] = ghq_options((1, 2, 3))
    result = compute_posterior(**args)
    assert result["status"] == "failed"
    assert result["state_results"][0]["status"] == "failed"
    assert result["state_results"][1]["posterior_probability"] == 0.


def test_actual_zero_dimensional_posterior_has_no_fabricated_stability():
    args = arguments()
    args["model"]["parameters"]["values"]["omega"] = 0.
    result = compute_posterior(**args)
    assert result["status"] == "success", result
    assert result["method_details"]["posterior_stability_passed"] is None
    assert result["method_details"]["posterior_order_results"] == []


@pytest.mark.parametrize("all_zero", [False, True])
def test_zero_dimensional_true_zero_evidence_is_distinguished_from_node_failure(monkeypatch, all_zero):
    args = arguments()
    args["model"]["parameters"]["values"]["omega"] = 0.
    def bind(**kwargs):
        zero = all_zero or kwargs["candidate_state"]["state_id"] == "a"
        return lambda x: -math.inf if zero else 0.
    monkeypatch.setattr(computation, "_bind_likelihood", bind)
    result = compute_posterior(**args)
    assert all(row["status"] == "success" for row in result["state_results"])
    if all_zero:
        assert result["status"] == "failed"
        assert all(row["posterior_probability"] is None for row in result["state_results"])
    else:
        assert result["status"] == "success"
        assert [r["posterior_probability"] for r in result["state_results"]] == [0., 1.]
