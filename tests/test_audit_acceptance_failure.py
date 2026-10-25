"""Retain shared NC-03 failure/nonrenormalization checks on native Laplace."""

import math

import numpy as np
import pytest

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.posterior import computation
from test_posterior_computation import analytic_inputs, analytic_logs


def test_nc03_real_full_posterior_positive_control():
    result = compute_posterior(**analytic_inputs())
    assert result["status"] == "success", result
    np.testing.assert_allclose([r["log_marginal_likelihood"] for r in result["state_results"]],
                               analytic_logs(), atol=1e-7, rtol=0)
    np.testing.assert_allclose([r["posterior_probability"] for r in result["state_results"]],
                               [.4756947064736979, .5243052935263021], atol=5e-8, rtol=0)


@pytest.mark.parametrize("scope", ["initial_point", "one_state", "all_states"])
@pytest.mark.parametrize("failure", ["nan", "positive_infinity", "exception"])
def test_nc03_prediction_failure_never_becomes_a_normalized_zero(scope, failure):
    args = analytic_inputs()
    original = args["model"]["predict"]
    visits = []
    def failing_predict(**kwargs):
        state_id = kwargs["candidate_state"]["state_id"]
        effect = kwargs["individual_parameters"]["segments"][0]["values"]["effect"]
        broken = scope == "all_states" or (state_id == "a" and (scope == "one_state" or abs(effect) < 1e-14))
        visits.append((state_id, broken))
        if broken and failure == "exception":
            raise RuntimeError("NC03 deliberate prediction failure")
        predictions = original(**kwargs)
        if broken:
            predictions["values"][0] = math.nan if failure == "nan" else math.inf
        return predictions
    args["model"]["predict"] = failing_predict
    result = compute_posterior(**args)
    assert result["status"] == "failed", result
    assert "not normalized" in result["failure_reason"]
    assert [row["state_id"] for row in result["state_results"]] == ["a", "b"]
    assert all(row["posterior_probability"] is None for row in result["state_results"])
    a, b = result["state_results"]
    assert a["status"] == "failed" and a["log_marginal_likelihood"] is None and a["failure_reason"]
    if scope == "all_states":
        assert b["status"] == "failed" and b["log_marginal_likelihood"] is None
    else:
        assert b["status"] == "success"
        assert b["log_marginal_likelihood"] == pytest.approx(analytic_logs()[1], abs=1e-7)
    assert any(broken for _, broken in visits)


@pytest.mark.parametrize("all_zero", [False, True])
def test_zero_dimensional_true_zero_is_distinct_from_failure(monkeypatch, all_zero):
    args = analytic_inputs()
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
