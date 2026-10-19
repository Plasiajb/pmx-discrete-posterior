from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.posterior import computation
from test_input_candidates import event


def analytic_inputs():
    # Test-only smooth positive predictor: at y=0 its log likelihood is quadratic in eta.
    def relations(*, parameters, iiv_values, evaluation_interval, **kwargs):
        values = parameters["values"]
        return {"units": {"effect": "mg/L", "b": "mg/L", "V": "L", "time_scale": "h"},
                "segments": [{"start": evaluation_interval[0], "end": evaluation_interval[1], "occasion_id": None,
                              "values": {"effect": values["scale"] * iiv_values["u"], "b": values["b"],
                                         "V": values["V"], "time_scale": values["time_scale"]}}]}

    def predict(*, individual_parameters, candidate_state, prediction_records, individual_data):
        p = individual_parameters["segments"][0]["values"]
        d = candidate_state["events"][0]["amount"] / p["V"]
        return {"record_ids": [r["record_id"] for r in prediction_records], "units": ["mg/L"] * len(prediction_records),
                "values": [math.hypot(p["b"], d + r["time"] / p["time_scale"] * p["effect"]) for r in prediction_records]}

    return {
        "model": {"parameters": {"values": {"omega": 1, "sigma": 1, "scale": 1, "b": 1, "V": 1, "time_scale": 1},
                                 "units": {"omega": "1", "sigma": "mg/L", "scale": "mg/L", "b": "mg/L", "V": "L", "time_scale": "h"}},
                  "parameter_model": {"covariate_names": [], "relations": relations}, "predict": predict,
                  "iiv": {"names": ["u"], "distribution": "normal", "covariance_parameters": [["omega"]]}, "iov": None,
                  "random_effects_dependence": "independent_blocks",
                  "residual_model": {"distribution": "normal", "error_type": "additive", "parameter_names": {"sd": "sigma"},
                                     "dependence": "conditionally_independent"}},
        "individual_data": {"covariates": {}, "occasions": [], "dosing_prehistory": {"events": []},
                            "initialization": {"mode": "zero", "start_time": 0, "event_phase": "before_events"}},
        "observations": [{"record_id": f"r{t}", "time": t, "value": 0, "time_unit": "h", "value_unit": "mg/L"} for t in (1, 2)],
        "candidate_states": [{"state_id": sid, "events": [event(time=0, amount=d)]} for sid, d in (("a", 1), ("b", 2))],
        "priors": {"a": .3, "b": .7}, "integration_method": "laplace",
    }


def row_map(result):
    return {row["state_id"]: row for row in result["state_results"]}


def analytic_logs(variance=1):
    # Complete square in -(eta^2/variance + (d+eta)^2 + (d+2*eta)^2)/2.
    if variance == 0:
        return np.array([-1 - math.log(2 * math.pi) - d*d for d in (1, 2)])
    h = 1 / variance + 5
    return np.array([-1 - math.log(2 * math.pi) - .5*math.log(variance*h) - d*d + 4.5*d*d/h for d in (1, 2)])


def test_analytic_joint_history_through_real_model_and_integrator():
    args = analytic_inputs()
    before = deepcopy(args)
    result = computation.compute_posterior(**args)
    assert result["status"] == "success", result
    rows = result["state_results"]
    logs = analytic_logs()
    weights = logs + np.log([.3, .7])
    expected = np.exp(weights - logsumexp(weights))
    np.testing.assert_allclose([r["log_marginal_likelihood"] for r in rows], logs, atol=1e-7, rtol=0)
    np.testing.assert_allclose([r["posterior_probability"] for r in rows], expected, atol=5e-8, rtol=0)
    for row, mode in zip(rows, (-.5, -1)):
        np.testing.assert_allclose(row["method_details"]["conditional_mode"], [mode], atol=1e-6, rtol=0)
    # Integrating each record with an independent eta incorrectly reverses the ranking.
    wrong_logs = np.array([-1 - math.log(2*math.pi) - .5*math.log(10) - .35*d*d for d in (1, 2)])
    wrong_weights = wrong_logs + np.log([.3, .7])
    wrong = np.exp(wrong_weights - logsumexp(wrong_weights))
    assert expected[0] < expected[1] and wrong[0] > wrong[1]
    assert args == before


@pytest.mark.parametrize("variance", [0, .3])
def test_zero_dimension_and_nonunit_variance_keep_normalization_constants(variance):
    args = analytic_inputs()
    args["model"]["parameters"]["values"]["omega"] = variance
    result = computation.compute_posterior(**args)
    assert result["status"] == "success", result
    np.testing.assert_allclose([r["log_marginal_likelihood"] for r in result["state_results"]], analytic_logs(variance), atol=1e-7, rtol=0)
    assert all(r["method_details"]["dimension"] == (0 if variance == 0 else 1) for r in result["state_results"])


def test_prior_changes_do_not_enter_integrals_and_order_is_preserved():
    args = analytic_inputs()
    first = row_map(computation.compute_posterior(**args))
    args["priors"] = {"b": .2, "a": .8}
    args["candidate_states"].reverse()
    args["observations"].reverse()
    result = computation.compute_posterior(**args)
    assert result["status"] == "success", result
    assert [row["state_id"] for row in result["state_results"]] == ["b", "a"]
    second = row_map(result)
    for sid in first:
        assert first[sid]["log_marginal_likelihood"] == second[sid]["log_marginal_likelihood"]
        assert first[sid]["method_details"]["conditional_mode"] == second[sid]["method_details"]["conditional_mode"]
    odds = .8/.2 * math.exp(analytic_logs()[0] - analytic_logs()[1])
    assert second["a"]["posterior_probability"] / second["b"]["posterior_probability"] == pytest.approx(odds, abs=1e-7)


def test_zero_prior_skips_model_and_integrator_without_removing_candidate(monkeypatch):
    args = analytic_inputs()
    args["priors"] = {"a": 0, "b": 1}
    original_predict, original_integrate = args["model"]["predict"], computation.integrate
    calls = []
    def predict(**kwargs):
        assert kwargs["candidate_state"]["state_id"] == "b"
        return original_predict(**kwargs)
    def integrate(**kwargs):
        calls.append(1)
        return original_integrate(**kwargs)
    args["model"]["predict"] = predict
    monkeypatch.setattr(computation, "integrate", integrate)
    result = computation.compute_posterior(**args)
    assert result["status"] == "success", result
    assert calls == [1]
    rows = row_map(result)
    assert rows["a"] == {"state_id": "a", "prior_probability": 0., "posterior_probability": 0.,
                         "status": "skipped_zero_prior", "log_marginal_likelihood": None, "failure_reason": None}
    assert rows["b"]["posterior_probability"] == 1


@pytest.mark.parametrize("prior", [.3, np.nextafter(0., 1.)])
def test_positive_candidate_model_failure_retains_successful_later_integrals(prior):
    args = analytic_inputs()
    args["priors"] = {"a": prior, "b": 1-prior}
    original = args["model"]["predict"]
    def predict(**kwargs):
        if kwargs["candidate_state"]["state_id"] == "a":
            raise RuntimeError("deliberate candidate failure")
        return original(**kwargs)
    args["model"]["predict"] = predict
    result = computation.compute_posterior(**args)
    assert result["status"] == "failed"
    a, b = result["state_results"]
    assert a["status"] == "failed" and a["log_marginal_likelihood"] is None
    assert "deliberate candidate failure" in a["failure_reason"]
    assert b["status"] == "success" and math.isfinite(b["log_marginal_likelihood"])
    assert a["posterior_probability"] is b["posterior_probability"] is None


@pytest.mark.parametrize("problem", ["effects", "residual", "ghq", "dimension", "hessian", "invalid_zero_prior_state"])
def test_invalid_shared_inputs_fail_before_any_integration(monkeypatch, problem):
    args = analytic_inputs()
    def forbidden(**kwargs):
        pytest.fail("invalid input reached integration")
    monkeypatch.setattr(computation, "integrate", forbidden)
    if problem == "effects":
        del args["model"]["random_effects_dependence"]
    elif problem == "residual":
        del args["model"]["residual_model"]["dependence"]
    elif problem == "ghq":
        args["integration_method"] = "ghq"
    elif problem == "dimension":
        args["numerical_options"] = {"optimization": {"initial_point": [0, 0]}}
    elif problem == "hessian":
        args["model"]["parameters"]["values"]["omega"] = 0
        args["numerical_options"] = {"derivatives": {"rtol": 0}}
    else:
        args["priors"] = {"a": 0, "b": 1}
        args["candidate_states"][0]["events"][0]["amount"] = -1
    with pytest.raises(InputError):
        computation.compute_posterior(**args)


@pytest.mark.parametrize("logs,expected_status", [([-math.inf, 0], "success"), ([-math.inf, -math.inf], "failed"),
                                                 ([0, -1000], "failed"), ([1e308, -1e308], "failed")])
def test_true_zero_undefined_evidence_and_underflow_keep_integral_values(monkeypatch, logs, expected_status):
    args = analytic_inputs()
    values = iter(logs)
    def integrate(**kwargs):
        return {"status": "success", "log_marginal_likelihood": next(values), "failure_reason": None}
    monkeypatch.setattr(computation, "integrate", integrate)
    result = computation.compute_posterior(**args)
    assert result["status"] == expected_status
    assert [r["log_marginal_likelihood"] for r in result["state_results"]] == logs
    if expected_status == "failed":
        assert all(r["posterior_probability"] is None for r in result["state_results"])
        assert "normalization" in result["failure_reason"]
    else:
        assert [r["posterior_probability"] for r in result["state_results"]] == [0, 1]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, "raise", "malformed"])
def test_bad_integrator_output_is_failure_not_zero_and_later_states_still_run(monkeypatch, bad):
    args = analytic_inputs()
    calls = []
    def integrate(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            if bad == "raise":
                raise RuntimeError("integrator failure")
            if bad == "malformed":
                return {}
            return {"status": "success", "log_marginal_likelihood": bad, "failure_reason": None}
        return {"status": "success", "log_marginal_likelihood": -2., "failure_reason": None}
    monkeypatch.setattr(computation, "integrate", integrate)
    result = computation.compute_posterior(**args)
    assert result["status"] == "failed" and len(calls) == 2
    a, b = result["state_results"]
    assert a["log_marginal_likelihood"] is None and a["failure_reason"]
    assert b["log_marginal_likelihood"] == -2 and b["status"] == "success"
    assert a["posterior_probability"] is b["posterior_probability"] is None


def test_zero_prior_stays_zero_when_all_positive_prior_evidence_is_zero(monkeypatch):
    args = analytic_inputs()
    args["priors"] = {"a": 0, "b": 1}
    monkeypatch.setattr(computation, "integrate", lambda **kwargs: {"status": "success", "log_marginal_likelihood": -math.inf, "failure_reason": None})
    result = computation.compute_posterior(**args)
    assert result["status"] == "failed"
    assert result["state_results"][0]["posterior_probability"] == 0
    assert result["state_results"][1]["posterior_probability"] is None


def test_mixed_skip_failure_success_preserve_diagnostics_without_aliasing(monkeypatch):
    args = analytic_inputs()
    args["candidate_states"].append({"state_id": "c", "events": [event(time=0, amount=3)]})
    args["priors"] = {"a": 0, "b": .4, "c": .6}
    outputs = [
        {"status": "failed", "log_marginal_likelihood": None, "failure_reason": "curvature unresolved",
         "method_details": {"stage": "curvature", "attempts": [1]}},
        {"status": "success", "log_marginal_likelihood": -3., "failure_reason": None,
         "method_details": {"stage": "complete", "mode": [0.1]}},
    ]
    iterator = iter(outputs)
    monkeypatch.setattr(computation, "integrate", lambda **kwargs: next(iterator))
    result = computation.compute_posterior(**args)
    assert result["status"] == "failed"
    assert [r["posterior_probability"] for r in result["state_results"]] == [0, None, None]
    for row, expected in zip(result["state_results"][1:], outputs):
        assert all(row[key] == value for key, value in expected.items())
    result["state_results"][1]["method_details"]["attempts"].append(2)
    assert outputs[0]["method_details"]["attempts"] == [1]
