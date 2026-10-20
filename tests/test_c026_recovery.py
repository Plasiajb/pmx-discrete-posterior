"""Full c026 recovery against independent local Laplace, not exact integrals.

References: independent Bateman/normal objective differentiated at 50 and 80
decimal digits. These values are assertions only, never optimizer starts or
curvature inputs.
"""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.interfaces.inputs import validate_inputs
from pmx_discrete_posterior.interfaces.results import validate_integration_result, validate_posterior_result
from pmx_discrete_posterior.models.prediction import prepare_prediction_records
from pmx_discrete_posterior.models.random_effects import resolve_random_effects
from pmx_discrete_posterior.numerics import integrate
from pmx_discrete_posterior.posterior.computation import _bind_likelihood


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_INPUTS = ROOT / "tests/fixtures/c026_scientific_inputs.json"
SPEC = importlib.util.spec_from_file_location("c026_reference_models", ROOT / "examples/reference_models.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def load_c026_inputs():
    """Build c026 from the portable OXC reference without numerical overrides."""
    inputs = reference.load_reference("OXC")
    model = inputs["model"]
    # The frozen c026 case differs from OXC_C003 only in its residual SD/variance.
    model["parameters"]["values"]["sigma"] = 1e-6
    model["random_effects_dependence"] = "independent_blocks"
    model["residual_model"]["dependence"] = "conditionally_independent"
    inputs["priors"] = {state["state_id"]: .25 for state in inputs["candidate_states"]}
    return inputs


def scientific_snapshot(inputs):
    """Snapshot all scientific inputs, with portable identifiers for callbacks."""
    mapped = deepcopy({key: value for key, value in inputs.items() if key != "model"})
    mapped["model"] = deepcopy({
        key: value for key, value in inputs["model"].items()
        if key not in ("predict", "parameter_model")
    })
    mapped["model"]["parameter_model"] = {
        "covariate_names": list(inputs["model"]["parameter_model"]["covariate_names"]),
        "relations_source": "examples/reference_models.py",
    }
    mapped["model"]["predict_source"] = "examples/reference_models.py"
    return mapped


# Chronological event order: none, 0h, 12h, both.
INDEPENDENT_LOGS = [-16.67946178899348, -6.192804874472976,
                    -1.5499239982109334, -2.541412623553703]
INDEPENDENT_POSTERIOR = [1.9464368664784453e-7, .006974893060111453,
                         .7242943510375628, .26873056125863914]


def assert_recovery_warning(recovery):
    warning = recovery["warning"]
    assert isinstance(warning, str) and warning.strip()
    for phrase in ("precision-limited gradient", "unchanged endpoint", "independent",
                   "curvature-relative stationarity", "local consistency checks",
                   "optimizer convergence and global optimality are not claimed"):
        assert phrase in warning.lower()


@pytest.fixture(scope="module")
def c026_run():
    inputs = load_c026_inputs()
    before = scientific_snapshot(inputs)
    # Exercise the public entry point with untouched default numerical options.
    result = compute_posterior(**inputs, integration_method="laplace")
    return inputs, before, result


def test_c026_science_and_default_numerical_contract_unchanged(c026_run):
    inputs, before, result = c026_run
    assert before == json.loads(SCIENTIFIC_INPUTS.read_text(encoding="utf-8"))
    assert scientific_snapshot(inputs) == before
    assert set(inputs) == {"model", "individual_data", "observations", "candidate_states", "priors"}
    assert [[e["time"] for e in s["events"]] for s in inputs["candidate_states"]] == [[], [0], [12], [0, 12]]
    values = inputs["model"]["parameters"]["values"]
    assert values["sigma"] == 1e-6 and values["omega_0_0"] == .0122457142458844
    validate_posterior_result(result, candidate_states=inputs["candidate_states"], priors=inputs["priors"])
    assert result["integration_method"] == "laplace"
    assert [r["state_id"] for r in result["state_results"]] == [s["state_id"] for s in inputs["candidate_states"]]
    for row in result["state_results"]:
        details = row["method_details"]
        assert row["prior_probability"] == .25
        assert details["dimension"] == 1 and details["coordinate_system"] == "prior_standardized"
        assert details["initial_point"] == details["standardized_initial_point"] == [0.]
        assert details["optimization"]["backend_options"]["gtol"] == 1e-6
        assert details["curvature_log_integral_atol"] == details["stationarity_log_integral_atol"] == 1e-3


def test_c026_full_posterior_recovers_independent_local_laplace(c026_run):
    inputs, _, result = c026_run
    validate_posterior_result(result, candidate_states=inputs["candidate_states"], priors=inputs["priors"])
    assert result["status"] == "success", result
    assert result["failure_reason"] is None
    rows = result["state_results"]
    assert len(rows) == 4 and all(r["status"] == "success" and r["failure_reason"] is None for r in rows)
    probabilities = [r["posterior_probability"] for r in rows]
    np.testing.assert_allclose([r["log_marginal_likelihood"] for r in rows], INDEPENDENT_LOGS, atol=1e-5, rtol=0)
    np.testing.assert_allclose(probabilities, INDEPENDENT_POSTERIOR, atol=5e-6, rtol=0)
    assert all(p > 0 for p in probabilities)
    assert math.fsum(probabilities) == pytest.approx(1., abs=8 * math.ulp(1.), rel=0)
    assert int(np.argmax(probabilities)) == 2
    for row in rows:
        details = row["method_details"]
        assert all(math.isfinite(row[key]) for key in
                   ("log_marginal_likelihood", "prior_probability", "posterior_probability"))
        for key in ("conditional_mode", "standardized_mode"):
            assert np.asarray(details[key]).shape == (1,)
            assert np.all(np.isfinite(details[key]))
        assert all(math.isfinite(details[key]) for key in
                   ("objective_value", "standardized_objective_value",
                    "log_determinant", "standardized_log_determinant"))
        assert 0 <= details["estimated_log_integral_error_from_hessian"] <= 1e-3
        assert 0 <= details["estimated_log_integral_error_from_stationarity"] <= 1e-3
        recovery = details["precision_recovery"]
        assert recovery["status"] == "accepted"
        assert recovery["optimization_status"] == "failed"
        assert isinstance(recovery["failure_reason"], str) and recovery["failure_reason"].strip()
        assert recovery["gradient_checks"] and recovery["local_probes"]
        assert_recovery_warning(recovery)
        failure = details["optimization"]["gradient_failure"]
        assert failure["phase"] in ("search", "stationarity")
        assert np.asarray(failure["point"]).shape == (1,) and np.all(np.isfinite(failure["point"]))
        assert failure["result"]["status"] == "failed"
        reason = failure["result"]["failure_reason"]
        assert isinstance(reason, str) and reason.strip()
        assert reason in recovery["failure_reason"]


def test_c026_model_failure_cannot_be_recovered_or_partially_normalized():
    inputs = load_c026_inputs()
    predict = inputs["model"]["predict"]
    failed_id = inputs["candidate_states"][0]["state_id"]

    def fail_one_candidate(**kwargs):
        if kwargs["candidate_state"]["state_id"] == failed_id:
            raise RuntimeError("deliberate c026 model execution failure")
        return predict(**kwargs)

    inputs["model"]["predict"] = fail_one_candidate
    result = compute_posterior(**inputs, integration_method="laplace")
    validate_posterior_result(result, candidate_states=inputs["candidate_states"], priors=inputs["priors"])
    assert result["status"] == "failed" and result["failure_reason"]
    rows = result["state_results"]
    assert [r["state_id"] for r in rows] == [s["state_id"] for s in inputs["candidate_states"]]
    assert all(r["posterior_probability"] is None for r in rows)
    assert rows[0]["status"] == "failed" and rows[0]["log_marginal_likelihood"] is None
    assert "deliberate c026 model execution failure" in rows[0]["failure_reason"]
    assert rows[0]["method_details"].get("precision_recovery", {}).get("status") != "accepted"
    assert all(r["status"] == "success" and math.isfinite(r["log_marginal_likelihood"]) for r in rows[1:])


@pytest.mark.parametrize("option_case,tolerance", [
    ("omitted", None), ("empty", None), ("gtol_only", None),
    ("explicit_default", 1e-5), ("explicit_tight", 1e-8), ("normalized_defaults", 1e-5),
])
def test_c026_tolerance_provenance_direct_matches_posterior(c026_run, option_case, tolerance):
    inputs, before, default_result = c026_run
    checked = validate_inputs(**inputs, integration_method="laplace", numerical_options=None)
    kwargs = {}
    if option_case == "normalized_defaults":
        # Passing normalized options back through a public API makes the limit explicit.
        kwargs["numerical_options"] = checked["numerical_options"]
    elif option_case != "omitted":
        options = {} if option_case == "empty" else {"optimization": {"backend_options": {"gtol": 1e-6}}}
        if tolerance is not None:
            options["optimization"]["stationarity_tolerance"] = tolerance
        kwargs["numerical_options"] = options
    original_kwargs = deepcopy(kwargs)
    supplied = kwargs.get("numerical_options", {}).get("optimization", {})
    assert ("stationarity_tolerance" in supplied) == (tolerance is not None)

    posterior = default_result if option_case == "omitted" else compute_posterior(
        **inputs, integration_method="laplace", **kwargs)
    validate_posterior_result(posterior, candidate_states=inputs["candidate_states"], priors=inputs["priors"])
    expected_status = "success" if tolerance is None else "failed"
    assert posterior["status"] == expected_status, posterior
    rows = posterior["state_results"]
    assert [r["state_id"] for r in rows] == [s["state_id"] for s in inputs["candidate_states"]]
    model, data = checked["model"], checked["individual_data"]
    effects = resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
                                     occasions=data["occasions"], dependence=model["random_effects_dependence"])
    request = prepare_prediction_records(checked["observations"], data)
    for state, row, expected_log in zip(checked["candidate_states"], rows, INDEPENDENT_LOGS):
        likelihood = _bind_likelihood(model=model, individual_data=data, observations=checked["observations"],
            candidate_state=state, random_effect_distribution=effects, prediction_request=request)
        direct = integrate(conditional_log_likelihood=likelihood, random_effect_distribution=effects,
                           integration_method="laplace", **kwargs)
        validate_integration_result(direct)
        assert direct["status"] == row["status"] == expected_status, (direct, row)
        assert direct["failure_reason"] == row["failure_reason"]
        for result in (direct, row):
            details = result["method_details"]
            assert details["optimization"]["backend_options"]["gtol"] == 1e-6
            recovery = details["precision_recovery"]
            assert recovery["optimization_status"] == "failed"
            assert details["optimization"]["gradient_failure"]["result"]["status"] == "failed"
            if tolerance is None:
                assert recovery["status"] == "accepted"
                assert_recovery_warning(recovery)
                assert result["log_marginal_likelihood"] == pytest.approx(expected_log, abs=1e-5, rel=0)
            else:
                assert result["log_marginal_likelihood"] is None
                assert "explicit stationarity_tolerance" in result["failure_reason"]
                assert details["optimization"]["stationarity_tolerance"] == tolerance
                assert recovery["status"] == "rejected"
                assert math.isfinite(recovery["gradient_inf_with_error"])
                assert recovery["gradient_inf_with_error"] > tolerance
        if tolerance is None:
            assert direct["log_marginal_likelihood"] == pytest.approx(row["log_marginal_likelihood"], abs=1e-12, rel=0)
    if tolerance is None:
        np.testing.assert_allclose([r["posterior_probability"] for r in rows], INDEPENDENT_POSTERIOR, atol=5e-6, rtol=0)
    else:
        assert all(r["posterior_probability"] is None for r in rows)
    assert kwargs == original_kwargs
    assert scientific_snapshot(inputs) == before
