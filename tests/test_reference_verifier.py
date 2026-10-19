"""Verifier unit tests use synthetic rows; integration CI runs the real CLI."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("portable_reference_verifier", ROOT / "tools/verify_reference.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def reference_result(name):
    states = verifier.load_states(name)
    rows = []
    for index, state in enumerate(states):
        events = state["dose_events" if name == "OXC" else "events"]
        rows.append({
            "state_id": state["state_event_id"], "status": "success", "failure_reason": None,
            "prior_probability": .25,
            "log_marginal_likelihood": verifier.REFERENCE_LOGS[name][index],
            "posterior_probability": verifier.REFERENCE_PROBABILITIES[name][index],
            "events": [{"time": event["time_h"], "amount": event["amount_mg"], "amount_unit": "mg",
                        "route": "oral", "compartment": "depot"} for event in events],
        })
    return {"model": name, "status": "success", "integration_method": "laplace", "state_results": rows}


@pytest.mark.parametrize("name", ["OXC", "TAC"])
def test_verifier_checks_all_labels_with_absolute_thresholds(name):
    report = verifier.verify_result(name, reference_result(name))
    assert [row["manuscript_id"] for row in report["states"]] == list(verifier.LABELS)
    assert [row["machine_bitstring"] for row in report["states"]] == ["00", "10", "01", "11"]
    assert report["log_atol"] == (1e-6 if name == "OXC" else 1e-3)
    assert report["probability_atol"] == 5e-6 and report["rtol"] == 0


@pytest.mark.parametrize("name", ["OXC", "TAC"])
@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("field", ["log_marginal_likelihood", "posterior_probability"])
def test_verifier_rejects_out_of_threshold_value_in_every_state(name, index, field):
    result = reference_result(name)
    threshold = verifier.LOG_ATOL[name] if field == "log_marginal_likelihood" else 5e-6
    result["state_results"][index][field] += 1.01 * threshold
    with pytest.raises(ValueError, match="absolute error"):
        verifier.verify_result(name, result)


@pytest.mark.parametrize("name", ["OXC", "TAC"])
@pytest.mark.parametrize("mutation", ["reordered", "missing", "duplicate", "events", "failed", "method", "prior"])
def test_verifier_rejects_identity_mapping_and_failed_computation(name, mutation):
    result = reference_result(name)
    rows = result["state_results"]
    if mutation == "reordered":
        rows.reverse()
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "events":
        rows[1]["events"] = []
    elif mutation == "failed":
        result["status"] = "failed"
    elif mutation == "method":
        result["integration_method"] = "ghq"
    else:
        rows[0]["prior_probability"] = .5
    with pytest.raises(ValueError):
        verifier.verify_result(name, result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True, "0.1"])
def test_verifier_rejects_nonfinite_or_non_numeric_probability(value):
    result = reference_result("OXC")
    result["state_results"][0]["posterior_probability"] = value
    with pytest.raises(ValueError):
        verifier.verify_result("OXC", result)
