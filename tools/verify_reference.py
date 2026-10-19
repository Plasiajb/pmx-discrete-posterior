"""Run the real reference CLI and verify all four state identities and values.

The constants are comparison oracles, never substitute computation results.
Independent Laplace references are not exact nonlinear integrals.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]
LOG_ATOL = {"OXC": 1e-6, "TAC": 1e-3}
PROBABILITY_ATOL = 5e-6
REFERENCE_HASHES = {
    "OXC": "67d268cbc4e7268474753403ea195ec52a5ce95986e40772e2e1a8ab03db3d5a",
    "TAC": "3aa3ae1408c8c8707e3644f1924217350847b3a3555d9dcec90b89a574ab14e6",
}
# Independently solved Laplace references retained from the accepted baseline.
REFERENCE_LOGS = {
    "OXC": [-6.5855336131149524, -3.6440125863671535, -2.1658162679584247, -2.5637698888076980],
    "TAC": [-3.0602575684754125, -2.8464740513320637, -2.7852536319272970, -2.8186652168365800],
}
REFERENCE_PROBABILITIES = {
    "OXC": [.006296560090573311, .11928606261935508, .5230728861787735, .35134449111129823],
    "TAC": [.20711804665707417, .2564854717642472, .27267835198432405, .2637181295943546],
}
LABELS = ("omega_00", "omega_01", "omega_10", "omega_11")
BITS = ("00", "10", "01", "11")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, label):
    check(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label}: not a number")
    check(math.isfinite(value), f"{label}: nonfinite number")
    return value


def load_states(name, root=ROOT):
    payload = (root / "tests" / "fixtures" / f"{name}_C003.json").read_bytes()
    check(hashlib.sha256(payload).hexdigest() == REFERENCE_HASHES[name], f"{name}: frozen fixture changed")
    raw = json.loads(payload)
    states = raw["candidate_states" if name == "OXC" else "states"]
    check(len(states) == 4, f"{name}: expected all four reference states")
    for state, label, bits in zip(states, LABELS, BITS, strict=True):
        check(state["manuscript_id"] == label, f"{name}: manuscript label mismatch")
        check(state["machine_bitstring"] == bits, f"{name}: chronological bit label mismatch")
        check(label == "omega_" + bits[::-1], f"{name}: manuscript/chronological label mismatch")
    return states


def verify_result(name, result, root=ROOT):
    states = load_states(name, root)
    check(result.get("model") == name, "Wrong reference model")
    check(result.get("status") == "success", "Reference computation did not succeed")
    check(result.get("integration_method") == "laplace", "Expected actual Laplace computation")
    check(result.get("failure_reason") is None, "Successful result has a failure reason")
    rows = result.get("state_results")
    check(isinstance(rows, list) and len(rows) == 4, "Expected all four result rows")
    errors = []
    probabilities = []
    for index, (state, row) in enumerate(zip(states, rows, strict=True)):
        label = LABELS[index]
        check(row.get("state_id") == state["state_event_id"], f"{label}: state identity/order mismatch")
        old_events = state["dose_events" if name == "OXC" else "events"]
        events = [
            {"time": event["time_h"], "amount": event["amount_mg"], "amount_unit": "mg",
             "route": "oral", "compartment": "depot"}
            for event in old_events
        ]
        check(row.get("events") == events, f"{label}: event mapping mismatch")
        check(row.get("status") == "success" and row.get("failure_reason") is None,
              f"{label}: state failed")
        check(row.get("prior_probability") == .25, f"{label}: changed reference prior")
        log_value = finite_number(row.get("log_marginal_likelihood"), f"{label} log integral")
        probability = finite_number(row.get("posterior_probability"), f"{label} probability")
        check(0 <= probability <= 1, f"{label}: invalid probability")
        log_error = abs(log_value - REFERENCE_LOGS[name][index])
        probability_error = abs(probability - REFERENCE_PROBABILITIES[name][index])
        check(log_error <= LOG_ATOL[name], f"{label}: log absolute error {log_error} > {LOG_ATOL[name]}")
        check(probability_error <= PROBABILITY_ATOL,
              f"{label}: probability absolute error {probability_error} > {PROBABILITY_ATOL}")
        probabilities.append(probability)
        errors.append({"manuscript_id": label, "machine_bitstring": BITS[index],
                       "state_id": row["state_id"], "log_absolute_error": log_error,
                       "probability_absolute_error": probability_error})
    check(abs(math.fsum(probabilities) - 1) <= 8 * math.ulp(1.0), "Probabilities do not sum to one")
    return {"model": name, "status": "verified", "log_atol": LOG_ATOL[name],
            "probability_atol": PROBABILITY_ATOL, "rtol": 0, "states": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("OXC", "TAC"))
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()
    if args.require_installed:
        import pmx_discrete_posterior

        imported = Path(pmx_discrete_posterior.__file__).resolve()
        check(imported.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()),
              f"Source checkout masks installed package: {imported}")
        check(version("pmx-discrete-posterior") == "0.2.0", "Wrong installed distribution version")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    run = subprocess.run(
        [sys.executable, "-B", str(ROOT / "examples" / "posterior_reference.py"), args.model],
        cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=1800,
    )
    check(run.returncode == 0, f"Reference CLI exited {run.returncode}: {run.stderr}\n{run.stdout}")
    result = json.loads(run.stdout)
    verification = verify_result(args.model, result)
    print(json.dumps({"verification": verification, "computed_result": result}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
