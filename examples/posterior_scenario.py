"""Run one manuscript sensitivity case through the installed Laplace API."""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

import numpy as np

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.interfaces.results import validate_posterior_result
from pmx_discrete_posterior.interfaces.validation import InputError

if __package__:
    from .scenario_inputs import list_scenarios, load_scenario, read_json
else:
    from scenario_inputs import list_scenarios, load_scenario, read_json


def _failed_result(inputs, reason):
    rows = []
    for state in inputs["candidate_states"]:
        prior = inputs["priors"][state["state_id"]]
        rows.append({"state_id": state["state_id"], "prior_probability": prior,
                     "posterior_probability": None if prior else 0.0,
                     "log_marginal_likelihood": None,
                     "status": "failed" if prior else "skipped_zero_prior",
                     "failure_reason": reason if prior else None})
    return {"status": "failed", "integration_method": "laplace",
            "failure_reason": reason, "state_results": rows}


def run_scenario(scenario, *, numerical_options=None):
    """Dispatch once, preserving None/options provenance and every candidate.

    No default option expansion, cached posterior, retry or partial normalization
    is performed here. Invalid API results become a complete failed state vector.
    """
    inputs = scenario["inputs"]
    options = deepcopy(numerical_options)
    try:
        if options is not None and (not isinstance(options, dict) or "ghq" in options):
            raise InputError("this reader accepts only Laplace numerical_options (an object or null)")
        result = compute_posterior(**inputs, integration_method="laplace", numerical_options=options)
        result = validate_posterior_result(result, candidate_states=inputs["candidate_states"], priors=inputs["priors"])
        if result["integration_method"] != "laplace":
            raise InputError("Laplace dispatch returned another integration method")
    except Exception as exc:
        result = _failed_result(inputs, f"{type(exc).__name__}: {exc}")
    states = {state["state_id"]: state for state in inputs["candidate_states"]}
    return {"schema": "manuscript_scenario_laplace_v1", "identity": deepcopy(scenario["identity"]),
            "mapping_adjustments": deepcopy(scenario["mapping_adjustments"]),
            "numerical_options_requested": deepcopy(numerical_options),
            "runtime_versions": {name: version(name) for name in ("pmx-discrete-posterior", "numpy", "scipy")},
            **result,
            "state_results": [{**row, "events": deepcopy(states[row["state_id"]]["events"])} for row in result["state_results"]]}


def _json_safe(value):
    """Keep diagnostics in strict JSON; explicitly spell nonfinite diagnostic values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _emit(payload, output):
    text = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        # Exclusive creation protects scientific input files and previous results.
        with Path(output).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list identities from the 218-case catalog; no integration")
    listing.add_argument("--model", type=str.upper, choices=("OXC", "TAC"))
    run = commands.add_parser("run", help="compute one current-API Laplace posterior")
    run.add_argument("model", type=str.upper, choices=("OXC", "TAC"))
    run.add_argument("case", nargs="?", help="case number or Cnnn; default C003")
    run.add_argument("--input", type=Path, help="custom raw OXC/TAC JSON instead of a catalog case")
    run.add_argument("--numerical-options", type=Path, help="current-API Laplace options JSON; omission passes None")
    run.add_argument("--output", type=Path, help="new JSON file; default stdout; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.command == "run" and args.input is not None and args.case is not None:
        parser.error("choose a catalog case or --input, not both")
    try:
        if args.command == "list":
            cases = list_scenarios(args.model)
            _emit({"count": len(cases), "cases": cases}, None)
            return 0
        if args.output is not None:
            if args.output.exists():
                raise InputError("--output already exists; choose a new path")
            if not args.output.parent.is_dir():
                raise InputError("--output parent directory does not exist")
        scenario = load_scenario(args.model, args.case, input_json=args.input)
        options = None if args.numerical_options is None else read_json(args.numerical_options)[0]
        result = run_scenario(scenario, numerical_options=options)
        _emit(result, args.output)
        return 0 if result["status"] == "success" else 1
    except (InputError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
