"""Reviewer CLI for the current frozen OXC/TAC Python-Laplace adapters."""
from __future__ import annotations

import os

# Set these before any adapter, package or numerical-library import.
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)
for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib
import json
import math
from pathlib import Path
import pprint
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
CSV_FIELDS = (
    "case_id", "model", "role", "status", "eligible", "outcome", "state_index",
    "state_id", "label", "prior", "posterior_probability", "events_json",
    "published_input_sha256", "input_sha256", "result_json",
)


def _check_loaded(prefix, root):
    for name, module in tuple(sys.modules.items()):
        if name == prefix or name.startswith(prefix + "."):
            source = getattr(module, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(root):
                raise RuntimeError(f"Foreign {name} already imported; start a fresh Python process")


def adapters():
    """Never fall back to another checkout or an installed scientific package."""
    if not (SRC / "pmx_discrete_posterior" / "multieta.py").is_file():
        raise RuntimeError(
            "This manuscript example requires the legacy package. "
            "Use a separate checkout of tag manuscript-20261018; "
            "v0.2 examples are in examples/posterior_reference.py."
        )
    if not (SRC / "pmx_discrete_posterior" / "__init__.py").is_file():
        raise RuntimeError("Missing repository src; retain the complete checkout, not just this folder")
    _check_loaded("pmx_discrete_posterior", SRC)
    _check_loaded("model_adapters", HERE / "model_adapters")
    for path in (SRC, HERE):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    module = importlib.import_module("model_adapters")
    _check_loaded("model_adapters", HERE / "model_adapters")
    return module


def _runtime():
    adapters()
    for name in ("numpy", "scipy", "mpmath", "pmx_discrete_posterior"):
        importlib.import_module(name)
    _check_loaded("pmx_discrete_posterior", SRC)


def compute_input(model, inputs):
    """Copyable entry: complete inputs only, no expected probabilities."""
    _runtime()
    return adapters().compute(model, inputs)


def _json(value):
    return json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n"


def _write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_json(value))


def _output_path(value):
    path = Path(value).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists; choose a new name: {path}")
    path = path.resolve()
    for protected in (HERE, SRC, REPO / "tests", REPO / ".git"):
        if path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError("Output must be disjoint from demo, src, tests and Git metadata")
    if not path.parent.is_dir():
        raise ValueError(f"Output parent must already exist: {path.parent}")
    return path


def _selected(args, entries):
    selected = [entry for entry in entries
                if (not args.model or entry["model"] == args.model)
                and (not args.role or entry["role"] == args.role)]
    if args.case:
        patterns = [part.strip().upper() for item in args.case for part in item.split(",")]
        for pattern in patterns:
            if not pattern or not any(fnmatch.fnmatchcase(e["id"], pattern) for e in selected):
                raise ValueError(f"Case filter {pattern!r} matches nothing within the model/role selection")
        selected = [entry for entry in selected
                    if any(fnmatch.fnmatchcase(entry["id"], p) for p in patterns)]
    if not selected:
        raise ValueError("No cases selected")
    return selected


def _input_states(model, inputs):
    if model == "OXC":
        priors = {row["state_event_id"]: row["probability"] for row in inputs["priors"]}
        return [dict(state_id=s["state_event_id"], label=s["manuscript_id"],
                     prior=priors[s["state_event_id"]], events=s["dose_events"])
                for s in inputs["candidate_states"]]
    return [dict(state_id=s["state_event_id"], label=s["manuscript_id"],
                 prior=s["prior"], events=s["events"]) for s in inputs["states"]]


def _failure(status, exc):
    return dict(status=status, eligible=False, posterior=None,
                error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())


def run_one(entry):
    """Retain the adapter result verbatim; add separate orchestration metadata."""
    started = time.monotonic()
    inputs, states = None, []
    try:
        loaded, inputs = adapters().load_case(entry["id"])
        states = _input_states(loaded["model"], inputs)
    except Exception as exc:
        result, outcome = _failure("FAILED_INPUT", exc), "input_error"
    else:
        try:
            result = compute_input(loaded["model"], inputs)
            if not isinstance(result, dict):
                raise TypeError("Adapter must return a result dictionary")
            _json(result)
        except Exception as exc:
            result, outcome = _failure("FAILED_EXECUTION", exc), "execution_error"
        else:
            outcome = "numerical_failure"
            if result.get("eligible") is True:
                posterior = result.get("posterior")
                ids = [s["state_id"] for s in states]
                complete = (result.get("status") in {"PASS", "PASS_LAPLACE"}
                            and isinstance(posterior, dict) and set(posterior) == set(ids)
                            and len(ids) == len(set(ids))
                            and all(isinstance(p, (int, float)) and math.isfinite(p)
                                    and 0 <= p <= 1 for p in posterior.values())
                            and abs(math.fsum(posterior.values()) - 1) <= 1e-10)
                outcome = "eligible" if complete else "contract_error"
    return dict(case_id=entry["id"], model=entry["model"], role=entry["role"],
                source_case_id=entry["source_case_id"],
                published_input_sha256=entry["input_sha256"],
                outcome=outcome, result=result, inputs=inputs, input_states=states,
                elapsed_seconds=time.monotonic() - started)


def _csv_rows(record):
    result = record["result"]
    for index, state in enumerate(record["input_states"] or [{}]):
        yield dict(
            case_id=record["case_id"], model=record["model"], role=record["role"],
            status=result.get("status"), eligible=result.get("eligible", False),
            outcome=record["outcome"], state_index=index if state else "",
            state_id=state.get("state_id", ""), label=state.get("label", ""),
            prior=state.get("prior"), events_json=json.dumps(state.get("events", [])),
            posterior_probability=(result["posterior"][state["state_id"]]
                                   if record["outcome"] == "eligible" else None),
            published_input_sha256=record["published_input_sha256"],
            input_sha256=result.get("input_sha256", ""),
            result_json=f"{record['case_id']}.json",
        )


def _source_hashes():
    paths = [Path(__file__), HERE / "catalog.json", HERE / "requirements.txt"]
    paths += sorted((HERE / "model_adapters").glob("*.py"))
    paths += sorted((SRC / "pmx_discrete_posterior").glob("*.py"))
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def execute(entries, output):
    output = _output_path(output)
    _runtime()
    summary = dict(schema="manuscript_current_cli_v1", status="RUNNING",
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   selected_ids=[e["id"] for e in entries], selected_count=len(entries),
                   completed_count=0, outcome_counts={}, cases=[],
                   route="Python-Laplace", comparison_performed=False,
                   python=sys.version, executable=sys.executable,
                   package_source=str(Path(sys.modules["pmx_discrete_posterior"].__file__).resolve()),
                   threads={name: os.environ[name] for name in THREAD_VARIABLES},
                   source_sha256=_source_hashes())
    output.mkdir()  # Exclusive creation: never adopt, clean or overwrite an existing run.
    _write_json(output / "run_started.json", summary)
    counts = Counter()
    interrupted = False
    try:
        with (output / "posterior.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            stream.flush()
            for entry in entries:
                print(f"Computing {entry['id']} ...", file=sys.stderr, flush=True)
                record = run_one(entry)
                _write_json(output / f"{entry['id']}.json", record)
                writer.writerows(_csv_rows(record))
                stream.flush()
                counts[record["outcome"]] += 1
                summary["cases"].append(dict(case_id=entry["id"], outcome=record["outcome"],
                                             status=record["result"].get("status"),
                                             result_json=f"{entry['id']}.json"))
                print(f"{entry['id']}: {record['result'].get('status')} ({record['outcome']})",
                      file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        interrupted = True
        summary["status"] = "INTERRUPTED"
    except Exception as exc:
        summary["status"] = "FAILED_EXECUTION"
        summary["error"] = _failure("FAILED_EXECUTION", exc)
    else:
        summary["status"] = "COMPLETE" if counts["eligible"] == len(entries) else "COMPLETE_WITH_FAILURES"
    summary.update(completed_count=sum(counts.values()), outcome_counts=dict(counts),
                   finished_utc=datetime.now(timezone.utc).isoformat())
    _write_json(output / "status.json", summary)
    print(_json(dict(status=summary["status"], output=str(output),
                     completed_count=summary["completed_count"], outcome_counts=dict(counts))), end="")
    if interrupted:
        return 130
    if summary["status"] == "FAILED_EXECUTION" or any(counts[k] for k in ("input_error", "execution_error", "contract_error")):
        return 2
    return 0 if summary["status"] == "COMPLETE" else 1


def export_python(entry, inputs):
    return '''"""Complete input example; run with --repo pointing to this checkout."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root (default: cwd)")
args = parser.parse_args()
path = args.repo.resolve() / "examples/manuscript_current/run_demo.py"
spec = importlib.util.spec_from_file_location("current_reviewer_demo", path)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)  # Sets single-thread limits before importing NumPy.
''' + f"\nMODEL = {entry['model']!r}\nINPUTS = " + pprint.pformat(inputs, sort_dicts=False, width=100) + '''

try:
    result = demo.compute_input(MODEL, INPUTS)
except Exception as exc:
    print(json.dumps({"status": "FAILED_EXECUTION", "eligible": False, "posterior": None,
                      "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
    raise SystemExit(2)
print(json.dumps(result, indent=2, allow_nan=False))
raise SystemExit(0 if result.get("eligible") is True else 1)
'''


def parser():
    root = argparse.ArgumentParser(
        description="Current frozen OXC/TAC cases; real Python-Laplace computation only.",
        epilog="References: OXC_C003, TAC_C003. TAC_C078 is retained, not forced to fail. "
               "Exit: 0 eligible/inspection, 1 numerical rejection, 2 input/runtime/output error, 130 interrupted.")
    commands = root.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List catalog records, without inference or input verification")
    listing.add_argument("--json", action="store_true", help="Print complete selected catalog entries as JSON")
    batch = commands.add_parser("batch", help="Compute all selected cases sequentially (default: all 218)")
    for sub in (listing, batch):
        sub.add_argument("--model", type=str.upper, choices=("OXC", "TAC"), help="Filter model (default: both)")
        sub.add_argument("--role", choices=("reference", "sensitivity"), help="Filter role (default: both)")
        sub.add_argument("--case", action="append", help="ID or quoted glob; repeat or comma-separate (default: all)")
    run = commands.add_parser("run", help="Compute one case, retaining full diagnostics")
    show = commands.add_parser("show", help="Verify input hash and print catalog entry plus complete input JSON; no inference")
    export = commands.add_parser("export-python", help="Print a runnable Python example with complete embedded inputs")
    for sub in (run, show, export):
        sub.add_argument("case_id", help="Catalog ID, for example OXC_C003 or TAC_C003")
    for sub in (run, batch):
        sub.add_argument("--output", required=True, help="New directory under an existing parent; no default, no overwrite. "
                         "Writes per-case JSON, posterior.csv and status.json; use a new name for each run.")
    export.add_argument("--output", help="New .py file under an existing parent; default: stdout; never overwrite")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        api = adapters()
        if args.command in {"list", "batch"}:
            entries = _selected(args, api.catalog())
            if args.command == "batch":
                return execute(entries, args.output)
            if args.json:
                print(_json(entries), end="")
            else:
                print("CASE_ID\tMODEL\tROLE\tSOURCE_CASE\tSERIES")
                for entry in entries:
                    series = ", ".join(dict.fromkeys(str(m.get("series", "")) for m in entry.get("memberships", [])))
                    print(f"{entry['id']}\t{entry['model']}\t{entry['role']}\t{entry['source_case_id']}\t{series}")
                print(f"{len(entries)} catalog records; no computation performed.", file=sys.stderr)
            return 0
        if args.command == "run":
            matches = [e for e in api.catalog() if e["id"] == args.case_id.upper()]
            if not matches:
                raise ValueError(f"Unknown case {args.case_id!r}; use list")
            return execute(matches, args.output)
        entry, inputs = api.load_case(args.case_id)
        if args.command == "show":
            print(_json(dict(catalog_entry=entry, inputs=inputs)), end="")
        else:
            source = export_python(entry, inputs)
            if args.output:
                path = _output_path(args.output)
                with path.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(source)
                print(str(path))
            else:
                print(source, end="")
        return 0
    except Exception as exc:
        kind = "DEPENDENCY_ERROR" if isinstance(exc, ModuleNotFoundError) else "CLI_ERROR"
        print(_json(dict(status=kind, eligible=False, error_type=type(exc).__name__, error=str(exc))),
              file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
