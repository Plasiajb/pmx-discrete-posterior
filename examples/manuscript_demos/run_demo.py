"""Bounded reviewer replay. New wrapper; frozen scientific sources are unchanged."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_subset():
    manifest = read(ROOT / "subset_manifest.json")
    expected = manifest["files"]
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "frozen").rglob("*") if p.is_file()}
    require(actual == set(expected), "Frozen subset inventory mismatch")
    for name, record in expected.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT / "frozen"), "Unsafe subset member")
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                "Frozen subset hash mismatch: " + name)
    return {"status": "PASS", "files": len(expected),
            "subset_manifest_sha256": digest(ROOT / "subset_manifest.json"),
            "source_archive": manifest["source_archive"],
            "source_archive_sha256": manifest["source_archive_sha256"],
            "scope": manifest["scope"]}


def output_path(value):
    path = Path(value).resolve()
    require(not path.is_relative_to(ROOT) and not ROOT.is_relative_to(path),
            "Output must be outside and disjoint from the demo source tree")
    require(not path.exists(), "Output must be a new directory")
    return path


def project_modules(drug):
    allowed = ROOT / "frozen" / drug
    selected = {}
    names = {"portable_adapter", "portable_support", "portable_gates", "supported_inference",
             "adapter", "routes", "scenarios"}
    for name, module in list(sys.modules.items()):
        if name in names or name == "pmx_discrete_posterior" or name.startswith("pmx_discrete_posterior."):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(allowed), "Foreign project module rejected: " + name)
            selected[name] = {"member": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
    return selected


def valid_vector(vector, ids):
    return (list(vector) == list(ids) and all(math.isfinite(v) and 0 <= v <= 1 for v in vector.values())
            and abs(math.fsum(vector.values()) - 1) <= 1e-10)


def comparison(actual, expected, tolerance):
    require(list(actual) == list(expected), "Saved reference state order differs")
    finite = all(math.isfinite(v) for v in (*actual.values(), *expected.values()))
    delta = max(abs(actual[s] - expected[s]) for s in actual) if finite else None
    return {"saved_values": expected, "max_absolute_delta": delta,
            "absolute_tolerance": tolerance, "passes": finite and delta <= tolerance}


def saved_native(folder, ids, expected):
    lines = (folder / "model.phm").read_text(encoding="utf-8-sig").splitlines()
    header = next(line.split() for line in lines if line.strip().startswith("SUBJECT_NO"))
    records = [dict(zip(header, map(float, line.split()))) for line in lines
               if line.strip() and not line.lstrip().startswith(("TABLE", "SUBJECT_NO"))]
    require(len(records) == len(ids) and [r["SUBPOP"] for r in records] == list(range(1, len(ids) + 1)),
            "Saved NONMEM subpopulation mapping differs")
    posterior = dict(zip(ids, (r["PMIX"] for r in records)))
    require(valid_vector(posterior, ids), "Invalid saved NONMEM vector")
    require(comparison(posterior, expected, 1e-14)["passes"], "Saved NONMEM/reference mismatch")
    return {"label": "SAVED NONMEM PMIX - NOT FRESH; no NONMEM run performed",
            "native_execution": False, "posterior": posterior,
            "normalization_applied": False,
            "files": [p.relative_to(ROOT).as_posix() for p in sorted(folder.iterdir()) if p.is_file()]}


def print_inputs(drug, inputs):
    print(f"{drug.upper()} reviewer demo | fixed input | Python routes only", flush=True)
    spec = inputs["complete_frozen_input"]
    print("Case: " + inputs["case"])
    print("Units: " + json.dumps(inputs["units"]))
    print("Model: " + json.dumps(spec["model"] if drug == "oxc" else spec["model_config"]))
    print("Patient/covariates: " + json.dumps(spec["patient"] if drug == "oxc" else spec["covariates"]))
    observations = spec["observation_events"] if drug == "oxc" else {
        "times_h": spec["model_config"]["observation_times_h"], "dv_ng_ml": spec["dv_ng_ml"]}
    print("Observations: " + json.dumps(observations))
    print("Common history: " + json.dumps(inputs["common_history"]))
    print("Variance/optimizer configuration: " + json.dumps(inputs["configuration"]))
    print("Residual variance: " + inputs["residual_variance"])
    print("State bits are latest-to-older; 1=given, 0=missed.")
    for state in inputs["state_order"]:
        prior = spec["prior_vector"][state] if drug == "oxc" else spec["priors"][spec["state_ids"].index(state)]
        events = inputs["actual_candidate_events"][state] if drug == "oxc" else next(
            s["recent_events"] for s in spec["state_mapping"] if s["state_id"] == state)
        print(f"  {state}: prior={prior}; recent events=" + json.dumps(events))
    print("Complete inputs and diagnostic records will be saved in report.json.", flush=True)


def print_result(result):
    print(f"\n{result['label']}: eligible={result['eligible']}; comparison_pass={result['passes']}")
    for state, probability in result["fresh"]["posterior"].items():
        print(f"  {state}: {probability:.17g}")
    for name, check in result["comparisons"].items():
        print(f"  Saved {name}: max_abs_delta={check['max_absolute_delta']}; tolerance={check['absolute_tolerance']}")
    diagnostics = result["fresh"].get("diagnostics", result["fresh"].get("state_diagnostics"))
    if diagnostics:
        for state, diag in diagnostics.items():
            hessian = diag.get("hessian_min_eigenvalue", diag.get("hessian"))
            print(f"  {state}: optimizer_success={diag['optimizer_success']}; "
                  f"eta_hat={diag['eta_hat']}; Hessian_min={hessian}; warnings={diag['warnings']}")
    else:
        print(f"  Prior-centered GHQ: {result['fresh']['nodes_per_dimension']} nodes/dimension; "
              f"{result['fresh']['node_count']} nodes/state; valid={result['fresh']['valid']}")


def run_oxc(route):
    root = ROOT / "frozen/oxc"
    project_modules("oxc")
    sys.path[:0] = [str(root / "source"), str(root / "vendor")]
    import portable_adapter as adapter
    import portable_support as support

    project_modules("oxc")
    spec = read(root / "current_input.json")
    parent = read(root / "parent_input.json")
    selection = read(root / "selected_records.json")
    case = selection["case"]
    require(case["case"] == "SCN003" and not case["numeric_target_changed"], "Unsupported OXC case")
    require(case["numeric_target_sha256"] == case["prior_numeric_target_sha256"], "OXC target identity mismatch")
    require(spec == parent, "Selected V26 SCN003 numeric input differs from its bound parent")
    require(spec["model"]["theta_wt_exp_hill"] == 219, "Current Hill=219 is required")
    require(digest(root / "current_input.json") == case["public_input_sha256"], "Current input identity differs")
    ids = spec["candidate_definition"]["candidate_order"]
    states, config, _ = support.inference_inputs(spec)
    model = adapter.model(spec, stable=True)
    inputs = {"case": "SCN003", "analysis_id": case["analysis_id"],
              "current_Hill": 219, "adapter": adapter.ADAPTER_VERSION,
              "state_order": ids, "complete_frozen_input": spec,
              "actual_candidate_events": support.build_candidate_events(spec),
              "common_history": {"type": "analytic steady-state pre-zero history", "dose_mg": 300,
                                 "interval_h": 12, "last_dose_h": -12},
              "units": {"dose": "mg", "time": "h", "concentration": "mg/L",
                        "clearance": "L/h", "volume": "L", "weight": "kg"},
              "configuration": dataclasses.asdict(config),
              "residual_variance": "additive_sd^2 + (proportional_sd * prediction)^2",
              "additive_variance_mg2_L2": config.additive_sd ** 2,
              "eta_zero_predictions_mg_L": {s.state_id: model.predictions(s.state_id, (0.0,)) for s in states}}
    print_inputs("oxc", inputs)
    results = {}
    for chosen in (("laplace", "ghq") if route == "both" else (route,)):
        source_route = "package_laplace" if chosen == "laplace" else "aghq41"
        started = time.perf_counter()
        fresh = adapter.evaluate(spec, source_route, nodes=41, stable=True)
        elapsed = time.perf_counter() - started
        saved = read(root / "reference" / (source_route + ".json"))
        binding = next(b for b in selection["repeat_1_bindings"] if b["route"] == source_route)
        require(saved["input_sha256"] == binding["executed_input_sha256"], "Saved OXC input identity mismatch")
        require(digest(root / "reference" / (source_route + ".json")) == binding["public_result_sha256"],
                "Saved OXC result identity mismatch")
        eligible = bool(fresh["eligible"] and valid_vector(fresh["posterior"], ids))
        comparisons = {"posterior": comparison(fresh["posterior"], saved["posterior"], 1e-10),
                       "log_evidence": comparison(fresh["log_marginals"], saved["log_marginals"], 1e-10)}
        results[chosen] = {"label": "FRESH Python " + source_route,
                           "elapsed_seconds": elapsed, "eligible": eligible,
                           "centering": "mode/Hessian adaptive GHQ" if chosen == "ghq" else "Laplace mode/Hessian",
                           "nodes": 41 if chosen == "ghq" else None,
                           "saved_reference": {"type": "saved single run, repeat 1, not a five-run mean",
                                               "binding": binding},
                           "fresh": fresh, "comparisons": comparisons,
                           "passes": eligible and all(c["passes"] for c in comparisons.values())}
    native_result = read(root / "saved_nonmem/result.json")
    require(native_result["native_eligible"] and native_result["native_input_precision_eligible"],
            "Saved native eligibility missing")
    native = saved_native(root / "saved_nonmem", ids,
                          dict(zip(ids, (r["PMIX"] for r in native_result["phm"]["rows"]))))
    native["saved_diagnostics"] = native_result["diagnostic_gate"]
    native["reference_type"] = "V26-bound single saved repeat 1; verified_reuse"
    return inputs, results, native


def run_tac(route):
    root = ROOT / "frozen/tac"
    project_modules("tac")
    sys.path[:0] = [str(root / "scripts"), str(root / "source/study_adapters")]
    from adapter import public_package
    from scenarios import case_from_settings
    from supported_inference import covariance_gate, inference_entry

    public_package()
    project_modules("tac")
    setting = next(c for c in read(root / "tacrolimus/scenario_settings.json")["cases"] if c["case_id"] == "TAC001")
    require(setting["truth_label_declared"] is False, "TAC fixed-data scope changed")
    case = case_from_settings(setting)
    config = case.error_config.public_config()
    require(tuple(map(tuple, setting["omega_covariance"])) == config.omega_covariance,
            "Recorded and executed covariance differ")
    covariance_gate(config.omega_covariance, 2)
    coverage = next(c for c in rows(root / "tacrolimus/scenario_coverage.csv") if c["case_id"] == "TAC001")
    order = int(coverage["ghq_order"])
    require(order == 241, "TAC001 saved reference requires the original GHQ241 route")
    model = case.model()
    inputs = {"case": "TAC001", "complete_frozen_input": setting,
              "state_order": list(case.state_ids), "configuration": dataclasses.asdict(config),
              "units": {"dose": "mg", "time": "h", "concentration": "ng/mL", "clearance": "L/h",
                        "volume": "L", "concentration_conversion": "mg/L multiplied by 1000 to ng/mL"},
              "common_history": {"type": "finite, zero initial condition; no SS or ADDL",
                                 "count": 120, "dose_mg": 5, "interval_h": 12,
                                 "first_dose_h": -1440, "last_dose_h": -12},
              "covariance_gate": "finite, dimension-matched, symmetric, Cholesky positive definite",
              "residual_variance": "additive_sd^2 + (proportional_sd * prediction)^2",
              "additive_variance_ng2_mL2": config.additive_sd ** 2,
              "eta_zero_predictions_ng_mL": {s: model.predictions(s, (0.0, 0.0)) for s in case.state_ids}}
    print_inputs("tac", inputs)
    expected = [r for r in rows(root / "tacrolimus/posterior_vectors.csv") if r["case_id"] == "TAC001"]
    ghq_rows = [r for r in rows(root / "tacrolimus/ghq_orders.csv")
                if r["case_id"] == "TAC001" and int(r["order"]) == order]
    results = {}
    for chosen in (("laplace", "ghq") if route == "both" else (route,)):
        route_name = "tacrolimus_laplace" if chosen == "laplace" else "tacrolimus_prior_ghq"
        started = time.perf_counter()
        fresh = inference_entry(route_name, config=config, eta_dimension=2, case=case, nodes=order)
        elapsed = time.perf_counter() - started
        expected_rows = [r for r in expected if r["route"] == "Laplace"] if chosen == "laplace" else ghq_rows
        comparisons = {"posterior": comparison(fresh["posterior"],
                        {r["state_id"]: float(r["posterior"]) for r in expected_rows}, 1e-7)}
        if chosen == "ghq":
            comparisons["log_evidence"] = comparison(fresh["state_log_evidence"],
                {r["state_id"]: float(r["log_evidence"]) for r in ghq_rows}, 1e-6)
            saved_final = {r["state_id"]: float(r["posterior"]) for r in expected if r["route"] == "GHQ"}
            require(comparison(comparisons["posterior"]["saved_values"], saved_final, 1e-14)["passes"],
                    "Saved GHQ order is not the current final reference")
        eligible = bool(fresh.get("valid_for_comparison", fresh.get("valid", False))
                        and valid_vector(fresh["posterior"], case.state_ids))
        results[chosen] = {"label": "FRESH Python " + route_name,
                           "elapsed_seconds": elapsed, "eligible": eligible,
                           "centering": "N(0,Omega) prior tensor GHQ" if chosen == "ghq" else "Laplace mode/Hessian",
                           "nodes_per_dimension": order if chosen == "ghq" else None,
                           "saved_reference": "tacrolimus/ghq_orders.csv, TAC001 order=241" if chosen == "ghq"
                                              else "tacrolimus/posterior_vectors.csv, TAC001 route=Laplace",
                           "fresh": fresh, "comparisons": comparisons,
                           "passes": eligible and all(c["passes"] for c in comparisons.values())}
    native_folder = root / "tacrolimus/models/TAC001"
    native = saved_native(native_folder, list(case.state_ids),
                          {r["state_id"]: float(r["posterior"]) for r in expected if r["route"] == "NONMEM"})
    native["saved_diagnostics"] = read(native_folder / "native_diagnostics.json")
    require(native["saved_diagnostics"]["eligible"], "Saved native diagnostic eligibility missing")
    native["reference_type"] = "saved TAC001 reference; historical/v1_1/archive/ESM_2_v16.zip"
    return inputs, results, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="demo", required=True)
    commands.add_parser("verify", help="Check only the bundled subset hashes; no inference")
    for drug, case in (("oxc", "SCN003"), ("tac", "TAC001")):
        command = commands.add_parser(drug)
        command.add_argument("--case", choices=(case,), default=case)
        command.add_argument("--route", choices=("both", "laplace", "ghq"), default="both")
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(sys.version_info >= (3, 12), "Python 3.12 or newer is required by the frozen OXC source")
    require(__debug__, "Do not use Python -O: frozen dependency assertions must remain enabled")
    verified = verify_subset()
    if args.demo == "verify":
        print(json.dumps(verified, indent=2))
        return 0
    output = output_path(args.output)
    import numpy
    import scipy

    inputs, results, native = (run_oxc if args.demo == "oxc" else run_tac)(args.route)
    passed = all(r["passes"] for r in results.values())
    report = {"schema": "manuscript-demo-report-v1", "status": "PASS" if passed else "FAIL",
              "demo": args.demo, "case": args.case, "wrapper_revision": "1",
              "core_version": "0.1.4", "native_execution": False, "full_grid_execution": False,
              "clinical_accuracy_claim": False, "inputs": inputs, "routes": results,
              "saved_nonmem": native, "subset_verification": verified,
              "runtime": {"Python": platform.python_version(), "numpy": numpy.__version__,
                          "scipy": scipy.__version__, "system": platform.system()},
              "loaded_project_modules": project_modules(args.demo)}
    verify_subset()
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(serialized, encoding="utf-8")
    ids = inputs["state_order"]
    with (output / "posterior.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "state_id", "posterior", "eligible"])
        for route, result in results.items():
            for state in ids:
                writer.writerow([result["label"], state, result["fresh"]["posterior"][state], result["eligible"]])
        for state in ids:
            writer.writerow([native["label"], state, native["posterior"][state], "saved_only"])
    for result in results.values():
        print_result(result)
    print("\n" + native["label"])
    print(json.dumps(native["posterior"]))
    print(f"\n{report['status']}: report.json and posterior.csv written to the requested new output directory.")
    return 0 if passed else 1


def public_error(exc):
    """Keep the useful failure cause without exposing a host filesystem path."""
    detail = str(exc).replace("\n", " ").replace("\r", " ")
    if re.search(r"[A-Za-z]:[\\/]|\\\\|/(?:Users|home|tmp|var|opt|mnt)/", detail):
        detail = "Host-path details omitted."
    if isinstance(exc, ImportError):
        action = "Use Python 3.12 and install requirements.txt into this same Python environment."
    elif isinstance(exc, OSError):
        action = "Check that the complete demo directory is readable and --output is a new writable directory outside it."
    elif isinstance(exc, RuntimeError):
        action = "Check NumPy/SciPy versions and runtime permissions, then run the verify command; do not relax numerical tolerances."
    else:
        action = "Check the fixed-case command and run verify to check the bundled input/source files."
    return f"ERROR [{type(exc).__name__}]: {detail or 'No further detail was supplied.'} {action}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, ImportError, OSError, RuntimeError) as exc:
        print(public_error(exc), file=sys.stderr)
        raise SystemExit(1)
