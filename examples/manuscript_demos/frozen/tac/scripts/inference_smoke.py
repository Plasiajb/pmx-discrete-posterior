"""Source-isolated selected inference checks; no native execution or full panel."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys

from supported_inference import covariance_gate, inference_entry

sys.dont_write_bytecode = True
CASES = ("TAC001", "TAC046", "SCN001", "SCN043")
POSTERIOR_TOL = 1e-7
LOG_EVIDENCE_TOL = 1e-6


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap(root):
    sys.path.insert(0, str(root/"source/study_adapters"))
    import adapter
    pmx = adapter.public_package()
    isolated_modules(root)
    return pmx


def isolated_modules(root):
    names = {"adapter", "scenarios", "routes", "oxc_original", "oxc_corrected", "oxc_ghq",
             "oxc_factory", "precision_definitions", "run_cp01a_package_laplace_smoke"}
    found = {}
    for name, module in list(sys.modules.items()):
        if name in names or name == "pmx_discrete_posterior" or name.startswith("pmx_discrete_posterior."):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root/"source"), "Project module outside extracted source: " + name)
            found[name] = path.relative_to(root).as_posix()
    return found


def posterior_delta(actual, expected):
    require(list(actual) == list(expected), "Candidate order differs")
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in actual.values()), "Invalid fresh posterior")
    require(abs(math.fsum(actual.values())-1) <= 1e-10, "Fresh posterior has invalid sum")
    delta = max(abs(actual[k]-expected[k]) for k in actual)
    require(delta <= POSTERIOR_TOL, "Fresh posterior exceeds declared regression tolerance")
    return delta


def log_delta(actual, expected):
    require(list(actual) == list(expected), "State evidence order differs")
    require(all(math.isfinite(v) for v in actual.values()), "Nonfinite fresh state evidence")
    delta = max(abs(actual[k]-expected[k]) for k in actual)
    require(delta <= LOG_EVIDENCE_TOL, "Fresh state evidence exceeds declared regression tolerance")
    return delta


def tacrolimus(root, cid):
    from scenarios import case_from_settings
    isolated_modules(root)
    setting = next(r for r in load(root/"tacrolimus/scenario_settings.json")["cases"] if r["case_id"] == cid)
    require(setting["truth_label_declared"] is False, "Truth-label scope changed")
    case = case_from_settings(setting)
    config = case.error_config.public_config()
    coverage = next(r for r in rows(root/"tacrolimus/scenario_coverage.csv") if r["case_id"] == cid)
    order = int(coverage["ghq_order"])
    # Both selected cases used the executed hermgauss route at 241, not the 641 continuation.
    require(order == 241, "Selected final order changed; do not substitute a different integrator")
    laplace = inference_entry("tacrolimus_laplace",config=config,eta_dimension=2,case=case)
    require(laplace["valid_for_comparison"], "Fresh Laplace diagnostic gate failed")
    ghq = inference_entry("tacrolimus_prior_ghq",config=config,eta_dimension=2,case=case,nodes=order)
    expected = [r for r in rows(root/"tacrolimus/posterior_vectors.csv") if r["case_id"] == cid]
    laplace_expected = {r["state_id"]:float(r["posterior"]) for r in expected if r["route"] == "Laplace"}
    selected = [r for r in rows(root/"tacrolimus/ghq_orders.csv") if r["case_id"] == cid and int(r["order"]) == order]
    ghq_expected = {r["state_id"]:float(r["posterior"]) for r in selected}
    evidence_expected = {r["state_id"]:float(r["log_evidence"]) for r in selected}
    return {"case_id":cid,"status":"PASS","ghq_order":order,
            "ghq_centering":"independent prior-centered tensor GHQ",
            "laplace_max_posterior_delta":posterior_delta(laplace["posterior"], laplace_expected),
            "ghq_max_posterior_delta":posterior_delta(ghq["posterior"], ghq_expected),
            "ghq_max_log_evidence_delta":log_delta(ghq["state_log_evidence"], evidence_expected),
            "laplace_posterior":laplace["posterior"],"ghq_posterior":ghq["posterior"],
            "ghq_state_log_evidence":ghq["state_log_evidence"],
            "laplace_diagnostic_gate":True,"covariance_gate":"symmetry and Cholesky"}


def oxc(root, cid, pmx):
    from oxc_corrected import make_model, inference_inputs
    isolated_modules(root)
    n = load(root/"inference_inputs"/(cid+".json"))
    require(n["truth_label_declared"] is False, "Truth-label scope changed")
    if cid == "SCN043":
        require(n["input_version"] == "E01_HISTORY300_V1", "Corrected input required")
        require(n["e01"]["history"]["last_dose_time_h"] == -24, "Corrected common history must end at -24 h")
    model = make_model(n, original=cid == "SCN001")
    states, config, observations = inference_inputs(n)
    laplace = inference_entry("laplace",config=config,eta_dimension=1,
        observations=observations,states=states,model=model)
    require(not laplace.indeterminate, "Fresh OXC Laplace indeterminate")
    require(all(s.diagnostics.valid_laplace and s.diagnostics.optimizer_success
                and not s.diagnostics.boundary_hit and s.diagnostics.hessian_min_eigenvalue > 0
                for s in laplace.states), "Fresh OXC Laplace diagnostic gate failed")
    posterior, evidence, diagnostics = inference_entry("adaptive_ghq",config=config,eta_dimension=1,
        observations=observations,states=states,model=model,nodes=121)
    require(all(d["valid_mode"] and d["optimizer_success"] for d in diagnostics.values()), "OXC GHQ diagnostic gate failed")
    base = root/"archive/v14_oxc"
    laplace_rows = [r for r in rows(base/"scenario_vectors.csv") if r["scenario_id"] == cid and r["route"] == "package_laplace"]
    ghq_rows = [r for r in rows(base/"ghq_five_order_posterior_long.csv") if r["scenario_id"] == cid and int(r["nodes"]) == 121]
    require(all(r["selected_input_sha256"] == n["selected_input_sha256"] for r in laplace_rows), "OXC input identity mismatch")
    require(all(r["input_sha256"] == n["selected_input_sha256"] for r in ghq_rows), "OXC GHQ input identity mismatch")
    return {"case_id":cid,"status":"PASS","input_version":n["input_version"],
            "ghq_order":121,"ghq_centering":"historical adaptive mode/Hessian; not independent prior-centered GHQ",
            "laplace_max_posterior_delta":posterior_delta(laplace.posterior,{r["candidate_id"]:float(r["normalized_mean"]) for r in laplace_rows}),
            "ghq_max_posterior_delta":posterior_delta(posterior,{r["candidate_id"]:float(r["posterior"]) for r in ghq_rows}),
            "ghq_max_log_evidence_delta":log_delta(evidence,{r["candidate_id"]:float(r["log_marginal"]) for r in ghq_rows}),
            "laplace_posterior":laplace.posterior,"ghq_posterior":posterior,"ghq_state_log_evidence":evidence,
            "laplace_diagnostic_gate":True,"covariance_gate":"symmetry and Cholesky",
            "native_execution":False,"C3_result_validation":False}


def run(root, selected=CASES):
    require(bool(selected) and len(set(selected)) == len(selected) and set(selected) <= set(CASES), "Only four bounded smoke cases are permitted")
    import numpy as np
    import scipy
    root = root.resolve()
    pmx = bootstrap(root)
    results = [tacrolimus(root,cid) if cid.startswith("TAC") else oxc(root,cid,pmx) for cid in selected]
    return {"status":"PASS","scope":"Fresh-extraction source-isolated inference, not an independent environment or full historical reproduction",
            "versions":{"Python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "posterior_absolute_tolerance":POSTERIOR_TOL,"state_log_evidence_absolute_tolerance":LOG_EVIDENCE_TOL,
            "results":results,"loaded_project_modules":isolated_modules(root),
            "covariance_safeguard":"Wrapper rejects nonsymmetric/non-PD input before calling frozen v0.1.4; generic release bug is not fixed",
            "native_execution":False,"full_panel_execution":False,"clinical_accuracy_claim":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cases",nargs="+",choices=CASES,default=CASES)
    args = parser.parse_args()
    print(json.dumps(run(args.root,args.cases),indent=2,allow_nan=False))
