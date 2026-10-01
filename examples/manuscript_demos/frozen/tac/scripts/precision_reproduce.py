"""Reconstruct published input-precision variants and summaries; optional inference replay."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def require(ok,message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def evaluate(n,cid):
    from adapter import public_package
    from oxc_corrected import make_model, inference_inputs
    from oxc_factory import build_pmx_inputs
    from supported_inference import inference_entry
    pmx = public_package()
    if cid in ("SCN043","SCN044","SCN045"):
        model = make_model(n)
        states,config,observations = inference_inputs(n)
    else:
        _,model,states,observations = build_pmx_inputs(n,"package_laplace","Nelder-Mead")
        config = pmx.MultiEtaLaplaceConfig(
            omega_covariance=((float(n["model"]["iiv"]["omega_variance"]),),),
            additive_sd=float(n["model"]["ruv"]["additive_sd_mg_l"]),
            proportional_sd=float(n["model"]["ruv"]["proportional_sd"]),optimizer_method="Nelder-Mead")
    result = inference_entry("laplace",config=config,eta_dimension=1,
        observations=observations,states=states,model=model)
    p,logs,_ = inference_entry("adaptive_ghq",config=config,eta_dimension=1,
        observations=observations,states=states,model=model,nodes=41)
    return {"package_laplace":{"posterior_raw":result.posterior,
              "state_log_evidence":{s:d.laplace_log_likelihood for s,d in result.diagnostics.items()}},
            "ghq41":{"posterior_raw":p,"state_log_evidence":logs}}


def run(root,recompute=False):
    root = root.resolve()
    sys.path.insert(0,str(root/"source/study_adapters"))
    from precision_definitions import variants, comparison
    from reaggregate import distances
    base = root/"input_precision"
    identity,summary = load(base/"identity.json"),load(base/"summary.json")
    require(len(set(identity["case_ids"]))==30 and len(identity["rounded_case_ids"])==26,"Panel identity mismatch")
    with (base/"paired_comparisons.csv").open(encoding="utf-8",newline="") as stream:
        table = {(r["case_id"],r["variant"],r["route"]):r for r in csv.DictReader(stream)}
    results = []
    vectors = components = 0
    max_reconstruction_error = 0.
    max_fresh_posterior_delta = max_fresh_evidence_delta = 0.
    imported = {}
    versions = None
    if recompute:
        from inference_smoke import bootstrap
        bootstrap(root)
        import numpy, scipy, platform
        versions = {"Python":platform.python_version(),"numpy":numpy.__version__,"scipy":scipy.__version__}
    for cid in identity["case_ids"]:
        case = load(base/"cases"/(cid+".json"))
        reconstructed,deltas,roundtrip = variants(case["inputs"]["json"],case["native_literals"])
        require(reconstructed==case["inputs"],"Decoded input-variant reconstruction differs: "+cid)
        require(deltas==case["field_deltas"] and roundtrip==case["native_SIGMA_to_existing_SD_API_variance_roundtrip_delta"],
                "Numeric field delta reconstruction differs")
        for variant,payload in case["variants"].items():
            fresh = evaluate(reconstructed[variant],cid) if recompute else None
            order = payload["state_order"]
            for route,value in payload["routes"].items():
                vectors += 1
                components += len(order)
                p,logs = value["posterior_raw"],value["state_log_evidence"]
                require(list(p)==list(logs)==order and value["eligible"] and value["mode_valid"],"Saved vector/diagnostic identity differs")
                total = math.fsum(p.values())
                require(all(math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(total-1)<=1e-6,"Invalid saved probability vector")
                require(value["posterior_normalized"]=={s:p[s]/total for s in order},"Repeat normalization differs")
                weights = {s:logs[s]+math.log(case["inputs"][variant]["prior_vector"][s]) for s in order}
                high = max(weights.values())
                linear = {s:math.exp(weights[s]-high) for s in order}
                denominator = math.fsum(linear.values())
                max_reconstruction_error = max(max_reconstruction_error,
                    max(abs(linear[s]/denominator-value["posterior_normalized"][s]) for s in order))
                for row in value["all_pair_log_evidence_contrasts"]:
                    require(row["log_evidence_left_minus_right"]==logs[row["left"]]-logs[row["right"]],"Saved evidence contrast differs")
                if fresh:
                    require(list(fresh[route]["posterior_raw"])==order,"Fresh state order differs")
                    pd = max(abs(fresh[route]["posterior_raw"][s]-p[s]) for s in order)
                    ed = max(abs(fresh[route]["state_log_evidence"][s]-logs[s]) for s in order)
                    require(math.isfinite(pd) and math.isfinite(ed),"Nonfinite fresh replay difference")
                    max_fresh_posterior_delta = max(max_fresh_posterior_delta,pd)
                    max_fresh_evidence_delta = max(max_fresh_evidence_delta,ed)
        for variant in ("native_dv_only","native_all_numeric"):
            for route in ("package_laplace","ghq41"):
                left,right = (case["variants"][v]["routes"][route] for v in ("json",variant))
                actual = comparison(left,right)
                require(actual==case["comparisons"][variant][route],"Full paired comparison differs")
                row = table[(cid,variant,route)]
                p,q = left["posterior_normalized"],right["posterior_normalized"]
                independent = distances(list(p),list(p.values()),list(q.values()))
                for field,key in (("max_abs","max_abs"),("TV","TV"),("L2","L2")):
                    require(abs(independent[key]-float(row[field]))<=2e-14,"Independent distance mismatch")
                for field in ("max_abs","TV","L2","max_abs_log_evidence_change","max_abs_log_evidence_contrast_change"):
                    require(actual[field]==float(row[field]),"Full-precision CSV mismatch")
                for key in ("exact_MAP","tolerant_MAP","exact_rank","tolerant_rank"):
                    require((actual[key+"_before"]!=actual[key+"_after"])==(row[key+"_changed"]=="True"),"CSV rank/MAP change mismatch")
                results.append({"case":cid,"variant":variant,"route":route,**actual})
    maxima = []
    for row in summary["summaries"]:
        selected = [v for v in results if v["route"]==row["route"] and v["variant"]==row["variant"]]
        require(len(selected)==row["planned"]==row["eligible"]==30,"Precision denominator mismatch")
        for field in ("max_abs","max_abs_log_evidence_change","max_abs_log_evidence_contrast_change"):
            worst = max(selected,key=lambda r:r[field])
            expected = row["worst"][field] if field=="max_abs" else row[field]
            require(worst[field]==expected,"Precision summary maximum mismatch")
        for key in ("exact_MAP","tolerant_MAP","exact_rank","tolerant_rank"):
            require([r["case"] for r in selected if r[key+"_before"]!=r[key+"_after"]]==row[key+"_changes"],"Precision summary change count mismatch")
        maxima.append({"variant":row["variant"],"route":row["route"],"case":row["worst"]["case"],
            "max_posterior_change":row["worst"]["max_abs"],
            "max_state_log_evidence_change":row["max_abs_log_evidence_change"],
            "max_log_contrast_change":row["max_abs_log_evidence_contrast_change"]})
    require(vectors==180 and components==744 and len(results)==120 and max_reconstruction_error<1e-12,"Saved precision verification failed")
    replay = load(base/"archived_json_replay.json")
    require(len(replay)==60 and all(r["max_abs"]==0 and all(v==0 for v in r["component_deltas"].values()) for r in replay),"Archive replay record mismatch")
    if recompute:
        from inference_smoke import isolated_modules
        imported = isolated_modules(root)
        # This is an implementation replay, not an evidence-convergence threshold.
        require(max_fresh_posterior_delta<=1e-7 and max_fresh_evidence_delta<=1e-6,
                "Fresh implementation replay differs; retain saved evidence and report environment variation")
    return {"status":"PASS","cases":30,"variants":3,"vectors":vectors,"components":components,
        "paired_comparisons":len(results),"archived_exact_replay_records":60,
        "posterior_reconstruction_max_abs":max_reconstruction_error,"maxima":maxima,
        "new_inference_evaluations":180 if recompute else 0,"native_runs":0,
        "fresh_posterior_max_abs":max_fresh_posterior_delta if recompute else None,
        "fresh_log_evidence_max_abs":max_fresh_evidence_delta if recompute else None,
        "versions":versions,"loaded_project_modules":imported,
        "boundary":"Paired precision sensitivity, not a new C3 gate, five-repeat evidence or absolute evidence equivalence"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--recompute",action="store_true",help="Replay the bounded 30-case/180-evaluation Python panel with NumPy/SciPy")
    args = parser.parse_args()
    print(json.dumps(run(args.root,args.recompute),indent=2,allow_nan=False))
