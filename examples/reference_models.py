"""功能 / Function: 展示OXC/TAC参考输入到新模型层的连接 / Demonstrate reference inputs through the new model layer.
输入 / Inputs: 本地已冻结C003参考JSON / Frozen local C003 reference JSON.
输出 / Outputs: 显式model、个体数据、观测与候选；仅预测和条件似然 / Explicit inputs, predictions and conditional likelihoods only.
"""

import json
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

from pmx_discrete_posterior.interfaces.validation import InputError, scalar
from pmx_discrete_posterior.models.likelihood import conditional_log_likelihood
from pmx_discrete_posterior.models.parameter_resolution import resolve_parameters
from pmx_discrete_posterior.models.prediction import evaluate_prediction, prepare_prediction_records
from pmx_discrete_posterior.models.prediction.linear import LinearPredictor
from pmx_discrete_posterior.models.random_effects import resolve_random_effects
from pmx_discrete_posterior.models.residuals import resolve_residuals


def _event(time, amount):
    """功能 / Function: 显式口服事件 / Explicit oral event.
    输入 / Inputs: h、mg. 输出 / Outputs: 事件映射 / Event mapping.
    """
    return dict(time=time, amount=amount, amount_unit="mg", route="oral", compartment="depot")


def _covariance_refs(matrix, prefix, values, units):
    """功能 / Function: 将已冻结协方差移入唯一参数源 / Store frozen covariance in the single parameter source.
    输入 / Inputs: 对称矩阵、前缀、参数映射 / Symmetric matrix and parameter maps.
    输出 / Outputs: 引用矩阵；values/units补充引用值 / Reference matrix and updated maps.
    """
    matrix = np.asarray(matrix)
    if not np.array_equal(matrix, matrix.T):
        raise InputError("reference covariance is not symmetric")
    refs = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            if matrix[i, j] == 0:
                row.append(None)
            else:
                name = f"{prefix}_{min(i,j)}_{max(i,j)}"
                values[name], units[name] = float(matrix[i, j]), "1"
                row.append(name)
        refs.append(row)
    return refs


def _oxc_relations(*, parameters, covariates, iiv_values, iov_values, occasions, evaluation_interval):
    """功能 / Function: OXC体重及清除率IIV关系 / OXC weight and clearance IIV relationship.
    输入 / Inputs: 已给参数、BW、当前CL效应 / Parameters, BW and current CL effect.
    输出 / Outputs: CL、V、ka分段 / Resolved CL, V and ka segment.
    """
    p, bw = parameters["values"], scalar(covariates["BW"], positive=True)
    exponent = p["k0"] - p["kmax"] * expit(p["gamma"] * math.log(bw / p["k50"]))
    cl = p["CL_p"] * (bw/p["normalization_weight"])**exponent * math.exp(iiv_values["CL"])
    values = {"CL": scalar(cl, positive=True), "V": scalar(p["V_p"], positive=True), "ka": scalar(p["ka"], positive=True)}
    return {"units": {"CL": "L/h", "V": "L", "ka": "1/h"}, "segments": [
        {"start": evaluation_interval[0], "end": max(evaluation_interval[1], evaluation_interval[0]+1),
         "occasion_id": None, "values": values}]}


def _oxc_system(*, values, units):
    """功能 / Function: OXC一房室口服系统 / OXC one-compartment oral system.
    输入 / Inputs: 个体CL、V、ka及单位 / Individual kinetics and units.
    输出 / Outputs: 线性房室系统 / Linear compartment system.
    """
    if units != {"CL": "L/h", "V": "L", "ka": "1/h"}:
        raise InputError("OXC units differ")
    cl, volume, ka = (scalar(values[k], positive=True) for k in ("CL", "V", "ka"))
    return {"state_names": ["depot", "central"], "amount_unit": "mg", "time_unit": "h", "concentration_unit": "mg/L",
            "matrix": [[-ka, 0], [ka, -cl/volume]], "observation_vector": [0, 1/volume],
            "inputs": {("oral", "depot"): {"fractions": [1, 0], "lag": 0}}}


def _tac_relations(*, parameters, covariates, iiv_values, iov_values, occasions, evaluation_interval):
    """功能 / Function: TAC协变量、IIV及IOV关系 / TAC covariate, IIV and IOV relationships.
    输入 / Inputs: 参数、协变量及按occasion索引效应 / Parameters, covariates and occasion-indexed effects.
    输出 / Outputs: 每个occasion的CL/V1/V2/Q/ka/lag / Occasion-specific resolved kinetics.
    """
    p, c, segments = parameters["values"], covariates, []
    for occasion in occasions:
        io = iov_values[occasion["occasion_id"]]
        cl = (p["CL"] * p["theta_CYP3A5"]**c["X"] * (1+p["theta_HEM"]*(c["HEM"]-.33))
              * (c["WT"]/p["normalization_weight"])**p["WT_exponent"]
              * (1+p["theta_POD"]*(min(c["POD"], p["POD_cap"])-22.7)) * math.exp(iiv_values["CL"]+io["CL"]))
        v1 = p["V1"] * (1+p["theta_Pred"]*(c["Pred"]-155.5)) * math.exp(iiv_values["V1"]+io["V1"])
        values = {"CL": cl, "V1": v1, "V2": p["V2"]*math.exp(iiv_values["V2"]), "Q": p["Q"],
                  "ka": p["ka"]*math.exp(iiv_values["ka"]), "lag": p["lag"]}
        values = {k: scalar(v, positive=(k != "lag"), minimum=0) for k, v in values.items()}
        segments.append({**occasion, "values": values})
    return {"units": {"CL": "L/h", "V1": "L", "V2": "L", "Q": "L/h", "ka": "1/h", "lag": "h"}, "segments": segments}


def _tac_system(*, values, units):
    """功能 / Function: TAC两房室口服系统 / TAC two-compartment oral system.
    输入 / Inputs: 已解析动力学 / Resolved kinetics.
    输出 / Outputs: 药量mg、浓度ng/mL的系统 / System with mg amounts and ng/mL concentration.
    """
    if units != {"CL": "L/h", "V1": "L", "V2": "L", "Q": "L/h", "ka": "1/h", "lag": "h"}:
        raise InputError("TAC units differ")
    cl, v1, v2, q, ka = (scalar(values[k], positive=True) for k in ("CL", "V1", "V2", "Q", "ka"))
    return {"state_names": ["depot", "central", "peripheral"], "amount_unit": "mg", "time_unit": "h", "concentration_unit": "ng/mL",
            "matrix": [[-ka, 0, 0], [ka, -(cl+q)/v1, q/v2], [0, q/v1, -q/v2]],
            "observation_vector": [0, 1000/v1, 0],
            "inputs": {("oral", "depot"): {"fractions": [1, 0, 0], "lag": values["lag"]}}}


def load_reference(name):
    """功能 / Function: 载入本地参考用例并显式映射 / Load and explicitly map a local reference case.
    输入 / Inputs: OXC或TAC / OXC or TAC. 输出 / Outputs: 全部模型层输入 / Complete model-layer inputs.
    """
    if name not in ("OXC", "TAC"):
        raise InputError("reference name must be OXC or TAC")
    source = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / f"{name}_C003.json"
    expected_hash = {"OXC": "67d268cbc4e7268474753403ea195ec52a5ce95986e40772e2e1a8ab03db3d5a",
                     "TAC": "3aa3ae1408c8c8707e3644f1924217350847b3a3555d9dcec90b89a574ab14e6"}
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash[name]:
        raise InputError("frozen reference changed; modify the returned explicit model inputs instead")
    raw = json.loads(payload.decode("utf-8"))
    if name == "OXC":
        values = {**raw["model"]["fixed"], **{k: v for k, v in raw["model"]["parameters"].items() if k != "BW"}, "sigma": raw["residual"]["sd"]}
        units = {k: "1" for k in values}
        units.update(CL_p="L/h", V_p="L", ka="1/h", k50="kg", normalization_weight="kg", sigma="mg/L")
        iiv_names, covariates = ["CL"], {"BW": raw["model"]["parameters"]["BW"]}
        iiv_matrix = raw["variability"]["joint_omega"]
        iov, occasions = None, []
        shared, dose, period = raw["initialization"]["shared_last_dose_h"], 300, 12
        candidates = [{"state_id": s["state_event_id"], "events": [_event(e["time_h"], e["amount_mg"]) for e in s["dose_events"]]} for s in raw["candidate_states"]]
        observations = [{"record_id": str(i), "time": r["time_h"], "value": r["concentration_mg_L"], "time_unit": "h", "value_unit": "mg/L"} for i, r in enumerate(raw["observations"])]
        relations, predictor, error_type, role = _oxc_relations, LinearPredictor(_oxc_system), "additive", "sd"
    else:
        covariates = {k: raw["fixed"][k] for k in ("X", "HEM", "POD", "Pred")}
        covariates["WT"] = raw["parameters"]["WT"]
        values = {**{k: v for k, v in raw["fixed"].items() if k not in covariates},
                  **{k: v for k, v in raw["parameters"].items() if k != "WT"}, "sigma": raw["residual_sd"]}
        units = {k: "1" for k in values}
        units.update(CL="L/h", V1="L", V2="L", Q="L/h", ka="1/h", lag="h", normalization_weight="kg",
                     theta_POD="1/day", POD_cap="day", theta_Pred="mL/ng")
        iiv_names, iiv_matrix = ["CL", "V1", "V2", "ka"], np.array(raw["omega"])[:4, :4]
        shared, dose, period = raw["shared_last"], raw["shared_dose"], raw["tau"]
        boundaries = [shared, *raw["slots"], max(48, max(raw["times"])+1)]
        occasions = [{"occasion_id": f"occ_{i}", "start": a, "end": b} for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        iov = {"names": ["CL", "V1"], "distribution": "normal", "occasion_key": "occasion_id",
               "covariance_parameters": _covariance_refs(np.array(raw["omega"])[4:6, 4:6], "pi", values, units)}
        candidates = [{"state_id": s["state_event_id"], "events": [_event(e["time_h"], e["amount_mg"]) for e in s["events"]]} for s in raw["states"]]
        observations = [{"record_id": str(i), "time": t, "value": y, "time_unit": "h", "value_unit": "ng/mL"} for i, (t, y) in enumerate(zip(raw["times"], raw["observations"]))]
        relations, predictor, error_type, role = _tac_relations, LinearPredictor(_tac_system), "proportional", "coefficient_sd"
    refs = _covariance_refs(iiv_matrix, "omega", values, units)
    model = {"parameters": {"values": values, "units": units}, "parameter_model": {"covariate_names": list(covariates), "relations": relations},
             "iiv": {"names": iiv_names, "distribution": "normal", "covariance_parameters": refs}, "iov": iov,
             "residual_model": {"distribution": "normal", "error_type": error_type, "parameter_names": {role: "sigma"}}, "predict": predictor}
    periodic = _event(shared, dose)
    periodic["offset"] = 0
    del periodic["time"]
    individual_data = {"covariates": covariates, "occasions": occasions,
        "dosing_prehistory": {"events": [_event(shared, dose)], "regimen": {"period": period, "anchor_time": shared, "time_unit": "h", "events": [periodic]}},
        "initialization": {"mode": "steady_state", "start_time": shared, "event_phase": "before_events",
                           "steady_state": {"parameter_rule": "fixed_reference", "reference_occasion_id": occasions[0]["occasion_id"] if occasions else None}}}
    return {"model": model, "individual_data": individual_data, "observations": observations, "candidate_states": candidates}


def evaluate_reference(case, candidate_state, random_effect_values):
    """功能 / Function: 串联一次模型层评价 / Evaluate the complete model layer once.
    输入 / Inputs: 参考输入、一个候选、当前效应向量 / Reference inputs, one candidate and current effects.
    输出 / Outputs: 预测和条件对数似然，不是后验 / Predictions and conditional log-likelihood, not posterior probabilities.
    """
    model, data = case["model"], case["individual_data"]
    distribution = resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
                                          occasions=data["occasions"], dependence="independent_blocks")
    request = prepare_prediction_records(case["observations"], data)
    individual_parameters = resolve_parameters(parameter_model=model["parameter_model"], parameters=model["parameters"],
        covariates=data["covariates"], **distribution.expand(random_effect_values), occasions=data["occasions"], evaluation_interval=request["evaluation_interval"])
    predictions = evaluate_prediction(predict=model["predict"], individual_parameters=individual_parameters, candidate_state=candidate_state,
        prediction_records=request["prediction_records"], individual_data=data)
    residual_distribution = resolve_residuals(residual_model=model["residual_model"], parameters=model["parameters"], predictions=predictions,
        prediction_records=request["prediction_records"], dependence="conditionally_independent")
    return predictions, conditional_log_likelihood(observations=case["observations"], predictions=predictions, residual_distribution=residual_distribution)


def main():
    """功能 / Function: 打印两个参考模型的条件评价 / Print conditional reference evaluations.
    输入 / Inputs: 本地参考文件，当前效应全零 / Local references at zero random effects.
    输出 / Outputs: JSON浓度与条件似然；不输出后验 / JSON concentrations and conditional likelihoods, not posteriors.
    """
    output = {}
    for name, dimension in (("OXC", 1), ("TAC", 10)):
        case = load_reference(name)
        output[name] = []
        for state in case["candidate_states"]:
            predictions, likelihood = evaluate_reference(case, state, np.zeros(dimension))
            output[name].append({"state_id": state["state_id"], "predictions": predictions["values"].tolist(),
                                 "units": predictions["units"], "conditional_log_likelihood": likelihood})
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
