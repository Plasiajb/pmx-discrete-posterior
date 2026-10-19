"""功能 / Function: 验收CALL-01/02/03调用边界，不审计其他问题 / Audit only caller boundaries CALL-01/02/03.
输入 / Inputs: 当前新包、只读C003输入、内存变体 / Current package, read-only C003 inputs and in-memory variants.
输出 / Outputs: 真实快速计算、异常与CLI证据；由run_tests.py归档 / Fast real computations, errors and CLI evidence archived by run_tests.py.
"""

from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import Mock

import numpy as np
import pytest

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.interfaces.validation import InputError, ModelExecutionError
from pmx_discrete_posterior.models.random_effects import resolve_random_effects
from pmx_discrete_posterior.posterior import computation


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SPEC = importlib.util.spec_from_file_location("audit_callers_reference", ROOT / "examples/reference_models.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def _case(name="OXC", *, zero_dimension=False):
    """功能 / Function: 准备真实模型快速调用 / Prepare a real model for fast calls.
    输入 / Inputs: OXC/TAC及是否固定效应为零 / Model name and whether effects are fixed to zero.
    输出 / Outputs: 显式独立性声明和ID先验；不修改参考文件 / Explicit dependencies and ID-keyed priors, without fixture writes.
    """
    case = reference.load_reference(name)
    model = case["model"]
    model["random_effects_dependence"] = "independent_blocks"
    model["residual_model"]["dependence"] = "conditionally_independent"
    if zero_dimension:
        for level in ("iiv", "iov"):
            if model[level] is not None:
                for row in model[level]["covariance_parameters"]:
                    for key in row:
                        if key is not None:
                            model["parameters"]["values"][key] = 0.0
    case["priors"] = {state["state_id"]: weight for state, weight in
                      zip(case["candidate_states"], (.1, .2, .3, .4), strict=True)}
    case["integration_method"] = "laplace"
    return case


def _rows(result):
    """功能 / Function: 按ID索引真实结果 / Index actual results by ID.
    输入 / Inputs: 后验结果 / Posterior result. 输出 / Outputs: ID到行映射 / ID-to-row mapping.
    """
    return {row["state_id"]: row for row in result["state_results"]}


@pytest.mark.parametrize("name,level", [("OXC", "iiv"), ("TAC", "iiv"), ("TAC", "iov")])
@pytest.mark.parametrize("declaration", ["missing", None, "", "zero_mean_uniform_log_effects", "NORMAL"])
def test_call01_invalid_distribution_rejected_before_dispatch(monkeypatch, name, level, declaration):
    """功能 / Function: 拒绝缺失及不支持的分布 / Reject missing and unsupported distributions.
    输入 / Inputs: IIV/IOV声明内存变体 / In-memory IIV/IOV mutations.
    输出 / Outputs: InputError且零积分调用；mock仅核对边界 / InputError and zero dispatches; mock checks only the boundary.
    """
    case = _case(name)
    if declaration == "missing":
        del case["model"][level]["distribution"]
    else:
        case["model"][level]["distribution"] = declaration
    sentinel = Mock(side_effect=AssertionError("invalid declaration reached integration"))
    monkeypatch.setattr(computation, "integrate", sentinel)
    with pytest.raises(InputError, match="distribution|explicitly declared"):
        compute_posterior(**case)
    sentinel.assert_not_called()
    with pytest.raises(InputError, match="distribution|explicitly declared"):
        reference.evaluate_reference(case, case["candidate_states"][0], np.zeros(1 if name == "OXC" else 10))


@pytest.mark.parametrize("name,dimension", [("OXC", 1), ("TAC", 10)])
def test_call01_supported_normal_model_executes_conditionally(name, dimension):
    """功能 / Function: 验证显式normal仍能真实执行 / Check explicit normal still executes.
    输入 / Inputs: 未改参考模型和零效应点 / Unchanged reference model at zero effects.
    输出 / Outputs: 正态密度、四候选有限浓度及似然；非完整积分 / Normal density and four finite predictions/likelihoods, not full integration.
    """
    case = _case(name)
    model = case["model"]
    distribution = resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
        occasions=case["individual_data"]["occasions"], dependence=model["random_effects_dependence"])
    assert len(distribution.effect_keys) == dimension
    expected_density = -.5 * (dimension * math.log(2 * math.pi) + np.linalg.slogdet(distribution.covariance)[1])
    assert distribution.log_density(np.zeros(dimension)) == pytest.approx(expected_density, abs=1e-12, rel=0)
    for state in case["candidate_states"]:
        predictions, likelihood = reference.evaluate_reference(case, state, np.zeros(dimension))
        assert np.isfinite(predictions["values"]).all() and (predictions["values"] > 0).all()
        assert math.isfinite(likelihood)
        print("NORMAL_CONDITIONAL", name, state["state_id"], predictions["values"].tolist(), likelihood)


@pytest.mark.parametrize("mutation", ["missing_distribution", "unsupported_distribution", "swapped_labels", "swapped_bits", "changed_events"])
@pytest.mark.parametrize("name", ["OXC", "TAC"])
def test_call01_call02_frozen_loader_blocks_mutated_payload(monkeypatch, name, mutation):
    """功能 / Function: 核对冻结导入保护 / Check frozen import protection.
    输入 / Inputs: 内存读取替身，不改真实文件 / In-memory read replacement, never file mutation.
    输出 / Outputs: hash拒绝；仅证明加载边界，不声称语义解析器已实现 / Hash rejection only, not semantic-parser validation.
    """
    if name == "TAC" and "distribution" in mutation:
        # TAC原始schema没有分布字段；注入字段也必须被hash拒绝 / TAC has no such source field; injected fields must fail the hash.
        mutation = "unsupported_distribution"
    path = ROOT / "tests/fixtures" / f"{name}_C003.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    states = raw["candidate_states" if name == "OXC" else "states"]
    if mutation == "missing_distribution":
        del raw["variability"]["distribution"]
    elif mutation == "unsupported_distribution":
        raw.setdefault("variability", {})["distribution"] = "zero_mean_uniform_log_effects"
    elif mutation in ("swapped_labels", "swapped_bits"):
        key = "manuscript_id" if mutation == "swapped_labels" else "machine_bitstring"
        states[1][key], states[2][key] = states[2][key], states[1][key]
    else:
        states[1]["dose_events" if name == "OXC" else "events"][0]["time_h"] = 1.0
    payload = json.dumps(raw).encode("utf-8")
    real_read = Path.read_bytes

    def read_variant(current):
        """功能 / Function: 仅替换目标读取 / Replace only the target read.
        输入 / Inputs: 路径 / Path. 输出 / Outputs: 内存变体或原字节 / Mutated or original bytes.
        """
        return payload if current.resolve() == path.resolve() else real_read(current)

    monkeypatch.setattr(Path, "read_bytes", read_variant)
    with pytest.raises(InputError, match="frozen reference changed"):
        reference.load_reference(name)


@pytest.mark.parametrize("name", ["OXC", "TAC"])
def test_call02_frozen_ids_labels_and_events_are_consistent(name):
    """功能 / Function: 核对参考标签及转换后的事件 / Check reference labels and mapped events.
    输入 / Inputs: 两个只读C003 JSON / Two read-only C003 JSON files.
    输出 / Outputs: 逐ID、逐事件对应证明；不推广到218份旧输入 / Per-ID/event evidence, not a scan of all 218 old inputs.
    """
    raw = json.loads((ROOT / "tests/fixtures" / f"{name}_C003.json").read_text(encoding="utf-8"))
    original = raw["candidate_states" if name == "OXC" else "states"]
    mapped = reference.load_reference(name)["candidate_states"]
    assert len(mapped) == len(original) == 4
    for old, new in zip(original, mapped, strict=True):
        assert old["manuscript_id"] == "omega_" + old["machine_bitstring"][::-1]
        events = old["dose_events" if name == "OXC" else "events"]
        slots = old["nominal_slots_h"] if name == "OXC" else raw["slots"]
        assert [e["nominal_slot_h"] for e in events] == [t for t, bit in zip(slots, old["machine_bitstring"], strict=True) if bit == "1"]
        assert new["state_id"] == old["state_event_id"]
        assert new["events"] == [{"time": e["time_h"], "amount": e["amount_mg"], "amount_unit": "mg",
                                  "route": "oral", "compartment": "depot"} for e in events]
        assert set(new) == {"state_id", "events"}


@pytest.mark.parametrize("key", ["manuscript_id", "machine_bitstring", "label"])
def test_call02_legacy_label_fields_are_not_silently_consumed(monkeypatch, key):
    """功能 / Function: 拒绝未支持的标签字段 / Reject unsupported label fields.
    输入 / Inputs: 显式候选加入错标签 / Explicit candidate with a wrong extra label.
    输出 / Outputs: InputError且不进入积分；仅契约mock / InputError before integration; contract-only mock.
    """
    case = _case()
    case["candidate_states"][1][key] = "omega_10" if key != "machine_bitstring" else "01"
    sentinel = Mock(side_effect=AssertionError("unsupported label reached integration"))
    monkeypatch.setattr(computation, "integrate", sentinel)
    with pytest.raises(InputError, match="unsupported"):
        compute_posterior(**case)
    sentinel.assert_not_called()


def test_call02_real_oxc_zero_dim_order_identity_and_events():
    """功能 / Function: 用真实OXC零维验证ID与事件 / Verify ID/event binding with real zero-dimensional OXC.
    输入 / Inputs: 非均匀先验、重排、改名、事件互换 / Nonuniform priors, reordering, renaming and swapped events.
    输出 / Outputs: 对数似然随事件而非位字符串变化；不替代OXC参考积分 / Likelihood follows events, not bits; not the full OXC reference integral.
    """
    case = _case(zero_dimension=True)
    original = compute_posterior(**case)
    assert original["status"] == "success", original
    base = _rows(original)
    logs = []
    for state in case["candidate_states"]:
        _, likelihood = reference.evaluate_reference(case, state, np.zeros(0))
        logs.append(likelihood)
        assert base[state["state_id"]]["log_marginal_likelihood"] == pytest.approx(likelihood, abs=1e-12, rel=0)
        assert base[state["state_id"]]["method_details"]["dimension"] == 0
    weights = [p * math.exp(value - max(logs)) for p, value in zip((.1, .2, .3, .4), logs, strict=True)]
    expected = [weight / math.fsum(weights) for weight in weights]
    assert [row["posterior_probability"] for row in original["state_results"]] == pytest.approx(expected, abs=1e-12, rel=0)
    reordered = deepcopy(case)
    reordered["candidate_states"].reverse()
    reordered["priors"] = dict(reversed(list(reordered["priors"].items())))
    result = compute_posterior(**reordered)
    assert result["status"] == "success"
    assert [row["state_id"] for row in result["state_results"]] == [s["state_id"] for s in reordered["candidate_states"]]
    for sid, row in _rows(result).items():
        assert row["posterior_probability"] == pytest.approx(base[sid]["posterior_probability"], abs=1e-12, rel=0)
    renamed = deepcopy(case)
    ids = ("omega_11", "omega_10", "omega_01", "arbitrary_candidate")
    for state, new_id in zip(renamed["candidate_states"], ids, strict=True):
        state["state_id"] = new_id
    renamed["priors"] = dict(zip(ids, (.1, .2, .3, .4), strict=True))
    result = compute_posterior(**renamed)
    assert result["status"] == "success"
    assert [r["log_marginal_likelihood"] for r in result["state_results"]] == pytest.approx(logs, abs=1e-12, rel=0)
    swapped = deepcopy(case)
    states = swapped["candidate_states"]
    states[1]["events"], states[2]["events"] = states[2]["events"], states[1]["events"]
    result = compute_posterior(**swapped)
    assert result["status"] == "success"
    assert [r["log_marginal_likelihood"] for r in result["state_results"]] == pytest.approx([logs[0], logs[2], logs[1], logs[3]], abs=1e-12, rel=0)
    print("OPAQUE_ID_CONTRACT", "legacy-looking IDs are accepted as opaque strings; dose events control calculations")


def test_call02_shifted_times_and_additional_candidate_execute_without_bit_inference():
    """功能 / Function: 验证编辑窗口事件不依赖位串 / Check edited candidate events without bit parsing.
    输入 / Inputs: OXC零维，移时剂量和额外候选 / Zero-dimensional OXC with shifted dose and extra candidate.
    输出 / Outputs: 真实有限结果与显式事件似然相同 / Real finite results matching explicit-event likelihoods.
    """
    case = _case(zero_dimension=True)
    baseline = compute_posterior(**case)
    case["candidate_states"][1]["events"][0]["time"] = 1.5
    extra = deepcopy(case["candidate_states"][1])
    extra["state_id"] = "extra_half_dose"
    extra["events"][0]["amount"] = 150.0
    case["candidate_states"].append(extra)
    case["priors"] = {s["state_id"]: .2 for s in case["candidate_states"]}
    result = compute_posterior(**case)
    assert result["status"] == "success", result
    assert len(result["state_results"]) == 5
    assert result["state_results"][1]["log_marginal_likelihood"] != baseline["state_results"][1]["log_marginal_likelihood"]
    for state, row in zip(case["candidate_states"], result["state_results"], strict=True):
        _, likelihood = reference.evaluate_reference(case, state, np.zeros(0))
        assert row["state_id"] == state["state_id"]
        assert row["log_marginal_likelihood"] == pytest.approx(likelihood, abs=1e-12, rel=0)


@pytest.mark.parametrize("problem", ["negative_sd", "missing_residual", "missing_events"])
def test_call03_static_input_error_is_not_numerical_failure(monkeypatch, problem):
    """功能 / Function: 非法输入不得伪装为数值失败 / Keep invalid input separate from numerical failure.
    输入 / Inputs: 负SD或缺字段 / Negative SD or missing fields.
    输出 / Outputs: InputError且零积分调用；仅分发边界mock / InputError and no integration; dispatch-only mock.
    """
    case = _case()
    if problem == "negative_sd":
        case["model"]["parameters"]["values"]["sigma"] = -1.0
    elif problem == "missing_residual":
        del case["model"]["residual_model"]
    else:
        del case["candidate_states"][0]["events"]
    sentinel = Mock(side_effect=AssertionError("static error reached integration"))
    monkeypatch.setattr(computation, "integrate", sentinel)
    with pytest.raises(InputError):
        compute_posterior(**case)
    sentinel.assert_not_called()


@pytest.mark.parametrize("callback", ["predict", "relations"])
def test_call03_callback_exception_is_model_execution_error(callback):
    """功能 / Function: 核对模型异常类型及传播 / Check model exception type and propagation.
    输入 / Inputs: 零维真实模型及故障回调 / Real zero-dimensional model with a fault-injected callback.
    输出 / Outputs: 底层ModelExecutionError，上层failed且保留原因；不证明数值正确性 / Typed lower-level error and failed upper result; no numerical-validity claim.
    """
    case = _case(zero_dimension=True)

    def fail(**kwargs):
        """功能 / Function: 注入可识别运行异常 / Inject an identifiable runtime error.
        输入 / Inputs: 回调参数 / Callback arguments. 输出 / Outputs: RuntimeError / RuntimeError.
        """
        raise RuntimeError("CALL03_RUNTIME_SENTINEL")

    if callback == "predict":
        case["model"]["predict"] = fail
    else:
        case["model"]["parameter_model"]["relations"] = fail
    with pytest.raises(ModelExecutionError, match="CALL03_RUNTIME_SENTINEL") as caught:
        reference.evaluate_reference(case, case["candidate_states"][0], np.zeros(0))
    assert isinstance(caught.value.__cause__, RuntimeError)
    result = compute_posterior(**case)
    assert result["status"] == "failed"
    assert len(result["state_results"]) == 4
    for row in result["state_results"]:
        assert row["posterior_probability"] is None and row["log_marginal_likelihood"] is None
        assert "ModelExecutionError" in row["failure_reason"] and "CALL03_RUNTIME_SENTINEL" in row["failure_reason"]
    print("MODEL_FAILURE", json.dumps(result, allow_nan=False))


def test_call03_real_numerical_budget_rejection_is_not_input_or_callback_error():
    """功能 / Function: 核对数值拒绝的独立原因 / Check the distinct numerical rejection reason.
    输入 / Inputs: 合法OXC及不足节点预算 / Valid OXC and insufficient node budget.
    输出 / Outputs: node_budget失败、全行保留、无后验；不审计GHQ准确性 / Budget failure, retained rows and no posterior; not a GHQ accuracy audit.
    """
    case = _case()
    case["integration_method"] = "ghq"
    case["numerical_options"] = {"ghq": {"variant": "prior_centered", "orders": [3, 5, 7],
        "log_integral_atol": 1e-6, "posterior_atol": 1e-6, "max_nodes": 1}}
    result = compute_posterior(**case)
    assert result["status"] == "failed"
    assert len(result["state_results"]) == 4
    for row in result["state_results"]:
        assert row["method_details"]["stage"] == "node_budget"
        assert row["posterior_probability"] is None and row["log_marginal_likelihood"] is None
        assert "InputError" not in row["failure_reason"] and "ModelExecutionError" not in row["failure_reason"]
    print("NUMERICAL_REJECTION", json.dumps(result, allow_nan=False))


def _run_python(arguments):
    """功能 / Function: 用指定Python启动真实子进程 / Start a real process with the specified Python.
    输入 / Inputs: 参数列表 / Argument list. 输出 / Outputs: 完整stdout/stderr/退出码，均由测试历史收集 / Captured streams/status preserved in test history.
    """
    command = [str(PYTHON), "-B", *arguments]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "examples"),
             "PYTHONIOENCODING": "utf-8"})
    print("CLI_RECEIPT", json.dumps({"command": command, "cwd": str(ROOT), "returncode": run.returncode,
                                    "stdout": run.stdout, "stderr": run.stderr}, ensure_ascii=True))
    return run


@pytest.mark.parametrize("arguments", [["UNKNOWN"], ["OXC", "--method", "ghq"], ["OXC", "--max-nodes", "1"]])
def test_call03_cli_argument_errors_are_nonzero(arguments):
    """功能 / Function: 核对现有argparse约定 / Check the current argparse convention.
    输入 / Inputs: 非法模型名或参数组合 / Invalid name or option combination.
    输出 / Outputs: 退出2及usage，无数值结果 / Exit 2 and usage, without numerical results.
    """
    run = _run_python(["examples/posterior_reference.py", *arguments])
    assert run.returncode == 2
    assert "usage:" in run.stderr and run.stdout == ""


def test_call03_cli_rejects_invalid_numeric_input_without_claiming_numeric_result():
    """功能 / Function: 记录新CLI对语义输入错误的实际约定 / Record the new CLI's semantic-input-error behavior.
    输入 / Inputs: 所有字段齐全但负容差 / Complete flags with a negative tolerance.
    输出 / Outputs: 非零和InputError，无failed结果JSON；不强加旧退出2 / Nonzero with InputError and no failed-result JSON; no legacy exit-2 requirement.
    """
    run = _run_python(["examples/posterior_reference.py", "OXC", "--method", "ghq", "--ghq-orders", "3", "5", "7",
        "--log-integral-atol", "-1", "--posterior-atol", "1e-6", "--max-nodes", "1"])
    assert run.returncode != 0
    assert "InputError" in run.stderr and run.stdout == ""


def test_call03_cli_real_numerical_rejection_exits_one():
    """功能 / Function: 真实CLI数值拒绝不能成功退出 / Real CLI numerical rejection must not exit successfully.
    输入 / Inputs: 合法OXC及不足节点预算 / Valid OXC with insufficient budget.
    输出 / Outputs: 退出1和含node_budget的failed JSON / Exit 1 and failed JSON with node_budget.
    """
    run = _run_python(["examples/posterior_reference.py", "OXC", "--method", "ghq", "--ghq-orders", "3", "5", "7",
        "--log-integral-atol", "1e-6", "--posterior-atol", "1e-6", "--max-nodes", "1"])
    assert run.returncode == 1 and run.stderr == ""
    result = json.loads(run.stdout)
    assert result["status"] == "failed"
    expected = {s["state_id"]: s["events"] for s in reference.load_reference("OXC")["candidate_states"]}
    for row in result["state_results"]:
        assert row["posterior_probability"] is None
        assert row["method_details"]["stage"] == "node_budget"
        assert row["events"] == expected[row["state_id"]]


@pytest.mark.parametrize("scenario", ["success_reordered", "negative_sd", "missing_field", "runtime_error"])
def test_call02_call03_cli_with_in_memory_loader_boundary(scenario):
    """功能 / Function: 执行真实示例入口的可编辑对象边界 / Execute the real example entry at its editable-object boundary.
    输入 / Inputs: runpy加内存loader注入，OXC零维或故障 / runpy with an injected loader, zero-dimensional OXC or fault.
    输出 / Outputs: 真实计算/序列化/退出证据；不声称CLI支持自定义文件或完整参考积分 / Real computation/serialization/exits, not custom-file CLI support or full reference integration.
    """
    script = textwrap.dedent(f'''\
        import runpy
        import sys
        import reference_models

        real_load = reference_models.load_reference

        def loaded(name):
            """功能 / Function: 内存变体 / In-memory variant.
            输入 / Inputs: 名称 / Name. 输出 / Outputs: 真实模型变体 / Real model variant.
            """
            case = real_load(name)
            for row in case["model"]["iiv"]["covariance_parameters"]:
                for key in row:
                    if key is not None:
                        case["model"]["parameters"]["values"][key] = 0.0
            scenario = {scenario!r}
            if scenario == "negative_sd":
                case["model"]["parameters"]["values"]["sigma"] = -1.0
            elif scenario == "missing_field":
                del case["model"]["residual_model"]
            elif scenario == "runtime_error":
                def fail(**kwargs):
                    """功能 / Function: 故障注入 / Fault injection.
                    输入 / Inputs: 回调参数 / Callback args. 输出 / Outputs: RuntimeError.
                    """
                    raise RuntimeError("CALL03_CLI_RUNTIME_SENTINEL")
                case["model"]["predict"] = fail
            else:
                case["candidate_states"].reverse()
            return case

        reference_models.load_reference = loaded
        sys.argv = ["posterior_reference.py", "OXC"]
        runpy.run_path("examples/posterior_reference.py", run_name="__main__")
    ''')
    run = _run_python(["-c", script])
    if scenario in ("negative_sd", "missing_field"):
        assert run.returncode != 0 and "InputError" in run.stderr and run.stdout == ""
        return
    result = json.loads(run.stdout)
    assert run.stderr == ""
    if scenario == "runtime_error":
        assert run.returncode == 1 and result["status"] == "failed"
        assert all("ModelExecutionError" in r["failure_reason"] and r["posterior_probability"] is None for r in result["state_results"])
    else:
        assert run.returncode == 0 and result["status"] == "success"
        case = _case(zero_dimension=True)
        case["priors"] = {s["state_id"]: .25 for s in case["candidate_states"]}
        expected_result = compute_posterior(**case)
        assert expected_result["status"] == "success"
        expected = _rows(expected_result)
        states = list(reversed(case["candidate_states"]))
        for state, row in zip(states, result["state_results"], strict=True):
            assert row["state_id"] == state["state_id"] and row["events"] == state["events"]
            assert row["posterior_probability"] == pytest.approx(expected[row["state_id"]]["posterior_probability"], abs=1e-12, rel=0)


def test_call02_reference_models_cli_runs_real_fast_examples():
    """功能 / Function: 运行OXC/TAC快速条件示例 / Run fast OXC/TAC conditional examples.
    输入 / Inputs: 未修改reference_models.py / Unmodified reference_models.py.
    输出 / Outputs: 真实预测与ID顺序；不进行TAC积分 / Real predictions and ID order without TAC integration.
    """
    run = _run_python(["examples/reference_models.py"])
    assert run.returncode == 0 and run.stderr == ""
    result = json.loads(run.stdout)
    assert set(result) == {"OXC", "TAC"}
    for name, dimension in (("OXC", 1), ("TAC", 10)):
        case = reference.load_reference(name)
        assert len(result[name]) == 4
        for state, row in zip(case["candidate_states"], result[name], strict=True):
            predictions, likelihood = reference.evaluate_reference(case, state, np.zeros(dimension))
            assert row["state_id"] == state["state_id"]
            assert row["predictions"] == pytest.approx(predictions["values"].tolist(), abs=1e-12, rel=0)
            assert row["conditional_log_likelihood"] == pytest.approx(likelihood, abs=1e-12, rel=0)
