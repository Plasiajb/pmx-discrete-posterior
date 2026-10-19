"""功能 / Function: 独立复测CALL-03最小修复 / Independently test the minimal CALL-03 repair.
输入 / Inputs: 真实示例入口和内存OXC变体 / Real example entry and in-memory OXC variants.
输出 / Outputs: 类型化拒绝、执行计数和快速正向证据 / Typed rejection, execution counts and fast positive evidence.
"""

from copy import deepcopy
import importlib.util
import json
import sys
import textwrap
from unittest.mock import Mock

import pytest

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.posterior import computation
from test_audit_acceptance_callers import ROOT, _case, _run_python, reference


MISSING = object()


@pytest.fixture
def example(monkeypatch):
    """功能 / Function: 载入真实示例且隔离命令行 / Load the real example with isolated CLI arguments.
    输入 / Inputs: pytest补丁作用域 / Pytest patch scope.
    输出 / Outputs: 未替换校验或计算的示例模块 / Example module with real validation and computation.
    """
    monkeypatch.setitem(sys.modules, "reference_models", reference)
    spec = importlib.util.spec_from_file_location("call03_example", ROOT / "examples/posterior_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["posterior_reference.py", "OXC"])
    return module


@pytest.mark.parametrize("path,value,message,posterior_calls", [
    pytest.param(("model",), MISSING, "model must be a mapping", 0, id="model_missing"),
    pytest.param(("model",), None, "model must be a mapping", 0, id="model_none"),
    pytest.param(("model",), [], "model must be a mapping", 0, id="model_nonmapping"),
    pytest.param(("model", "residual_model"), MISSING, "missing=.*residual_model", 0, id="residual_missing"),
    pytest.param(("model", "residual_model"), None, "residual_model must be a mapping", 0, id="residual_none"),
    pytest.param(("model", "residual_model"), [], "residual_model must be a mapping", 0, id="residual_nonmapping"),
    pytest.param(("candidate_states",), MISSING, "candidate_states must be a nonempty", 0, id="candidates_missing"),
    pytest.param(("candidate_states",), [], "candidate_states must be a nonempty", 0, id="candidates_empty"),
    pytest.param(("candidate_states",), {}, "candidate_states must be a nonempty", 0, id="candidates_nonsequence"),
    pytest.param(("candidate_states", 0, "state_id"), MISSING, "missing=.*state_id", 0, id="state_id_missing"),
    pytest.param(("candidate_states", 0, "state_id"), "", "identifier must be a nonempty", 0, id="state_id_empty"),
    pytest.param(("candidate_states", 0), None, "candidate_state must be a mapping", 0, id="candidate_nonmapping"),
    pytest.param(("candidate_states", 0, "events"), None, "candidate events must be a sequence", 0, id="events_invalid"),
    pytest.param(("model", "random_effects_dependence"), "correlated_blocks", "model.random_effects_dependence", 0,
                 id="random_dependence_unsupported"),
    pytest.param(("model", "random_effects_dependence"), None, "model.random_effects_dependence", 0,
                 id="random_dependence_none"),
    pytest.param(("model", "residual_model", "dependence"), "correlated", "residual_model.dependence", 0,
                 id="residual_dependence_unsupported"),
    pytest.param(("model", "residual_model", "dependence"), None, "residual_model.dependence", 0,
                 id="residual_dependence_none"),
    pytest.param(("observations",), MISSING, "records must be a nonempty", 1, id="observations_missing"),
    pytest.param(("individual_data",), MISSING, "individual_data must be a mapping", 1, id="individual_data_missing"),
])
def test_invalid_loader_fields_raise_input_error_before_execution(
    example, monkeypatch, capsys, path, value, message, posterior_calls
):
    """功能 / Function: 检查输入错误不进入执行 / Keep input errors outside model execution.
    输入 / Inputs: 单字段变体及真实main / Single-field mutation and real main.
    输出 / Outputs: InputError、零模型/积分调用且无结果JSON / InputError, zero model/integration calls and no result JSON.
    """
    case = _case(zero_dimension=True)
    relations = Mock(side_effect=AssertionError("CALL03_RELATIONS_EXECUTED"))
    predict = Mock(side_effect=AssertionError("CALL03_PREDICT_EXECUTED"))
    case["model"]["parameter_model"]["relations"] = relations
    case["model"]["predict"] = predict
    target = case
    for key in path[:-1]:
        target = target[key]
    if value is MISSING:
        del target[path[-1]]
    else:
        target[path[-1]] = deepcopy(value)

    loader = Mock(return_value=case)
    posterior = Mock(wraps=example.compute_posterior)
    integration = Mock(side_effect=AssertionError("CALL03_INTEGRATION_EXECUTED"))
    monkeypatch.setattr(example, "load_reference", loader)
    monkeypatch.setattr(example, "compute_posterior", posterior)
    monkeypatch.setattr(computation, "integrate", integration)
    with pytest.raises(InputError, match=message) as caught:
        example.main()
    assert type(caught.value) is InputError
    loader.assert_called_once_with("OXC")
    assert posterior.call_count == posterior_calls
    relations.assert_not_called()
    predict.assert_not_called()
    integration.assert_not_called()
    if path[-1] in ("dependence", "random_effects_dependence"):
        assert target[path[-1]] == value
    output = capsys.readouterr()
    assert output.out == output.err == ""
    print("CALL03_REJECTION", path, str(caught.value), "posterior_calls", posterior.call_count,
          "relations/predict/integration", 0, 0, 0)


@pytest.mark.parametrize("declared", [False, True], ids=["reference_settings_added_to_copy", "explicit_settings_preserved"])
def test_fast_reference_is_unchanged_and_matches_direct_entry(example, monkeypatch, capsys, declared):
    """功能 / Function: 核对副本及真实调用一致性 / Check copy ownership and real entry equivalence.
    输入 / Inputs: 四候选零维OXC，缺省或已声明的参考独立性 / Four-state zero-dimensional OXC with absent or explicit reference settings.
    输出 / Outputs: 原对象不变、等先验和逐ID后验一致；不替代参考积分 / Unchanged source, equal priors and per-ID parity; not full reference integration.
    """
    case = _case(zero_dimension=True)
    case["candidate_states"].reverse()
    case["priors"] = {state["state_id"]: .25 for state in case["candidate_states"]}
    expected = compute_posterior(**deepcopy(case))
    assert expected["status"] == "success"
    if not declared:
        del case["model"]["random_effects_dependence"]
        del case["model"]["residual_model"]["dependence"]
    # 回调按身份保留，数据深复制；另查预测器状态 / Preserve callback identities, copy data, and check predictor state separately.
    callbacks = (case["model"]["predict"], case["model"]["parameter_model"]["relations"])
    before = deepcopy(case, {id(callback): callback for callback in callbacks})
    predictor_state = deepcopy(vars(callbacks[0]))
    assert case == before
    posterior = Mock(wraps=example.compute_posterior)
    monkeypatch.setattr(example, "load_reference", Mock(return_value=case))
    monkeypatch.setattr(example, "compute_posterior", posterior)
    assert example.main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    result = json.loads(output.out)
    assert case == before
    assert vars(case["model"]["predict"]) == predictor_state
    posterior.assert_called_once()
    supplied = posterior.call_args.kwargs
    assert supplied["model"] is not case["model"]
    assert supplied["model"]["residual_model"] is not case["model"]["residual_model"]
    assert supplied["candidate_states"] is not case["candidate_states"]
    assert supplied["model"]["predict"] is case["model"]["predict"]
    assert supplied["model"]["parameter_model"]["relations"] is case["model"]["parameter_model"]["relations"]
    assert supplied["individual_data"] is case["individual_data"]
    assert supplied["observations"] is case["observations"]
    assert result["status"] == "success" and result["model"] == "OXC"
    assert len(result["state_results"]) == 4
    for state, row, direct in zip(case["candidate_states"], result["state_results"], expected["state_results"], strict=True):
        assert row["state_id"] == direct["state_id"] == state["state_id"]
        assert row["events"] == state["events"]
        assert row["prior_probability"] == .25
        assert row["status"] == "success"
        assert direct["method_details"]["dimension"] == 0
        for key in ("log_marginal_likelihood", "posterior_probability"):
            assert row[key] == pytest.approx(direct[key], abs=1e-12, rel=0)
    print("CALL03_FAST_POSITIVE", json.dumps(result, allow_nan=False), "source_unchanged", True)


@pytest.mark.parametrize("fault", ["missing_state_id", "unsupported_dependence"])
def test_real_process_keeps_input_error_and_nonzero_exit(fault):
    """功能 / Function: 独立核对真实进程失败契约 / Independently check the real process failure contract.
    输入 / Inputs: 仅内存loader故障注入 / In-memory loader fault injection only.
    输出 / Outputs: 非零、InputError及零执行计数；不要求旧退出码2 / Nonzero, InputError and zero execution counts; no legacy exit-2 requirement.
    """
    script = textwrap.dedent(f'''\
        import json
        import runpy
        import sys
        from unittest.mock import Mock
        import reference_models
        from pmx_discrete_posterior.posterior import computation

        case = reference_models.load_reference("OXC")
        relations = Mock(side_effect=AssertionError("CALL03_RELATIONS_EXECUTED"))
        predict = Mock(side_effect=AssertionError("CALL03_PREDICT_EXECUTED"))
        integration = Mock(side_effect=AssertionError("CALL03_INTEGRATION_EXECUTED"))
        case["model"]["parameter_model"]["relations"] = relations
        case["model"]["predict"] = predict
        computation.integrate = integration
        if {fault!r} == "missing_state_id":
            del case["candidate_states"][0]["state_id"]
        else:
            case["model"]["residual_model"]["dependence"] = "correlated"
        reference_models.load_reference = Mock(return_value=case)
        sys.argv = ["posterior_reference.py", "OXC"]
        try:
            runpy.run_path("examples/posterior_reference.py", run_name="__main__")
        finally:
            print("CALL03_EXECUTION_COUNTS", json.dumps([relations.call_count, predict.call_count, integration.call_count]),
                  file=sys.stderr)
    ''')
    run = _run_python(["-c", script])
    assert run.returncode != 0 and run.stdout == ""
    assert "pmx_discrete_posterior.interfaces.validation.InputError:" in run.stderr
    assert "CALL03_EXECUTION_COUNTS [0, 0, 0]" in run.stderr
    assert "CALL03_RELATIONS_EXECUTED" not in run.stderr
    assert "CALL03_PREDICT_EXECUTED" not in run.stderr
    assert "CALL03_INTEGRATION_EXECUTED" not in run.stderr
