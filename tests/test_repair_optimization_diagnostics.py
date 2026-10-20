"""功能 / Function: 核验优化失败证据而不改变接纳规则 / Check failure evidence without changing acceptance.
输入 / Inputs: 可控SciPy与求导结果 / Controlled optimizer and derivative outputs.
输出 / Outputs: 返回点、试探点、最佳点及失败原因断言 / Returned, trial, best-point and failure assertions.
"""

import json

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pmx_discrete_posterior.numerics import optimization


def test_precision_loss_preserves_returned_point_and_centered_value(monkeypatch):
    """功能 / Function: 保留拒绝点并区分中心化值 / Preserve rejected point and distinguish centered value.
    输入 / Inputs: 精度损失返回 / Precision-loss return. 输出 / Outputs: 原失败状态及诊断 / Failure and diagnostics.
    """
    def backend(fun, point, **kwargs):
        fun(np.array([1.]))
        fun(np.array([3.]))
        return OptimizeResult(x=np.array([1.]), fun=-4., jac=np.array([2e-7]),
                              success=False, status=2, message="precision loss", nit=2, nfev=3, njev=1)
    monkeypatch.setattr(optimization, "scipy_minimize", backend)
    result = optimization.minimize(lambda x: 5 + (x[0]-1)**2, [3.])
    details = result["method_details"]
    assert result["status"] == "failed"
    assert result["minimizer"] is None and result["objective_value"] is None
    assert details["backend_result"]["point"] == [1.]
    assert details["backend_result"]["centered_objective_value"] == -4.
    assert details["backend_result"]["objective_value"] == 5.
    assert details["backend_result"]["status"] == 2
    assert details["last_objective_evaluation"] == {"point": [3.], "value": 9.}
    assert details["best_objective_evaluation"] == {"point": [1.], "value": 5.}


def test_gradient_failure_keeps_trial_and_derivative_diagnostics(monkeypatch):
    """功能 / Function: 区分梯度试探点与优化返回点 / Distinguish derivative trial from a solver result.
    输入 / Inputs: 后端内部求导失败 / Inner derivative failure.
    输出 / Outputs: 无假返回点，保留原始求导检查 / No invented return point; retained derivative checks.
    """
    def gradient(objective, point, options=None):
        return {"status": "failed", "gradient": None, "estimated_error": np.array([4e-6]),
                "failure_reason": "roundoff gate", "method_details": {"nfev": 12, "roundoff_screen": 4e-6}}
    def backend(fun, point, jac, **kwargs):
        fun(np.array([.1]))
        jac(np.array([.2]))
        raise AssertionError("gradient failure must abort")
    monkeypatch.setattr(optimization, "scipy_minimize", backend)
    monkeypatch.setattr(optimization, "checked_gradient", gradient)
    result = optimization.minimize(lambda x: x @ x, [0.])
    details = result["method_details"]
    assert result["status"] == "failed"
    assert "backend_result" not in details
    assert details["gradient_failure"]["point"] == [.2]
    assert details["gradient_failure"]["phase"] == "search"
    assert details["gradient_failure"]["result"]["method_details"]["roundoff_screen"] == 4e-6
    assert details["last_objective_evaluation"]["point"] == [.1]
    assert details["best_objective_evaluation"]["point"] == [0.]
    assert json.loads(json.dumps(details))["gradient_failure"]["result"]["estimated_error"] == [4e-6]


def test_false_success_keeps_rejected_point_without_promoting_it(monkeypatch):
    """功能 / Function: 独立验收失败仍保留后端返回 / Retain a backend return rejected by independent verification.
    输入 / Inputs: 误导性零梯度 / Misleading zero gradient. 输出 / Outputs: 明确失败 / Explicit failure.
    """
    result = optimization.minimize(lambda x: (x[0]-6)**2, [5.], gradient=lambda x: np.zeros(1))
    assert result["status"] == "failed"
    assert result["method_details"]["backend_result"]["point"] == [5.]
    assert result["method_details"]["backend_result"]["objective_value"] == pytest.approx(1.)
    assert result["minimizer"] is None


def test_bad_initial_evaluation_does_not_invent_a_finite_candidate():
    """功能 / Function: 初始评价错误不产生候选解 / Invalid initial evaluation produces no finite candidate.
    输入 / Inputs: NaN目标 / NaN objective. 输出 / Outputs: 无最佳有限点 / No best finite point.
    """
    result = optimization.minimize(lambda x: np.nan, [0.])
    details = result["method_details"]
    assert result["status"] == "failed"
    assert "backend_result" not in details
    assert "best_objective_evaluation" not in details
    assert details["last_objective_request"] == [0.]


@pytest.mark.parametrize("kind", ["exception", "wrong_shape", "mutating"])
def test_callback_gradient_failure_is_retained(kind):
    """功能 / Function: 记录梯度回调失败 / Retain gradient callback failures.
    输入 / Inputs: 异常、错误形状或修改输入 / Exception, wrong shape or mutation.
    输出 / Outputs: 失败原因、阶段和坐标 / Failure reason, phase and point.
    """
    def gradient(point):
        if kind == "exception":
            raise RuntimeError("callback fault")
        if kind == "wrong_shape":
            return [0., 0.]
        point[0] += 1
        return [0.]
    result = optimization.minimize(lambda x: x@x, [2.], gradient=gradient)
    assert result["status"] == "failed"
    record = result["method_details"]["gradient_failure"]
    assert record["point"] == [2.] and record["source"] == "callback"
    assert record["phase"] == "search"
    assert record["failure_reason"]
    assert "backend_result" not in result["method_details"]
    json.dumps(result)


def test_independent_gradient_failure_preserves_isolated_serializable_evidence(monkeypatch):
    """功能 / Function: 归档独立验收失败且不共享数组 / Archive independent-check failure without shared arrays.
    输入 / Inputs: 可修改的求导结果 / Mutable derivative result.
    输出 / Outputs: 不随后续修改改变的证据 / Evidence unaffected by later mutation.
    """
    derivative = {"status": "failed", "gradient": None, "estimated_error": np.array([4e-6]),
                  "failure_reason": "unresolved", "method_details": {"nfev": 3, "backend_status": [[-2]]}}
    monkeypatch.setattr(optimization, "checked_gradient", lambda *args: derivative)
    result = optimization.minimize(lambda x: x@x, [0.], gradient=lambda x: np.zeros(1))
    record = result["method_details"]["gradient_failure"]
    derivative["estimated_error"][:] = 100
    derivative["method_details"]["backend_status"][0][0] = 100
    assert result["status"] == "failed" and result["minimizer"] is None
    assert record["phase"] == "stationarity" and record["point"] == [0.]
    assert record["result"]["estimated_error"] == [4e-6]
    assert record["result"]["method_details"]["backend_status"] == [[-2]]
    json.dumps(result)


def test_derivative_record_retains_gradient_without_array_aliasing():
    """功能 / Function: 序列化有数值梯度的诊断 / Serialize diagnostics containing a numerical gradient.
    输入 / Inputs: 求导数组 / Derivative arrays. 输出 / Outputs: 独立列表 / Independent lists.
    """
    original = {"gradient": np.array([1., 2.]), "estimated_error": np.array([.1, .2])}
    record = optimization._derivative_record(original)
    original["gradient"][:] = 10
    assert record == {"gradient": [1., 2.], "estimated_error": [.1, .2]}
    json.dumps(record)


def test_derivative_probes_do_not_replace_primary_best_and_counts_are_preserved(monkeypatch):
    """功能 / Function: 隔离主评价记录和求导探针并核对次数 / Separate primary records from probes and verify counts.
    输入 / Inputs: 探针碰到更低目标值 / Probe with a lower objective.
    输出 / Outputs: 最佳主点不变，评价数相加正确 / Unchanged best primary point and correct evaluation totals.
    """
    calls = []
    def objective(point):
        calls.append(point.tolist())
        return (point[0]-100)**2
    def gradient(callback, point, options=None):
        callback(np.array([100.]))
        return {"status": "success", "gradient": np.array([-196.]), "estimated_error": np.array([0.]),
                "failure_reason": None, "method_details": {"nfev": 1}}
    def backend(fun, point, jac, **kwargs):
        value = fun(np.array([2.]))
        derivative = jac(np.array([2.]))
        return OptimizeResult(x=np.array([2.]), fun=value, jac=derivative, success=False, status=2,
                              message="test stop", nit=1)
    monkeypatch.setattr(optimization, "checked_gradient", gradient)
    monkeypatch.setattr(optimization, "scipy_minimize", backend)
    result = optimization.minimize(objective, [0.])
    details = result["method_details"]
    assert details["best_objective_evaluation"] == {"point": [2.], "value": 9604.}
    assert details["objective_nfev"] == 3
    assert details["derivative_objective_nfev"] == 1
    assert details["nfev"] == len(calls) == 4
