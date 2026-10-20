"""功能 / Function: 核验受精度限制导数的诊断表示 / Verify precision-limited derivative diagnostics.
输入 / Inputs: 可控SciPy返回和真实异常 / Controlled backend results and real errors.
输出 / Outputs: 候选诊断保留但不伪装成已通过梯度 / Candidate metadata without claiming a passed gradient.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pmx_discrete_posterior.numerics import derivatives


def backend(monkeypatch, status, error=1.0, value=2.0):
    """功能 / Function: 提供有限节点与可控后端状态 / Supply finite probes and controlled status.
    输入 / Inputs: patcher、状态及估计 / Status and estimates. 输出 / Outputs: 替身 / Test backend.
    """
    def run(function, point, **kwargs):
        function(np.array([[point[0] - .1, point[0] + .1]]))
        return SimpleNamespace(df=np.array([value]), error=np.array([error]),
                               success=np.array([status == 0]), status=np.array([status]))
    monkeypatch.setattr(derivatives, "jacobian", run)


@pytest.mark.parametrize("status", [-1, -2, 0])
def test_precision_limited_finite_estimate_is_diagnostic_not_success(monkeypatch, status):
    backend(monkeypatch, status)
    result = derivatives.gradient(lambda x: x[0]**2, np.array([1.]))
    assert result["status"] == "failed"
    assert result["gradient"] is None
    candidate = result["method_details"]["precision_limited_candidate"]
    assert candidate["gradient"] == [2.]
    assert candidate["estimated_error"] == [1.]


@pytest.mark.parametrize("status,value,error", [(-3, 2., 1.), (-2, np.nan, 1.), (-2, 2., np.inf)])
def test_nonfinite_or_invalid_backend_cannot_supply_candidate(monkeypatch, status, value, error):
    backend(monkeypatch, status, value=value, error=error)
    result = derivatives.gradient(lambda x: x[0]**2, np.array([1.]))
    assert result["status"] == "failed" and result["gradient"] is None
    assert "precision_limited_candidate" not in result["method_details"]


def test_callback_failure_cannot_supply_candidate():
    def broken(x):
        raise ValueError("model failure")
    result = derivatives.gradient(broken, np.array([1.]))
    assert result["status"] == "failed"
    assert "precision_limited_candidate" not in result["method_details"]


def test_ordinary_success_does_not_need_recovery_metadata():
    result = derivatives.gradient(lambda x: x[0]**2, np.array([1.]))
    assert result["status"] == "success"
    assert "precision_limited_candidate" not in result["method_details"]
