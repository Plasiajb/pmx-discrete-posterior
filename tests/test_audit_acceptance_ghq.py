"""功能 / Function: 验收NC-03完整失败传播 / Accept NC-03 failure propagation end to end.
输入 / Inputs: 解析模型、真实GHQ及故意失败预测 / Analytic model, real GHQ and deliberately failing predictions.
输出 / Outputs: 正常积分核对或明确失败断言 / Checks of valid integrals or explicit failure.
"""

import math

import numpy as np
import pytest

from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics.ghq import integrate
from test_ghq_options import ghq_options
from test_posterior_computation import analytic_inputs, analytic_logs


def ghq_arguments():
    """功能 / Function: 配置已知解析例 / Configure a known analytic example.
    输入 / Inputs: 无 / None. 输出 / Outputs: 含中心节点的明确GHQ配置 / Explicit GHQ settings with a central node.
    """
    args = analytic_inputs()
    args["integration_method"] = "ghq"
    args["numerical_options"] = ghq_options((65, 97, 129))
    return args


def test_nc03_real_full_posterior_positive_control():
    """功能 / Function: 核验真实全链路正向对照 / Check the real end-to-end positive control.
    输入 / Inputs: 解析二状态例 / Analytic two-state example.
    输出 / Outputs: 绝对积分与完整后验一致 / Matching absolute integrals and full posterior.
    """
    result = compute_posterior(**ghq_arguments())
    assert result["status"] == "success", result
    np.testing.assert_allclose(
        [row["log_marginal_likelihood"] for row in result["state_results"]],
        analytic_logs(), atol=1e-10, rtol=0,
    )
    np.testing.assert_allclose(
        [row["posterior_probability"] for row in result["state_results"]],
        [.4756947064736979, .5243052935263021], atol=1e-10, rtol=0,
    )


@pytest.mark.parametrize("scope", ["central_node", "one_state", "all_states"])
@pytest.mark.parametrize("failure", ["nan", "positive_infinity", "exception"])
def test_nc03_prediction_failure_never_becomes_a_normalized_zero(scope, failure):
    """功能 / Function: 检查节点失败直达整体失败 / Check propagation from a failed node to overall failure.
    输入 / Inputs: 单中心节点、单候选或全部候选的非法预测 / Invalid prediction at one central node, one state or all states.
    输出 / Outputs: 保留候选与原因，不部分归一化 / Preserved candidates and reasons, without partial normalization.
    """
    args = ghq_arguments()
    original = args["model"]["predict"]
    visits = []

    def failing_predict(**kwargs):
        """功能 / Function: 在真实预测边界注入错误 / Inject a fault at the real prediction boundary.
        输入 / Inputs: 已解析参数与候选 / Resolved parameters and candidate.
        输出 / Outputs: 原预测或测试指定错误 / Original prediction or the specified test fault.
        """
        state_id = kwargs["candidate_state"]["state_id"]
        effect = kwargs["individual_parameters"]["segments"][0]["values"]["effect"]
        broken = scope == "all_states" or (
            state_id == "a" and (scope == "one_state" or abs(effect) < 1e-14)
        )
        visits.append((state_id, broken))
        if broken and failure == "exception":
            raise RuntimeError("NC03 deliberate prediction failure")
        predictions = original(**kwargs)
        if broken:
            predictions["values"][0] = math.nan if failure == "nan" else math.inf
        return predictions

    args["model"]["predict"] = failing_predict
    result = compute_posterior(**args)
    assert result["status"] == "failed", result
    assert "not normalized" in result["failure_reason"]
    assert [row["state_id"] for row in result["state_results"]] == ["a", "b"]
    assert all(row["posterior_probability"] is None for row in result["state_results"])
    a, b = result["state_results"]
    assert a["status"] == "failed" and a["log_marginal_likelihood"] is None
    assert a["failure_reason"]
    if scope == "all_states":
        assert b["status"] == "failed" and b["log_marginal_likelihood"] is None
    else:
        assert b["status"] == "success"
        assert b["log_marginal_likelihood"] == pytest.approx(analytic_logs()[1], abs=1e-10)
    assert any(broken for _, broken in visits)
    if scope == "central_node":
        assert any(sid == "a" and not broken for sid, broken in visits)


def test_nc03_legitimate_zero_density_is_not_a_failed_node():
    """功能 / Function: 区分合法负无穷与非法节点 / Distinguish valid negative infinity from a failed node.
    输入 / Inputs: E[2 I(eta>0)]解析目标及偶数阶 / Analytic E[2 I(eta>0)] and even orders.
    输出 / Outputs: 对数积分为零且稳定 / Stable zero log integral.
    """
    distribution = RandomEffectDistribution([("iiv", None, "eta")], np.eye(1))
    result = integrate(
        conditional_log_likelihood=lambda x: -math.inf if x[0] < 0 else math.log(2),
        random_effect_distribution=distribution,
        integration_method="ghq", numerical_options=ghq_options(),
    )
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(0, abs=1e-13)


def test_nc03_all_sampled_nodes_zero_is_not_evidence_of_zero_integral():
    """功能 / Function: 拒绝有限采样全零的错误结论 / Reject a zero-integral conclusion from finitely sampled zeros.
    输入 / Inputs: 全节点返回负无穷的回调 / Callback returning negative infinity at all sampled nodes.
    输出 / Outputs: failed/None而非合法零证据 / failed/None rather than valid zero evidence.
    """
    distribution = RandomEffectDistribution([("iiv", None, "eta")], np.eye(1))
    result = integrate(
        conditional_log_likelihood=lambda x: -math.inf,
        random_effect_distribution=distribution,
        integration_method="ghq", numerical_options=ghq_options(),
    )
    assert result["status"] == "failed"
    assert result["log_marginal_likelihood"] is None
    assert "do not establish" in result["failure_reason"]
