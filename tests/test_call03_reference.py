"""功能 / Function: 核验修复后真实示例的参考后验 / Verify repaired reference CLI posteriors.
输入 / Inputs: OXC/TAC冻结参考与既有独立基准 / Frozen OXC/TAC inputs and existing independent references.
输出 / Outputs: 退出状态、候选事件与完整概率核对 / Exit, event and full probability assertions.
"""

import json
import math

import numpy as np
import pytest

from cli_helpers import cli
from test_posterior_binding import declared_reference
from test_posterior_reference_models import INDEPENDENT_LOGS, LEGACY_LOGS, LEGACY_POSTERIORS


@pytest.mark.parametrize("name,log_atol,probability_atol", [("OXC", 1e-6, 5e-7), pytest.param("TAC", 1e-3, 5e-4, marks=pytest.mark.slow)])
def test_call03_repaired_cli_preserves_full_reference(name, log_atol, probability_atol):
    """功能 / Function: 运行真实Laplace示例，不替换计算器 / Run the real Laplace example without replacing the calculator.
    输入 / Inputs: 模型名及既有核验容差 / Model name and unchanged reference tolerances.
    输出 / Outputs: 成功JSON与逐状态基准一致；不代表GHQ验收 / Successful JSON matches per-state references, not GHQ acceptance.
    """
    run = cli(name, timeout=1800 if name == "TAC" else 90)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    print("CALL03_REFERENCE", name, json.dumps(result, allow_nan=False))
    assert result["model"] == name
    assert result["status"] == "success" and result["integration_method"] == "laplace"
    states = declared_reference(name)["candidate_states"]
    rows = result["state_results"]
    assert len(rows) == len(states) == 4
    assert [r["state_id"] for r in rows] == [s["state_id"] for s in states]
    assert [r["events"] for r in rows] == [s["events"] for s in states]
    assert all(r["status"] == "success" and r["prior_probability"] == .25 for r in rows)

    # 功能 / Function: 复用独立积分常数和原门槛 / Reuse independent integrals and original thresholds.
    # 输入 / Inputs: 输出向量、既有参考 / Output vectors and archived references.
    # 输出 / Outputs: 对数积分及完整后验均匹配 / Matching log integrals and complete posteriors.
    logs = [r["log_marginal_likelihood"] for r in rows]
    probabilities = [r["posterior_probability"] for r in rows]
    weights = [math.exp(value) for value in INDEPENDENT_LOGS[name]]
    expected = [weight / math.fsum(weights) for weight in weights]
    np.testing.assert_allclose(logs, INDEPENDENT_LOGS[name], atol=log_atol, rtol=0)
    np.testing.assert_allclose(probabilities, expected, atol=probability_atol, rtol=0)
    np.testing.assert_allclose(logs, LEGACY_LOGS[name], atol=1e-5, rtol=0)
    np.testing.assert_allclose(probabilities, LEGACY_POSTERIORS[name], atol=5e-6, rtol=0)
    assert math.fsum(probabilities) == pytest.approx(1., abs=8 * math.ulp(1.), rel=0)
