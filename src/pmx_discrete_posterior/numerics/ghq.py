"""功能 / Function: 先验中心正态张量积GHQ / Prior-centered normal tensor-product GHQ.
输入 / Inputs: 条件对数似然、正态效应分布和显式求积设置 / Conditional log-likelihood, normal effect distribution and explicit quadrature settings.
输出 / Outputs: 阶数估计及检查；不包含离散先验 / Order estimates and checks, without discrete priors.
"""

import itertools
import math

import numpy as np
from scipy.linalg import cholesky
from scipy.special import logsumexp, roots_hermitenorm

from ..interfaces.inputs.integration_method import validate_integration_method
from ..interfaces.inputs.numerical_options import _positive_integer, validate_numerical_options
from ..interfaces.results import IntegrationResult, validate_integration_result
from ..interfaces.validation import InputError
from ..models.random_effects import RandomEffectDistribution
from .laplace import _direct_log_likelihood


def _normal_rule(order):
    """功能 / Function: 用SciPy生成标准正态节点和对数权重 / Generate standard-normal nodes and log weights using SciPy.
    输入 / Inputs: 正整数阶数 / Positive integer order.
    输出 / Outputs: 节点及规范化对数权重；权重下溢时失败而非删除节点 / Nodes and normalized log weights; underflow fails rather than dropping nodes.
    """
    order = _positive_integer(order, "GHQ order")
    nodes, weights = roots_hermitenorm(order)
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise FloatingPointError("GHQ nodes/weights are not representable; no zero-weight nodes discarded")
    # e^(-z^2/2)权重除以sqrt(2*pi)，已含正态先验 / Divide by sqrt(2*pi); weights already include the normal prior.
    log_weights = np.log(weights) - .5 * math.log(2 * math.pi)
    if abs(float(logsumexp(log_weights))) > 32 * np.finfo(float).eps:
        raise FloatingPointError("GHQ normal weights do not sum to one within float64 resolution")
    return nodes, log_weights


def _node_likelihood(callback, point, details):
    """功能 / Function: 评价一个节点并拒绝非法值或输入修改 / Evaluate a node and reject invalid values or input mutation.
    输入 / Inputs: 条件似然、有限节点、累计计数 / Conditional likelihood, finite node and cumulative counter.
    输出 / Outputs: 有限对数似然或明确零似然-inf；异常继续传播 / Finite log-likelihood or explicit zero -inf; exceptions propagate.
    """
    if not np.all(np.isfinite(point)):
        raise FloatingPointError("non-finite transformed GHQ node")
    original = point.copy()
    details["evaluations"] += 1
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = callback(point)
    if point.shape != original.shape or not np.array_equal(point, original):
        raise FloatingPointError("conditional_log_likelihood mutated a GHQ node")
    value = validate_integration_result({"status": "success", "log_marginal_likelihood": value,
                                         "failure_reason": None})["log_marginal_likelihood"]
    if math.isfinite(value):
        details["order_max_abs_log_likelihood"] = max(details.get("order_max_abs_log_likelihood", 0.), abs(value))
    return value


def _evaluate_order(callback, mean, factor, order, details):
    """功能 / Function: 逐节点计算一个完整张量阶数 / Evaluate one complete tensor-product order node by node.
    输入 / Inputs: 似然、正态均值/Cholesky因子、阶数和累计计数 / Likelihood, normal mean/Cholesky factor, order and evaluation counter.
    输出 / Outputs: 单阶对数积分；有限节点全零不能证明真零积分 / Single-order log integral; all-zero finite nodes do not prove zero evidence.
    不构建完整网格，按1024项分块对数求和 / No full mesh allocation; sum log terms in blocks of 1024.
    """
    nodes, log_weights = _normal_rule(order)
    details["current_order"] = order
    details["order_max_abs_log_likelihood"] = 0.
    chunk, chunk_weights, centers, totals = [], [], [], []
    for index, coordinates in enumerate(itertools.product(range(order), repeat=len(mean))):
        details["node_index"] = index
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            point = mean + factor @ nodes[list(coordinates)]
        value = _node_likelihood(callback, point.copy(), details)
        if index == 0 and value != _node_likelihood(callback, point.copy(), details):
            raise FloatingPointError("conditional_log_likelihood is not deterministic at the repeated GHQ node")
        chunk.append(value)
        chunk_weights.append(math.fsum(float(log_weights[i]) for i in coordinates))
        if len(chunk) == 1024:
            center, total = _centered_chunk(chunk, chunk_weights)
            centers.append(center)
            totals.append(total)
            chunk.clear()
            chunk_weights.clear()
    if chunk:
        center, total = _centered_chunk(chunk, chunk_weights)
        centers.append(center)
        totals.append(total)
    center = max(centers)
    if not math.isfinite(center):
        raise FloatingPointError("all GHQ nodes have zero likelihood; finite nodes do not establish a zero integral")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = center + float(logsumexp((np.asarray(centers) - center) + totals))
    if not math.isfinite(result):
        raise FloatingPointError("all GHQ nodes have zero likelihood; finite nodes do not establish a zero integral")
    # 同时考虑节点值、权重与相消后的还原；经验筛查不是严格误差界 / Screen node values, weights and reconstruction after cancellation; not a rigorous error bound.
    magnitude = max(1., details["order_max_abs_log_likelihood"],
                    len(mean) * float(np.max(np.abs(log_weights))), abs(result))
    reduction_depth = 1 + math.ceil(math.log2(order**len(mean) + 1))
    details["estimated_log_integral_roundoff"] = (8 * np.finfo(float).eps * reduction_depth) * magnitude
    return result


def _centered_chunk(values, weights):
    """功能 / Function: 加权前消除共同对数偏移 / Remove a common log offset before adding weights.
    输入 / Inputs: 有限或-inf似然和对数权重 / Finite or -inf likelihoods and log weights.
    输出 / Outputs: 中心与相对对数和；全零块保持-inf / Center and relative log sum; all-zero blocks retain -inf.
    """
    center = max(values)
    if center == -math.inf:
        return -math.inf, -math.inf
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return center, float(logsumexp((np.asarray(values) - center) + weights))


def integrate(*, conditional_log_likelihood, random_effect_distribution,
              integration_method, numerical_options=None) -> IntegrationResult:
    """功能 / Function: 以先验中心GHQ计算并核验逐状态对数积分 / Compute and check a per-state log integral using prior-centered GHQ.
    输入 / Inputs: 确定性条件似然、models解析的正态分布、ghq方法、显式阶数/容差/预算 / Deterministic likelihood, models-resolved normal distribution, ghq method and explicit orders/tolerances/budget.
    输出 / Outputs: 最后两次阶数变化均达标才success；执行或预算失败保留逐阶估计 / Success requires both final order changes to pass; execution/budget failures retain order estimates.
    稳定性是经验检查，不是严格积分误差上界；不使用优化/Hessian / Stability is empirical, not a rigorous error bound; optimization/Hessian are not used.
    """
    if validate_integration_method(integration_method) != "ghq":
        raise InputError("ghq.integrate requires integration_method='ghq'")
    if not callable(conditional_log_likelihood):
        raise InputError("conditional_log_likelihood must be callable")
    # 均值与协方差本身不能证明正态；首版仅接收现有正态模型对象 / Moments alone do not establish normality; accept the existing normal model object only.
    if not isinstance(random_effect_distribution, RandomEffectDistribution):
        raise InputError("prior_centered GHQ requires the models-resolved normal RandomEffectDistribution")
    options = validate_numerical_options(numerical_options)
    if "ghq" not in options:
        raise InputError("GHQ requires explicit numerical_options.ghq settings")
    settings = options["ghq"]
    mean = random_effect_distribution.mean
    dimension = len(mean)
    details = {"integration_method": "ghq", **settings, "dimension": dimension,
               "scope": "direct" if dimension == 0 else "prior_centered_tensor",
               "evaluations": 0, "order_results": [], "stability_passed": False,
               "stage": "direct_likelihood" if dimension == 0 else "quadrature"}
    result = {"status": "failed", "log_marginal_likelihood": None,
              "failure_reason": None, "method_details": details}
    try:
        if dimension == 0:
            if settings["max_nodes"] < 2:
                raise FloatingPointError("GHQ max_nodes cannot cover two direct likelihood checks")

            def counted_direct(point):
                """功能 / Function: 记录共享零维核验的调用次数 / Count evaluations in the shared zero-dimensional check.
                输入 / Inputs: 空向量 / Empty vector. 输出 / Outputs: 原似然结果 / Original likelihood result.
                """
                details["evaluations"] += 1
                return conditional_log_likelihood(point)

            value = _direct_log_likelihood(counted_direct, random_effect_distribution)
            details["stability_passed"] = None  # 无积分，不声称做过阶数核验 / No integration, hence no order-stability claim.
        else:
            factor = cholesky(random_effect_distribution.covariance, lower=True, check_finite=True)
            for order in settings["orders"]:
                cost = order**dimension + 1
                if cost > settings["max_nodes"] - details["evaluations"]:
                    details["stage"] = "node_budget"
                    details["next_order"] = order
                    details["next_order_evaluations"] = cost
                    raise FloatingPointError("GHQ max_nodes would be exceeded before the next complete order")
                before = details["evaluations"]
                value = _evaluate_order(conditional_log_likelihood, mean, factor, order, details)
                previous = details["order_results"][-1]["log_marginal_likelihood"] if details["order_results"] else None
                delta = abs(value - previous) if previous is not None else None
                details["order_results"].append({"order": order, "log_marginal_likelihood": value,
                                                 "change": delta, "evaluations": details["evaluations"] - before,
                                                 "estimated_log_integral_roundoff": details["estimated_log_integral_roundoff"]})
                if max(math.ulp(value), details["estimated_log_integral_roundoff"]) > settings["log_integral_atol"] / 4:
                    details["stage"] = "floating_point_resolution"
                    raise FloatingPointError("float64 log-integral resolution/roundoff cannot support the requested stability tolerance")
            details["stage"] = "order_stability"
            details["stability_passed"] = all(row["change"] <= settings["log_integral_atol"]
                                               for row in details["order_results"][-2:])
            if not details["stability_passed"]:
                raise FloatingPointError("GHQ final two log-integral changes exceed log_integral_atol")
        result.update(status="success", log_marginal_likelihood=value)
        details["stage"] = "complete"
    except Exception as exc:
        result["failure_reason"] = f"{details['stage']}: {type(exc).__name__}: {exc}"
    return validate_integration_result(result)
