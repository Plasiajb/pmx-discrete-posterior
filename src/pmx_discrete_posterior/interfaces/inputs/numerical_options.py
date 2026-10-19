"""功能 / Function: 复用已验证数值设置的输入契约 / Input contracts for tested numerical settings.
输入 / Inputs: 优化、求导和可选起点 / Optimization, differentiation and optional initial point.
输出 / Outputs: 校验后的独立配置，不执行优化或积分 / Independent configuration without numerical execution.
"""

from numbers import Integral

import numpy as np

from ..validation import InputError, fields, numeric_array, scalar


def _positive_integer(value, name):
    """功能 / Function: 拒绝隐式取整 / Reject implicit integer coercion.
    输入 / Inputs: 正整数 / Positive integer. 输出 / Outputs: int或InputError / int or InputError.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise InputError(f"{name} must be a positive integer")
    return int(value)


def resolve_derivative_options(derivative_options):
    """功能 / Function: 解析SciPy设置 / Resolve SciPy settings.
    输入 / Inputs: 映射或None / Mapping or None. 输出 / Outputs: 有效设置 / Validated settings.
    """
    options = {} if derivative_options is None else derivative_options
    fields(options, (), ("backend", "backend_options", "atol", "rtol"), name="derivative_options")
    if not isinstance(options.get("backend", "scipy"), str) or options.get("backend", "scipy") != "scipy":
        raise InputError("only the tested scipy differentiation backend is supported")
    backend = options.get("backend_options", {})
    fields(backend, (), ("initial_step", "step_factor", "maxiter", "order"), name="backend_options")
    settings = {"initial_step": scalar(backend.get("initial_step", 0.5), positive=True, name="initial_step"),
                "step_factor": scalar(backend.get("step_factor", 2.0), positive=True, name="step_factor"),
                "maxiter": _positive_integer(backend.get("maxiter", 10), "maxiter"),
                "order": _positive_integer(backend.get("order", 8), "order"),
                "atol": scalar(options.get("atol", 1e-6), minimum=0, name="atol"),
                "rtol": scalar(options.get("rtol", 1e-6), minimum=0, name="rtol")}
    if settings["step_factor"] <= 1:
        raise InputError("step_factor must exceed 1 in this decreasing-step implementation")
    if settings["order"] not in (2, 4, 6, 8):
        raise InputError("tested even difference orders are 2, 4, 6 and 8")
    if settings["maxiter"] < 2:
        raise InputError("maxiter must allow at least two derivative estimates")
    if settings["atol"] == 0 and settings["rtol"] == 0:
        raise InputError("at least one derivative tolerance must be positive")
    return settings


def resolve_optimization_options(optimization_options):
    """功能 / Function: 解析有限的BFGS设置 / Resolve a bounded set of BFGS options.
    输入 / Inputs: 优化设置 / Optimization settings.
    输出 / Outputs: 后端设置与驻点容差 / Backend options and stationarity tolerance.
    """
    options = {} if optimization_options is None else optimization_options
    fields(options, (), ("method", "backend_options", "stationarity_tolerance"), name="optimization_options")
    if not isinstance(options.get("method", "BFGS"), str) or options.get("method", "BFGS") != "BFGS":
        raise InputError("only unconstrained BFGS is supported; no implicit bounds or fallback")
    tolerance = scalar(options.get("stationarity_tolerance", 1e-5), positive=True, name="stationarity_tolerance")
    backend = options.get("backend_options", {})
    fields(backend, (), ("maxiter", "gtol", "xrtol", "c1", "c2", "finite_diff_rel_step"), name="backend_options")
    parsed = {"maxiter": _positive_integer(backend.get("maxiter", 1000), "maxiter"),
              "gtol": scalar(backend.get("gtol", min(1e-7, tolerance / 10)), positive=True, name="gtol"),
              "xrtol": scalar(backend.get("xrtol", 0), minimum=0, name="xrtol"),
              "c1": scalar(backend.get("c1", 1e-4), positive=True, name="c1"),
              "c2": scalar(backend.get("c2", 0.9), positive=True, name="c2"),
              "norm": np.inf}
    if not 0 < parsed["c1"] < parsed["c2"] < 1:
        raise InputError("BFGS requires 0 < c1 < c2 < 1")
    if "finite_diff_rel_step" in backend:
        raise InputError("finite_diff_rel_step would be ignored by checked differentiation; use a gradient callback for custom settings")
    return parsed, tolerance


def resolve_hessian_options(derivative_options):
    """功能 / Function: 检查嵌套Hessian后端的额外容差条件 / Check the nested Hessian backend's tolerance requirement.
    输入 / Inputs: 求导设置 / Derivative options.
    输出 / Outputs: 已解析设置或InputError；不限制仅梯度调用 / Resolved settings or InputError; gradient-only calls unaffected.
    """
    settings = resolve_derivative_options(derivative_options)
    if settings["rtol"] < 100 * np.finfo(float).eps:
        raise InputError("Hessian rtol must be at least 100 times float64 epsilon for the nested backend")
    return settings


def validate_numerical_options(numerical_options):
    """功能 / Function: 检查当前已实现的数值选项 / Validate currently implemented numerical options.
    输入 / Inputs: Laplace优化/求导组，或显式GHQ组 / Laplace optimization/derivative groups, or an explicit GHQ group.
    输出 / Outputs: 可直接分发的独立设置副本 / Independent settings ready for dispatch.
    先验中心GHQ不使用优化/求导设置，不允许混填 / Prior-centered GHQ does not consume optimization/derivative settings; mixed groups are rejected.
    """
    options = {} if numerical_options is None else numerical_options
    fields(options, (), ("optimization", "derivatives", "ghq"), name="numerical_options")
    if "ghq" in options:
        if set(options) != {"ghq"}:
            raise InputError("prior_centered GHQ accepts only the ghq group, not optimization or derivatives")
        return {"ghq": resolve_ghq_options(options["ghq"])}
    optimization = options.get("optimization", {})
    fields(optimization, (), ("method", "initial_point", "backend_options", "stationarity_tolerance"), name="optimization")
    backend, tolerance = resolve_optimization_options({k: v for k, v in optimization.items() if k != "initial_point"})
    result_optimization = {"method": "BFGS", "stationarity_tolerance": tolerance,
                           "backend_options": {k: v for k, v in backend.items() if k != "norm"}}
    if "initial_point" in optimization:
        result_optimization["initial_point"] = numeric_array(optimization["initial_point"], name="initial_point")
    derivative_options = options.get("derivatives", {})
    fields(derivative_options, (), ("backend", "backend_options", "atol", "rtol"), name="derivatives")
    derivatives = resolve_derivative_options(derivative_options)
    result_derivatives = {"backend": "scipy", "atol": derivatives["atol"], "rtol": derivatives["rtol"],
                          "backend_options": {k: v for k, v in derivatives.items() if k not in ("atol", "rtol")}}
    return {"optimization": result_optimization, "derivatives": result_derivatives}


def resolve_ghq_options(ghq_options):
    """功能 / Function: 校验显式GHQ阶数与接纳设置 / Validate explicit GHQ orders and acceptance settings.
    输入 / Inputs: prior_centered、至少三阶、两个绝对容差及每候选累计调用上限 / prior_centered, at least three orders, two absolute tolerances and a per-candidate cumulative evaluation cap.
    输出 / Outputs: 独立配置；不自动选择阶数、容差或预算 / Independent settings, without inferred orders, tolerances or budgets.
    max_nodes包含节点评价及每阶首节点的一次重复核验 / max_nodes includes node evaluations and one repeated first-node check per order.
    """
    fields(ghq_options, ("variant", "orders", "log_integral_atol", "posterior_atol", "max_nodes"), (), name="ghq")
    if not isinstance(ghq_options["variant"], str) or ghq_options["variant"] != "prior_centered":
        raise InputError("only prior_centered GHQ is supported; no adaptive fallback")
    orders = ghq_options["orders"]
    if not isinstance(orders, (list, tuple)) or len(orders) < 3:
        raise InputError("GHQ orders must be a sequence of at least three positive integers")
    orders = [_positive_integer(order, "GHQ order") for order in orders]
    if any(a >= b for a, b in zip(orders, orders[1:])):
        raise InputError("GHQ orders must be strictly increasing")
    posterior_atol = scalar(ghq_options["posterior_atol"], positive=True, name="posterior_atol")
    if posterior_atol >= 1:
        raise InputError("posterior_atol must be less than 1")
    return {"variant": "prior_centered", "orders": orders,
            "log_integral_atol": scalar(ghq_options["log_integral_atol"], positive=True, name="log_integral_atol"),
            "posterior_atol": posterior_atol,
            "max_nodes": _positive_integer(ghq_options["max_nodes"], "max_nodes")}
