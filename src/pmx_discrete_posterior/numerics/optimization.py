"""功能 / Function: 通过SciPy搜索并独立核验驻点 / Search with SciPy and independently check stationarity.
输入 / Inputs: 确定性目标、向量起点和优化设置 / Deterministic objective, vector start and options.
输出 / Outputs: 接纳的驻点或明确失败；不证明全局最小值 / Accepted stationary point or failure, not proof of a global minimum.
"""

from copy import deepcopy

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from ..interfaces.inputs.numerical_options import resolve_optimization_options as _options
from ..interfaces.validation import InputError, numeric_array
from .derivatives import _EvaluationError, _ScalarObjective, gradient as checked_gradient


def _derivative_record(derivative):
    """功能 / Function: 将求导失败证据转换为可归档副本 / Convert derivative evidence to an archivable copy.
    输入 / Inputs: 求导结果 / Derivative result.
    输出 / Outputs: 数组字段转换为列表；不修改原结果 / Array fields as lists without mutating the original result.
    """
    record = deepcopy(derivative)
    for key in ("gradient", "estimated_error"):
        if record.get(key) is not None:
            record[key] = np.asarray(record[key]).tolist()
    return record


def minimize(objective, initial_point, optimization_options=None, gradient=None):
    """功能 / Function: 搜索无约束局部驻点并检验梯度 / Search an unconstrained local stationary point and verify its gradient.
    输入 / Inputs: objective(d)->标量、initial_point(d)、设置、可选梯度(d)->(d) / Objective, start, options, optional gradient.
    输出 / Outputs: minimizer、objective_value、status、failure_reason；失败位置和值均为None / Point and value are None on failure.
    """
    point = numeric_array(initial_point, name="initial_point")
    if point.size == 0:
        raise InputError("no active coordinates: evaluate the likelihood directly")
    if gradient is not None and not callable(gradient):
        raise InputError("gradient must be callable or None")
    backend, tolerance = _options(optimization_options)
    checked = _ScalarObjective(objective)
    derivative_nfev = 0
    details = {"method": "BFGS", "backend_options": backend, "stationarity_tolerance": tolerance,
               "gradient_source": "callback" if gradient is not None else "checked_scipy_jacobian"}
    result = {"minimizer": None, "objective_value": None, "status": "failed",
              "failure_reason": None, "method_details": details}

    def evaluate(x, *, fresh=False):
        """功能 / Function: 记录主优化目标评价，不混入求导探针 / Record primary objective calls, excluding derivative probes.
        输入 / Inputs: 坐标及缓存策略 / Coordinates and cache policy.
        输出 / Outputs: 原目标值；分别保留请求、成功评价和最佳评价 / Original value; separate request, last and best records.
        """
        details["last_objective_request"] = np.asarray(x).tolist()
        value = checked(x, fresh=fresh)
        record = {"point": np.asarray(x).tolist(), "value": value}
        details["last_objective_evaluation"] = record
        best = details.get("best_objective_evaluation")
        if best is None or value < best["value"]:
            details["best_objective_evaluation"] = deepcopy(record)
        return value

    try:
        origin = evaluate(point)
        evaluate(point, fresh=True)

        def centered(x):
            """功能 / Function: 降低常数对优化的影响 / Reduce constant-offset effects in optimization.
            输入 / Inputs: 坐标 / Coordinates. 输出 / Outputs: 减去起点值的目标 / Objective minus its initial value.
            """
            return evaluate(x) - origin

        def jacobian(x):
            """功能 / Function: 接入已核验求导或验证梯度回调 / Use checked differentiation or validate a gradient callback.
            输入 / Inputs: 坐标 / Coordinates. 输出 / Outputs: 同形有限向量 / Finite vector of the same shape.
            """
            nonlocal derivative_nfev
            details["last_gradient_request"] = np.asarray(x).tolist()
            if gradient is None:
                derivative = checked_gradient(objective, x, {"atol": min(1e-8, backend["gtol"] / 10)})
                derivative_nfev += derivative["method_details"].get("nfev", 0)
                if derivative["status"] != "success":
                    details["gradient_failure"] = {
                        "point": np.asarray(x).tolist(), "phase": "search", "result": _derivative_record(derivative)}
                    raise _EvaluationError("optimization gradient failed: " + derivative["failure_reason"])
                return derivative["gradient"]
            argument = x.copy()
            try:
                value = numeric_array(gradient(argument), name="gradient output")
            except Exception as exc:
                details["gradient_failure"] = {
                    "point": np.asarray(x).tolist(), "phase": "search", "source": "callback",
                    "failure_reason": f"{type(exc).__name__}: {exc}"}
                raise _EvaluationError(f"gradient callback failed: {type(exc).__name__}: {exc}") from exc
            if value.shape != point.shape or not np.array_equal(argument, x):
                details["gradient_failure"] = {
                    "point": np.asarray(x).tolist(), "phase": "search", "source": "callback",
                    "failure_reason": "gradient callback returned a wrong shape or mutated its input"}
                raise _EvaluationError("gradient callback returned a wrong shape or mutated its input")
            return value

        solution = scipy_minimize(centered, point, method="BFGS", jac=jacobian, options=backend)
        details.update(backend_success=bool(solution.success), backend_message=str(solution.message),
                       iterations=int(solution.nit), nfev=checked.nfev)
        # 使用缓存而非相加恢复原目标，且不为诊断增加评价 / Use cached uncentered value without extra evaluations.
        details["backend_result"] = {
            "point": np.asarray(solution.x).tolist(),
            "centered_objective_value": float(solution.fun),
            "objective_value": checked.cache.get(np.asarray(solution.x, dtype=float).tobytes()),
            "status": None if solution.get("status") is None else int(solution.status),
            "gradient": None if solution.get("jac") is None else np.asarray(solution.jac).tolist(),
        }
        if not solution.success:
            result["failure_reason"] = "optimizer did not converge: " + str(solution.message)
            return result
        value = evaluate(solution.x, fresh=True)
        # 不复用优化器梯度或逆Hessian作验收 / Do not reuse optimizer derivatives for acceptance.
        verification = checked_gradient(objective, solution.x)
        derivative_nfev += verification["method_details"].get("nfev", 0)
        details["stationarity_check"] = verification["method_details"]
        if verification["status"] != "success":
            details["gradient_failure"] = {
                "point": np.asarray(solution.x).tolist(), "phase": "stationarity",
                "result": _derivative_record(verification)}
            result["failure_reason"] = "stationarity could not be verified: " + verification["failure_reason"]
            return result
        bound = float(np.max(np.abs(verification["gradient"]) + verification["estimated_error"]))
        details["gradient_inf_with_error"] = bound
        if bound > tolerance:
            result["failure_reason"] = "optimizer point failed independent stationarity check"
            return result
        result.update(minimizer=solution.x.copy(), objective_value=value, status="success")
    except (_EvaluationError, FloatingPointError, OverflowError) as exc:
        result["failure_reason"] = str(exc)
    finally:
        details.update(nfev=checked.nfev + derivative_nfev,
                       objective_nfev=checked.nfev, derivative_objective_nfev=derivative_nfev)
    return result
