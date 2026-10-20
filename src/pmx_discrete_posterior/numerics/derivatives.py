"""功能 / Function: 适配并核验SciPy求导 / Adapt and check SciPy differentiation.
输入 / Inputs: 确定性标量函数、同坐标评价点 / Deterministic scalar function and point.
输出 / Outputs: 导数和经验误差估计，不是严格误差界 / Derivative and empirical error, not a rigorous bound.
"""

import math

import numpy as np
from scipy.differentiate import hessian as scipy_hessian, jacobian

from ..interfaces.inputs.numerical_options import resolve_derivative_options as _options
from ..interfaces.inputs.numerical_options import resolve_hessian_options
from ..interfaces.validation import InputError, numeric_array, scalar


class _EvaluationError(RuntimeError):
    """功能 / Function: 标识目标执行失败 / Identify objective execution failure.
    输入 / Inputs: 原因 / Reason. 输出 / Outputs: 异常 / Exception.
    """


class _ScalarObjective:
    """功能 / Function: 校验回调并缓存重复节点 / Check callbacks and cache repeated nodes.
    输入 / Inputs: 确定性函数 / Deterministic callable.
    输出 / Outputs: 有限标量及评价计数 / Finite scalar and evaluation count.
    """

    def __init__(self, objective):
        """功能 / Function: 保存回调 / Store callback.
        输入 / Inputs: objective / Callable. 输出 / Outputs: 空缓存 / Empty cache.
        """
        if not callable(objective):
            raise InputError("objective must be callable")
        self.objective = objective
        self.cache = {}
        self.nfev = 0
        self.max_magnitude = 0.0

    def __call__(self, point, *, fresh=False):
        """功能 / Function: 执行一次评价 / Execute one evaluation.
        输入 / Inputs: 向量；fresh绕过缓存 / Vector; fresh bypasses cache.
        输出 / Outputs: 标量或异常 / Scalar or evaluation error.
        """
        point = np.asarray(point, dtype=float)
        if not np.isfinite(point).all():
            raise _EvaluationError("non-finite evaluation point")
        key = point.tobytes()
        if not fresh and key in self.cache:
            return self.cache[key]
        argument = point.copy()
        try:
            self.nfev += 1
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value = scalar(self.objective(argument), name="objective output")
        except Exception as exc:
            raise _EvaluationError(f"objective failed: {type(exc).__name__}: {exc}") from exc
        if not np.array_equal(argument, point):
            raise _EvaluationError("objective mutated its input")
        if key in self.cache and value != self.cache[key]:
            raise _EvaluationError("objective is not deterministic at the repeated point")
        self.cache[key] = value
        self.max_magnitude = max(self.max_magnitude, abs(value))
        return value




def _differentiate(objective, evaluation_point, derivative_options, degree=1):
    """功能 / Function: 以两组步长核验导数 / Check derivatives using two step schedules.
    输入 / Inputs: 目标、向量及设置 / Objective, vector and settings.
    输出 / Outputs: 结果；失败导数为None / Result; derivative is None on failure.
    """
    point = numeric_array(evaluation_point, name="evaluation_point")
    if point.size == 0:
        raise InputError("no active coordinates: evaluate the likelihood directly")
    settings = resolve_hessian_options(derivative_options) if degree == 2 else _options(derivative_options)
    checked = _ScalarObjective(objective)
    key = "gradient" if degree == 1 else "hessian"
    backend = jacobian if degree == 1 else scipy_hessian
    details = {"backend": f"scipy.differentiate.{'jacobian' if degree == 1 else 'hessian'}", "settings": settings,
               "verification_step_multiplier": 1.6}
    result = {key: None, "estimated_error": None, "status": "failed",
              "failure_reason": None, "method_details": details}
    try:
        # 先检查坐标能否分辨整个步长预算 / Check coordinate resolution across the planned step budget.
        log_min_step = (math.log(settings["initial_step"])
                        - (settings["maxiter"] + settings["order"] // 2) * math.log(settings["step_factor"]))
        coordinate_resolution = float(np.max(np.abs(np.spacing(point))))
        if log_min_step <= math.log(max(np.finfo(float).tiny, coordinate_resolution)) + math.log(8):
            result["failure_reason"] = "derivative step budget is not resolved in the evaluation coordinates"
            return result
        origin = checked(point)
        checked(point, fresh=True)
        minimum_observed_step = math.inf
        cancellation_scale = 8 * max(coordinate_resolution,
                                     np.finfo(float).eps * settings["initial_step"])

        def vectorized(points):
            """功能 / Function: 将批量节点适配为标量调用 / Adapt batched nodes to scalar calls.
            输入 / Inputs: 形状(d, ...) / Shape (d, ...).
            输出 / Outputs: 去常数后的批量值 / Centered values with matching batch shape.
            """
            nonlocal minimum_observed_step
            flat = np.asarray(points).reshape(point.size, -1)
            distances = np.abs(flat - point[:, None])
            resolved = distances[distances > cancellation_scale]
            if resolved.size:
                minimum_observed_step = min(minimum_observed_step, float(resolved.min()))
            return np.asarray([checked(p) - origin for p in flat.T]).reshape(points.shape[1:])

        runs = [backend(vectorized, point, tolerances={"atol": settings["atol"], "rtol": settings["rtol"]},
                         initial_step=settings["initial_step"] * multiplier,
                         step_factor=settings["step_factor"], maxiter=settings["maxiter"], order=settings["order"])
                for multiplier in (1.0, 1.6)]
        first, second = runs
        first_value = np.asarray(first.df if degree == 1 else first.ddf)
        estimate = np.asarray(second.df if degree == 1 else second.ddf)
        # 额外经验检查，不是严格误差界 / Additional empirical checks, not error bounds.
        if not math.isfinite(minimum_observed_step):
            result["failure_reason"] = "no resolved perturbation was evaluated"
            return result
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            floor = float(8 * np.finfo(float).eps * checked.max_magnitude / np.float64(minimum_observed_step) ** degree)
        error = np.maximum.reduce([first.error, second.error, np.abs(first_value - estimate),
                                   np.full_like(estimate, floor)])
        if degree == 2:
            error = np.maximum(error, np.abs(estimate - estimate.T))
            error = np.maximum(error, error.T)
        tolerance = settings["atol"] + settings["rtol"] * np.abs(estimate)
        details.update(nfev=checked.nfev, backend_status=[r.status.tolist() for r in runs], roundoff_screen=floor,
                       minimum_observed_step=minimum_observed_step)
        result["estimated_error"] = error
        if not all(np.all(r.success) for r in runs):
            result["failure_reason"] = "derivative backend did not converge for every component and step schedule"
        elif not np.isfinite(estimate).all() or not np.isfinite(error).all():
            result["failure_reason"] = "non-finite derivative or error estimate"
        elif not np.all(error <= tolerance):
            result["failure_reason"] = "derivative failed step agreement or floating-point resolution check"
        else:
            # 仅在非对称差异通过检查后对称化 / Symmetrize only after checking the discrepancy.
            value = estimate.copy() if degree == 1 else estimate / 2 + estimate.T / 2
            if not np.isfinite(value).all():
                result["failure_reason"] = "non-finite final derivative"
            else:
                result[key] = value
                result["status"] = "success"
        # 功能 / Function: 保留有限的受限梯度供独立积分诊断，不升级求导状态 / Preserve a finite limited gradient for independent integral diagnostics, without upgrading derivative status.
        # 输入 / Inputs: 两组实际估计、误差和后端状态 / Two estimates, errors and backend states.
        # 输出 / Outputs: 仅诊断候选；模型异常、非有限值和未知后端失败不保留 / Diagnostic candidate only; exclude model errors, nonfinite values and unknown failures.
        if (degree == 1 and result["status"] == "failed"
                and np.isfinite(first_value).all() and np.isfinite(estimate).all()
                and np.isfinite(error).all() and np.all(error >= 0)
                and all(np.isin(r.status, (0, -1, -2)).all() for r in runs)):
            details["precision_limited_candidate"] = {
                "gradient": estimate.tolist(), "estimated_error": error.tolist(),
            }
    except (_EvaluationError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        details["nfev"] = checked.nfev
    return result


def gradient(objective, evaluation_point, derivative_options=None):
    """功能 / Function: 求同坐标梯度 / Differentiate in the input coordinates.
    输入 / Inputs: objective(d-vector)->finite scalar、评价点、可选设置 / Objective, point, options.
    输出 / Outputs: gradient(d)、estimated_error(d)、status、failure_reason；失败为None / Gradient is None on failure.
    """
    return _differentiate(objective, evaluation_point, derivative_options)


def hessian(objective, evaluation_point, derivative_options=None):
    """功能 / Function: 计算并核验Hessian；不以正定性代替准确性 / Compute and check Hessian, not just definiteness.
    输入 / Inputs: objective(d-vector)->finite scalar、同坐标点及设置 / Objective, point in the same coordinates, options.
    输出 / Outputs: hessian(d,d)、estimated_error(d,d)、status、failure_reason；失败矩阵为None / Matrix is None on failure.
    """
    return _differentiate(objective, evaluation_point, derivative_options, degree=2)
