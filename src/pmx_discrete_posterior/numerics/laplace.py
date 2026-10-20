"""功能 / Function: 对单个候选作局部Laplace积分 / Local Laplace integration for one candidate.
输入 / Inputs: 条件对数似然、已解析随机效应分布及数值设置 / Conditional log-likelihood, resolved effect distribution and options.
输出 / Outputs: 对数边际似然或明确失败；不加离散先验 / Marginal log-likelihood or explicit failure, without discrete priors.
"""

import math

import numpy as np
from scipy.linalg import cholesky, norm

from ..interfaces.inputs.integration_method import validate_integration_method
from ..interfaces.inputs.numerical_options import resolve_hessian_options, validate_numerical_options
from ..interfaces.protocols import ConditionalLogLikelihood, RandomEffectDistributionProtocol
from ..interfaces.results import IntegrationResult, validate_integration_result
from ..interfaces.validation import InputError, numeric_array, scalar
from .coordinates import PriorCoordinates
from .derivatives import _EvaluationError, _ScalarObjective, hessian
from .optimization import minimize


CURVATURE_LOG_INTEGRAL_ATOL = 1e-3
STATIONARITY_LOG_INTEGRAL_ATOL = 1e-3


def _joint_objective(conditional_log_likelihood, random_effect_distribution):
    """功能 / Function: 构造负对数联合密度，不重复计入效应先验 / Build the negative log joint density, counting the effect density once.
    输入 / Inputs: 观测似然回调及规范化分布 / Observation likelihood callback and normalized distribution.
    输出 / Outputs: 同坐标标量目标；复用既有回调检查 / Scalar objective in the same coordinates, reusing callback checks.
    """
    likelihood = _ScalarObjective(conditional_log_likelihood)
    density = _ScalarObjective(random_effect_distribution.log_density)

    def objective(random_effect_values):
        """功能 / Function: 合并观测似然与效应密度 / Combine observation likelihood and effect density.
        输入 / Inputs: 活动效应向量 / Active-effect vector.
        输出 / Outputs: -(log likelihood + log density)，保留归一化常数 / Negative joint log density, retaining normalizing constants.
        """
        # 内层强制重评，避免缓存掩盖外层重复点检查 / Re-evaluate inside so caches do not mask outer determinism checks.
        return -math.fsum((likelihood(random_effect_values, fresh=True), density(random_effect_values, fresh=True)))

    return objective


def _direct_log_likelihood(conditional_log_likelihood, random_effect_distribution):
    """功能 / Function: 零维时直接评价观测似然 / Evaluate the likelihood directly in zero dimensions.
    输入 / Inputs: 接收空向量的似然及零维分布 / Likelihood accepting an empty vector and a zero-dimensional distribution.
    输出 / Outputs: 对数似然，可含明确真零的-inf；执行异常向上传播 / Log-likelihood, including explicit true-zero -inf; execution failures propagate.
    只有零维路径可直接确认整个积分等于此值 / Only the zero-dimensional path establishes that the integral equals this value.
    """
    if not callable(conditional_log_likelihood):
        raise InputError("conditional_log_likelihood must be callable")
    density = _ScalarObjective(random_effect_distribution.log_density)
    values = []
    for _ in range(2):
        point = np.zeros(0)
        if density(point, fresh=True) != 0:
            raise _EvaluationError("a normalized zero-dimensional distribution must have log density zero")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value = conditional_log_likelihood(point)
        if point.shape != (0,):
            raise _EvaluationError("conditional_log_likelihood mutated its empty input")
        values.append(validate_integration_result({"status": "success", "log_marginal_likelihood": value,
                                                   "failure_reason": None})["log_marginal_likelihood"])
    if values[0] != values[1]:
        raise _EvaluationError("conditional_log_likelihood is not deterministic at the repeated point")
    return values[0]


def _curvature_terms(matrix, estimated_error, gradient_inf_with_error):
    """功能 / Function: 检查局部正曲率并稳定计算logdet / Check positive local curvature and compute a stable log determinant.
    输入 / Inputs: 对称Hessian、逐项经验误差及已有梯度包络 / Symmetric Hessian, elementwise empirical errors and the existing gradient envelope.
    输出 / Outputs: logdet、曲率误差筛查量和局部驻点偏移筛查量；曲率误差超1e-3拒绝 / Log determinant, curvature-error screen and local mode-displacement screen; reject curvature error above 1e-3.
    不加jitter、不截特征值；误差估计不是严格界，也不衡量Laplace截断误差 / No jitter or eigenvalue clipping; estimates are not strict bounds or Laplace approximation errors.
    """
    matrix = numeric_array(matrix, ndim=2, name="hessian")
    error = numeric_array(estimated_error, ndim=2, name="hessian estimated_error")
    gradient_bound = scalar(gradient_inf_with_error, name="gradient_inf_with_error", minimum=0)
    dimension = matrix.shape[0]
    if dimension == 0 or matrix.shape != (dimension, dimension) or error.shape != matrix.shape:
        raise _EvaluationError("Hessian and estimated_error must be nonempty square matrices of the same shape")
    if not np.array_equal(matrix, matrix.T) or not np.array_equal(error, error.T) or np.any(error < 0):
        raise _EvaluationError("Hessian must be symmetric and its error must be symmetric and nonnegative")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0):
        raise _EvaluationError("Laplace Hessian must be positive definite")
    # 只缩放线性代数，不变换模型或优化坐标 / Scale only linear algebra, not model or optimization coordinates.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scale = np.sqrt(diagonal)
        scaled = (matrix / scale[:, None]) / scale[None, :]
        scaled_error = (error / scale[:, None]) / scale[None, :]
        factor = cholesky(scaled, lower=True, check_finite=True)
        eigenvalues = np.linalg.eigvalsh(scaled)
        resolution = np.finfo(float).eps * dimension * float(np.max(np.abs(eigenvalues)))
        if eigenvalues[0] <= resolution:
            raise _EvaluationError("Laplace curvature is numerically unresolved; no regularization applied")
        curvature_uncertainty = float(norm(scaled_error, ord=2)) + resolution
        relative_error = curvature_uncertainty / float(eigenvalues[0])
        if not math.isfinite(relative_error) or relative_error >= 1:
            raise _EvaluationError("Hessian uncertainty does not resolve positive curvature")
        log_integral_error = -.5 * dimension * math.log1p(-relative_error)
        if log_integral_error > CURVATURE_LOG_INTEGRAL_ATOL:
            raise _EvaluationError(f"estimated Hessian contribution to log integral {log_integral_error:.6g} exceeds 1e-3")
        log_determinant = math.fsum(math.log(float(x)) for x in diagonal)
        log_determinant += 2 * math.fsum(math.log(float(x)) for x in np.diag(factor))
        # 复用梯度包络与局部曲率，不求逆或重新优化 / Reuse gradient envelope and local curvature, without inversion or reoptimization.
        scaled_gradient_norm = float(norm(gradient_bound / scale))
        mode_displacement_error = .5 * (scaled_gradient_norm / math.sqrt(float(eigenvalues[0]) - curvature_uncertainty))**2
    if not math.isfinite(log_determinant) or not math.isfinite(mode_displacement_error):
        raise _EvaluationError("non-finite local Laplace diagnostics")
    return log_determinant, log_integral_error, mode_displacement_error


def integrate(
    *, conditional_log_likelihood: ConditionalLogLikelihood,
    random_effect_distribution: RandomEffectDistributionProtocol,
    integration_method: str, numerical_options=None,
) -> IntegrationResult:
    """功能 / Function: 复用优化与Hessian，计算单候选的局部Laplace积分 / Compute one candidate's local Laplace integral using existing optimization and Hessian implementations.
    输入 / Inputs: 确定性似然、同坐标分布、显式laplace方法及已有数值选项 / Deterministic likelihood, same-coordinate distribution, explicit laplace method and existing options.
    输出 / Outputs: IntegrationResult；非法设置抛InputError，执行失败返回failed/None及原因 / IntegrationResult; invalid settings raise InputError, execution failures return failed/None with reasons.
    起点和返回众数用原始效应坐标；优化与求导步长用协方差标准化坐标 / Starts and returned modes use original effects; optimization and derivative steps use covariance-standardized coordinates.
    无多起点或自动重试，不证明全局众数 / No multistart or automatic retry; no global-mode guarantee.
    """
    method = validate_integration_method(integration_method)
    if method != "laplace":
        raise InputError("laplace.integrate requires integration_method='laplace'; no GHQ fallback")
    if not callable(conditional_log_likelihood):
        raise InputError("conditional_log_likelihood must be callable")
    try:
        mean = numeric_array(random_effect_distribution.mean, name="random_effect_distribution.mean")
        dimension = len(random_effect_distribution.effect_keys)
        if not callable(random_effect_distribution.log_density) or mean.shape != (dimension,):
            raise InputError("random-effect distribution has incompatible coordinates or log_density")
    except (AttributeError, TypeError) as exc:
        raise InputError("random_effect_distribution must expose mean, effect_keys and log_density") from exc
    options = validate_numerical_options(numerical_options)
    if "ghq" in options:
        raise InputError("Laplace does not consume GHQ options")
    optimization_options = options["optimization"]
    point = optimization_options.pop("initial_point", mean).copy()
    if point.shape != mean.shape:
        raise InputError("initial_point must match the active random-effect dimension")
    coordinates = None
    if dimension:
        try:
            coordinates = PriorCoordinates(mean, random_effect_distribution.covariance)
        except AttributeError as exc:
            raise InputError("random_effect_distribution must expose its active covariance") from exc
    derivative_options = options["derivatives"]
    resolve_hessian_options(derivative_options)
    details = {"integration_method": method, "dimension": dimension, "initial_point": point.tolist(),
               "scope": "direct" if dimension == 0 else "single_local_mode",
               "curvature_log_integral_atol": CURVATURE_LOG_INTEGRAL_ATOL,
               "stationarity_log_integral_atol": STATIONARITY_LOG_INTEGRAL_ATOL,
               "stage": "direct_likelihood" if dimension == 0 else "optimization"}
    result: IntegrationResult = {"status": "failed", "log_marginal_likelihood": None,
                                 "failure_reason": None, "method_details": details}
    try:
        if dimension == 0:
            value = _direct_log_likelihood(conditional_log_likelihood, random_effect_distribution)
        else:
            original_objective = _joint_objective(conditional_log_likelihood, random_effect_distribution)

            def objective(standardized_point):
                """功能 / Function: 在标准化坐标中评价同一积分目标 / Evaluate the same integration target in standardized coordinates.
                输入 / Inputs: z(d) / Standardized effects.
                输出 / Outputs: U(mean+Lz)-log|L|；保留原始密度回调 / Transformed negative log density retaining the original density callback.
                """
                return math.fsum((original_objective(coordinates.to_original(standardized_point)),
                                  -coordinates.log_abs_det))

            standardized_start = coordinates.to_standard(point)
            details.update(coordinate_system="prior_standardized",
                           original_coordinate_system="random_effects",
                           standardized_initial_point=standardized_start.tolist(),
                           log_abs_det_coordinate_transform=coordinates.log_abs_det)
            optimum = minimize(objective, standardized_start, optimization_options)
            details["optimization"] = optimum["method_details"]
            details["optimization"]["coordinate_system"] = "prior_standardized"
            if optimum["status"] != "success":
                raise _EvaluationError(optimum["failure_reason"])
            mode = optimum["minimizer"]
            details.update(conditional_mode=coordinates.to_original(mode).tolist(),
                           standardized_mode=mode.tolist(),
                           objective_value=math.fsum((optimum["objective_value"], coordinates.log_abs_det)),
                           standardized_objective_value=optimum["objective_value"], stage="hessian")
            curvature = hessian(objective, mode, derivative_options)
            details["hessian"] = curvature["method_details"]
            details["hessian"]["coordinate_system"] = "prior_standardized"
            if curvature["status"] != "success":
                raise _EvaluationError(curvature["failure_reason"])
            details["stage"] = "curvature"
            logdet, error, mode_error = _curvature_terms(curvature["hessian"], curvature["estimated_error"],
                                                        optimum["method_details"]["gradient_inf_with_error"])
            details.update(log_determinant=math.fsum((logdet, -2*coordinates.log_abs_det)),
                           standardized_log_determinant=logdet, estimated_log_integral_error_from_hessian=error,
                           estimated_log_integral_error_from_stationarity=mode_error, stage="stationarity")
            if mode_error > STATIONARITY_LOG_INTEGRAL_ATOL:
                raise _EvaluationError(f"estimated local mode contribution to log integral {mode_error:.6g} exceeds 1e-3")
            details["stage"] = "formula"
            # 同一z坐标的密度与曲率，Jacobian已在目标中计入 / Density and curvature both in z; the objective already includes the Jacobian.
            value = math.fsum((-optimum["objective_value"], .5 * dimension * math.log(2 * math.pi), -.5 * logdet))
            if not math.isfinite(value):
                raise _EvaluationError("non-finite Laplace log integral; not a proven zero likelihood")
        result.update(status="success", log_marginal_likelihood=value)
        details["stage"] = "complete"
    except Exception as exc:
        result["failure_reason"] = f"{details['stage']}: {type(exc).__name__}: {exc}"
    return validate_integration_result(result)
