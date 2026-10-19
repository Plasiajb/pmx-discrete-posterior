"""功能 / Function: 模型层的最小数据契约 / Minimal model data contracts.
输入 / Inputs: 映射、标量、数组 / Mappings, scalars and arrays.
输出 / Outputs: 校验后的副本；非法输入报错 / Validated copies or explicit errors.
"""

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy.linalg import cholesky


class InputError(ValueError):
    """功能 / Function: 标识非法输入 / Identify invalid input.
    输入 / Inputs: 原因 / Reason. 输出 / Outputs: 异常 / Exception.
    """


class ModelExecutionError(RuntimeError):
    """功能 / Function: 标识回调或模型计算失败 / Identify model execution failure.
    输入 / Inputs: 原因 / Reason. 输出 / Outputs: 异常 / Exception.
    """


def fields(value, required, optional=(), *, name="record"):
    """功能 / Function: 拒绝缺失或未知字段 / Reject missing or unknown fields.
    输入 / Inputs: 映射及字段集合 / Mapping and field sets.
    输出 / Outputs: 原映射 / Original mapping, or InputError.
    """
    if not isinstance(value, Mapping):
        raise InputError(f"{name} must be a mapping")
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing or unknown:
        raise InputError(f"{name}: missing={missing}, unsupported={unknown}")
    return value


def scalar(value, *, name="value", minimum=None, positive=False):
    """功能 / Function: 检查实数及范围 / Check real scalar and range.
    输入 / Inputs: 数值、可选下限 / Value and optional lower bound.
    输出 / Outputs: 有限float / Finite float, or InputError.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InputError(f"{name} must be a real number, not a string or bool")
    try:
        value = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise InputError(f"{name} cannot be represented as a finite float") from exc
    if not np.isfinite(value) or (positive and value <= 0):
        raise InputError(f"{name} must be finite" + (" and positive" if positive else ""))
    if minimum is not None and value < minimum:
        raise InputError(f"{name} must be >= {minimum}")
    return value


def probability_scalar(value, *, name="probability"):
    """功能 / Function: 检查单项概率，拒绝转换掩盖越界或非零值 / Validate a probability without hiding invalid bounds or nonzero values during conversion.
    输入 / Inputs: [0,1]内实数，不接收布尔值；允许严格零 / Real number in [0,1], excluding booleans; exact zero is allowed.
    输出 / Outputs: float；非零值若下溢为零则抛InputError，不截断 / Float; raise InputError if a nonzero value underflows to zero, without clipping.
    """
    checked = scalar(value, name=name)
    if value < 0 or value > 1:
        raise InputError(f"{name} must be a probability in [0, 1]")
    if value != 0 and checked == 0:
        raise InputError(f"{name}: nonzero probability cannot be represented as a nonzero float")
    return checked


def identifier(value, *, name="identifier"):
    """功能 / Function: 校验显式名称 / Validate an explicit name.
    输入 / Inputs: 字符串 / String. 输出 / Outputs: 非空名称 / Nonempty name.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InputError(f"{name} must be a nonempty unpadded string")
    return value


def names(values, *, name="names", allow_empty=False):
    """功能 / Function: 保持名称顺序并排除重复 / Preserve order and reject duplicates.
    输入 / Inputs: 名称序列 / Name sequence.
    输出 / Outputs: 名称元组 / Tuple of names.
    """
    if isinstance(values, (str, bytes)):
        raise InputError(f"{name} must be a sequence, not a string")
    try:
        result = tuple(identifier(v, name=name) for v in values)
    except TypeError as exc:
        raise InputError(f"{name} must be a sequence") from exc
    if (not result and not allow_empty) or len(set(result)) != len(result):
        raise InputError(f"{name} must be nonempty and unique (unless empty is allowed)")
    return result


def numeric_array(value, *, name="array", ndim=1):
    """功能 / Function: 建立有限数值数组副本 / Build a finite numeric array copy.
    输入 / Inputs: 数值序列及维度 / Numeric sequence and dimension.
    输出 / Outputs: float数组 / Float array, or InputError.
    """
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{name} must be a rectangular numeric array") from exc
    if raw.ndim != ndim or raw.dtype.kind not in "iuf":
        raise InputError(f"{name} must have {ndim} dimensions and real numeric entries")
    if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
        raise InputError(f"{name} must not contain boolean entries, including mixed numeric sequences")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise InputError(f"{name} must contain only finite values")
    return result


def validate_parameters(parameters):
    """功能 / Function: 核对参数值和单位一一对应 / Validate value-unit correspondence.
    输入 / Inputs: parameters.values与units；无量纲单位写1 / Values and units; use '1' for dimensionless.
    输出 / Outputs: 独立字典副本 / Independent dictionaries; no unit conversion.
    """
    fields(parameters, {"values", "units"}, name="parameters")
    values, units = parameters["values"], parameters["units"]
    if not isinstance(values, Mapping) or not isinstance(units, Mapping):
        raise InputError("parameter values and units must be mappings")
    keys = names(values, name="parameter names", allow_empty=True)
    if set(keys) != set(units):
        raise InputError("every parameter needs exactly one unit")
    return {"values": {k: scalar(values[k], name=k) for k in keys},
            "units": {k: identifier(units[k], name=f"unit of {k}") for k in keys}}


def validate_covariance(covariance):
    """功能 / Function: 复用正态协方差接纳检查 / Share normal-covariance admissibility checks.
    输入 / Inputs: 有限实数方阵；严格零方差可固定 / Finite square matrix; exact zero variances may be fixed.
    输出 / Outputs: active索引、活动covariance及Cholesky factor / Active indices, covariance and factor, or InputError.
    不加jitter，不修补非对称或秩亏 / No jitter, symmetrization or rank repair.
    """
    covariance = numeric_array(covariance, ndim=2, name="covariance")
    if covariance.shape[0] != covariance.shape[1]:
        raise InputError("covariance must be square")
    if not np.array_equal(covariance, covariance.T):
        raise InputError("covariance must be symmetric; no silent symmetrization")
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0):
        raise InputError("variances cannot be negative")
    zero = diagonal == 0
    if np.any(covariance[zero, :] != 0):
        raise InputError("zero-variance coordinates must have zero covariance")
    active = np.flatnonzero(~zero)
    matrix = covariance[np.ix_(active, active)]
    try:
        if len(active):
            sd = np.sqrt(np.diag(matrix))
            with np.errstate(over="ignore", invalid="ignore"):
                correlation = (matrix / sd[:, None]) / sd[None, :]
            if not np.isfinite(correlation).all():
                raise InputError("active correlation is not numerically representable")
            eigenvalues = np.linalg.eigvalsh(correlation)
            rank_threshold = np.finfo(float).eps * len(active) * max(abs(eigenvalues))
            if eigenvalues[0] <= rank_threshold:
                raise InputError("active correlation is singular or numerically unresolved; no jitter")
        factor = cholesky(matrix, lower=True) if len(active) else np.zeros((0, 0))
    except np.linalg.LinAlgError as exc:
        raise InputError("active covariance must be positive definite; no jitter") from exc
    return {"active": active, "covariance": matrix, "factor": factor}
