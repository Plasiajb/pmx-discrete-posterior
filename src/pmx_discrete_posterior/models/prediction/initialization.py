"""功能 / Function: 线性房室系统条件周期初态 / Conditional periodic initialization for linear compartments.
输入 / Inputs: 当前参数的速率矩阵、周期给药和滞后 / Current rate matrix, periodic inputs and lags.
输出 / Outputs: 边界前药量与未触发输入 / Pre-event amounts and pending inputs; no implicit future dosing.
"""

from decimal import Decimal
import math
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, expm, solve

from ...interfaces.validation import InputError, ModelExecutionError, fields, numeric_array, scalar


def clock_add(left, right):
    """功能 / Function: 十进制时间相加 / Add decimal event times.
    输入 / Inputs: 有限时间 / Finite times. 输出 / Outputs: 有限时间和 / Finite sum, not an epsilon adjustment.
    """
    return scalar(float(Decimal(str(scalar(left))) + Decimal(str(scalar(right)))), name="event time")


def validate_matrix(matrix):
    """功能 / Function: 校验线性房室速率矩阵 / Validate a linear compartment rate matrix.
    输入 / Inputs: 同一时间单位下的方阵 / Square matrix in the common inverse-time unit.
    输出 / Outputs: 有限Metzler矩阵副本 / Finite Metzler matrix copy.
    """
    matrix = numeric_array(matrix, ndim=2, name="rate matrix")
    if not len(matrix) or matrix.shape[0] != matrix.shape[1]:
        raise InputError("rate matrix must be nonempty and square")
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.)
    if (off_diagonal < 0).any() or (np.diag(matrix) > 0).any():
        raise InputError("compartment rates require nonnegative off-diagonals and nonpositive diagonals")
    return matrix


def advance(matrix, state, duration):
    """功能 / Function: 连续传播药物状态 / Propagate drug amounts continuously.
    输入 / Inputs: 速率矩阵、药量向量、非负时长 / Rate matrix, amount vector, nonnegative duration.
    输出 / Outputs: 更新药量，无截断或清零 / Updated amounts without clipping or reset.
    """
    matrix = validate_matrix(matrix)
    state = numeric_array(state, name="drug state")
    duration = scalar(duration, name="duration", minimum=0)
    if state.shape != (len(matrix),) or (state < 0).any():
        raise InputError("drug amounts must match compartments and be nonnegative")
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = expm(matrix * duration) @ state
        except (FloatingPointError, ValueError) as exc:
            raise ModelExecutionError("linear propagation failed") from exc
    if not np.isfinite(result).all() or (result < 0).any():
        raise ModelExecutionError("invalid propagated drug amounts; no clipping")
    return result


def periodic_state(*, matrix, period, anchor_time, start_time, input_events):
    """功能 / Function: 对历史周期输入求和 / Sum periodic inputs strictly before the boundary.
    输入 / Inputs: 稳定速率矩阵，周期/锚点/起点；每事件offset、lag、values药量向量 / Stable rates, clocks, input vectors.
    输出 / Outputs: values与pending_inputs；边界后的计划给药不自动添加 / Amounts and pending releases, not future administrations.
    """
    matrix = validate_matrix(matrix)
    period = scalar(period, name="period", positive=True)
    anchor_time, start_time = scalar(anchor_time), scalar(start_time)
    if not isinstance(input_events, (list, tuple)) or not input_events:
        raise InputError("periodic initialization needs input events")
    if np.any(np.linalg.eigvals(matrix).real >= 0):
        raise InputError("periodic steady state requires asymptotically stable dynamics")
    # 功能 / Function: 用增广指数计算expm(A*tau)-I / Compute expm(A*tau)-I by augmentation.
    # 输入 / Inputs: 当前速率 / Current rates. 输出 / Outputs: 稳态算子，避免近单位矩阵相减 / Operator without near-identity subtraction.
    n = len(matrix)
    augmented = np.zeros((2*n, 2*n))
    augmented[:n, :n] = matrix * period
    augmented[:n, n:] = matrix * period
    operator = -expm(augmented)[:n, n:]
    state = np.zeros(len(matrix))
    pending = []
    for event in input_events:
        fields(event, {"offset", "lag", "values"}, name="resolved periodic input")
        offset = scalar(event["offset"], minimum=0)
        lag = scalar(event["lag"], minimum=0)
        vector = numeric_array(event["values"], name="input amounts")
        if offset >= period or vector.shape != state.shape or (vector < 0).any():
            raise InputError("invalid periodic offset or input vector")
        base = Decimal(str(anchor_time)) + Decimal(str(offset))
        boundary, tau, delay = Decimal(str(start_time)), Decimal(str(period)), Decimal(str(lag))
        last_admin = math.ceil((boundary - base) / tau) - 1
        last_release = min(last_admin, math.ceil((boundary - base - delay) / tau) - 1)
        if last_admin - last_release > 100000:
            raise InputError("more than 100000 pending historical inputs; not truncated")
        # 功能 / Function: 无穷历史几何级数 / Infinite-history geometric series.
        # 输入 / Inputs: 最后已触发输入 / Last released input. 输出 / Outputs: 起点前药量 / Pre-boundary amounts.
        age = float(boundary - (base + delay + last_release * tau))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                summed = solve(operator, vector)
            state += advance(matrix, summed, age)
        except (np.linalg.LinAlgError, LinAlgWarning) as exc:
            raise ModelExecutionError("ill-conditioned periodic-state solve") from exc
        for n in range(last_release + 1, last_admin + 1):
            release = scalar(float(base + delay + n * tau), name="pending release time")
            pending.append({"time": release, "values": vector.copy()})
    if not np.isfinite(state).all() or (state < 0).any():
        raise ModelExecutionError("invalid periodic state")
    return {"values": state, "pending_inputs": sorted(pending, key=lambda event: event["time"])}
