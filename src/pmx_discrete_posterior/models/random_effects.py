"""功能 / Function: 构造零均值正态IIV/IOV / Resolve zero-mean normal IIV/IOV.
输入 / Inputs: 分布声明、协方差参数引用、occasion / Distribution, covariance references, occasions.
输出 / Outputs: 活动分布和具名效应映射 / Active distribution and named-effect mapping.
"""

import math

import numpy as np
from scipy.linalg import block_diag, solve_triangular

from ..interfaces.inputs.model_inputs import validate_random_effect_spec
from ..interfaces.validation import InputError, numeric_array, validate_covariance, validate_parameters
from .parameter_resolution import validate_occasions


class RandomEffectDistribution:
    """功能 / Function: 保存活动正态坐标和确定性零分量 / Hold active normal coordinates and fixed zeros.
    输入 / Inputs: 全效应键及协方差 / Full effect keys and covariance.
    输出 / Outputs: 密度与展开方法 / Log-density and named-coordinate expansion.
    """

    def __init__(self, effect_keys, covariance):
        """功能 / Function: 校验协方差并降去严格零方差 / Validate covariance and remove exact zero variances.
        输入 / Inputs: (level, occasion, name)键，方差/协方差矩阵 / Keys and variance-covariance matrix.
        输出 / Outputs: 已校验对象；其他秩亏矩阵拒绝 / Validated object; other singular matrices rejected.
        """
        self._all_keys = tuple(effect_keys)
        for key in self._all_keys:
            if len(key) != 3 or key[0] not in ("iiv", "iov") or not isinstance(key[2], str):
                raise InputError("invalid random-effect key")
            if (key[0] == "iiv" and key[1] is not None) or (key[0] == "iov" and not isinstance(key[1], str)):
                raise InputError("effect key has an invalid occasion")
        if len(set(self._all_keys)) != len(self._all_keys):
            raise InputError("duplicate random-effect keys")
        covariance = numeric_array(covariance, ndim=2, name="covariance")
        if covariance.shape != (len(self._all_keys), len(self._all_keys)):
            raise InputError("covariance shape does not match effect keys")
        checked = validate_covariance(covariance)
        self._active, self._covariance, self._factor = checked["active"], checked["covariance"], checked["factor"]
        self._log_normalizer = len(self._active) * math.log(2 * math.pi) / 2 + np.log(np.diag(self._factor)).sum()

    @property
    def effect_keys(self):
        """功能 / Function: 活动坐标顺序 / Active coordinate order.
        输入 / Inputs: 对象 / Object. 输出 / Outputs: 不可变键元组 / Immutable keys.
        """
        return tuple(self._all_keys[i] for i in self._active)

    @property
    def mean(self):
        """功能 / Function: 正态均值 / Normal mean.
        输入 / Inputs: 对象 / Object. 输出 / Outputs: 零向量副本 / Zero-vector copy.
        """
        return np.zeros(len(self._active))

    @property
    def covariance(self):
        """功能 / Function: 活动协方差 / Active covariance.
        输入 / Inputs: 对象 / Object. 输出 / Outputs: 独立矩阵副本 / Independent matrix copy.
        """
        return self._covariance.copy()

    def _point(self, values):
        """功能 / Function: 验证积分坐标 / Validate integration coordinates.
        输入 / Inputs: 活动向量 / Active vector. 输出 / Outputs: 有限副本 / Finite copy.
        """
        point = numeric_array(values, name="random_effect_values")
        if point.shape != (len(self._active),):
            raise InputError("random-effect vector has the wrong dimension")
        return point

    def log_density(self, random_effect_values):
        """功能 / Function: 计算规范化正态对数密度 / Compute normalized normal log-density.
        输入 / Inputs: 活动效应向量 / Active effects.
        输出 / Outputs: 标量；零维为0 / Scalar; zero in zero dimensions.
        """
        point = self._point(random_effect_values)
        whitened = solve_triangular(self._factor, point, lower=True) if len(point) else point
        with np.errstate(over="ignore", invalid="ignore"):
            result = -self._log_normalizer - float(whitened @ whitened) / 2
        if not np.isfinite(result):
            raise InputError("random-effect density overflow; not a proven zero density")
        return float(result)

    def expand(self, random_effect_values):
        """功能 / Function: 活动坐标展开为IIV/IOV / Expand active coordinates into IIV/IOV.
        输入 / Inputs: 当前活动向量 / Current active vector.
        输出 / Outputs: iiv_values、iov_values；固定零恢复 / Named effects with fixed zeros restored.
        """
        point = self._point(random_effect_values)
        full = np.zeros(len(self._all_keys))
        full[self._active] = point
        iiv, iov = {}, {}
        for (level, occasion, name), value in zip(self._all_keys, full):
            if level == "iiv":
                iiv[name] = float(value)
            else:
                iov.setdefault(occasion, {})[name] = float(value)
        return {"iiv_values": iiv, "iov_values": iov}


def _resolve_block(spec, parameters, *, is_iov):
    """功能 / Function: 解析协方差引用 / Resolve covariance references.
    输入 / Inputs: 正态块定义和参数 / Normal block and parameters.
    输出 / Outputs: 效应名、方差矩阵 / Effect names and covariance matrix.
    """
    spec = validate_random_effect_spec(spec, parameters=parameters, level="iov" if is_iov else "iiv")
    effect_names = spec["names"]
    refs = spec["covariance_parameters"]
    covariance = np.zeros((len(refs), len(refs)))
    for i, row in enumerate(refs):
        for j, ref in enumerate(row):
            if ref is None:  # 功能 / Function: 显式零协方差 / Explicit zero covariance.
                continue
            covariance[i, j] = parameters["values"][ref]
    return effect_names, covariance


def resolve_random_effects(*, iiv, iov, parameters, occasions, dependence):
    """功能 / Function: 组装声明独立的正态效应块 / Assemble explicitly independent normal blocks.
    输入 / Inputs: IIV/IOV或None，参数、occasion、dependence='independent_blocks'.
    输出 / Outputs: RandomEffectDistribution；跨组相关不支持 / Resolved distribution; cross-block dependence rejected.
    """
    if dependence != "independent_blocks":
        raise InputError("declare dependence='independent_blocks'; cross-block dependence is unsupported")
    parameters = validate_parameters(parameters)
    occasions = validate_occasions(occasions)
    keys, matrices = [], []
    if iiv is not None:
        effect_names, covariance = _resolve_block(iiv, parameters, is_iov=False)
        keys.extend(("iiv", None, name) for name in effect_names)
        matrices.append(covariance)
    if iov is not None:
        if not occasions:
            raise InputError("IOV requires explicit occasions")
        effect_names, covariance = _resolve_block(iov, parameters, is_iov=True)
        for occasion in occasions:
            keys.extend(("iov", occasion["occasion_id"], name) for name in effect_names)
            matrices.append(covariance)
    covariance = block_diag(*matrices) if matrices else np.zeros((0, 0))
    return RandomEffectDistribution(keys, covariance)
