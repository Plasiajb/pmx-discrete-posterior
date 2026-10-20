"""功能 / Function: 用分布协方差构造可逆计算坐标 / Build reversible computation coordinates from a distribution covariance.
输入 / Inputs: 活动坐标的均值和正定协方差 / Active-coordinate mean and positive-definite covariance.
输出 / Outputs: eta = mean + L z及其逆映射和log|L| / Forward/inverse mappings and log|L|.
不假定密度为正态，不丢弃活动坐标 / Does not assume Gaussian density or remove active coordinates.
"""

import math

import numpy as np
from scipy.linalg import solve_triangular

from ..interfaces.validation import InputError, numeric_array, validate_covariance


class PriorCoordinates:
    """功能 / Function: 复用协方差校验及SciPy三角求解 / Reuse covariance validation and SciPy triangular solves.
    输入 / Inputs: 均值(d)、协方差(d,d) / Mean and covariance.
    输出 / Outputs: 保留原始效应维度的仿射变换 / Affine transform preserving the active-effect dimension.
    """

    def __init__(self, mean, covariance):
        """功能 / Function: 建立独立的变换参数 / Build independent transform parameters.
        输入 / Inputs: 活动分布元数据 / Active distribution metadata.
        输出 / Outputs: Cholesky因子及一次性Jacobian / Cholesky factor and one Jacobian term.
        """
        self._mean = numeric_array(mean, name="coordinate mean")
        matrix = numeric_array(covariance, ndim=2, name="coordinate covariance")
        if matrix.shape != (self._mean.size, self._mean.size):
            raise InputError("coordinate covariance must match the active mean dimension")
        resolved = validate_covariance(matrix)
        if len(resolved["active"]) != self._mean.size:
            raise InputError("coordinate covariance must already contain active coordinates only")
        self._factor = resolved["factor"]
        self.log_abs_det = math.fsum(math.log(float(x)) for x in np.diag(self._factor))

    def to_original(self, standardized_point):
        """功能 / Function: 将计算坐标映射至模型效应 / Map computation coordinates to model effects.
        输入 / Inputs: z(d) / Standardized vector.
        输出 / Outputs: eta(d)，新数组；溢出报错 / Original vector, newly allocated; overflow fails.
        """
        point = numeric_array(standardized_point, name="standardized_point")
        if point.shape != self._mean.shape:
            raise InputError("standardized_point must match the active dimension")
        with np.errstate(over="raise", invalid="raise"):
            return self._mean + self._factor @ point

    def to_standard(self, original_point):
        """功能 / Function: 变换用户起点及原始效应 / Transform user starts and original effects.
        输入 / Inputs: eta(d) / Original vector.
        输出 / Outputs: z(d)，不显式求逆 / Standardized vector without explicit inversion.
        """
        point = numeric_array(original_point, name="original_point")
        if point.shape != self._mean.shape:
            raise InputError("original_point must match the active dimension")
        if not point.size:
            return point
        with np.errstate(over="raise", invalid="raise"):
            result = solve_triangular(self._factor, point - self._mean, lower=True, check_finite=True)
        if not np.isfinite(result).all():
            raise FloatingPointError("standardized coordinate is not numerically representable")
        return result
