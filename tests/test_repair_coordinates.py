"""功能 / Function: 验证可逆的协方差坐标变换 / Verify reversible covariance-based coordinates.
输入 / Inputs: 相关协方差、极小正方差及非法输入 / Correlated covariance, tiny variances and invalid inputs.
输出 / Outputs: 映射、Jacobian和隔离性断言 / Mapping, Jacobian and isolation assertions.
"""

import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics.coordinates import PriorCoordinates


def test_correlated_round_trip_and_jacobian():
    """功能 / Function: 核验非零均值与相关性 / Verify shift and correlation.
    输入 / Inputs: 二维正定矩阵 / SPD matrix. 输出 / Outputs: 往返和行列式相等 / Round trip and determinant equality.
    """
    mean = np.array([2., -3.])
    covariance = np.array([[4., 1.], [1., .5]])
    transform = PriorCoordinates(mean, covariance)
    z = np.array([.3, -.8])
    eta = mean + np.linalg.cholesky(covariance) @ z
    np.testing.assert_allclose(transform.to_original(z), eta, atol=1e-15)
    np.testing.assert_allclose(transform.to_standard(eta), z, atol=1e-15)
    assert transform.log_abs_det == pytest.approx(.5*np.linalg.slogdet(covariance)[1])


@pytest.mark.parametrize("variance", [1e-8, 1e-100, 1e-300])
def test_tiny_positive_variances_are_not_fixed(variance):
    """功能 / Function: 保留微小正方差 / Preserve tiny positive variance.
    输入 / Inputs: 非零方差 / Nonzero variance. 输出 / Outputs: 非零映射及精确尺度 / Nonzero mapped coordinate and scale.
    """
    transform = PriorCoordinates([0.], [[variance]])
    eta = transform.to_original([1.])
    assert eta[0] > 0
    assert eta[0] == pytest.approx(math.sqrt(variance), rel=1e-14, abs=0)
    np.testing.assert_allclose(transform.to_standard(eta), [1.], rtol=1e-14)
    assert transform.log_abs_det == pytest.approx(.5*math.log(variance))


def test_empty_transform_has_unit_jacobian():
    """功能 / Function: 支持空映射 / Support empty transform.
    输入 / Inputs: 零维 / Zero dimension. 输出 / Outputs: 空向量和零logJacobian / Empty vector and zero log Jacobian.
    """
    transform = PriorCoordinates([], np.zeros((0, 0)))
    assert transform.to_original([]).shape == (0,)
    assert transform.to_standard([]).shape == (0,)
    assert transform.log_abs_det == 0


@pytest.mark.parametrize("mean,covariance", [
    ([0.], [[0.]]), ([0., 0.], [[1., 0.], [0., 0.]]),
    ([0.], [[-1.]]), ([0.], [[np.nan]]), ([True], [[1.]]),
    ([0.], [[1., 0.], [0., 1.]]), ([0., 0.], [[1., .2], [.3, 1.]]),
    ([0., 0.], [[1., 1.], [1., 1.]]), ([np.inf], [[1.]]),
])
def test_invalid_or_unresolved_active_covariance_is_rejected(mean, covariance):
    """功能 / Function: 不自行丢弃坐标或补曲率 / Do not drop coordinates or repair covariance.
    输入 / Inputs: 非法活动分布 / Invalid active distribution. 输出 / Outputs: InputError.
    """
    with pytest.raises(InputError):
        PriorCoordinates(mean, covariance)


@pytest.mark.parametrize("point", [[1., 2.], [[1.]], [np.nan], [True]])
@pytest.mark.parametrize("method", ["to_standard", "to_original"])
def test_mapping_validates_point_shape_and_values(point, method):
    """功能 / Function: 验证映射输入 / Validate mapping inputs.
    输入 / Inputs: 非法向量 / Invalid vector. 输出 / Outputs: InputError.
    """
    with pytest.raises(InputError):
        getattr(PriorCoordinates([0.], [[1.]]), method)(point)


def test_inputs_and_returned_values_are_not_shared():
    """功能 / Function: 拒绝数组别名导致的隐式修改 / Prevent implicit changes through array aliases.
    输入 / Inputs: 可修改数组 / Mutable arrays. 输出 / Outputs: 独立变换状态 / Independent transform state.
    """
    mean, covariance, z = np.array([2.]), np.array([[4.]]), np.array([3.])
    transform = PriorCoordinates(mean, covariance)
    mean[:] = 100
    covariance[:] = 100
    eta = transform.to_original(z)
    np.testing.assert_array_equal(z, [3.])
    assert eta.tolist() == [8.]
    eta[:] = -100
    assert transform.to_original(z).tolist() == [8.]


def test_unrepresentable_output_is_not_silently_returned():
    """功能 / Function: 映射溢出明确失败 / Fail explicitly on mapping overflow.
    输入 / Inputs: 过大坐标 / Excessive coordinate. 输出 / Outputs: 浮点错误 / Floating-point error.
    """
    transform = PriorCoordinates([0.], [[1e300]])
    with pytest.raises(FloatingPointError):
        transform.to_original([1e200])


def test_unrepresentable_inverse_is_not_silently_returned():
    """功能 / Function: 逆映射溢出明确失败 / Fail explicitly on inverse-map overflow.
    输入 / Inputs: 有限但不可缩放的坐标 / Finite but unrepresentable scaled coordinate.
    输出 / Outputs: 浮点错误而非无穷大 / Floating-point error instead of infinity.
    """
    transform = PriorCoordinates([0.], [[1e-300]])
    with pytest.raises(FloatingPointError):
        transform.to_standard([1e300])


def test_inverse_subtraction_overflow_is_explicit():
    """功能 / Function: 检查去均值溢出 / Check overflow while subtracting the mean.
    输入 / Inputs: 相反极大坐标 / Opposite extreme coordinates. 输出 / Outputs: 浮点错误 / Floating-point error.
    """
    with pytest.raises(FloatingPointError):
        PriorCoordinates([-1e308], [[1.]]).to_standard([1e308])
