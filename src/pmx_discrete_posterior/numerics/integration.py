"""功能 / Function: 选择已实现的积分方法 / Select the implemented integration method.
输入 / Inputs: 原有统一积分参数 / Existing unified integration arguments.
输出 / Outputs: 原积分器结果，不重复计算或转换 / Original integrator result without recalculation or conversion.
"""

from ..interfaces.inputs.integration_method import validate_integration_method
from . import laplace


def integrate(*, conditional_log_likelihood, random_effect_distribution,
              integration_method, numerical_options=None):
    """功能 / Function: 按显式方法分派一次，不回退 / Dispatch once by the explicit method, without fallback.
    输入 / Inputs: 似然、效应分布、方法及设置 / Likelihood, effect distribution, method and settings.
    输出 / Outputs: 所选积分器的IntegrationResult / Selected integrator's IntegrationResult.
    """
    method = validate_integration_method(integration_method)
    return laplace.integrate(conditional_log_likelihood=conditional_log_likelihood,
                             random_effect_distribution=random_effect_distribution,
                             integration_method=method, numerical_options=numerical_options)
