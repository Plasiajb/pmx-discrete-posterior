"""功能 / Function: 积分近似名称契约 / Integration-approximation name contract.
输入 / Inputs: 明确的方法标识 / Explicit method identifier.
输出 / Outputs: 已校验标识；不执行或声明后端可用 / Validated name, not execution or backend availability.
"""

from ..validation import InputError


def validate_integration_method(integration_method):
    """功能 / Function: 区分Laplace与GHQ选择 / Validate the Laplace/GHQ selection.
    输入 / Inputs: 精确小写字符串laplace或ghq / Exact lowercase string laplace or ghq.
    输出 / Outputs: 原标识或InputError，不静默替换 / Original identifier or InputError, no fallback.
    接受名称不表示相应积分器已实现 / Accepting a name does not imply an implemented integrator.
    """
    if not isinstance(integration_method, str) or integration_method not in ("laplace", "ghq"):
        raise InputError("integration_method must be laplace or ghq")
    return integration_method
