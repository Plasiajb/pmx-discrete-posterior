"""功能 / Function: 积分近似名称契约 / Integration-approximation name contract.
输入 / Inputs: 明确的方法标识 / Explicit method identifier.
输出 / Outputs: 已校验标识；不执行或声明后端可用 / Validated name, not execution or backend availability.
"""

from ..validation import InputError


def validate_integration_method(integration_method):
    """功能 / Function: 校验唯一原生Laplace路线 / Validate the sole native Laplace route.
    输入 / Inputs: 精确小写字符串laplace / Exact lowercase string laplace.
    输出 / Outputs: 原标识或InputError，不静默替换 / Original identifier or InputError, no fallback.
    外部R aghq不是本包后端 / External R aghq is not a package backend.
    """
    if isinstance(integration_method, str) and integration_method == "ghq":
        raise InputError("native GHQ has been removed; external R aghq is a separate documented workflow, "
                         "not a package backend; integration_method must be 'laplace' (no fallback)")
    if not isinstance(integration_method, str) or integration_method != "laplace":
        raise InputError("integration_method must be 'laplace'; external R aghq is not a package backend")
    return integration_method
