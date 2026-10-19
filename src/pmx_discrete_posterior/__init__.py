"""功能 / Function: 离散状态后验计算开发包 / Discrete-state posterior development package.
输入 / Inputs: 显式模型、个体数据、观测及候选 / Explicit model, individual data, observations and candidates.
输出 / Outputs: 输入/结果校验、模型评价、Laplace/GHQ积分与完整后验 / Input/result validation, model evaluation, Laplace/GHQ integration and complete posteriors.
公开入口compute_posterior支持Laplace和先验中心GHQ / Public compute_posterior supports Laplace and prior-centered GHQ.
"""

__all__ = ["compute_posterior"]


def __getattr__(name):
    """功能 / Function: 按需导出同一后验函数，保持接口导入独立 / Lazily export the same posterior function, preserving isolated interface imports.
    输入 / Inputs: 待读取的包属性名 / Requested package attribute name.
    输出 / Outputs: compute_posterior原函数；未知属性抛AttributeError，不包装计算 / Original compute_posterior function; unknown names raise AttributeError, without wrapping execution.
    """
    if name != "compute_posterior":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .posterior import compute_posterior
    globals()[name] = compute_posterior
    return compute_posterior
