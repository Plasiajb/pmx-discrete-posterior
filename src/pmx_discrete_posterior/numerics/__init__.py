"""功能 / Function: 导出逐候选积分入口 / Export per-candidate integration.
输入 / Inputs: 条件似然、效应分布及显式方法 / Conditional likelihood, effect distribution and explicit method.
输出 / Outputs: Laplace对数边际似然及状态 / Laplace log marginal likelihood and status.
"""

from .integration import integrate

__all__ = ["integrate"]
