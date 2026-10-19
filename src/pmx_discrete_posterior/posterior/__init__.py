"""功能 / Function: 候选状态的后验计算 / Posterior computation across candidate states.
输入 / Inputs: 模型、观测、候选及先验 / Model, observations, candidates and priors.
输出 / Outputs: 完整候选集合的概率或明确失败 / Probabilities for the complete candidate set or explicit failure.
"""

from .computation import compute_posterior

__all__ = ["compute_posterior"]
