"""功能 / Function: 约定并校验输入，不执行模型 / Define and validate inputs without evaluating models.
输入 / Inputs: 模型、个体、观测、候选、先验及计算配置 / Model, individual data, observations, candidates, priors and settings.
输出 / Outputs: 输入校验、调用协议、基础结果类型及校验 / Input validation, call protocols and basic result types/validation.
protocols仅声明签名，results不执行积分或后验归一化 / Protocols declare signatures; results do not integrate or normalize posteriors.
"""
