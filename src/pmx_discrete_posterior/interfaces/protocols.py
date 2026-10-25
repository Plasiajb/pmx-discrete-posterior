"""功能 / Function: 声明回调签名，不增加执行包装 / Declare callback signatures without execution wrappers.
输入 / Inputs: 按既有接口提供的函数或可调用对象 / Functions or callable objects using the existing interfaces.
输出 / Outputs: 供类型检查和阅读使用的Protocol / Protocols for static typing and source navigation.
协议不做运行时验收；值、单位和输出检查由既有执行层完成 / Protocols do not validate runtime values, units or results.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .results import IntegrationResult


class ParameterRelations(Protocol):
    """功能 / Function: 约定个体参数关系回调 / Specify the individual-parameter callback.
    输入 / Inputs: 模型参数、协变量、当前具名IIV/IOV及时间背景 / Model parameters, covariates, named current effects and time context.
    输出 / Outputs: units与覆盖评价区间的segments / Units and segments covering the evaluation interval.
    用户无需继承；resolve_parameters负责调用及检查 / No inheritance required; resolve_parameters calls and validates it.
    """

    def __call__(
        self, *, parameters: Mapping[str, Any], covariates: Mapping[str, float],
        iiv_values: Mapping[str, float], iov_values: Mapping[str, Mapping[str, float]],
        occasions: Sequence[Mapping[str, Any]], evaluation_interval: tuple[float, float],
    ) -> Mapping[str, Any]:
        """功能 / Function: 将当前效应作用一次，形成个体参数 / Apply current effects once to resolve individual parameters.
        输入 / Inputs: 显式参数值/单位、协变量、IIV/IOV值、occasion及闭合评价范围 / Explicit values/units, covariates, effects, occasions and evaluation bounds.
        输出 / Outputs: units与segments；每段含start/end/occasion_id/values / Units and segments with start/end/occasion_id/values.
        不抽样，不读取观测浓度或状态先验 / No sampling, observed concentrations or state priors.
        """
        ...


class Predict(Protocol):
    """功能 / Function: 约定无残差的结构浓度预测 / Specify noise-free structural concentration prediction.
    输入 / Inputs: 已解析个体参数、候选事件、预测请求和个体背景 / Resolved parameters, candidate events, prediction requests and context.
    输出 / Outputs: 按请求顺序对应的record_ids、values、units / Record IDs, values and units aligned to requests.
    既有evaluate_prediction执行调用与输出检查 / Existing evaluate_prediction calls and validates the predictor.
    """

    def __call__(
        self, *, individual_parameters: Mapping[str, Any], candidate_state: Mapping[str, Any],
        prediction_records: Sequence[Mapping[str, Any]], individual_data: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """功能 / Function: 根据候选历史预测指定时点浓度 / Predict concentrations at requested times for a candidate history.
        输入 / Inputs: 参数分段、候选事件、不含实测值的请求、初始化/共同前史/occasion / Parameters, events, value-free requests and initialization/history/occasions.
        输出 / Outputs: 同长度非负浓度向量、逐记录标识和单位 / Same-length nonnegative concentrations with record identities and units.
        初始化在预测器内部完成；不重复施加IIV/IOV、不抽样残差 / Initialize inside the predictor; do not reapply IIV/IOV or sample residuals.
        显式event_phase须遵循或报错，不得静默忽略 / Honor explicit event_phase or reject it; never silently ignore it.
        before_events先读出再切换和输入；after_events相反 / before_events reads before switches/inputs; after_events reads after them.
        """
        ...


class ConditionalLogLikelihood(Protocol):
    """功能 / Function: 约定已绑定候选状态的条件似然函数 / Specify a conditional likelihood bound to one candidate.
    输入 / Inputs: 按分布坐标排列的一维活动随机效应向量 / One-dimensional active random effects in distribution order.
    输出 / Outputs: 含观测分布归一化常数的标量对数似然 / Scalar observation log-likelihood including normalization constants.
    由后续调度层绑定已有模型链，不是重新实现models.likelihood / Future orchestration binds the existing model chain, not a second likelihood formula.
    """

    def __call__(self, random_effect_values: NDArray[np.float64]) -> float:
        """功能 / Function: 在给定效应值下评价同一候选的观测 / Evaluate observations for the same candidate at given effects.
        输入 / Inputs: shape=(d,)；单效应仍为向量，零维为长度0 / Shape (d,); one effect remains a vector, no effects means length zero.
        输出 / Outputs: 条件观测对数似然，不含随机效应密度或状态先验 / Conditional observation log-likelihood, excluding effect density and state priors.
        当前高斯实现要求有限值，模型错误不得转换为零似然 / Current Gaussian implementation requires finite values; model errors are not zero likelihoods.
        """
        ...


class RandomEffectLogDensity(Protocol):
    """功能 / Function: 约定随机效应对数密度函数 / Specify a random-effect log-density function.
    输入 / Inputs: 活动效应向量 / Active-effect vector.
    输出 / Outputs: 规范化联合对数密度 / Normalized joint log-density.
    直接复用RandomEffectDistribution.log_density，无额外包装 / Use RandomEffectDistribution.log_density directly without a wrapper.
    """

    def __call__(self, random_effect_values: NDArray[np.float64]) -> float:
        """功能 / Function: 评价连续随机效应密度 / Evaluate the continuous random-effect density.
        输入 / Inputs: shape=(d,)且与effect_keys顺序一致 / Shape (d,) in effect_keys order.
        输出 / Outputs: 标量对数密度；零维为0，不含观测或状态先验 / Scalar log-density; zero for d=0, excluding observations and state priors.
        """
        ...


class RandomEffectDistributionProtocol(Protocol):
    """功能 / Function: 描述既有分布对象的读取接口 / Describe the existing distribution object's read interface.
    输入 / Inputs: 已解析的随机效应分布 / A resolved random-effect distribution.
    输出 / Outputs: 活动坐标、均值、协方差、密度及具名展开 / Active coordinates, mean, covariance, density and named expansion.
    不重建分布；首版由models.random_effects提供正态实现 / Does not rebuild distributions; models.random_effects supplies the current normal implementation.
    """

    @property
    def effect_keys(self) -> tuple[tuple[str, str | None, str], ...]:
        """功能 / Function: 声明活动坐标次序 / Specify active coordinate order.
        输入 / Inputs: 对象 / Object. 输出 / Outputs: (level, occasion_id, name)键元组 / Tuple of coordinate keys.
        """
        ...

    @property
    def mean(self) -> NDArray[np.float64]:
        """功能 / Function: 声明均值向量 / Specify the mean vector.
        输入 / Inputs: 对象 / Object. 输出 / Outputs: shape=(d,)副本 / Copy with shape (d,).
        """
        ...

    @property
    def covariance(self) -> NDArray[np.float64]:
        """功能 / Function: 声明活动协方差 / Specify active covariance.
        输入 / Inputs: 对象 / Object. 输出 / Outputs: shape=(d,d)副本 / Copy with shape (d,d).
        """
        ...

    def log_density(self, random_effect_values: NDArray[np.float64]) -> float:
        """功能 / Function: 评价规范化联合密度 / Evaluate normalized joint density.
        输入 / Inputs: 活动效应向量 / Active effects. 输出 / Outputs: 标量对数密度 / Scalar log-density.
        """
        ...

    def expand(self, random_effect_values: NDArray[np.float64]) -> Mapping[str, Any]:
        """功能 / Function: 将活动坐标恢复为具名效应 / Expand active coordinates into named effects.
        输入 / Inputs: 活动向量 / Active vector. 输出 / Outputs: iiv_values、iov_values，含确定性零分量 / Named IIV/IOV including fixed zeros.
        """
        ...


class Integrator(Protocol):
    """功能 / Function: 约定逐候选边际似然积分接口 / Specify per-candidate marginal-likelihood integration.
    输入 / Inputs: 条件似然回调、随机效应分布、方法名及数值设置 / Conditional likelihood, effect distribution, method and options.
    输出 / Outputs: 对数边际似然及计算状态 / Log marginal likelihood and calculation status.
    仅定义接口；numerics仅实现原生Laplace / Interface only; numerics implements native Laplace only.
    """

    def __call__(
        self, *, conditional_log_likelihood: ConditionalLogLikelihood,
        random_effect_distribution: RandomEffectDistributionProtocol,
        integration_method: Literal["laplace"],
        numerical_options: Mapping[str, Any] | None = None,
    ) -> IntegrationResult:
        """功能 / Function: 对当前候选的随机效应积分 / Integrate random effects for the current candidate.
        输入 / Inputs: 不含任何先验的条件观测似然、已解析效应分布、显式积分方法 / Observation likelihood without priors, resolved effect distribution and explicit method.
        输出 / Outputs: log_marginal_likelihood、status、failure_reason及可选method_details / Marginal log-likelihood, status, reason and optional method details.
        真零似然可为-inf；数值失败必须为failed/None，零先验跳过由调度层表示 / True zero may be -inf; numerical failure is failed/None, and zero-prior skips belong to orchestration.
        效应密度仅计入一次；状态先验和跨候选归一化留给posterior / Include effect density once; state priors and normalization belong to posterior.
        """
        ...
