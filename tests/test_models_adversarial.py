"""功能 / Function: 独立检查模型契约边界 / Independently probe model contracts.
输入 / Inputs: 本文件构造的显式输入和回调 / Explicit local inputs and callbacks.
输出 / Outputs: pytest断言；由主代理归档执行 / Assertions for archived execution by the main agent.
"""

import math
from copy import deepcopy
from unittest.mock import Mock

import pytest

from pmx_discrete_posterior.interfaces.validation import (
    InputError,
    ModelExecutionError,
    scalar,
)
from pmx_discrete_posterior.models.likelihood import conditional_log_likelihood
from pmx_discrete_posterior.models.parameter_resolution import (
    parameters_at,
    resolve_parameters,
)
from pmx_discrete_posterior.models.prediction import evaluate_prediction
from pmx_discrete_posterior.models.residuals import resolve_residuals


def _resolved_parameters():
    """功能 / Function: 构造参数段 / Build a parameter segment.
    输入 / Inputs: 无 / None. 输出 / Outputs: 独立参数字典 / Fresh parameter mapping.
    """
    return {
        "units": {"V": "L"},
        "segments": [
            {"start": 0, "end": 8, "occasion_id": None, "values": {"V": 10}}
        ],
    }


def _resolution_arguments(relations):
    """功能 / Function: 构造参数解析输入 / Build resolution inputs.
    输入 / Inputs: 回调 / Callback. 输出 / Outputs: 合法独立输入 / Fresh valid arguments.
    """
    return {
        "parameter_model": {"covariate_names": ["BW"], "relations": relations},
        "parameters": {
            "values": {"V": 10, "BW_REF": 70},
            "units": {"V": "L", "BW_REF": "kg"},
        },
        "covariates": {"BW": 35},
        "iiv_values": {},
        "iov_values": {},
        "occasions": [],
        "evaluation_interval": (0, 6),
    }


def _likelihood_arguments():
    """功能 / Function: 构造非时间排序记录 / Build records in nonchronological order.
    输入 / Inputs: 无 / None. 输出 / Outputs: 三组对齐输入 / Three aligned input mappings.
    """
    return {
        "observations": [
            {"record_id": "late", "time": 6, "value": 14, "time_unit": "h", "value_unit": "mg/L"},
            {"record_id": "early", "time": 2, "value": 5, "time_unit": "h", "value_unit": "mg/L"},
        ],
        "predictions": {
            "record_ids": ["late", "early"],
            "values": [12, 4],
            "units": ["mg/L", "mg/L"],
        },
        "residual_distribution": {
            "distribution": "normal",
            "dependence": "conditionally_independent",
            "record_ids": ["late", "early"],
            "means": [12, 4],
            "sd": [2, 0.5],
            "units": ["mg/L", "mg/L"],
        },
    }


def _prediction_arguments(predict):
    """功能 / Function: 构造预测协议输入 / Build prediction protocol inputs.
    输入 / Inputs: 预测回调 / Predictor. 输出 / Outputs: 无观测值的请求 / Requests without observed values.
    """
    observations = _likelihood_arguments()["observations"]
    return {
        "predict": predict,
        "individual_parameters": _resolved_parameters(),
        "candidate_state": {"state_id": "explicit-state", "events": []},
        "prediction_records": [
            {key: value for key, value in record.items() if key != "value"}
            for record in observations
        ],
        "individual_data": {
            "covariates": {"BW": 35},
            "occasions": [],
            "dosing_prehistory": {"events": []},
            "initialization": {"mode": "zero", "start_time": 0, "event_phase": "before_events"},
        },
    }


def test_unrepresentable_real_scalar_is_classified_as_input_error():
    """功能 / Function: 检查浮点转换溢出分类 / Check float-conversion overflow classification.
    输入 / Inputs: 超出float范围的整数 / Integer beyond float range.
    输出 / Outputs: InputError而非裸OverflowError / InputError, not raw OverflowError.
    """
    with pytest.raises(InputError):
        scalar(10**1000, name="V")


def test_missing_evaluation_interval_is_rejected_before_relations():
    """功能 / Function: 检查非区间输入分类 / Check invalid interval classification.
    输入 / Inputs: None区间 / None interval. 输出 / Outputs: InputError且回调不执行 / InputError before callback.
    """
    relations = Mock(return_value=_resolved_parameters())
    arguments = _resolution_arguments(relations)
    arguments["evaluation_interval"] = None
    with pytest.raises(InputError):
        resolve_parameters(**arguments)
    relations.assert_not_called()


def test_unknown_callback_segment_field_is_execution_failure():
    """功能 / Function: 拒绝回调结果中的未知字段 / Reject unknown callback-result fields.
    输入 / Inputs: 参数段额外效应字段 / Extra effect field in a parameter segment.
    输出 / Outputs: ModelExecutionError保留InputError原因 / Execution error chained from InputError.
    """
    result = _resolved_parameters()
    result["segments"][0]["iiv_values"] = {"eta_V": 0.1}
    relations = Mock(return_value=result)
    with pytest.raises(ModelExecutionError) as caught:
        resolve_parameters(**_resolution_arguments(relations))
    assert isinstance(caught.value.__cause__, InputError)
    relations.assert_called_once()


def test_parameter_callback_arithmetic_error_retains_original_cause():
    """功能 / Function: 保留参数计算失败 / Preserve parameter arithmetic failure.
    输入 / Inputs: 抛出浮点错误的回调 / Callback raising a floating-point error.
    输出 / Outputs: ModelExecutionError及原始异常 / Execution error with original cause.
    """
    failure = FloatingPointError("parameter exponential overflow")
    relations = Mock(side_effect=failure)
    with pytest.raises(ModelExecutionError) as caught:
        resolve_parameters(**_resolution_arguments(relations))
    assert caught.value.__cause__ is failure
    relations.assert_called_once()


def test_resolved_covariate_iiv_iov_effects_are_not_applied_again_by_prediction():
    """功能 / Function: 跨层核对参数只作用一次 / Check effects apply once across layers.
    输入 / Inputs: 乱序IOV键、两个occasion及预测替身 / Shuffled IOV keys, two occasions and predictor spy.
    输出 / Outputs: 一次解析、参数原样传入预测 / One resolution and unchanged predictor parameters.
    """
    def relations(**inputs):
        """功能 / Function: 计算具名体积 / Resolve named volumes.
        输入 / Inputs: 协变量与效应 / Covariates and effects. 输出 / Outputs: 两段体积 / Two volume segments.
        """
        values = inputs["parameters"]["values"]
        base = values["V"] * inputs["covariates"]["BW"] / values["BW_REF"]
        base *= math.exp(inputs["iiv_values"]["eta_V"])
        return {
            "units": {"V": "L"},
            "segments": [
                {
                    **occasion,
                    "values": {
                        "V": base * math.exp(inputs["iov_values"][occasion["occasion_id"]]["kappa_V"])
                    },
                }
                for occasion in inputs["occasions"]
            ],
        }

    relation_spy = Mock(side_effect=relations)
    arguments = _resolution_arguments(relation_spy)
    arguments["iiv_values"] = {"eta_V": math.log(2)}
    arguments["iov_values"] = {
        "a": {"kappa_V": math.log(5)},
        "z": {"kappa_V": math.log(3)},
    }
    arguments["occasions"] = [
        {"occasion_id": "z", "start": 0, "end": 4},
        {"occasion_id": "a", "start": 4, "end": 8},
    ]
    resolved = resolve_parameters(**arguments)
    assert [segment["values"]["V"] for segment in resolved["segments"]] == pytest.approx([30, 50])
    predictor = Mock(return_value=_likelihood_arguments()["predictions"])
    prediction_arguments = _prediction_arguments(predictor)
    prediction_arguments["individual_parameters"] = resolved
    prediction_arguments["individual_data"]["occasions"] = deepcopy(arguments["occasions"])
    result = evaluate_prediction(**prediction_arguments)
    relation_spy.assert_called_once()
    predictor.assert_called_once()
    received = predictor.call_args.kwargs["individual_parameters"]
    assert received == resolved
    assert received is not resolved
    assert [parameters_at(received, time)["values"]["V"] for time in (6, 2)] == pytest.approx([50, 30])
    assert result["record_ids"] == ("late", "early")
    assert result["values"].tolist() == [12, 4]


def test_permuted_prediction_output_is_not_silently_realigned():
    """功能 / Function: 拒绝完整但错序的输出 / Reject complete but permuted output.
    输入 / Inputs: ID和值共同反序 / IDs and values reversed together.
    输出 / Outputs: 回调执行错误，不自动排序 / Execution error without automatic sorting.
    """
    result = _likelihood_arguments()["predictions"]
    for key in ("record_ids", "values", "units"):
        result[key].reverse()
    predictor = Mock(return_value=result)
    with pytest.raises(ModelExecutionError) as caught:
        evaluate_prediction(**_prediction_arguments(predictor))
    assert isinstance(caught.value.__cause__, InputError)
    predictor.assert_called_once()


def test_failed_predictor_cannot_mutate_inputs_for_the_next_evaluation():
    """功能 / Function: 隔离失败回调的写入 / Isolate writes by a failing callback.
    输入 / Inputs: 修改嵌套输入后报错的预测器 / Predictor mutating nested inputs before failure.
    输出 / Outputs: 原输入不变且保留失败原因 / Unchanged inputs and preserved failure cause.
    """
    failure = ArithmeticError("structural calculation failed")

    def predict(**inputs):
        """功能 / Function: 模拟破坏性失败 / Simulate a destructive failure.
        输入 / Inputs: 隔离副本 / Isolated copies. 输出 / Outputs: 异常 / Exception.
        """
        inputs["individual_parameters"]["segments"][0]["values"]["V"] = 999
        inputs["prediction_records"][0]["time"] = 999
        inputs["candidate_state"]["events"].clear()
        inputs["individual_data"]["occasions"].append(
            {"occasion_id": "changed", "start": 0, "end": 8}
        )
        raise failure

    arguments = _prediction_arguments(predict)
    arguments["candidate_state"]["events"] = [
        {"time": 0, "amount": 100, "amount_unit": "mg", "route": "oral", "compartment": "depot"}
    ]
    before = deepcopy(arguments)
    with pytest.raises(ModelExecutionError) as caught:
        evaluate_prediction(**arguments)
    assert caught.value.__cause__ is failure
    assert arguments == before


def test_candidate_prior_field_is_rejected_before_prediction():
    """功能 / Function: 禁止候选字段夹带先验 / Reject a prior hidden in candidate fields.
    输入 / Inputs: 额外prior_probability / Extra prior_probability.
    输出 / Outputs: InputError且预测器不运行 / InputError without predictor execution.
    """
    predictor = Mock(return_value=_likelihood_arguments()["predictions"])
    arguments = _prediction_arguments(predictor)
    arguments["candidate_state"]["prior_probability"] = 0.25
    with pytest.raises(InputError):
        evaluate_prediction(**arguments)
    predictor.assert_not_called()


def test_observed_value_in_prediction_request_is_rejected_before_callback():
    """功能 / Function: 拒绝请求中夹带实测值 / Reject observed values in prediction requests.
    输入 / Inputs: 请求附加value / Extra request value.
    输出 / Outputs: InputError且预测器不运行 / InputError without predictor execution.
    """
    predictor = Mock(return_value=_likelihood_arguments()["predictions"])
    arguments = _prediction_arguments(predictor)
    arguments["prediction_records"][0]["value"] = 999
    with pytest.raises(InputError):
        evaluate_prediction(**arguments)
    predictor.assert_not_called()


def test_residual_parameter_references_reject_extra_variance_role():
    """功能 / Function: 拒绝未知残差角色 / Reject an unknown residual parameter role.
    输入 / Inputs: 同时提供sd及variance引用 / Both sd and variance references.
    输出 / Outputs: InputError，不忽略额外字段 / InputError without silently dropping a field.
    """
    with pytest.raises(InputError):
        resolve_residuals(
            residual_model={
                "distribution": "normal",
                "error_type": "additive",
                "parameter_names": {"sd": "sigma", "variance": "sigma_squared"},
            },
            parameters={
                "values": {"sigma": 2, "sigma_squared": 4},
                "units": {"sigma": "mg/L", "sigma_squared": "mg^2/L^2"},
            },
            predictions=_likelihood_arguments()["predictions"],
            prediction_records=_prediction_arguments(None)["prediction_records"],
            dependence="conditionally_independent",
        )


def test_proportional_scale_overflow_is_rejected_without_substitution():
    """功能 / Function: 禁止返回溢出的残差尺度 / Reject an overflowed residual scale.
    输入 / Inputs: 有限正系数与有限正预测 / Finite positive coefficient and predictions.
    输出 / Outputs: 按尺度契约报InputError，不替换尺度 / InputError under the scale contract, no substitution.
    """
    predictions = _likelihood_arguments()["predictions"]
    predictions["values"] = [1e308, 4]
    with pytest.raises(InputError):
        resolve_residuals(
            residual_model={
                "distribution": "normal",
                "error_type": "proportional",
                "parameter_names": {"coefficient_sd": "sigma"},
            },
            parameters={"values": {"sigma": 10}, "units": {"sigma": "1"}},
            predictions=predictions,
            prediction_records=_prediction_arguments(None)["prediction_records"],
            dependence="conditionally_independent",
        )


def test_nonsequence_residual_units_are_classified_as_input_error():
    """功能 / Function: 检查残差单位类型错误分类 / Check residual unit type-error classification.
    输入 / Inputs: units=None / None units. 输出 / Outputs: InputError而非裸TypeError / InputError, not raw TypeError.
    """
    arguments = _likelihood_arguments()
    arguments["residual_distribution"]["units"] = None
    with pytest.raises(InputError):
        conditional_log_likelihood(**arguments)


def test_permuted_residual_distribution_is_rejected_even_with_matching_id_set():
    """功能 / Function: 防止残差与预测错配 / Prevent residual-prediction misalignment.
    输入 / Inputs: 仅残差数据整体反序 / Only residual data reversed together.
    输出 / Outputs: InputError，不能按位置错配 / InputError rather than positional misassociation.
    """
    arguments = _likelihood_arguments()
    residual = arguments["residual_distribution"]
    for key in ("record_ids", "means", "sd", "units"):
        residual[key].reverse()
    assert set(residual["record_ids"]) == set(arguments["predictions"]["record_ids"])
    with pytest.raises(InputError):
        conditional_log_likelihood(**arguments)


def test_likelihood_sum_overflow_is_not_zero_likelihood_when_each_term_is_finite():
    """功能 / Function: 区分累加溢出与零似然 / Distinguish sum overflow from zero likelihood.
    输入 / Inputs: 三个有限极端正态log密度 / Three finite extreme normal log-densities.
    输出 / Outputs: 单项有限，累加抛执行错误 / Finite individual terms, execution error on accumulation.
    """
    arguments = _likelihood_arguments()
    arguments["observations"] = [
        {"record_id": f"tail-{index}", "time": index, "value": 1.2e154,
         "time_unit": "h", "value_unit": "mg/L"}
        for index in range(3)
    ]
    ids = [record["record_id"] for record in arguments["observations"]]
    arguments["predictions"] = {"record_ids": ids, "values": [0, 0, 0], "units": ["mg/L"] * 3}
    arguments["residual_distribution"] = {
        "distribution": "normal", "dependence": "conditionally_independent",
        "record_ids": ids, "means": [0, 0, 0], "sd": [1, 1, 1], "units": ["mg/L"] * 3,
    }
    single = deepcopy(arguments)
    single["observations"] = single["observations"][:1]
    for key in ("record_ids", "values", "units"):
        single["predictions"][key] = single["predictions"][key][:1]
    for key in ("record_ids", "means", "sd", "units"):
        single["residual_distribution"][key] = single["residual_distribution"][key][:1]
    term = conditional_log_likelihood(**single)
    assert math.isfinite(term)
    assert term == pytest.approx(-7.2e307, rel=1e-14)
    with pytest.raises(ModelExecutionError) as caught:
        conditional_log_likelihood(**arguments)
    assert isinstance(caught.value.__cause__, OverflowError)


def test_positive_gaussian_log_likelihood_is_not_clipped_to_log_probability():
    """功能 / Function: 保留大于一的连续密度 / Preserve continuous densities above one.
    输入 / Inputs: 观测等于均值且SD很小 / Observations at means with very small SDs.
    输出 / Outputs: 有限正log似然，不截为零 / Finite positive log-likelihood without zero clipping.
    """
    arguments = _likelihood_arguments()
    for record, mean in zip(arguments["observations"], arguments["predictions"]["values"]):
        record["value"] = mean
    arguments["residual_distribution"]["sd"] = [1e-200, 2e-200]
    expected = -math.log(2 * math.pi) - math.log(1e-200) - math.log(2e-200)
    actual = conditional_log_likelihood(**arguments)
    assert math.isfinite(actual)
    assert actual > 0
    assert actual == pytest.approx(expected, rel=1e-14)
