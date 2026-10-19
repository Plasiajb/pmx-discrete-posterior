"""功能 / Function: 汇总输入校验及跨输入检查 / Validate combined inputs and their consistency.
输入 / Inputs: 显式模型与数据定义 / Explicit model and data definitions.
输出 / Outputs: 校验后的副本；不执行模型 / Validated copies without model execution.
"""

from ..validation import InputError
from .model_inputs import validate_model
from .observations import validate_observations
from .candidate_states import validate_candidate_states, validate_candidate_history
from .individual_data import validate_individual_data
from .priors import validate_priors
from .integration_method import validate_integration_method
from .numerical_options import validate_numerical_options, resolve_hessian_options


def validate_inputs(*, model, individual_data, observations, candidate_states, priors,
                    integration_method, numerical_options=None):
    """功能 / Function: 组合七组输入校验和必要交叉核对 / Combine seven input contracts and cross-checks.
    输入 / Inputs: 七组用户输入；不要求重复填写内部结果 / Seven user-input groups, not internal results.
    输出 / Outputs: 同名字段的数据副本及原回调 / Same-named validated data copies and original callbacks.
    不执行模型/积分，不生成默认先验或独立性声明 / No model/integration execution, inferred priors or independence.
    """
    checked_model = validate_model(model)
    records = validate_observations(observations)
    individual = validate_individual_data(individual_data, model=checked_model, observations=records)
    states = [validate_candidate_history(state, initialization_start=individual["initialization"]["start_time"],
                                         common_events=individual["dosing_prehistory"]["events"])
              for state in validate_candidate_states(candidate_states)]
    residual = checked_model["residual_model"]
    if residual["error_type"] == "additive":
        ref = residual["parameter_names"]["sd"]
        if checked_model["parameters"]["units"][ref] != records[0]["value_unit"]:
            raise InputError("additive residual SD and observation concentration units must match")
    method = validate_integration_method(integration_method)
    options = validate_numerical_options(numerical_options)
    if (method == "ghq") != ("ghq" in options):
        raise InputError("GHQ requires its explicit settings; Laplace does not consume GHQ options")
    active_effects = any(ref is not None and checked_model["parameters"]["values"][ref] > 0
                         for spec in (checked_model["iiv"], checked_model["iov"]) if spec is not None
                         for i, row in enumerate(spec["covariance_parameters"]) for ref in (row[i],))
    if method == "laplace" and active_effects:
        resolve_hessian_options(options["derivatives"])
    return {"model": checked_model, "individual_data": individual, "observations": records,
            "candidate_states": states, "priors": validate_priors(priors, candidate_states=states),
            "integration_method": method, "numerical_options": options}
