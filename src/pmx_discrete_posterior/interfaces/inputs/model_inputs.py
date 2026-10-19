"""功能 / Function: 校验模型输入；不执行PK预测 / Validate model inputs without executing PK predictions.
输入 / Inputs: 模型声明及参数引用 / Model declarations and parameter references.
输出 / Outputs: 数据副本，保留回调身份 / Data copies preserving callback identity.
"""

from ..validation import InputError, fields, names, scalar, validate_covariance, validate_parameters


def validate_parameter_model(parameter_model):
    """功能 / Function: 检查参数关系声明，不执行关系 / Check a relationship without executing it.
    输入 / Inputs: covariate_names及relations回调 / Covariate names and relations callback.
    输出 / Outputs: 名称元组与原回调 / Name tuple and original callable; invalid inputs raise InputError.
    """
    fields(parameter_model, {"covariate_names", "relations"}, name="parameter_model")
    if not isinstance(parameter_model["covariate_names"], (list, tuple)):
        raise InputError("covariate_names must be an ordered list or tuple")
    required = names(parameter_model["covariate_names"], name="covariate_names", allow_empty=True)
    if not callable(parameter_model["relations"]):
        raise InputError("parameter_model.relations must be callable")
    return {"covariate_names": required, "relations": parameter_model["relations"]}


def validate_random_effect_spec(spec, *, parameters, level):
    """功能 / Function: 核对随机效应声明及引用 / Check random-effect declarations and references.
    输入 / Inputs: spec或None、parameters、level='iiv'或'iov' / Specification or None, parameters and level.
    输出 / Outputs: 独立定义副本；不组装分布 / Independent specification, without assembling a distribution.
    复用协方差检查；occasion绑定仍由models处理 / Shares covariance checks; models binds actual occasions.
    """
    if not isinstance(level, str) or level not in ("iiv", "iov"):
        raise InputError("level must be iiv or iov")
    if spec is None:
        return None
    required = {"names", "distribution", "covariance_parameters"}
    if level == "iov":
        required.add("occasion_key")
    fields(spec, required, name=level)
    if not isinstance(spec["distribution"], str) or spec["distribution"] != "normal":
        raise InputError("only explicitly declared zero-mean normal effects are supported")
    if level == "iov" and (not isinstance(spec["occasion_key"], str) or spec["occasion_key"] != "occasion_id"):
        raise InputError("first version requires occasion_key='occasion_id'")
    if not isinstance(spec["names"], (list, tuple)):
        raise InputError("effect names must be an ordered list or tuple")
    effect_names = names(spec["names"], name="effect names")
    parameters = validate_parameters(parameters)
    refs = spec["covariance_parameters"]
    if not isinstance(refs, (list, tuple)) or len(refs) != len(effect_names):
        raise InputError("covariance_parameters must be a square matrix of parameter references")
    result = []
    for row in refs:
        if not isinstance(row, (list, tuple)) or len(row) != len(refs):
            raise InputError("covariance reference matrix must be square")
        copied = []
        for ref in row:
            if ref is not None:
                if not isinstance(ref, str) or ref not in parameters["values"]:
                    raise InputError("covariance entries must reference parameters or be explicit None zeros")
                if parameters["units"][ref] != "1":
                    raise InputError("normal log-effect variances/covariances must be dimensionless ('1')")
            copied.append(ref)
        result.append(copied)
    validate_covariance([[0.0 if ref is None else parameters["values"][ref] for ref in row] for row in result])
    validated = {"names": effect_names, "distribution": "normal", "covariance_parameters": result}
    if level == "iov":
        validated["occasion_key"] = "occasion_id"
    return validated


def validate_residual_model(residual_model, *, parameters):
    """功能 / Function: 核对残差规则及SD参数 / Check residual rules and SD parameters.
    输入 / Inputs: 正态additive或proportional声明、参数 / Normal additive/proportional declaration and parameters.
    输出 / Outputs: 独立定义副本；加性SD单位在有预测记录后匹配 / Copy; additive SD units are matched later to predictions.
    """
    fields(residual_model, {"distribution", "error_type", "parameter_names"}, {"dependence"}, name="residual_model")
    if "dependence" in residual_model and (not isinstance(residual_model["dependence"], str)
                                          or residual_model["dependence"] != "conditionally_independent"):
        raise InputError("residual_model.dependence must be conditionally_independent")
    if not isinstance(residual_model["distribution"], str) or residual_model["distribution"] != "normal":
        raise InputError("only explicitly declared normal residuals are supported")
    error_type = residual_model["error_type"]
    if not isinstance(error_type, str) or error_type not in ("additive", "proportional"):
        raise InputError("supported error_type values are additive and proportional; combined is not inferred")
    role = "sd" if error_type == "additive" else "coefficient_sd"
    fields(residual_model["parameter_names"], {role}, name="residual parameter references")
    parameters = validate_parameters(parameters)
    ref = residual_model["parameter_names"][role]
    if not isinstance(ref, str) or ref not in parameters["values"]:
        raise InputError("unknown residual parameter reference")
    scalar(parameters["values"][ref], name=role, positive=True)
    if error_type == "proportional" and parameters["units"][ref] != "1":
        raise InputError("residual SD unit does not match its declared role")
    checked = {"distribution": "normal", "error_type": error_type, "parameter_names": {role: ref}}
    if "dependence" in residual_model:
        checked["dependence"] = residual_model["dependence"]
    return checked


def validate_model(model):
    """功能 / Function: 校验模型六组静态输入，不调用用户模型 / Validate six static groups without model calls.
    输入 / Inputs: 六组模型数据及可选random_effects_dependence / Six model groups and optional random_effects_dependence.
    输出 / Outputs: 独立数据副本、原回调；非法定义抛InputError / Data copy, original callbacks, or InputError.
    顶层posterior要求两项独立性声明；底层调用仍可显式传参 / Posterior requires both dependence declarations; low-level calls may still pass them explicitly.
    """
    fields(model, {"parameters", "parameter_model", "iiv", "iov", "residual_model", "predict"},
           {"random_effects_dependence"}, name="model")
    if "random_effects_dependence" in model and (not isinstance(model["random_effects_dependence"], str)
                                                or model["random_effects_dependence"] != "independent_blocks"):
        raise InputError("model.random_effects_dependence must be independent_blocks")
    parameters = validate_parameters(model["parameters"])
    parameter_model = validate_parameter_model(model["parameter_model"])
    if not callable(model["predict"]):
        raise InputError("model.predict must be callable")
    checked = {"parameters": parameters, "parameter_model": parameter_model,
            "iiv": validate_random_effect_spec(model["iiv"], parameters=parameters, level="iiv"),
            "iov": validate_random_effect_spec(model["iov"], parameters=parameters, level="iov"),
            "residual_model": validate_residual_model(model["residual_model"], parameters=parameters),
            "predict": model["predict"]}
    if "random_effects_dependence" in model:
        checked["random_effects_dependence"] = model["random_effects_dependence"]
    return checked
