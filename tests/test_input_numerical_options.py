from copy import deepcopy

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.inputs.numerical_options import validate_numerical_options
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.numerics.derivatives import gradient, hessian
from pmx_discrete_posterior.numerics.optimization import minimize


def test_defaults_match_tested_numerics_and_no_initial_point_inferred():
    options = validate_numerical_options(None)
    assert options["optimization"]["method"] == "BFGS"
    assert options["derivatives"]["backend"] == "scipy"
    assert options["derivatives"]["backend_options"]["initial_step"] == .5
    assert "initial_point" not in options["optimization"]
    assert validate_numerical_options(options) == options
    result = minimize(lambda x: (x[0] - 2)**2, [0.], options["optimization"])
    assert result["status"] == "success"
    assert result["minimizer"][0] == pytest.approx(2)
    result = hessian(lambda x: (x[0] - 2)**2, [2.], options["derivatives"])
    assert result["status"] == "success"
    assert result["hessian"][0, 0] == pytest.approx(2)


def test_default_gtol_changes_only_search_stopping_tolerance():
    """功能 / Function: 默认停止精度与独立检查分离 / Separate the default stopping tolerance from independent checks.
    输入 / Inputs: 默认与显式旧设置 / Default and explicit former settings.
    输出 / Outputs: 仅gtol变化；原检查保留 / Only gtol changes; original checks retained.
    """
    from pmx_discrete_posterior.numerics.laplace import STATIONARITY_LOG_INTEGRAL_ATOL

    default = validate_numerical_options(None)
    old = validate_numerical_options({"optimization": {"backend_options": {"gtol": 1e-7}}})
    assert default["optimization"]["backend_options"]["gtol"] == 1e-6
    assert old["optimization"]["backend_options"]["gtol"] == 1e-7
    default["optimization"]["backend_options"]["gtol"] = 1e-7
    assert default == old
    assert default["optimization"]["stationarity_tolerance"] == 1e-5
    assert STATIONARITY_LOG_INTEGRAL_ATOL == 1e-3
    tighter = validate_numerical_options({"optimization": {"stationarity_tolerance": 1e-8}})
    assert tighter["optimization"]["backend_options"]["gtol"] == pytest.approx(1e-9)


def test_user_settings_are_copied_and_initial_point_is_separate_from_optimizer_options():
    original = {"optimization": {"initial_point": [1., 2.], "backend_options": {"maxiter": 75}},
                "derivatives": {"backend_options": {"initial_step": 1.}, "atol": 1e-7}}
    before = deepcopy(original)
    result = validate_numerical_options(original)
    result["optimization"]["initial_point"][0] = 99
    result["derivatives"]["backend_options"]["initial_step"] = 2
    assert original == before


@pytest.mark.parametrize("options", [
    {"ghq": {}}, {"unknown": 1}, {"optimization": {"method": "Nelder-Mead"}},
    {"optimization": {"initial_point": True}}, {"optimization": {"initial_point": [float("nan")]}},
    {"optimization": {"backend_options": {"maxiter": True}}},
    {"optimization": {"backend_options": {"bounds": [-5, 5]}}},
    {"optimization": {"backend_options": {"finite_diff_rel_step": .01}}},
    {"derivatives": {"backend": "auto"}}, {"derivatives": {"backend_options": {"order": 3}}},
    {"derivatives": {"backend_options": {"step_factor": 1}}},
    {"derivatives": {"atol": 0, "rtol": 0}},
    {"optimization": {"method": np.array(["BFGS", "BFGS"])}},
    {"derivatives": {"backend": np.array(["scipy", "scipy"])}},
    {"derivatives": None},
])
def test_unsupported_options_are_not_silently_dropped(options):
    with pytest.raises(InputError):
        validate_numerical_options(options)


def test_gradient_only_keeps_zero_relative_tolerance_available():
    options = validate_numerical_options({"derivatives": {"atol": 1e-6, "rtol": 0}})
    assert gradient(lambda x: x[0] ** 2, [1.], options["derivatives"])["status"] == "success"
