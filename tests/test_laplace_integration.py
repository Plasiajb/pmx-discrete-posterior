import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics import laplace


def distribution(covariance):
    return RandomEffectDistribution([("iiv", None, str(i)) for i in range(len(covariance))], covariance)


def integrate(likelihood, effects=None, options=None):
    return laplace.integrate(conditional_log_likelihood=likelihood,
                             random_effect_distribution=distribution([[.3]]) if effects is None else effects,
                             integration_method="laplace", numerical_options=options)


def test_scalar_linear_gaussian_integral_and_mode_match_analytic_values():
    result = integrate(lambda eta: norm.logpdf(.6, loc=1.5*eta[0], scale=.8))
    assert result["status"] == "success", result
    expected = norm.logpdf(.6, scale=np.sqrt(.8**2 + 1.5**2*.3))
    assert result["log_marginal_likelihood"] == pytest.approx(expected, abs=1e-8, rel=0)
    assert result["method_details"]["conditional_mode"] == pytest.approx([.3*1.5*.6/(.8**2+1.5**2*.3)], abs=1e-7, rel=0)


def test_correlated_multivariate_gaussian_integral_retains_all_constants():
    omega = np.array([[.4, .1, 0], [.1, .8, -.2], [0, -.2, 1.1]])
    design = np.array([[1., .3, -.2], [.4, -.5, .7]])
    residual = np.array([[.7, .1], [.1, .5]])
    y, offset = np.array([.5, -.2]), np.array([.1, -.4])
    result = integrate(lambda eta: multivariate_normal.logpdf(y, mean=offset+design@eta, cov=residual), distribution(omega))
    expected = multivariate_normal.logpdf(y, mean=offset, cov=residual+design@omega@design.T)
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(expected, abs=1e-7, rel=0)
    assert result["method_details"]["estimated_log_integral_error_from_hessian"] < 1e-3
    assert "hessian" not in result["method_details"]["hessian"]


@pytest.mark.parametrize("covariance", [[[.2]], [[2., .3], [.3, 1.]]])
def test_constant_likelihood_integrates_to_itself_under_normalized_effect_density(covariance):
    result = integrate(lambda eta: 2.3, distribution(covariance))
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(2.3, abs=1e-8, rel=0)


@pytest.mark.parametrize("value", [2.1, -10., -math.inf])
def test_zero_dimension_exact_path_never_calls_optimization_or_hessian(monkeypatch, value):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not execute numeric approximation")
    monkeypatch.setattr(laplace, "minimize", forbidden)
    monkeypatch.setattr(laplace, "hessian", forbidden)
    result = integrate(lambda eta: value, distribution(np.zeros((0, 0))))
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == value
    assert result["method_details"]["scope"] == "direct"


def test_common_likelihood_factor_shifts_log_integrals_without_changing_relative_weights():
    logs = []
    for offset in (0., -10.):
        row = []
        for observation in (.3, 1.2):
            result = integrate(lambda eta: norm.logpdf(observation, eta[0], .8) + offset)
            assert result["status"] == "success", result
            row.append(result["log_marginal_likelihood"])
        logs.append(np.asarray(row))
    np.testing.assert_allclose(logs[1] - logs[0], -10, atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.exp(logs[1]-logsumexp(logs[1])), np.exp(logs[0]-logsumexp(logs[0])), atol=1e-8, rtol=0)


def test_large_constant_hessian_counterexample_is_rejected_not_returned_as_valid_integral():
    result = integrate(lambda eta: -5e7 - eta[0]**2)
    assert result["status"] == "failed", result
    assert result["log_marginal_likelihood"] is None
    assert result["failure_reason"]


@pytest.mark.parametrize("value", [-math.inf, math.nan, math.inf, None, True, [1.]])
def test_active_dimension_bad_nodes_or_zero_at_evaluated_points_do_not_prove_zero_integral(value):
    result = integrate(lambda eta: value)
    assert result["status"] == "failed"
    assert result["log_marginal_likelihood"] is None
    assert result["method_details"]["stage"] == "optimization"


def test_callback_exception_is_preserved_as_failure():
    def broken(eta):
        raise InputError("model-specific failure")
    result = integrate(broken)
    assert result["status"] == "failed"
    assert "model-specific failure" in result["failure_reason"]
    result = integrate(broken, distribution(np.zeros((0, 0))))
    assert result["status"] == "failed"
    assert "model-specific failure" in result["failure_reason"]


def test_boundary_mode_does_not_produce_unconstrained_laplace_success():
    result = integrate(lambda eta: 0. if eta[0] >= 0 else -math.inf)
    assert result["status"] == "failed", result
    assert result["log_marginal_likelihood"] is None


def test_stationary_saddle_is_rejected_after_optimizer_success():
    effects = distribution(np.eye(2))
    result = integrate(lambda eta: -eta[0]**2 + eta[1]**2 - effects.log_density(eta), effects)
    assert result["status"] == "failed", result
    assert result["method_details"]["optimization"]["backend_success"]
    assert result["method_details"]["stage"] in ("hessian", "curvature")


@pytest.mark.parametrize("options", [{"optimization": {"initial_point": [0., 0.]}}, {"ghq": {}},
                                    {"derivatives": {"rtol": 0}}, {"unknown": {}}, {"optimization": {"method": "Nelder-Mead"}}])
def test_illegal_settings_raise_before_model_evaluation(options):
    def forbidden(eta):
        raise AssertionError("invalid input must not run the model")
    with pytest.raises(InputError):
        integrate(forbidden, options=options)


def test_explicit_initial_point_is_preserved_without_changing_input_options():
    options = {"optimization": {"initial_point": [.2]}}
    result = integrate(lambda eta: norm.logpdf(.6, eta[0], .8), options=options)
    assert result["status"] == "success", result
    assert result["method_details"]["initial_point"] == [.2]
    assert options == {"optimization": {"initial_point": [.2]}}


def test_ghq_never_falls_back_to_laplace():
    with pytest.raises(InputError, match="GHQ"):
        laplace.integrate(conditional_log_likelihood=lambda eta: 0., random_effect_distribution=distribution([[1.]]), integration_method="ghq")


@pytest.mark.parametrize("precision", [1., 3., .414])
def test_legacy_common_offset_counterexamples_fail_explicitly_at_default_settings(precision):
    result = integrate(lambda eta: -5e7 + .5*math.log(precision) - .5*(precision-1)*eta[0]**2,
                       distribution([[1.]]))
    assert result["status"] == "failed", result
    assert result["log_marginal_likelihood"] is None


def test_numerical_acceptance_does_not_claim_laplace_approximation_accuracy():
    from scipy.integrate import quad

    result = integrate(lambda eta: -.1*eta[0]**4, distribution([[1.]]))
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(0., abs=1e-7, rel=0)
    exact_integral, quad_error = quad(lambda x: math.exp(-.1*x**4)*norm.pdf(x), -np.inf, np.inf, epsabs=1e-12)
    assert quad_error < 1e-8
    assert abs(result["log_marginal_likelihood"] - math.log(exact_integral)) > .05


def test_public_entry_is_direct_alias_and_matches_integrator_keyword_contract():
    import inspect
    from pmx_discrete_posterior.numerics import integrate as public_integrate
    from pmx_discrete_posterior.numerics.integration import integrate as dispatcher
    from pmx_discrete_posterior.interfaces.protocols import Integrator

    assert public_integrate is dispatcher
    implementation = inspect.signature(public_integrate).parameters
    contract = dict(inspect.signature(Integrator.__call__).parameters)
    contract.pop("self")
    assert implementation.keys() == contract.keys()
    for name, parameter in implementation.items():
        assert parameter.kind == contract[name].kind
        assert parameter.default == contract[name].default
