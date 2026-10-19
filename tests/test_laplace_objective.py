import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics.derivatives import _EvaluationError
from pmx_discrete_posterior.numerics.laplace import _direct_log_likelihood, _joint_objective


def distribution(covariance):
    return RandomEffectDistribution([("iiv", None, str(i)) for i in range(len(covariance))], covariance)


def test_joint_density_keeps_normalizers_and_counts_effect_density_once():
    covariance = [[.4, .1], [.1, .8]]
    effects = distribution(covariance)
    point = np.array([.3, -.2])
    likelihood = lambda eta: norm.logpdf(1.2, loc=eta[0] + 2*eta[1], scale=.7)
    objective = _joint_objective(likelihood, effects)
    expected = -(likelihood(point) + multivariate_normal.logpdf(point, cov=covariance))
    assert objective(point) == pytest.approx(expected, abs=1e-14, rel=0)
    assert _joint_objective(lambda eta: likelihood(eta) - 4.5, effects)(point) == pytest.approx(expected + 4.5, abs=1e-14, rel=0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None, True, [1.]])
def test_active_dimension_invalid_or_zero_likelihood_is_not_a_global_zero_integral(value):
    with pytest.raises(_EvaluationError):
        _joint_objective(lambda eta: value, distribution([[1.]]))(np.zeros(1))


def test_joint_objective_detects_mutating_and_nondeterministic_callbacks():
    def mutating(eta):
        eta[0] = 1
        return 0.
    with pytest.raises(_EvaluationError, match="mutated"):
        _joint_objective(mutating, distribution([[1.]]))(np.zeros(1))
    calls = iter([0., 1.])
    objective = _joint_objective(lambda eta: next(calls), distribution([[1.]]))
    objective(np.zeros(1))
    with pytest.raises(_EvaluationError, match="not deterministic"):
        objective(np.zeros(1))


@pytest.mark.parametrize("value", [-5., 2.1, -math.inf])
def test_zero_dimensions_are_exact_including_explicit_true_zero(value):
    assert _direct_log_likelihood(lambda eta: value, distribution(np.zeros((0, 0)))) == value


@pytest.mark.parametrize("value", [math.nan, math.inf, None, True, np.array([1.])])
def test_zero_dimensions_reject_invalid_callback_outputs(value):
    with pytest.raises(InputError):
        _direct_log_likelihood(lambda eta: value, distribution(np.zeros((0, 0))))


def test_zero_dimension_callback_failure_is_not_translated_to_zero():
    def broken(eta):
        raise ArithmeticError("callback failed")
    with pytest.raises(ArithmeticError, match="callback failed"):
        _direct_log_likelihood(broken, distribution(np.zeros((0, 0))))
    calls = iter([0., 1.])
    with pytest.raises(_EvaluationError, match="not deterministic"):
        _direct_log_likelihood(lambda eta: next(calls), distribution(np.zeros((0, 0))))


def test_zero_dimension_rejects_mutation_and_non_normalized_density():
    def mutating(eta):
        eta.resize(1, refcheck=False)
        return 0.
    with pytest.raises(_EvaluationError, match="mutated"):
        _direct_log_likelihood(mutating, distribution(np.zeros((0, 0))))
    class Unnormalized:
        def log_density(self, eta):
            return 1.
    with pytest.raises(_EvaluationError, match="normalized"):
        _direct_log_likelihood(lambda eta: 0., Unnormalized())
