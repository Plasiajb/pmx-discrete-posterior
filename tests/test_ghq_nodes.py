import math

import numpy as np
import pytest
from scipy.linalg import cholesky

from pmx_discrete_posterior.numerics.ghq import _normal_rule, _evaluate_order


def integrate_order(callback, mean, covariance, order):
    details = {"evaluations": 0}
    value = _evaluate_order(callback, np.asarray(mean), cholesky(covariance, lower=True), order, details)
    return value, details


@pytest.mark.parametrize("order", [1, 3, 12, 32, 100])
def test_normal_weights_and_polynomial_moments(order):
    nodes, logs = _normal_rule(order)
    weights = np.exp(logs)
    assert weights.sum() == pytest.approx(1, abs=2e-15)
    assert weights @ nodes == pytest.approx(0, abs=1e-15)
    if order >= 3:
        assert weights @ nodes**2 == pytest.approx(1, abs=2e-14)
        assert weights @ nodes**4 == pytest.approx(3, abs=2e-13)


def test_correlated_nonzero_mean_and_normalization_constants():
    mean = np.array([.4, -.3])
    covariance = np.array([[.7, .25], [.25, .9]])
    vector = np.array([.2, -.5])
    value, details = integrate_order(lambda x: 1.2 + vector @ x, mean, covariance, 12)
    expected = 1.2 + vector @ mean + .5 * vector @ covariance @ vector
    assert value == pytest.approx(expected, abs=1e-13)
    assert details["evaluations"] == 12**2 + 1


@pytest.mark.parametrize("offset", [-10000., 10000.])
def test_log_sum_does_not_exponentiate_raw_likelihood(offset):
    value, _ = integrate_order(lambda x: offset, np.zeros(2), np.eye(2), 5)
    assert value == pytest.approx(offset, abs=2e-12)


def test_genuine_zero_at_some_nodes_is_allowed():
    value, _ = integrate_order(lambda x: -math.inf if x[0] < 0 else math.log(2), [0.], [[1.]], 4)
    assert value == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, [1.]])
def test_one_invalid_node_is_not_dropped(bad):
    with pytest.raises((ValueError, TypeError)):
        integrate_order(lambda x: bad if x[0] > 0 else 0., [0.], [[1.]], 3)


def test_one_exception_is_not_dropped():
    def likelihood(x):
        if x[0] > 0:
            raise RuntimeError("failed positive node")
        return 0.
    with pytest.raises(RuntimeError, match="failed positive node"):
        integrate_order(likelihood, [0.], [[1.]], 3)


def test_all_nodes_zero_is_not_a_proof_of_zero_integral():
    with pytest.raises(FloatingPointError, match="do not establish"):
        integrate_order(lambda x: -math.inf, [0.], [[1.]], 3)


def test_mutation_is_detected():
    def likelihood(x):
        x[:] = 0
        return 0.
    with pytest.raises(FloatingPointError, match="mutated"):
        integrate_order(likelihood, [0.], [[1.]], 3)


def test_repeated_node_nondeterminism_is_detected():
    values = iter([0., 1.])
    with pytest.raises(FloatingPointError, match="not deterministic"):
        integrate_order(lambda x: next(values), [0.], [[1.]], 3)


def test_underflowed_library_weights_are_rejected(monkeypatch):
    import pmx_discrete_posterior.numerics.ghq as ghq
    monkeypatch.setattr(ghq, "roots_hermitenorm", lambda n: (np.arange(n), np.zeros(n)))
    with pytest.raises(FloatingPointError, match="zero-weight nodes"):
        _normal_rule(3)


def test_chunked_multidimensional_sum():
    value, details = integrate_order(lambda x: 3., np.zeros(3), np.eye(3), 12)
    assert value == pytest.approx(3, abs=1e-13)
    assert details["evaluations"] == 1729


def test_actual_library_weight_underflow_is_not_silently_ignored():
    with pytest.raises(FloatingPointError, match="not representable"):
        _normal_rule(512)


def test_all_zero_chunk_does_not_discard_other_chunks():
    # Even-order rule is symmetric; all negative-half chunks can be zero, but E[2 I(Z>0)]=1.
    value, _ = integrate_order(lambda x: -math.inf if x[0] < 0 else math.log(2), np.zeros(3), np.eye(3), 16)
    assert value == pytest.approx(0, abs=2e-14)
