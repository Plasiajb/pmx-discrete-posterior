import math
from types import SimpleNamespace

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics.ghq import integrate
from test_ghq_options import ghq_options


def normal(covariance):
    covariance = np.asarray(covariance)
    return RandomEffectDistribution([("iiv", None, f"eta{i}") for i in range(len(covariance))], covariance)


def run(callback, covariance, options=None):
    return integrate(conditional_log_likelihood=callback, random_effect_distribution=normal(covariance),
                     integration_method="ghq", numerical_options=options or ghq_options())


def test_correlated_gaussian_observation_matches_analytic_integral():
    cov = np.array([[.3, .15], [.15, .7]])
    a = np.array([.6, -.2])
    y, sd = .4, .8
    ll = lambda x: -.5 * ((y-a@x)/sd)**2 - math.log(sd*math.sqrt(2*math.pi))
    result = run(ll, cov)
    variance = sd**2 + a @ cov @ a
    expected = -.5 * y*y/variance - .5 * math.log(2*math.pi*variance)
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(expected, abs=1e-11)
    assert result["method_details"]["evaluations"] == 12**2 + 20**2 + 32**2 + 3


def test_ten_dimensional_analytic_case_without_laplace():
    # Analytic n=1 -> n=2 change is about 1.925e-8, below the unchanged 1e-6 gate.
    vector = np.arange(1, 11) / 100000
    result = run(lambda x: 2 + vector @ x, np.eye(10), ghq_options((1, 2, 3)))
    assert result["status"] == "success", result
    assert result["log_marginal_likelihood"] == pytest.approx(2 + .5*vector@vector, abs=2e-9)


def test_ten_dimensional_coarse_ladder_is_not_misreported_as_stable():
    vector = np.arange(1, 11) / 100
    result = run(lambda x: 2 + vector @ x, np.eye(10), ghq_options((1, 2, 3)))
    assert result["status"] == "failed"
    assert result["method_details"]["order_results"][1]["change"] > 1e-2


def test_joint_density_is_not_multiplied_again():
    distribution = normal([[2.]])
    distribution.log_density = lambda x: pytest.fail("GHQ prior counted twice")
    result = integrate(conditional_log_likelihood=lambda x: -7., random_effect_distribution=distribution,
                       integration_method="ghq", numerical_options=ghq_options())
    assert result["status"] == "success"
    assert result["log_marginal_likelihood"] == pytest.approx(-7)


@pytest.mark.parametrize("value", [0., -123., -math.inf])
def test_zero_dimensional_uses_shared_direct_evaluation(value):
    result = run(lambda x: value, np.zeros((0, 0)))
    assert result["status"] == "success"
    assert result["log_marginal_likelihood"] == value
    assert result["method_details"]["evaluations"] == 2
    assert result["method_details"]["stability_passed"] is None
    assert result["method_details"]["order_results"] == []


def test_deterministic_coordinates_are_removed_by_models():
    result = run(lambda x: -.5*x[0]**2, [[0., 0.], [0., 1.]], ghq_options((24, 40, 64)))
    assert result["status"] == "success", result
    assert result["method_details"]["dimension"] == 1
    assert result["log_marginal_likelihood"] == pytest.approx(-.5*math.log(2), abs=1e-12)


def test_insufficient_order_stability_returns_failure_with_estimates():
    result = run(lambda x: -100*(x[0]-.4)**2, [[1.]], ghq_options((2, 3, 4)))
    assert result["status"] == "failed"
    assert result["log_marginal_likelihood"] is None
    assert len(result["method_details"]["order_results"]) == 3
    assert "log_integral_atol" in result["failure_reason"]


def test_two_final_comparisons_required_even_if_last_passes():
    # E[exp(.1Z)] has error >1e-6 at n=1; 2->3 nearly stable, but 1->2 is not.
    result = run(lambda x: .1*x[0], [[1.]], ghq_options((1, 2, 3)))
    assert result["status"] == "failed"
    rows = result["method_details"]["order_results"]
    assert rows[-1]["change"] < 1e-5
    assert rows[-2]["change"] > 1e-3


def test_budget_checked_before_allocation_or_callback():
    result = run(lambda x: pytest.fail("over budget"), np.eye(10), ghq_options((7, 9, 11), max_nodes=1000))
    assert result["status"] == "failed"
    assert result["method_details"]["evaluations"] == 0
    assert result["method_details"]["stage"] == "node_budget"


def test_partial_orders_preserved_and_budget_includes_repeat_checks():
    result = run(lambda x: 0., [[1.]], ghq_options((3, 5, 7), max_nodes=10))
    assert result["status"] == "failed"
    assert result["method_details"]["evaluations"] == 10
    assert [r["order"] for r in result["method_details"]["order_results"]] == [3, 5]


@pytest.mark.parametrize("bad", [math.inf, math.nan, -math.inf])
def test_bad_or_all_zero_nodes_fail_not_zero_evidence(bad):
    result = run(lambda x: bad, [[1.]])
    assert result["status"] == "failed"
    assert result["log_marginal_likelihood"] is None


def test_exception_at_later_node_retains_actual_evaluation_count():
    def callback(x):
        if x[0] > 0:
            raise RuntimeError("deliberate node failure")
        return 0.
    result = run(callback, [[1.]], ghq_options((3, 5, 7)))
    assert result["status"] == "failed"
    assert result["method_details"]["evaluations"] == 4
    assert "deliberate node failure" in result["failure_reason"]


def test_moments_do_not_imply_a_normal_distribution():
    fake = SimpleNamespace(mean=np.zeros(1), covariance=np.eye(1), effect_keys=("eta",), log_density=lambda x: 0)
    with pytest.raises(InputError, match="normal RandomEffectDistribution"):
        integrate(conditional_log_likelihood=lambda x: 0., random_effect_distribution=fake,
                  integration_method="ghq", numerical_options=ghq_options())


def test_settings_required_before_callback():
    with pytest.raises(InputError, match="explicit"):
        integrate(conditional_log_likelihood=lambda x: pytest.fail(), random_effect_distribution=normal([[1.]]),
                  integration_method="ghq")


def test_huge_constant_offset_cannot_produce_false_stability():
    result = run(lambda x: 1e16, [[1.]])
    assert result["status"] == "failed"
    assert "resolution" in result["failure_reason"]
