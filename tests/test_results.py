import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.results import (
    IntegrationResult, validate_integration_result, validate_posterior_result, validate_state_result,
)
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.numerics.derivatives import hessian
from pmx_discrete_posterior.numerics.optimization import minimize


def integration(status="success", value=2.5, reason=None, **extra):
    return dict(status=status, log_marginal_likelihood=value, failure_reason=reason, **extra)


@pytest.mark.parametrize("value", [-1e300, -10, 0, .2, 1000, np.float64(3)])
def test_finite_integration_results_allow_positive_log_densities(value):
    result = validate_integration_result(IntegrationResult(**integration(value=value)))
    assert isinstance(result, dict)
    assert result["log_marginal_likelihood"] == value


@pytest.mark.parametrize("value", [None, True, np.bool_(False), "1", [1], np.array([1]), math.nan, math.inf,
                                   "-inf", [-math.inf], np.array(-math.inf), np.array([-math.inf]), -10**400])
def test_invalid_integration_success_is_rejected(value):
    with pytest.raises(InputError):
        validate_integration_result(integration(value=value))


@pytest.mark.parametrize("value", [-math.inf, np.float64(-math.inf)])
def test_explicit_true_zero_likelihood_has_success_without_failure_reason(value):
    result = validate_integration_result(integration(value=value))
    assert result == integration(value=-math.inf)
    with pytest.raises(InputError):
        validate_integration_result(integration(value=value, reason="overflow"))


def test_failed_integration_preserves_reason_without_a_placeholder_number():
    original = integration("failed", None, "Hessian not positive definite", method_details={"eigenvalues": np.array([-1, 2])})
    result = validate_integration_result(original)
    assert result["log_marginal_likelihood"] is None
    assert result["failure_reason"] == original["failure_reason"]
    result["method_details"]["eigenvalues"][0] = 99
    assert original["method_details"]["eigenvalues"][0] == -1


@pytest.mark.parametrize("result", [
    integration("failed", -math.inf, "overflow"), integration("failed", 0, "failure"),
    integration("failed", None, None), integration("failed", None, " "), integration(reason="failure"),
    integration("pending"), integration(["success"]), integration(np.array(["success"])),
    integration(method_details=None), integration(extra="unknown"),
    {"status": "success", "log_marginal_likelihood": 1},
])
def test_inconsistent_integration_fields_are_rejected(result):
    with pytest.raises(InputError):
        validate_integration_result(result)


def test_low_level_success_cannot_be_passed_off_as_integration_success():
    curvature = hessian(lambda point: point[0] ** 2 - point[1] ** 2, [0., 0.])
    assert curvature["status"] == "success"
    assert np.linalg.eigvalsh(curvature["hessian"])[0] < 0
    with pytest.raises(InputError):
        validate_integration_result(curvature)
    optimum = minimize(lambda point: point[0] ** 2, [1.])
    assert optimum["status"] == "success"
    with pytest.raises(InputError):
        validate_integration_result(optimum)


def test_density_overflow_stays_an_error_not_a_valid_zero_likelihood():
    distribution = RandomEffectDistribution([("iiv", None, "CL")], [[1.]])
    with pytest.raises(InputError, match="overflow"):
        distribution.log_density([1e300])


def state(state_id="a", prior=.5, probability=.5, **changes):
    return {**integration(), "state_id": state_id, "prior_probability": prior,
            "posterior_probability": probability, **changes}


def test_state_result_is_flat_and_preserves_unavailable_posterior():
    ready = validate_state_result(state())
    assert ready["posterior_probability"] == .5
    assert "integration_result" not in ready
    partial = validate_state_result(state(probability=None))
    assert partial["status"] == "success"
    assert partial["log_marginal_likelihood"] == 2.5
    assert partial["posterior_probability"] is None
    failure = validate_state_result(state(probability=None, status="failed", log_marginal_likelihood=None, failure_reason="density overflow"))
    assert failure["posterior_probability"] is None
    assert failure["failure_reason"] == "density overflow"


@pytest.mark.parametrize("prior,probability", [(0, 0), (0, None), (1e-300, 0), (1, 1)])
def test_zero_and_small_positive_priors_remain_distinct(prior, probability):
    result = validate_state_result(state(prior=prior, probability=probability))
    assert result["prior_probability"] == prior
    assert result["posterior_probability"] == probability


@pytest.mark.parametrize("changes", [
    {"state_id": ""}, {"state_id": 1}, {"prior_probability": True}, {"prior_probability": -.1},
    {"prior_probability": 1.1}, {"prior_probability": np.nan}, {"posterior_probability": [1]},
    {"posterior_probability": -.1}, {"posterior_probability": 1.1}, {"posterior_probability": np.nan},
    {"prior_probability": 0, "posterior_probability": 1e-300},
    {"status": "failed", "failure_reason": "failure", "log_marginal_likelihood": None, "posterior_probability": 0},
])
def test_state_result_rejects_invalid_combinations(changes):
    with pytest.raises(InputError):
        validate_state_result(state(**changes))


def skipped(state_id="b", **changes):
    return state(state_id, prior=0, probability=0, status="skipped_zero_prior",
                 log_marginal_likelihood=None, **changes)


def test_skipped_zero_prior_is_a_state_result_not_an_integration_result():
    original = skipped(method_details={"policy": ["zero prior"]})
    result = validate_state_result(original)
    assert result["status"] == "skipped_zero_prior"
    assert result["prior_probability"] == result["posterior_probability"] == 0
    assert result["log_marginal_likelihood"] is result["failure_reason"] is None
    result["method_details"]["policy"].append("modified")
    assert original["method_details"]["policy"] == ["zero prior"]
    with pytest.raises(InputError):
        validate_integration_result(integration("skipped_zero_prior", None))


@pytest.mark.parametrize("changes", [
    {"prior_probability": 1e-300}, {"prior_probability": np.nextafter(0., 1.)}, {"prior_probability": .5},
    {"posterior_probability": None}, {"posterior_probability": 1e-300}, {"posterior_probability": False},
    {"log_marginal_likelihood": 0}, {"log_marginal_likelihood": -math.inf},
    {"failure_reason": "numerical failure"}, {"failure_reason": ""}, {"method_details": None},
])
def test_skip_rejects_positive_priors_missing_or_nonzero_posteriors_and_integral_values(changes):
    row = skipped()
    row.update(changes)
    with pytest.raises(InputError):
        validate_state_result(row)


STATES = [{"state_id": "a", "events": []}, {"state_id": "b", "events": []}]
PRIORS = {"a": .5, "b": .5}


def posterior(rows=None, status="success", reason=None, **extra):
    return dict(status=status, integration_method="laplace", failure_reason=reason,
                state_results=rows if rows is not None else [state("a"), state("b")], **extra)


def checked(result, priors=PRIORS):
    return validate_posterior_result(result, candidate_states=STATES, priors=priors)


def test_complete_posterior_preserves_order_and_ties_without_modifying_input():
    original = posterior([state("b", method_details={"orders": [5, 7]}), state("a")])
    result = checked(original)
    assert [row["state_id"] for row in result["state_results"]] == ["b", "a"]
    assert result["state_results"][0]["posterior_probability"] == .5
    result["state_results"][0]["method_details"]["orders"].append(9)
    assert original["state_results"][0]["method_details"]["orders"] == [5, 7]


@pytest.mark.parametrize("rows", [
    [state("a")], [state("a"), state("a")], [state("a"), state("c")],
    [state("a"), state("b"), state("c", prior=0, probability=0)],
])
def test_missing_duplicate_or_extra_candidate_is_not_normalized_away(rows):
    with pytest.raises(InputError, match="exactly once"):
        checked(posterior(rows))


@pytest.mark.parametrize("prior", [.5, 1e-300])
def test_positive_prior_failure_blocks_success_even_for_tiny_prior(prior):
    priors = {"a": 1-prior, "b": prior}
    rows = [state("a", prior=priors["a"], probability=1),
            state("b", prior=prior, probability=None, status="failed", log_marginal_likelihood=None, failure_reason="numerical failure")]
    with pytest.raises(InputError, match="complete successfully"):
        checked(posterior(rows), priors)
    with pytest.raises(InputError, match="all posterior"):
        checked(posterior(rows, status="failed", reason="a candidate failed"), priors)
    rows[0]["posterior_probability"] = None
    result = checked(posterior(rows, status="failed", reason="a candidate failed"), priors)
    assert result["state_results"][0]["status"] == "success"
    assert result["state_results"][0]["log_marginal_likelihood"] == 2.5
    assert result["state_results"][1]["failure_reason"] == "numerical failure"


def test_computed_zero_prior_state_is_retained_with_zero_posterior():
    rows = [state("a", prior=1, probability=1), state("b", prior=0, probability=0)]
    result = checked(posterior(rows), {"a": 1, "b": 0})
    assert len(result["state_results"]) == 2
    assert result["state_results"][1]["log_marginal_likelihood"] == 2.5


def test_skipped_zero_prior_is_retained_in_successful_complete_posterior():
    result = checked(posterior([state("a", prior=1, probability=1), skipped()]), {"a": 1, "b": 0})
    assert len(result["state_results"]) == 2
    assert result["state_results"][1]["status"] == "skipped_zero_prior"
    assert result["state_results"][1]["log_marginal_likelihood"] is None


def test_skipped_zero_prior_retains_zero_without_masking_failure_of_a_positive_prior_state():
    rows = [state("a", prior=1, probability=None, status="failed", log_marginal_likelihood=None,
                  failure_reason="integration failed"), skipped()]
    with pytest.raises(InputError, match="complete successfully"):
        checked(posterior(rows), {"a": 1, "b": 0})
    result = checked(posterior(rows, status="failed", reason="positive-prior state failed"), {"a": 1, "b": 0})
    assert [row["posterior_probability"] for row in result["state_results"]] == [None, 0]


def test_skipped_state_cannot_rescue_all_zero_evidence():
    rows = [state("a", prior=1, probability=0, log_marginal_likelihood=-math.inf), skipped()]
    with pytest.raises(InputError, match="positive evidence"):
        checked(posterior(rows), {"a": 1, "b": 0})
    rows[0]["posterior_probability"] = None
    result = checked(posterior(rows, status="failed", reason="zero evidence"), {"a": 1, "b": 0})
    assert [row["posterior_probability"] for row in result["state_results"]] == [None, 0]


def test_every_candidate_must_remain_present_even_when_one_would_be_skipped():
    with pytest.raises(InputError, match="exactly once"):
        checked(posterior([state("a", prior=1, probability=1)]), {"a": 1, "b": 0})


def test_true_zero_likelihood_can_have_zero_posterior_when_another_state_has_positive_evidence():
    rows = [state("a", probability=0, log_marginal_likelihood=-math.inf), state("b", probability=1)]
    result = checked(posterior(rows))
    assert result["state_results"][0]["status"] == "success"
    assert result["state_results"][0]["log_marginal_likelihood"] == -math.inf
    assert result["state_results"][0]["posterior_probability"] == 0


def test_true_zero_likelihood_cannot_receive_positive_posterior():
    with pytest.raises(InputError, match="true-zero-likelihood"):
        validate_state_result(state(log_marginal_likelihood=-math.inf))


def test_all_true_zero_likelihoods_require_failed_normalization():
    rows = [state(name, probability=0, log_marginal_likelihood=-math.inf) for name in ("a", "b")]
    with pytest.raises(InputError, match="positive evidence"):
        checked(posterior(rows))
    for row in rows:
        row["posterior_probability"] = None
    result = checked(posterior(rows, status="failed", reason="all positive-prior candidates have zero likelihood"))
    assert all(row["status"] == "success" for row in result["state_results"])
    assert all(row["posterior_probability"] is None for row in result["state_results"])


def test_finite_likelihood_of_zero_prior_state_does_not_rescue_zero_evidence():
    rows = [state("a", prior=1, probability=0, log_marginal_likelihood=-math.inf), state("b", prior=0, probability=0)]
    with pytest.raises(InputError, match="positive evidence"):
        checked(posterior(rows), {"a": 1, "b": 0})


def test_overall_failure_can_preserve_successful_integrals_without_posterior_values():
    result = checked(posterior([state("a", probability=None), state("b", probability=None)],
                               status="failed", reason="normalization could not be completed"))
    assert result["status"] == "failed"
    assert all(row["status"] == "success" and row["log_marginal_likelihood"] == 2.5 for row in result["state_results"])


def test_posterior_sum_roundoff_uses_existing_probability_tolerance_without_rescaling():
    probabilities = [.5, np.nextafter(.5, 1.)]
    result = checked(posterior([state("a", probability=probabilities[0]), state("b", probability=probabilities[1])]))
    assert [row["posterior_probability"] for row in result["state_results"]] == probabilities


@pytest.mark.parametrize("probabilities", [(0, 0), (.2, .2), (.5, .5 + 1e-13), (None, 1)])
def test_invalid_complete_posterior_vector_is_rejected(probabilities):
    with pytest.raises(InputError):
        checked(posterior([state("a", probability=probabilities[0]), state("b", probability=probabilities[1])]))


def test_reported_prior_must_match_original_input_exactly():
    with pytest.raises(InputError, match="reported priors"):
        checked(posterior([state("a", prior=.4), state("b", prior=.6)]))


@pytest.mark.parametrize("changes", [
    {"status": "pending"}, {"status": ["success"]}, {"failure_reason": "bad"},
    {"integration_method": "auto"}, {"state_results": []}, {"state_results": {}}, {"extra": 1},
])
def test_invalid_overall_result_fields_are_rejected(changes):
    result = posterior()
    result.update(changes)
    with pytest.raises(InputError):
        checked(result)
