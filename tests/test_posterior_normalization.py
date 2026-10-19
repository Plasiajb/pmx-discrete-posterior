from fractions import Fraction
import math

import mpmath as mp
import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.posterior.normalization import normalize


def states(ids):
    return [{"state_id": key, "events": []} for key in ids]


def test_hand_calculated_probabilities_and_input_nonmutation():
    logs, priors = {"a": math.log(.2), "b": math.log(.4)}, {"a": .25, "b": .75}
    original = (logs.copy(), priors.copy())
    result = normalize(logs, priors, candidate_states=states(["a", "b"]))
    assert result == pytest.approx({"a": 1/7, "b": 6/7}, abs=1e-15, rel=0)
    assert (logs, priors) == original


@pytest.mark.parametrize("shift", [0., -1000., 1000., -1e300, 1e300])
def test_equal_likelihoods_preserve_unequal_priors_even_at_huge_common_offsets(shift):
    priors = {"a": .1, "b": .2, "c": .7}
    result = normalize(dict.fromkeys(priors, shift), priors, candidate_states=states(priors))
    assert result == pytest.approx(priors, abs=1e-15, rel=0)


def test_common_shift_and_candidate_permutation_invariance():
    logs, priors = {"a": -4., "b": -2., "c": 0.}, {"a": .2, "b": .3, "c": .5}
    first = normalize(logs, priors, candidate_states=states(priors))
    second = normalize({k: v-5e7 for k, v in logs.items()}, priors, candidate_states=states(reversed(priors)))
    assert list(second) == ["c", "b", "a"]
    assert second == pytest.approx(first, abs=1e-15, rel=0)


@pytest.mark.parametrize("zero_log", [None, -math.inf, 1e300])
def test_zero_prior_is_retained_without_influencing_normalization(zero_log):
    result = normalize({"a": zero_log, "b": -3.}, {"a": 0., "b": 1.}, candidate_states=states(["a", "b"]))
    assert result == {"a": 0., "b": 1.}


def test_true_zero_likelihood_is_distinct_from_failed_or_missing_evaluation():
    assert normalize({"a": -math.inf, "b": -10.}, {"a": .5, "b": .5}, candidate_states=states(["a", "b"])) == {"a": 0., "b": 1.}
    with pytest.raises(InputError):
        normalize({"a": None, "b": -10.}, {"a": .5, "b": .5}, candidate_states=states(["a", "b"]))


def test_all_positive_prior_evidence_zero_is_undefined_even_with_finite_zero_prior_state():
    with pytest.raises(FloatingPointError, match="undefined"):
        normalize({"a": -math.inf, "b": 100.}, {"a": 1., "b": 0.}, candidate_states=states(["a", "b"]))


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, [0.], "-1"])
@pytest.mark.parametrize("prior", [0., .5])
def test_invalid_integral_is_not_silently_discarded_even_for_zero_prior(bad, prior):
    with pytest.raises(InputError):
        normalize({"a": bad, "b": 0.}, {"a": prior, "b": 1-prior}, candidate_states=states(["a", "b"]))


@pytest.mark.parametrize("logs", [{"a": 0.}, {"a": 0., "b": 0., "extra": 0.}, [0., 0.]])
def test_missing_extra_or_positional_integrals_rejected(logs):
    with pytest.raises(InputError):
        normalize(logs, {"a": .5, "b": .5}, candidate_states=states(["a", "b"]))


@pytest.mark.parametrize("priors", [{"a": 1., "b": 1.}, {"a": 0., "b": 0.}, {"a": -.1, "b": 1.1},
                                  {"a": Fraction(1, 10**400), "b": 1.}, {"a": .5}])
def test_invalid_prior_inputs_are_not_renormalized(priors):
    with pytest.raises(InputError):
        normalize({"a": 0., "b": 0.}, priors, candidate_states=states(["a", "b"]))


def test_subnormal_prior_can_be_compensated_by_likelihood_without_underflow():
    tiny = np.nextafter(0., 1.)
    result = normalize({"a": 745., "b": 0.}, {"a": tiny, "b": 1.}, candidate_states=states(["a", "b"]))
    with mp.workdps(80):
        weight = mp.mpf(float(tiny))*mp.exp(745)
        expected = float(weight/(1+weight))
    assert result["a"] == pytest.approx(expected, abs=1e-13, rel=0)


def test_representable_tiny_posterior_is_not_removed():
    result = normalize({"a": -740., "b": 0.}, {"a": .5, "b": .5}, candidate_states=states(["a", "b"]))
    assert result["a"] > 0 and result["b"] == 1.


def test_unrepresentable_nonzero_probability_fails_instead_of_becoming_true_zero():
    with pytest.raises(FloatingPointError, match="underflow"):
        normalize({"a": -1000., "b": 0.}, {"a": .5, "b": .5}, candidate_states=states(["a", "b"]))


def test_overflow_in_log_weight_difference_is_not_silently_converted_to_zero():
    with pytest.raises(FloatingPointError):
        normalize({"a": -1e308, "b": 1e308}, {"a": .5, "b": .5}, candidate_states=states(["a", "b"]))


def test_many_tied_candidates_remain_a_valid_distribution():
    ids = [str(i) for i in range(1000)]
    result = normalize(dict.fromkeys(ids, 1e200), dict.fromkeys(ids, .001), candidate_states=states(ids))
    assert math.fsum(result.values()) == pytest.approx(1., abs=8*math.ulp(1.), rel=0)


@pytest.mark.parametrize("log_b", [0., 1000.])
def test_smallest_positive_prior_is_not_skipped_or_prematurely_exponentiated(log_b):
    tiny = math.nextafter(0., 1.)
    result = normalize({"a": 0., "b": log_b}, {"a": 1., "b": tiny}, candidate_states=states(["a", "b"]))
    with mp.workdps(100):
        b_weight = mp.mpf(tiny) * mp.exp(log_b)
        expected = [float(1/(1+b_weight)), float(b_weight/(1+b_weight))]
    np.testing.assert_allclose(list(result.values()), expected, atol=0, rtol=2e-13)
    assert all(p > 0 for p in result.values())
    if log_b == 0:
        assert result["b"] == tiny


def test_smallest_positive_prior_cannot_mask_missing_integral():
    with pytest.raises(InputError):
        normalize({"a": 0., "b": None}, {"a": 1., "b": math.nextafter(0., 1.)}, candidate_states=states(["a", "b"]))
