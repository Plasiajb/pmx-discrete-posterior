from fractions import Fraction

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.inputs.priors import validate_priors
from pmx_discrete_posterior.interfaces.results import validate_posterior_result, validate_state_result
from pmx_discrete_posterior.interfaces.validation import InputError


STATES = [{"state_id": "a", "events": []}, {"state_id": "b", "events": []}]
TINY = Fraction(1, 10**400)


def state(state_id="a", prior=.5, probability=.5):
    return dict(state_id=state_id, prior_probability=prior, posterior_probability=probability,
                status="success", log_marginal_likelihood=0., failure_reason=None)


@pytest.mark.parametrize("value,other", [(TINY, 1), (-TINY, 1), (1 + TINY, 0)])
def test_input_priors_reject_conversion_that_changes_zero_or_hides_invalid_bounds(value, other):
    with pytest.raises(InputError):
        validate_priors({"a": value, "b": other}, candidate_states=STATES)


@pytest.mark.parametrize("field", ["prior_probability", "posterior_probability"])
@pytest.mark.parametrize("value", [TINY, -TINY, 1 + TINY])
def test_result_probabilities_reject_conversion_that_changes_zero_or_hides_invalid_bounds(field, value):
    row = state()
    row[field] = value
    with pytest.raises(InputError):
        validate_state_result(row)


def test_positive_exact_prior_must_not_be_silently_skipped_in_complete_output():
    rows = [state("a", prior=1, probability=1), state("b", prior=TINY, probability=0)]
    rows[1].update(status="skipped_zero_prior", log_marginal_likelihood=None)
    result = dict(status="success", integration_method="laplace", state_results=rows, failure_reason=None)
    with pytest.raises(InputError):
        validate_posterior_result(result, candidate_states=STATES, priors={"a": 1, "b": TINY})


@pytest.mark.parametrize("value", [Fraction(0), Fraction(1, 4), Fraction(1, 10**300), np.nextafter(0., 1.)])
def test_representable_probabilities_keep_their_zero_or_positive_status(value):
    result = validate_priors({"a": value, "b": 1-value}, candidate_states=STATES)
    assert result["a"] == float(value)
    assert (result["a"] == 0) == (value == 0)
    row = validate_state_result(state(prior=value, probability=value))
    assert row["prior_probability"] == row["posterior_probability"] == float(value)
