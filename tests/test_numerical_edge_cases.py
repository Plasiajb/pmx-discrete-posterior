import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution
from pmx_discrete_posterior.models.prediction.initialization import periodic_state


def test_scaled_rank_one_covariance_is_rejected_even_if_cholesky_rounds_positive():
    with pytest.raises(InputError):
        RandomEffectDistribution([("iiv", None, "CL"), ("iiv", None, "V")], [[2., 2.], [2., 2.]])


def test_slow_elimination_retains_finite_periodic_solution():
    result = periodic_state(matrix=[[-1e-20]], period=1, anchor_time=0, start_time=0,
                            input_events=[{"offset": 0, "lag": 0, "values": [1.]}])
    assert result["values"][0] == pytest.approx(1/math.expm1(1e-20), rel=1e-12)


def test_tiny_independent_variances_are_not_dropped():
    distribution = RandomEffectDistribution([("iiv", None, "CL"), ("iiv", None, "V")], [[1e-30, 0], [0, 1e-40]])
    assert len(distribution.effect_keys) == 2
    assert np.isfinite(distribution.log_density([0, 0]))
