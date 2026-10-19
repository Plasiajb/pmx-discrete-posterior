import math

import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.random_effects import RandomEffectDistribution, resolve_random_effects


def test_correlated_normal_closed_form_and_permutation():
    covariance = [[.04, .01], [.01, .09]]
    keys = [("iiv", None, "CL"), ("iiv", None, "V")]
    dist = RandomEffectDistribution(keys, covariance)
    expected = -math.log(2 * math.pi) - math.log(.0035) / 2 - 1.2
    assert dist.log_density([.2, -.3]) == pytest.approx(expected, abs=1e-12)
    reverse = RandomEffectDistribution(keys[::-1], np.array(covariance)[::-1, ::-1])
    assert reverse.log_density([-.3, .2]) == pytest.approx(expected, abs=1e-12)
    copy = dist.covariance
    copy[:] = 0
    assert dist.covariance[0, 0] == .04


@pytest.mark.parametrize("covariance", [
    [[-1, 0], [0, 1]], [[1, .1], [.2, 1]], [[1, 1], [1, 1]],
    [[0, .1], [.1, 1]], [[float("nan"), 0], [0, 1]], [[1, 2], [2, 1]],
])
def test_invalid_covariance_never_repaired(covariance):
    with pytest.raises(InputError):
        RandomEffectDistribution([("iiv", None, "CL"), ("iiv", None, "V")], covariance)


def test_deterministic_coordinates_and_zero_dimensions():
    dist = RandomEffectDistribution([("iiv", None, "CL"), ("iov", "o1", "V")], [[.04, 0], [0, 0]])
    assert dist.effect_keys == (("iiv", None, "CL"),)
    assert dist.expand([.2]) == {"iiv_values": {"CL": .2}, "iov_values": {"o1": {"V": 0.}}}
    zero = RandomEffectDistribution([("iiv", None, "CL")], [[0.]])
    assert zero.log_density([]) == 0
    assert zero.expand([])["iiv_values"] == {"CL": 0.}
    with pytest.raises(InputError):
        dist.log_density(.2)


def test_explicit_independent_iiv_iov_blocks():
    p = {"values": {"omega": .04, "pi": .09}, "units": {"omega": "1", "pi": "1"}}
    iiv = {"names": ["CL"], "distribution": "normal", "covariance_parameters": [["omega"]]}
    iov = {"names": ["CL"], "distribution": "normal", "covariance_parameters": [["pi"]], "occasion_key": "occasion_id"}
    occasions = [{"occasion_id": "o1", "start": -12, "end": 0}, {"occasion_id": "o2", "start": 0, "end": 24}]
    args = dict(iiv=iiv, iov=iov, parameters=p, occasions=occasions, dependence="independent_blocks")
    dist = resolve_random_effects(**args)
    assert dist.covariance.tolist() == [[.04, 0, 0], [0, .09, 0], [0, 0, .09]]
    assert dist.expand([1, 2, 3]) == {"iiv_values": {"CL": 1.}, "iov_values": {"o1": {"CL": 2.}, "o2": {"CL": 3.}}}
    args["dependence"] = None
    with pytest.raises(InputError):
        resolve_random_effects(**args)
    args["dependence"] = "independent_blocks"
    iiv["distribution"] = "uniform"
    with pytest.raises(InputError):
        resolve_random_effects(**args)


def test_missing_reference_or_non_dimensionless_rejected():
    spec = {"names": ["CL"], "distribution": "normal", "covariance_parameters": [["missing"]]}
    for params in ({"values": {}, "units": {}}, {"values": {"missing": 1}, "units": {"missing": "mg"}}):
        with pytest.raises(InputError):
            resolve_random_effects(iiv=spec, iov=None, parameters=params, occasions=[], dependence="independent_blocks")


def test_no_random_effects():
    dist = resolve_random_effects(iiv=None, iov=None, parameters={"values": {}, "units": {}},
                                  occasions=[], dependence="independent_blocks")
    assert dist.mean.shape == (0,)
    assert dist.log_density([]) == 0
