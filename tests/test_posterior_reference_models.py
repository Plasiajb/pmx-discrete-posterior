"""Full posterior regressions; independent Laplace references are not exact nonlinear integrals."""

import json
import math

import numpy as np
import pytest

from pmx_discrete_posterior import compute_posterior
from test_laplace_reference_models import LEGACY_LOGS
from test_posterior_binding import declared_reference


# Independent modes/curvature from test_laplace_reference_models, verified in the
# accepted v0.2 baseline before posterior implementation; no archive files are read.
# OXC: 60/80-digit analytic differentiation. TAC: separate model, optimizer and finite differences.
INDEPENDENT_LOGS = {
    "OXC": [-6.5855336131149524, -3.6440125863671535, -2.1658162679584247, -2.5637698888076980],
    "TAC": [-3.0602575684754125, -2.8464740513320637, -2.7852536319272970, -2.8186652168365800],
}
LEGACY_POSTERIORS = {
    "OXC": [.006296560090573311, .11928606261935508, .5230728861787735, .35134449111129823],
    "TAC": [.20711804665707417, .2564854717642472, .27267835198432405, .2637181295943546],
}


@pytest.mark.parametrize("name,dimension,log_atol,probability_atol", [("OXC", 1, 1e-6, 5e-7), pytest.param("TAC", 10, 1e-3, 5e-4, marks=pytest.mark.slow)])
def test_full_reference_posterior(name, dimension, log_atol, probability_atol):
    case = declared_reference(name)
    states = case["candidate_states"]
    # Historical labels omega_01 and omega_10 are not chronological bit labels.
    assert [[e["time"] for e in s["events"]] for s in states] == [[], [0.], [12.], [0., 12.]]
    result = compute_posterior(**case, priors={s["state_id"]: .25 for s in states}, integration_method="laplace")
    print(name, json.dumps({**result, "state_results": [{k: v for k, v in row.items() if k != "method_details"}
                                                      for row in result["state_results"]]}, allow_nan=False))
    assert result["status"] == "success", result
    rows = result["state_results"]
    assert [r["state_id"] for r in rows] == [s["state_id"] for s in states]
    logs = [r["log_marginal_likelihood"] for r in rows]
    probabilities = [r["posterior_probability"] for r in rows]
    weights = [math.exp(x) for x in INDEPENDENT_LOGS[name]]
    independent_probabilities = [w/math.fsum(weights) for w in weights]
    np.testing.assert_allclose(logs, INDEPENDENT_LOGS[name], atol=log_atol, rtol=0)
    np.testing.assert_allclose(probabilities, independent_probabilities, atol=probability_atol, rtol=0)
    np.testing.assert_allclose(logs, LEGACY_LOGS[name], atol=1e-5, rtol=0)
    np.testing.assert_allclose(probabilities, LEGACY_POSTERIORS[name], atol=5e-6, rtol=0)
    assert math.fsum(probabilities) == pytest.approx(1., abs=8*math.ulp(1.), rel=0)
    assert int(np.argmax(probabilities)) == 2
    for row in rows:
        details = row["method_details"]
        assert row["status"] == "success" and row["prior_probability"] == .25
        assert details["dimension"] == dimension
        assert details["estimated_log_integral_error_from_hessian"] <= 1e-3
        assert details["estimated_log_integral_error_from_stationarity"] <= 1e-3


def test_public_entry_is_a_direct_alias_not_a_wrapper():
    from pmx_discrete_posterior.posterior import compute_posterior as exported
    from pmx_discrete_posterior.posterior.computation import compute_posterior as implementation
    assert compute_posterior is exported is implementation
