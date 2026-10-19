import json
import math
import time

import numpy as np
import pytest
from scipy.integrate import quad

from pmx_discrete_posterior import compute_posterior
from test_ghq_options import ghq_options
from test_posterior_binding import declared_reference, bind


# Provenance: accepted OXC C003 GHQ r01 final level from the frozen manuscript
# reference results, retained as comparison constants without archive access.
# Original research GHQ is adaptive, not Laplace and not certified exact integration.
OXC_GHQ_LOGS = [-6.592470293298786, -3.6457203592562832, -2.1676862684648963, -2.5641361342479705]
OXC_GHQ_POSTERIOR = [.006261505558313793, .11924385460870279, .5228029819563068, .35169165787667644]


def reference_args(name, settings):
    case = declared_reference(name)
    return {**case, "priors": {s["state_id"]: .25 for s in case["candidate_states"]},
            "integration_method": "ghq", "numerical_options": settings}


def test_oxc_full_posterior_matches_archived_ghq():
    args = reference_args("OXC", ghq_options())
    start = time.perf_counter()
    result = compute_posterior(**args)
    print(json.dumps({"model": "OXC", "elapsed_seconds": time.perf_counter()-start, **result}, allow_nan=False))
    assert result["status"] == "success", result
    np.testing.assert_allclose([r["log_marginal_likelihood"] for r in result["state_results"]],
                               OXC_GHQ_LOGS, atol=1e-7, rtol=0)
    np.testing.assert_allclose([r["posterior_probability"] for r in result["state_results"]],
                               OXC_GHQ_POSTERIOR, atol=1e-7, rtol=0)
    assert result["method_details"]["posterior_stability_passed"] is True


def test_oxc_independent_scalar_quadrature_check():
    # Independent QUADPACK quadrature of the same bound likelihood, not an independent PK implementation.
    case = declared_reference("OXC")
    covariance_ref = case["model"]["iiv"]["covariance_parameters"][0][0]
    sd = math.sqrt(case["model"]["parameters"]["values"][covariance_ref])
    for state, expected in zip(case["candidate_states"], OXC_GHQ_LOGS):
        callback = bind(case, state)
        density_bound = 1 / (2.992 * math.sqrt(2*math.pi))
        integrand = lambda z: math.exp(callback(np.array([sd*z])) - .5*z*z) / math.sqrt(2*math.pi)
        integral, error = quad(integrand, -10, 10, epsabs=1e-11, epsrel=1e-10, limit=200)
        # Normal tail <= erfc(10/sqrt(2)); likelihood <= additive normal density maximum.
        tail_bound = density_bound * math.erfc(10/math.sqrt(2))
        assert error + tail_bound < 1e-9 * integral
        assert math.log(integral) == pytest.approx(expected, abs=1e-8)
        print(json.dumps({"model": "OXC", "state_id": state["state_id"], "quad_log_integral": math.log(integral),
                          "quad_estimated_error": error, "tail_bound": tail_bound}))


@pytest.mark.slow
def test_tac_full_ten_dimensional_bounded_smoke_not_accuracy_acceptance():
    # Exactly the full reference model, no altered variances, dimension reduction or approximate predictor.
    args = reference_args("TAC", ghq_options((1, 2, 3), log_integral_atol=1e-3, posterior_atol=1e-4, max_nodes=1100))
    start = time.perf_counter()
    result = compute_posterior(**args)
    print(json.dumps({"model": "TAC", "qualification": "bounded_execution_only_not_accuracy",
                      "elapsed_seconds": time.perf_counter()-start, **result}, allow_nan=False))
    assert result["status"] == "failed", result
    assert result["method_details"]["posterior_stability_passed"] is None
    for row in result["state_results"]:
        assert row["posterior_probability"] is None and row["log_marginal_likelihood"] is None
        details = row["method_details"]
        assert details["dimension"] == 10 and details["scope"] == "prior_centered_tensor"
        assert details["stage"] == "node_budget"
        assert details["evaluations"] == 1 + 1 + 2**10 + 1
        assert details["next_order"] == 3 and details["next_order_evaluations"] == 3**10 + 1
        assert [r["order"] for r in details["order_results"]] == [1, 2]
        assert all(math.isfinite(r["log_marginal_likelihood"]) for r in details["order_results"])
