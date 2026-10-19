"""Independent Laplace oracles; legacy comparisons are regression checks, not exact integrals."""

import json
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize

from fixtures.reference_adapters import oxc_model, tac_model
from pmx_discrete_posterior.numerics.laplace import integrate
from test_numerics_reference_models import ROOT, bound_objective, examples, reference_finite_differences


# Frozen remote_refs_r01/{OXC,TAC}_C003.json, documented in docs/laplace_implementation.md.
# State order is omega_00, omega_01, omega_10, omega_11, not chronological bit order.
LEGACY_LOGS = {
    "OXC": [-6.585533618609123, -3.6440125895619895, -2.1658162694420495, -2.5637698872053107],
    "TAC": [-3.0602579384807456, -2.8464748201961614, -2.785253940202942, -2.818666001079686],
}


def compute(name, index):
    case, effects, _ = bound_objective(name, index)
    state = case["candidate_states"][index]
    result = integrate(conditional_log_likelihood=lambda eta: examples.evaluate_reference(case, state, eta)[1],
                       random_effect_distribution=effects, integration_method="laplace")
    assert result["status"] == "success", result
    assert result["method_details"]["dimension"] == (1 if name == "OXC" else 10)
    assert result["method_details"]["estimated_log_integral_error_from_hessian"] <= 1e-3
    assert result["method_details"]["estimated_log_integral_error_from_stationarity"] <= 1e-3
    return case, state["state_id"], result


@pytest.mark.parametrize("index", range(4))
def test_oxc_log_integrals_against_independently_solved_high_precision_modes(index):
    _, state_id, result = compute("OXC", index)
    reference = oxc_model.Model(json.loads((ROOT / "tests/fixtures/OXC_C003.json").read_text()))
    solutions = []
    for digits in (60, 80):
        with mp.workdps(digits):
            objective = lambda eta: reference.objective_mp(state_id, eta)
            mode = mp.findroot(lambda eta: mp.diff(objective, eta), (mp.mpf("-.2"), mp.mpf(".2")))
            curvature = mp.diff(objective, mode, 2)
            assert curvature > 0
            assert abs(mp.diff(objective, mode)) < mp.mpf("1e-50")
            log_integral = -objective(mode) + mp.log(2*mp.pi)/2 - mp.log(curvature)/2
            solutions.append((float(mode), float(log_integral)))
    np.testing.assert_allclose(solutions[0], solutions[1], atol=1e-14, rtol=0)
    expected_mode, expected_log = solutions[1]
    details = result["method_details"]
    assert details["conditional_mode"][0] == pytest.approx(expected_mode, abs=1e-6, rel=0)
    assert result["log_marginal_likelihood"] == pytest.approx(expected_log, abs=1e-6, rel=0)
    assert result["log_marginal_likelihood"] == pytest.approx(LEGACY_LOGS["OXC"][index], abs=1e-5, rel=0)
    print("OXC", index, "new_log_integral", result["log_marginal_likelihood"], "independent_mp", expected_log,
          "legacy_difference", result["log_marginal_likelihood"] - LEGACY_LOGS["OXC"][index],
          "curvature_error_screen", details["estimated_log_integral_error_from_hessian"],
          "stationarity_error_screen", details["estimated_log_integral_error_from_stationarity"])


@pytest.mark.parametrize("index", range(4))
@pytest.mark.slow
def test_tac_log_integrals_against_separate_prediction_optimization_and_curvature(index):
    _, state_id, result = compute("TAC", index)
    raw = json.loads((ROOT / "tests/fixtures/TAC_C003.json").read_text())
    reference_case = tac_model.ResolvedCase.from_dict(raw)
    omega = np.asarray(raw["omega"])
    normalizer = .5*(len(omega)*math.log(2*math.pi)+np.linalg.slogdet(omega)[1])

    def objective(eta):
        prediction = tac_model.predict_expm(reference_case, state_id, eta, raw["times"])
        sd = raw["residual_sd"] * np.asarray(prediction)
        residual = (np.asarray(raw["observations"])-prediction)/sd
        return float(np.sum(np.log(sd)+.5*math.log(2*math.pi)+.5*residual**2)
                     +.5*eta @ np.linalg.solve(omega, eta)+normalizer)

    def gradient(eta, step=1e-4):
        # Test-only fourth-order stencil, independent of production scipy.differentiate.
        axes = np.eye(len(eta))*step
        return np.array([(-objective(eta+2*axis)+8*objective(eta+axis)-8*objective(eta-axis)+objective(eta-2*axis))
                         /(12*step) for axis in axes])

    optimum = scipy_minimize(objective, np.zeros(len(omega)), method="BFGS", jac=gradient,
                             options={"gtol": 1e-6, "maxiter": 1000})
    assert optimum.success, optimum.message
    mode = optimum.x
    assert np.max(np.abs(gradient(mode, 5e-5))) <= 1e-3
    coarse = reference_finite_differences(objective, mode, 1e-4)
    fine = reference_finite_differences(objective, mode, 5e-5)
    assert np.linalg.eigvalsh(fine).min() > 0
    logdet = np.linalg.slogdet(fine)[1]
    assert abs(np.linalg.slogdet(coarse)[1]-logdet) <= 1e-3
    expected_log = -objective(mode)+.5*len(mode)*math.log(2*math.pi)-.5*logdet
    details = result["method_details"]
    assert abs(details["objective_value"]-objective(mode)) <= 1e-3
    assert abs(details["log_determinant"]-logdet) <= 1e-3
    assert result["log_marginal_likelihood"] == pytest.approx(expected_log, abs=1e-3, rel=0)
    assert result["log_marginal_likelihood"] == pytest.approx(LEGACY_LOGS["TAC"][index], abs=1e-5, rel=0)
    print("TAC", index, "new_log_integral", result["log_marginal_likelihood"], "independent_fd", expected_log,
          "legacy_difference", result["log_marginal_likelihood"] - LEGACY_LOGS["TAC"][index],
          "curvature_error_screen", details["estimated_log_integral_error_from_hessian"],
          "stationarity_error_screen", details["estimated_log_integral_error_from_stationarity"])
