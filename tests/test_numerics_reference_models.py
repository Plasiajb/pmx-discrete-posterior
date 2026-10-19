"""Numerical integration contracts through the model layer, not posterior delivery."""

import importlib.util
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from fixtures.reference_adapters import oxc_model, tac_model
from pmx_discrete_posterior.models.random_effects import resolve_random_effects
from pmx_discrete_posterior.numerics.derivatives import gradient, hessian
from pmx_discrete_posterior.numerics.optimization import minimize

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("numerics_reference_examples", ROOT / "examples/reference_models.py")
examples = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(examples)


def bound_objective(name, state_index):
    case = examples.load_reference(name)
    model, data = case["model"], case["individual_data"]
    distribution = resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
                                          occasions=data["occasions"], dependence="independent_blocks")
    def objective(eta):
        _, ll = examples.evaluate_reference(case, case["candidate_states"][state_index], eta)
        return -(ll + distribution.log_density(eta))
    return case, distribution, objective


@pytest.mark.parametrize("state_index", range(4))
def test_oxc_mode_and_curvature_against_60_digit_reference(state_index):
    case, distribution, objective = bound_objective("OXC", state_index)
    optimum = minimize(objective, distribution.mean)
    assert optimum["status"] == "success", optimum
    point = optimum["minimizer"]
    curvature = hessian(objective, point)
    assert curvature["status"] == "success", curvature
    raw = json.loads((ROOT / "tests/fixtures/OXC_C003.json").read_text())
    old = oxc_model.Model(raw)
    state_id = case["candidate_states"][state_index]["state_id"]
    with mp.workdps(60):
        f = lambda eta: old.objective_mp(state_id, eta)
        ref_g = float(mp.diff(f, mp.mpf(point[0])))
        ref_h = float(mp.diff(f, mp.mpf(point[0]), 2))
        ref_value = float(f(mp.mpf(point[0])))
    assert abs(ref_g) / np.sqrt(ref_h) <= 1e-4
    assert abs(optimum["objective_value"] - ref_value) <= 1e-3
    assert abs(np.log(curvature["hessian"][0, 0] / ref_h)) <= 1e-3
    np.testing.assert_allclose(curvature["hessian"], [[ref_h]], rtol=1e-6, atol=1e-6)
    print("OXC", state_index, "mode", point, "H", curvature["hessian"], "H_ref", ref_h,
          "gradient_ref", ref_g, "nfev", curvature["method_details"]["nfev"])


def reference_finite_differences(objective, point, step):
    # Independent test-only stencil, never used by the new production implementation.
    dimension = point.size
    matrix = np.empty((dimension, dimension))
    f0 = objective(point)
    for i in range(dimension):
        ei = np.eye(dimension)[i] * step
        matrix[i, i] = (objective(point+ei) - 2*f0 + objective(point-ei)) / step**2
        for j in range(i):
            ej = np.eye(dimension)[j] * step
            matrix[i, j] = matrix[j, i] = (objective(point+ei+ej) - objective(point+ei-ej)
                                          - objective(point-ei+ej) + objective(point-ei-ej)) / (4*step**2)
    return matrix


@pytest.mark.parametrize("state_index", range(4))
@pytest.mark.slow
def test_tac_mode_and_hessian_against_independent_prediction_path(state_index):
    case, distribution, objective = bound_objective("TAC", state_index)
    optimum = minimize(objective, distribution.mean)
    assert optimum["status"] == "success", optimum
    point = optimum["minimizer"]
    curvature = hessian(objective, point)
    assert curvature["status"] == "success", curvature
    raw = json.loads((ROOT / "tests/fixtures/TAC_C003.json").read_text())
    old = tac_model.ResolvedCase.from_dict(raw)
    inverse = np.linalg.inv(np.asarray(raw["omega"]))
    normalizer = .5*(len(point)*np.log(2*np.pi) + np.linalg.slogdet(raw["omega"])[1])
    state_id = case["candidate_states"][state_index]["state_id"]
    def reference(eta):
        predicted = tac_model.predict_expm(old, state_id, eta, raw["times"])
        sd = raw["residual_sd"] * np.asarray(predicted)
        residual = (np.asarray(raw["observations"]) - predicted) / sd
        return float(np.sum(np.log(sd) + .5*np.log(2*np.pi) + .5*residual**2)
                     + .5*eta @ inverse @ eta + normalizer)
    assert abs(objective(point)-reference(point)) < 1e-9
    reference_g = gradient(reference, point)
    assert reference_g["status"] == "success", reference_g
    assert np.max(np.abs(reference_g["gradient"])) <= 1e-3
    coarse = reference_finite_differences(reference, point, 1e-4)
    fine = reference_finite_differences(reference, point, 5e-5)
    assert np.linalg.eigvalsh(fine).min() > 0
    assert abs(np.linalg.slogdet(coarse)[1]-np.linalg.slogdet(fine)[1]) <= 1e-3
    assert abs(np.linalg.slogdet(curvature["hessian"])[1]-np.linalg.slogdet(fine)[1]) <= 1e-3
    np.testing.assert_allclose(curvature["hessian"], fine, atol=1e-4, rtol=1e-4)
    print("TAC", state_index, "mode", point, "max_H_difference", np.max(abs(curvature["hessian"]-fine)),
          "logdet_difference", np.linalg.slogdet(curvature["hessian"])[1]-np.linalg.slogdet(fine)[1],
          "nfev", curvature["method_details"]["nfev"])
