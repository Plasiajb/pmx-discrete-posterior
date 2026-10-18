"""Public-entry regressions for inherited covariance audit finding P1-01."""

from dataclasses import replace
import math

import numpy as np
import pytest

import pmx_discrete_posterior as pmx


STATES = (pmx.DiscreteState("a", 0.4), pmx.DiscreteState("b", 0.6))
CONFIG = pmx.MultiEtaLaplaceConfig(
    omega_covariance=((0.4, 0.1), (0.1, 0.3)), additive_sd=1.0,
)
ENTRIES = ("posterior", "diagnostics", "ghq", "cached_config", "cached_omega")


class ConstantModel:
    def __init__(self):
        self.calls = []

    def predictions(self, state_id, eta):
        self.calls.append((state_id, tuple(eta)))
        return (0.0,)


def public_call(entry, config, model):
    if entry == "posterior":
        return pmx.compute_multieta_posterior((0.0,), STATES, model, config)
    if entry == "ghq":
        return pmx.gauss_hermite_posterior((0.0,), STATES, model, config, nodes=3)
    if entry == "cached_config":
        return pmx.multieta_state_diagnostics(
            (0.0,), STATES[0], model, config, np.eye(2), np.eye(2), 0.0,
        )
    if entry == "cached_omega":
        return pmx.multieta_state_diagnostics(
            (0.0,), STATES[0], model, CONFIG, config.omega_covariance, np.eye(2), 0.0,
        )
    return pmx.multieta_state_diagnostics((0.0,), STATES[0], model, config)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("omega,message", [
    pytest.param(((-1.0, 0.0), (0.0, -1.0)), "positive definite", id="negative-definite-positive-det"),
    pytest.param(np.diag((1.0, -1.0, -1.0)), "positive definite", id="indefinite-positive-det"),
    pytest.param(((1.0, 0.8), (0.0, 1.0)), "symmetric", id="asymmetric-upper"),
    pytest.param(((1.0, 0.0), (0.8, 1.0)), "symmetric", id="asymmetric-lower"),
    pytest.param(((math.nan, 0.0), (0.0, 1.0)), "finite", id="nan"),
    pytest.param(((math.inf, 0.0), (0.0, 1.0)), "finite", id="positive-infinity"),
    pytest.param(((1.0, -math.inf), (0.0, 1.0)), "finite", id="upper-negative-infinity"),
    pytest.param(((1.0, 1.0), (1.0, 1.0)), "positive definite", id="singular-correlated"),
    pytest.param(((1.0, 0.0), (0.0, 0.0)), "positive definite", id="zero-variance-no-dimension-removal"),
    pytest.param(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), "square", id="rectangular"),
    pytest.param((1.0, 1.0), "square", id="vector"),
    pytest.param(np.empty((0, 0)), "at least one ETA", id="empty-square"),
])
def test_invalid_covariance_rejected_before_inversion_or_model_evaluation(entry, omega, message, monkeypatch):
    def forbidden_inverse(*args, **kwargs):
        raise AssertionError("Invalid covariance reached inversion.")

    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    model = ConstantModel()
    with pytest.raises(ValueError, match=message):
        public_call(entry, replace(CONFIG, omega_covariance=omega), model)
    assert not model.calls


@pytest.mark.parametrize("entry", ("posterior", "diagnostics", "ghq"))
@pytest.mark.parametrize("off_diagonal,delta", [(0.0, 5e-13), (0.25, 1e-11)])
def test_within_declared_symmetry_tolerance_is_not_symmetrized(entry, off_diagonal, delta, monkeypatch):
    omega = np.array(((1.0, off_diagonal + delta), (off_diagonal, 1.0)))
    before = omega.tobytes()
    inverse_inputs = []
    original_inverse = np.linalg.inv

    def recording_inverse(value):
        inverse_inputs.append(value.tobytes())
        return original_inverse(value)

    monkeypatch.setattr(np.linalg, "inv", recording_inverse)
    public_call(entry, replace(CONFIG, omega_covariance=omega), ConstantModel())
    assert inverse_inputs and all(value == before for value in inverse_inputs)
    assert omega.tobytes() == before


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("off_diagonal,delta", [(0.0, 2e-12), (0.25, 1e-10)])
def test_outside_declared_symmetry_tolerance_is_rejected(entry, off_diagonal, delta):
    omega = ((1.0, off_diagonal + delta), (off_diagonal, 1.0))
    with pytest.raises(ValueError, match="symmetric"):
        public_call(entry, replace(CONFIG, omega_covariance=omega), ConstantModel())


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("scale", [1e-18, 1e-12, 1.0, 1e12])
@pytest.mark.parametrize("transpose", [False, True])
def test_material_asymmetry_rejected_independently_of_variance_scale(entry, scale, transpose):
    omega = scale * np.array(((1.0, 0.5), (0.0, 1.0)))
    if transpose:
        omega = omega.T.copy()
    model = ConstantModel()
    with pytest.raises(ValueError, match="symmetric in correlation units"):
        public_call(entry, replace(CONFIG, omega_covariance=omega), model)
    assert not model.calls


def tac_covariance(occasions):
    # Exact reference_matrices from phase 48 tables/TAC_variability_protocol_v4.json.
    # Four IIV effects and both IOV effects in every occasion are retained.
    omega = np.zeros((4 + 2 * occasions, 4 + 2 * occasions))
    omega[:4, :4] = (
        (0.08344460695463439, 0.0, 0.0, 0.0),
        (0.0, 0.19805053857439603, -0.01671233240450257, 0.13615474483258513),
        (0.0, -0.01671233240450257, 0.5873621303556349, -0.004502486522586959),
        (0.0, 0.13615474483258513, -0.004502486522586959, 0.20422654773754803),
    )
    for occasion in range(occasions):
        start = 4 + 2 * occasion
        omega[start:start + 2, start:start + 2] = ((0.08562800390349452, 0.0), (0.0, 0.955597979744738))
    return omega


@pytest.mark.parametrize("occasions", [3, 4])
def test_valid_full_tac_covariance_retains_all_dimensions_and_gaussian_normalization(occasions):
    omega = tac_covariance(occasions)
    before = omega.tobytes()
    config = replace(CONFIG, omega_covariance=omega)
    model = ConstantModel()
    result = public_call("posterior", config, model)
    diag = public_call("diagnostics", config, model)
    cached = pmx.multieta_state_diagnostics(
        (0.0,), STATES[0], model, config, omega, np.linalg.inv(omega), float(np.linalg.slogdet(omega)[1]),
    )
    assert not result.indeterminate
    assert result.posterior == pytest.approx({"a": 0.4, "b": 0.6}, abs=1e-12)
    assert result.log_evidence == pytest.approx(-0.5 * math.log(2.0 * math.pi), abs=2e-7)
    assert diag == cached == result.diagnostics["a"]
    assert all(len(eta) == 4 + 2 * occasions for _, eta in model.calls)
    assert omega.tobytes() == before


def test_full_tac_covariance_does_not_bypass_ghq_dimension_limit():
    model = ConstantModel()
    with pytest.raises(NotImplementedError, match="low-dimensional"):
        public_call("ghq", replace(CONFIG, omega_covariance=tac_covariance(3)), model)
    assert not model.calls
