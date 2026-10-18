from dataclasses import replace
import math

import numpy as np
import pytest

import pmx_discrete_posterior as pmx


class LinearModel:
    def __init__(self):
        self.calls = []

    def predictions(self, state_id, eta):
        self.calls.append((state_id, tuple(eta)))
        offset = 0.0 if state_id == "a" else 1.0
        return np.asarray(eta) + offset


class ConstantModel:
    def __init__(self, predictions):
        self.values = predictions

    def predictions(self, state_id, eta):
        return self.values[state_id]


STATES = (pmx.DiscreteState("a", 0.4), pmx.DiscreteState("b", 0.6))
CONFIG = pmx.MultiEtaLaplaceConfig(
    omega_covariance=((0.4, 0.1), (0.1, 0.3)), additive_sd=0.5,
)


def compute(config=CONFIG, model=None):
    return pmx.compute_multieta_posterior((0.3, -0.2), STATES, model or LinearModel(), config)


def normal_log_density(y, mean, covariance):
    delta = np.asarray(y) - np.asarray(mean)
    return float(-0.5 * (
        len(delta) * math.log(2 * math.pi) + np.linalg.slogdet(covariance)[1]
        + delta @ np.linalg.solve(covariance, delta)
    ))


def test_public_nonzero_start_is_used_for_each_state_without_mutation():
    start = np.array([0.7, -0.6])
    model = LinearModel()
    result = compute(replace(CONFIG, optimizer_start=start), model)
    for state in STATES:
        first = next(eta for state_id, eta in model.calls if state_id == state.state_id)
        assert first == tuple(start)
    np.testing.assert_array_equal(start, [0.7, -0.6])
    assert not result.indeterminate
    assert result.posterior == pytest.approx(compute().posterior, abs=1e-7)


def test_public_nelder_mead_simplex_and_tolerances_are_used_without_mutation():
    start = np.array([0.5, -0.4])
    simplex = np.array([[0.5, -0.4], [0.7, -0.4], [0.5, -0.2]])
    original = simplex.copy()
    options = {"initial_simplex": simplex, "xatol": 1e-10, "fatol": 1e-10, "adaptive": True}
    model = LinearModel()
    result = compute(replace(CONFIG, optimizer_start=start, optimizer_method="Nelder-Mead", optimizer_options=options), model)
    np.testing.assert_array_equal([eta for _, eta in model.calls[:3]], original)
    np.testing.assert_array_equal(simplex, original)
    assert "maxiter" not in options
    assert not result.indeterminate
    assert result.posterior == pytest.approx(compute().posterior, abs=1e-7)


def test_single_state_public_entry_accepts_explicit_start():
    model = LinearModel()
    diag = pmx.multieta_state_diagnostics((0.3, -0.2), STATES[0], model, replace(CONFIG, optimizer_start=(0.2, 0.8)))
    assert model.calls[0][1] == (0.2, 0.8)
    assert diag.valid_laplace


@pytest.mark.parametrize("bounds", [(-math.inf, math.inf), (-math.inf, 5.0), (-5.0, math.inf)])
def test_unbounded_and_one_sided_optimizer_bounds_remain_supported(bounds):
    result = compute(replace(CONFIG, eta_lower=bounds[0], eta_upper=bounds[1],
                             optimizer_method="Nelder-Mead", optimizer_start=(0.0, 0.0),
                             optimizer_options={"xatol": 1e-9, "fatol": 1e-10}))
    assert not result.indeterminate
    assert result.posterior == pytest.approx(compute().posterior, abs=1e-7)


@pytest.mark.parametrize("bounds", [(math.nan, 5.0), (-5.0, math.nan), (0.0, 0.0),
                                     (2.0, -2.0), (math.inf, math.inf), (-math.inf, -math.inf)])
def test_invalid_optimizer_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="ETA optimizer bounds"):
        compute(replace(CONFIG, eta_lower=bounds[0], eta_upper=bounds[1]))


@pytest.mark.parametrize("start", [(), (0.0,), (0.0, 0.0, 0.0), [[0.0, 0.0]], (math.nan, 0.0), (math.inf, 0.0), (6.0, 0.0), (-6.0, 0.0), "00", 0.0])
def test_invalid_starts_are_rejected_before_model_evaluation(start):
    model = LinearModel()
    with pytest.raises(ValueError, match="optimizer_start"):
        compute(replace(CONFIG, optimizer_start=start), model)
    assert not model.calls


@pytest.mark.parametrize("value", [0, -1, 1.5, True, math.inf])
def test_invalid_maxiter_is_rejected(value):
    with pytest.raises(ValueError, match="optimizer_maxiter"):
        compute(replace(CONFIG, optimizer_maxiter=value))


@pytest.mark.parametrize("value", [1, 1000, 100000])
def test_options_cannot_override_declared_maxiter_even_with_equal_value(value):
    with pytest.raises(ValueError, match="optimizer_options must not contain maxiter"):
        compute(replace(CONFIG, optimizer_options={"maxiter": value}))


@pytest.mark.parametrize("options", [[], {1: 2}, {"maxfev": 0}, {"maxfev": True}, {"xatol": -1}, {"fatol": math.nan}])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        compute(replace(CONFIG, optimizer_method="Nelder-Mead", optimizer_options=options))


@pytest.mark.parametrize("simplex", [
    [[0, 0], [1, 0]], [[0, 0], [1, 0], [0, math.nan]],
    [[0, 0], [6, 0], [0, 1]], [[0, 0], [1, 0], [2, 0]],
    [[1, 0], [1, 1], [0, 1]],
])
def test_invalid_simplex_is_rejected(simplex):
    with pytest.raises(ValueError, match="initial_simplex"):
        compute(replace(CONFIG, optimizer_method="Nelder-Mead", optimizer_options={"initial_simplex": simplex}))


@pytest.mark.parametrize("options", [{"initial_simplex": [[0, 0], [1, 0], [0, 1]]}, {"xatol": 1e-6}, {"fatol": 1e-6}])
def test_nelder_mead_only_options_rejected_for_other_methods(options):
    with pytest.raises(ValueError, match="Nelder-Mead"):
        compute(replace(CONFIG, optimizer_options=options))


def test_maxiter_failure_still_flags_top_level_indeterminate():
    result = compute(replace(CONFIG, optimizer_start=(2.0, -2.0), optimizer_method="Nelder-Mead", optimizer_maxiter=1))
    assert result.indeterminate
    assert all(not diag.optimizer_success for diag in result.diagnostics.values())
    assert any("optimizer_reported_unsuccessful" in warning for warning in result.warnings)


def test_boundary_solution_still_flags_top_level_indeterminate():
    result = compute(replace(CONFIG, eta_lower=-0.01, eta_upper=0.01))
    assert result.indeterminate
    assert any(diag.boundary_hit for diag in result.diagnostics.values())


@pytest.mark.parametrize("mode", ["legacy_combined", "additive_only"])
def test_linear_gaussian_absolute_evidence_hessian_and_prior_normalization(mode):
    config = replace(CONFIG, residual_mode=mode)
    result = compute(config)
    omega = np.asarray(CONFIG.omega_covariance)
    covariance = omega + CONFIG.additive_sd ** 2 * np.eye(2)
    hessian = np.linalg.inv(omega) + np.eye(2) / CONFIG.additive_sd ** 2
    log_weights = []
    for state in STATES:
        diag = result.diagnostics[state.state_id]
        mean = (0.0, 0.0) if state.state_id == "a" else (1.0, 1.0)
        expected = normal_log_density((0.3, -0.2), mean, covariance)
        assert diag.laplace_log_likelihood == pytest.approx(expected, abs=2e-7)
        assert diag.hessian_logdet == pytest.approx(np.linalg.slogdet(hessian)[1], abs=2e-7)
        assert diag.log_prior == pytest.approx(math.log(state.prior))
        log_weights.append(expected + math.log(state.prior))
    evidence = float(np.logaddexp.reduce(log_weights))
    assert result.log_evidence == pytest.approx(evidence, abs=2e-7)
    assert list(result.posterior.values()) == pytest.approx(np.exp(np.asarray(log_weights) - evidence), abs=1e-7)


def test_strict_proportional_density_normalization_and_no_negative_prediction_clipping():
    model = ConstantModel({"a": (-2.0, 4.0), "b": (-3.0, 5.0)})
    config = replace(CONFIG, additive_sd=0.0, proportional_sd=0.25, residual_mode="proportional_only", positive_floor=100.0)
    observations = (-2.3, 3.9)
    result = pmx.compute_multieta_posterior(observations, STATES, model, config)
    gh, log_marginals = pmx.gauss_hermite_posterior(observations, STATES, model, config, nodes=9)
    for state in STATES:
        pred = np.asarray(model.values[state.state_id])
        sd = 0.25 * np.abs(pred)
        diag = result.diagnostics[state.state_id]
        np.testing.assert_array_equal(diag.pred_at_eta_hat, pred)
        np.testing.assert_array_equal(diag.residual_sd_at_eta_hat, sd)
        expected = normal_log_density(observations, pred, np.diag(sd ** 2))
        assert diag.laplace_log_likelihood == pytest.approx(expected, abs=2e-7)
        assert log_marginals[state.state_id] == pytest.approx(expected, abs=1e-12)
    assert result.posterior == pytest.approx(gh, abs=1e-7)
    assert not result.indeterminate


def test_strict_additive_accepts_zero_and_negative_predictions_without_floor():
    config = replace(CONFIG, additive_sd=1e-9, residual_mode="additive_only", positive_floor=100.0)
    model = ConstantModel({"a": (0.0, -2.0), "b": (0.0, -2.0)})
    result = pmx.compute_multieta_posterior((0.0, -2.0), STATES, model, config)
    for diag in result.diagnostics.values():
        assert diag.residual_sd_at_eta_hat == (1e-9, 1e-9)
        expected = -math.log(2 * math.pi) - 2 * math.log(1e-9)
        assert diag.laplace_log_likelihood == pytest.approx(expected, abs=2e-6)
    assert result.posterior == pytest.approx({"a": 0.4, "b": 0.6})


@pytest.mark.parametrize("mode", ["additive_only", "proportional_only"])
@pytest.mark.parametrize("active", [0.0, -0.2, math.nan, math.inf])
def test_strict_active_sd_must_be_finite_and_positive(mode, active):
    add, prop = (active, 0.0) if mode == "additive_only" else (0.0, active)
    with pytest.raises(ValueError, match="active SD"):
        compute(replace(CONFIG, additive_sd=add, proportional_sd=prop, residual_mode=mode))


@pytest.mark.parametrize("mode", ["additive_only", "proportional_only"])
@pytest.mark.parametrize("inactive", [0.1, -0.1, math.nan, math.inf])
def test_strict_inactive_sd_must_be_zero(mode, inactive):
    add, prop = (0.2, inactive) if mode == "additive_only" else (inactive, 0.2)
    with pytest.raises(ValueError, match="inactive SD"):
        compute(replace(CONFIG, additive_sd=add, proportional_sd=prop, residual_mode=mode))


@pytest.mark.parametrize("mode", ["additive_only", "proportional_only"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_strict_nonfinite_predictions_are_rejected(mode, value):
    add, prop = (0.2, 0.0) if mode == "additive_only" else (0.0, 0.2)
    config = replace(CONFIG, additive_sd=add, proportional_sd=prop, residual_mode=mode)
    with pytest.raises(ValueError, match="finite model predictions"):
        compute(config, ConstantModel({"a": (value, 1.0)}))


@pytest.mark.parametrize("prediction,coefficient", [(0.0, 0.2), (1e-300, 1e-300), (1e300, 1e300)])
def test_strict_proportional_zero_underflow_or_infinite_sd_is_rejected(prediction, coefficient):
    config = replace(CONFIG, additive_sd=0.0, proportional_sd=coefficient, residual_mode="proportional_only")
    with pytest.raises(ValueError, match="strictly positive residual SDs"):
        compute(config, ConstantModel({"a": (prediction, 1.0)}))


def test_strict_nonfinite_log_density_is_rejected_without_floor():
    config = replace(CONFIG, additive_sd=1e-300, residual_mode="additive_only")
    with pytest.raises(ValueError, match="finite log observation density"):
        compute(config, ConstantModel({"a": (1e300, 1.0)}))


def test_quadrature_and_single_state_entry_enforce_strict_validation():
    config = replace(CONFIG, additive_sd=0.0, proportional_sd=0.2, residual_mode="proportional_only")
    model = ConstantModel({"a": (0.0, 1.0)})
    with pytest.raises(ValueError, match="strictly positive residual SDs"):
        pmx.gauss_hermite_posterior((0.0, 1.0), STATES[:1], model, config, nodes=3)
    with pytest.raises(ValueError, match="strictly positive residual SDs"):
        pmx.multieta_state_diagnostics((0.0, 1.0), STATES[0], model, config)


def test_unknown_residual_mode_is_rejected():
    with pytest.raises(ValueError, match="residual_mode"):
        compute(replace(CONFIG, residual_mode="additve_only"))


def test_prediction_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="predictions for"):
        compute(replace(CONFIG, residual_mode="additive_only"), ConstantModel({"a": (1.0,)}))


def test_legacy_floor_and_clipping_remain_the_default():
    config = replace(CONFIG, additive_sd=0.0, proportional_sd=0.2, positive_floor=0.1)
    assert config.residual_mode == "legacy_combined"
    model = ConstantModel({"a": (-2.0, 4.0), "b": (-2.0, 4.0)})
    result = compute(config, model)
    for diag in result.diagnostics.values():
        assert diag.pred_at_eta_hat == (-2.0, 4.0)
        assert diag.residual_sd_at_eta_hat == pytest.approx((0.1, math.sqrt(0.1 ** 2 + 0.8 ** 2)))
    assert result.posterior == pytest.approx({"a": 0.4, "b": 0.6})


def test_structural_zero_and_unnormalized_priors_are_unchanged():
    states = (pmx.DiscreteState("a", 2.0), pmx.DiscreteState("b", 0.0))
    result = pmx.compute_multieta_posterior((0.3, -0.2), states, LinearModel(), replace(CONFIG, residual_mode="additive_only"))
    assert result.posterior == {"a": 1.0, "b": 0.0}
    assert result.diagnostics["b"].log_prior == -math.inf
    assert result.log_evidence == pytest.approx(result.diagnostics["a"].laplace_log_likelihood + math.log(2.0))


def test_existing_positional_config_order_and_version_are_preserved():
    config = pmx.MultiEtaLaplaceConfig(((1.0,),), 0.5, 0.0, -5.0, 5.0, "L-BFGS-B", 1000, 1e-4, 1e-4, 1e-12, "structural_zero")
    assert config.optimizer_start is None
    assert config.optimizer_options is None
    assert config.residual_mode == "legacy_combined"
    assert pmx.__version__ == "0.1.4"
