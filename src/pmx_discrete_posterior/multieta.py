from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss

from .api import DiscreteState
from .engine import logsumexp


_OMEGA_SYMMETRY_RTOL = 1e-10
_OMEGA_SYMMETRY_ATOL = 1e-12


try:  # pragma: no cover - scipy availability is runtime-specific
    from scipy.optimize import minimize as _scipy_minimize
except Exception:  # pragma: no cover
    _scipy_minimize = None


class MultiEtaPredictionModel(Protocol):
    """Model interface for multi-ETA, multi-observation posterior calculations."""

    def predictions(self, state_id: str, eta: Sequence[float]) -> Sequence[float]:
        """Return model predictions for all observations in one discrete state."""
        ...


@dataclass(frozen=True)
class MultiEtaLaplaceConfig:
    """Numerical settings for multi-ETA Laplace approximation."""

    omega_covariance: Sequence[Sequence[float]]
    additive_sd: float
    proportional_sd: float = 0.0
    eta_lower: float = -5.0
    eta_upper: float = 5.0
    optimizer_method: str = "L-BFGS-B"
    optimizer_maxiter: int = 1000
    derivative_step_min: float = 1e-4
    derivative_step_scale: float = 1e-4
    positive_floor: float = 1e-12
    prior_policy: str = "structural_zero"
    optimizer_start: Sequence[float] | None = None
    optimizer_options: Mapping[str, Any] | None = None
    residual_mode: str = "legacy_combined"


@dataclass(frozen=True)
class MultiEtaStateDiagnostics:
    """Per-state diagnostics for multi-ETA Laplace posterior calculations."""

    state_id: str
    eta_hat: tuple[float, ...]
    psi_hat: float
    laplace_log_likelihood: float
    log_prior: float
    log_weight: float
    hessian_logdet: float
    hessian_min_eigenvalue: float
    pred_at_eta_hat: tuple[float, ...]
    residual_sd_at_eta_hat: tuple[float, ...]
    optimizer_success: bool
    optimizer_message: str
    boundary_hit: bool
    valid_laplace: bool
    indeterminate: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiEtaStatePosterior:
    """Posterior result for one state in a multi-ETA calculation."""

    state: DiscreteState
    posterior_probability: float
    diagnostics: MultiEtaStateDiagnostics


@dataclass(frozen=True)
class MultiEtaPosteriorResult:
    """Posterior probabilities over states for a multi-ETA observation vector."""

    observations: tuple[float, ...]
    states: tuple[MultiEtaStatePosterior, ...]
    log_evidence: float
    map_state_id: str | None
    source: str = "multi_eta_laplace_approximation"
    indeterminate: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def posterior(self) -> dict[str, float]:
        return {
            state_result.state.state_id: state_result.posterior_probability
            for state_result in self.states
        }

    @property
    def diagnostics(self) -> dict[str, MultiEtaStateDiagnostics]:
        return {
            state_result.state.state_id: state_result.diagnostics
            for state_result in self.states
        }


def compute_multieta_posterior(
    observations: Sequence[float],
    states: Sequence[DiscreteState],
    model: MultiEtaPredictionModel,
    config: MultiEtaLaplaceConfig,
) -> MultiEtaPosteriorResult:
    """Compute posterior probabilities using a multivariate Laplace approximation."""

    y = _coerce_observations(observations)
    omega, omega_inv, omega_logdet = _omega_matrices(config)
    if omega.shape[0] < 1:
        raise ValueError("omega_covariance must have at least one ETA dimension.")
    _validate_prior_policy(config)
    _validate_states(states)
    _validate_residual_mode(config)

    diagnostics = [
        multieta_state_diagnostics(y, state, model, config, omega, omega_inv, omega_logdet)
        for state in states
    ]
    log_evidence = logsumexp(diag.log_weight for diag in diagnostics)
    state_results: list[MultiEtaStatePosterior] = []
    for state, diag in zip(states, diagnostics):
        probability = (
            math.exp(diag.log_weight - log_evidence)
            if math.isfinite(diag.log_weight) and math.isfinite(log_evidence)
            else 0.0
        )
        state_results.append(MultiEtaStatePosterior(state, float(probability), diag))

    map_state_id = max(state_results, key=lambda item: item.posterior_probability).state.state_id
    return MultiEtaPosteriorResult(
        observations=tuple(float(value) for value in y),
        states=tuple(state_results),
        log_evidence=float(log_evidence),
        map_state_id=map_state_id,
        indeterminate=any(diag.indeterminate for diag in diagnostics),
        warnings=tuple(
            f"{diag.state_id}:{warning}"
            for diag in diagnostics
            for warning in diag.warnings
        ),
    )


def multieta_state_diagnostics(
    observations: Sequence[float],
    state: DiscreteState,
    model: MultiEtaPredictionModel,
    config: MultiEtaLaplaceConfig,
    omega: np.ndarray | None = None,
    omega_inv: np.ndarray | None = None,
    omega_logdet: float | None = None,
) -> MultiEtaStateDiagnostics:
    """Compute one state's marginal likelihood and diagnostics."""

    if _scipy_minimize is None:
        raise RuntimeError("scipy is required for multi-ETA optimization. Install the 'scipy' extra.")
    y = _coerce_observations(observations)
    if omega is None or omega_inv is None or omega_logdet is None:
        omega, omega_inv, omega_logdet = _omega_matrices(config)
    else:
        # Precomputed arguments must not bypass the public covariance guard.
        _validated_omega(config.omega_covariance)
        omega = _validated_omega(omega)
    dim = omega.shape[0]
    _validate_residual_mode(config)
    start, options = _optimizer_inputs(config, dim)
    bounds = [(float(config.eta_lower), float(config.eta_upper)) for _ in range(dim)]

    def objective(eta: Sequence[float]) -> float:
        return _negative_log_joint(y, state.state_id, eta, model, config, omega_inv, float(omega_logdet))

    result = _scipy_minimize(
        objective,
        start,
        method=config.optimizer_method,
        bounds=bounds,
        options=options,
    )
    eta_hat = np.asarray(result.x, dtype=float)
    psi_hat = float(result.fun)
    hessian = finite_difference_hessian(
        objective,
        eta_hat,
        step_min=float(config.derivative_step_min),
        step_scale=float(config.derivative_step_scale),
    )
    eigvals = np.linalg.eigvalsh((hessian + hessian.T) / 2.0)
    hessian_min = float(np.min(eigvals))
    sign, hessian_logdet = np.linalg.slogdet(hessian)
    warnings: list[str] = []
    hessian_valid = bool(sign > 0 and math.isfinite(float(hessian_logdet)) and hessian_min > 0.0)
    if not hessian_valid:
        warnings.append("nonpositive_or_nonfinite_hessian")
        hessian_logdet = math.inf
        log_likelihood = -math.inf
    else:
        log_likelihood = -psi_hat + 0.5 * dim * math.log(2.0 * math.pi) - 0.5 * float(hessian_logdet)

    boundary_tol = 1e-6
    boundary_hit = bool(
        np.any(np.abs(eta_hat - float(config.eta_lower)) <= boundary_tol)
        or np.any(np.abs(eta_hat - float(config.eta_upper)) <= boundary_tol)
    )
    if boundary_hit:
        warnings.append("eta_hat_at_optimizer_boundary")
    if not bool(result.success):
        warnings.append("optimizer_reported_unsuccessful")

    log_prior = _log_prior(state.prior, policy=config.prior_policy)
    log_weight = log_prior + log_likelihood if math.isfinite(log_prior) else -math.inf
    pred = np.asarray(model.predictions(state.state_id, eta_hat), dtype=float)
    residual_sd = _combined_residual_sd(pred, config)
    valid_laplace = bool(hessian_valid and result.success and not boundary_hit)
    return MultiEtaStateDiagnostics(
        state_id=state.state_id,
        eta_hat=tuple(float(value) for value in eta_hat),
        psi_hat=psi_hat,
        laplace_log_likelihood=float(log_likelihood),
        log_prior=float(log_prior),
        log_weight=float(log_weight),
        hessian_logdet=float(hessian_logdet),
        hessian_min_eigenvalue=hessian_min,
        pred_at_eta_hat=tuple(float(value) for value in pred),
        residual_sd_at_eta_hat=tuple(float(value) for value in residual_sd),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        boundary_hit=boundary_hit,
        valid_laplace=valid_laplace,
        indeterminate=not valid_laplace,
        warnings=tuple(warnings),
    )


def gauss_hermite_posterior(
    observations: Sequence[float],
    states: Sequence[DiscreteState],
    model: MultiEtaPredictionModel,
    config: MultiEtaLaplaceConfig,
    *,
    nodes: int = 35,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute a tensor-product Gauss-Hermite reference posterior under N(0, Omega)."""

    y = _coerce_observations(observations)
    _validate_prior_policy(config)
    _validate_states(states)
    _validate_residual_mode(config)
    omega, _, _ = _omega_matrices(config)
    dim = omega.shape[0]
    if dim > 3:
        raise NotImplementedError("Gauss-Hermite reference is intended for low-dimensional stress tests.")
    node_values, weights = hermgauss(int(nodes))
    chol = np.linalg.cholesky(omega)
    log_marginals: dict[str, float] = {}
    for state in states:
        terms: list[float] = []
        for index_tuple in np.ndindex(*(nodes for _ in range(dim))):
            z = np.asarray([node_values[index] for index in index_tuple], dtype=float)
            weight_log = sum(math.log(float(weights[index])) for index in index_tuple)
            eta = math.sqrt(2.0) * chol @ z
            terms.append(weight_log + _log_observation_likelihood(y, state.state_id, eta, model, config))
        log_marginals[state.state_id] = logsumexp(terms) - 0.5 * dim * math.log(math.pi)

    log_weights = {
        state.state_id: _log_prior(state.prior, policy=config.prior_policy) + log_marginals[state.state_id]
        for state in states
    }
    norm = logsumexp(log_weights.values())
    posterior = {
        state_id: math.exp(log_weight - norm) if math.isfinite(log_weight) else 0.0
        for state_id, log_weight in log_weights.items()
    }
    return posterior, log_marginals


def finite_difference_hessian(
    func: Any,
    x: Sequence[float],
    *,
    step_min: float = 1e-4,
    step_scale: float = 1e-4,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    dim = x_arr.size
    hessian = np.zeros((dim, dim), dtype=float)
    steps = np.maximum(float(step_min), np.abs(x_arr) * float(step_scale))
    f0 = float(func(x_arr))
    for i in range(dim):
        ei = np.zeros(dim, dtype=float)
        ei[i] = steps[i]
        hessian[i, i] = (float(func(x_arr + ei)) - 2.0 * f0 + float(func(x_arr - ei))) / (steps[i] ** 2)
        for j in range(i + 1, dim):
            ej = np.zeros(dim, dtype=float)
            ej[j] = steps[j]
            hessian[i, j] = hessian[j, i] = (
                float(func(x_arr + ei + ej))
                - float(func(x_arr + ei - ej))
                - float(func(x_arr - ei + ej))
                + float(func(x_arr - ei - ej))
            ) / (4.0 * steps[i] * steps[j])
    return hessian


def _negative_log_joint(
    observations: np.ndarray,
    state_id: str,
    eta: Sequence[float],
    model: MultiEtaPredictionModel,
    config: MultiEtaLaplaceConfig,
    omega_inv: np.ndarray,
    omega_logdet: float,
) -> float:
    eta_arr = np.asarray(eta, dtype=float)
    log_likelihood = _log_observation_likelihood(observations, state_id, eta_arr, model, config)
    dim = eta_arr.size
    eta_prior_neg_log = (
        0.5 * dim * math.log(2.0 * math.pi)
        + 0.5 * float(omega_logdet)
        + 0.5 * float(eta_arr @ omega_inv @ eta_arr)
    )
    return float(-log_likelihood + eta_prior_neg_log)


def _log_observation_likelihood(
    observations: np.ndarray,
    state_id: str,
    eta: Sequence[float],
    model: MultiEtaPredictionModel,
    config: MultiEtaLaplaceConfig,
) -> float:
    pred = np.asarray(model.predictions(state_id, eta), dtype=float)
    if pred.shape != observations.shape:
        raise ValueError(
            f"Model returned {pred.size} predictions for {observations.size} observations."
        )
    sd = _combined_residual_sd(pred, config)
    if config.residual_mode != "legacy_combined":
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value = float(np.sum(
                    -0.5 * math.log(2.0 * math.pi) - np.log(sd)
                    - 0.5 * ((observations - pred) / sd) ** 2
                ))
        except FloatingPointError as exc:
            raise ValueError("Strict residual mode requires a finite log observation density.") from exc
        if not math.isfinite(value):
            raise ValueError("Strict residual mode requires a finite log observation density.")
        return value
    return float(np.sum(-0.5 * math.log(2.0 * math.pi) - np.log(sd) - 0.5 * ((observations - pred) / sd) ** 2))


def _combined_residual_sd(pred: np.ndarray, config: MultiEtaLaplaceConfig) -> np.ndarray:
    if config.residual_mode != "legacy_combined":
        _validate_residual_mode(config)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Strict residual mode requires finite model predictions.")
        if config.residual_mode == "additive_only":
            sd = np.full_like(pred, float(config.additive_sd), dtype=float)
        else:
            # abs(pred) is the SD implied by (proportional_sd * pred)**2,
            # not clipping of the prediction used as the observation mean.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                sd = float(config.proportional_sd) * np.abs(pred)
        if not np.all(np.isfinite(sd) & (sd > 0.0)):
            raise ValueError("Strict residual mode requires finite, strictly positive residual SDs.")
        return sd
    floor = max(float(config.positive_floor), 1e-300)
    add = max(float(config.additive_sd), floor)
    prop = max(float(config.proportional_sd), 0.0)
    return np.maximum(np.sqrt(add * add + (prop * np.maximum(pred, 0.0)) ** 2), floor)


def _validate_residual_mode(config: MultiEtaLaplaceConfig) -> None:
    if config.residual_mode == "legacy_combined":
        return
    if config.residual_mode not in ("additive_only", "proportional_only"):
        raise ValueError("residual_mode must be 'legacy_combined', 'additive_only', or 'proportional_only'.")
    add, prop = float(config.additive_sd), float(config.proportional_sd)
    active, inactive = (add, prop) if config.residual_mode == "additive_only" else (prop, add)
    if not math.isfinite(active) or active <= 0.0:
        raise ValueError(f"{config.residual_mode} requires a finite, strictly positive active SD.")
    if inactive != 0.0:
        raise ValueError(f"{config.residual_mode} requires the inactive SD to be exactly zero.")


def _optimizer_inputs(config: MultiEtaLaplaceConfig, dim: int) -> tuple[np.ndarray, dict[str, Any]]:
    lower, upper = float(config.eta_lower), float(config.eta_upper)
    if math.isnan(lower) or math.isnan(upper) or lower >= upper:
        raise ValueError("ETA optimizer bounds must be ordered and non-NaN with eta_lower < eta_upper.")
    if dim < 1:
        raise ValueError("omega_covariance must have at least one ETA dimension.")
    if isinstance(config.optimizer_start, (str, bytes)):
        raise ValueError("optimizer_start must be a finite numeric vector, not a string.")
    start = np.zeros(dim, dtype=float) if config.optimizer_start is None else np.array(config.optimizer_start, dtype=float, copy=True)
    if start.shape != (dim,) or not np.all(np.isfinite(start)):
        raise ValueError(f"optimizer_start must be a finite vector with {dim} ETA entries.")
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("optimizer_start must lie within the declared ETA bounds.")
    if not isinstance(config.optimizer_maxiter, Integral) or isinstance(config.optimizer_maxiter, bool) or config.optimizer_maxiter <= 0:
        raise ValueError("optimizer_maxiter must be a positive integer.")
    if config.optimizer_options is not None and not isinstance(config.optimizer_options, Mapping):
        raise ValueError("optimizer_options must be a mapping or None.")
    options = dict(config.optimizer_options or {})
    if any(not isinstance(key, str) for key in options):
        raise ValueError("optimizer_options keys must be strings.")
    if "maxiter" in options:
        raise ValueError("Set optimizer_maxiter explicitly; optimizer_options must not contain maxiter.")
    if "initial_simplex" in options:
        if config.optimizer_method.lower() != "nelder-mead":
            raise ValueError("initial_simplex is only supported with optimizer_method='Nelder-Mead'.")
        simplex = np.array(options["initial_simplex"], dtype=float, copy=True)
        if simplex.shape != (dim + 1, dim) or not np.all(np.isfinite(simplex)):
            raise ValueError(f"initial_simplex must be finite with shape ({dim + 1}, {dim}).")
        if np.any(simplex < lower) or np.any(simplex > upper):
            raise ValueError("initial_simplex must lie within the declared ETA bounds.")
        if not np.array_equal(simplex[0], start):
            raise ValueError("initial_simplex[0] must equal optimizer_start (zero when omitted).")
        if np.linalg.matrix_rank(simplex[1:] - simplex[0]) != dim:
            raise ValueError("initial_simplex must be nondegenerate in all ETA dimensions.")
        options["initial_simplex"] = simplex
    for name in ("xatol", "fatol"):
        if name in options:
            if config.optimizer_method.lower() != "nelder-mead":
                raise ValueError(f"{name} is only supported with optimizer_method='Nelder-Mead'.")
            value = float(options[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            options[name] = value
    if "maxfev" in options:
        value = options["maxfev"]
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError("maxfev must be a positive integer.")
    options["maxiter"] = int(config.optimizer_maxiter)
    return start, options


def _coerce_observations(observations: Sequence[float]) -> np.ndarray:
    if isinstance(observations, (str, bytes)):
        raise ValueError("observations must be a numeric sequence, not a string.")
    y = np.asarray(list(observations), dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("observations must be a non-empty one-dimensional numeric sequence.")
    if not np.all(np.isfinite(y)):
        raise ValueError("observations must be finite.")
    return y


def _validated_omega(covariance: Sequence[Sequence[float]]) -> np.ndarray:
    omega = np.asarray(covariance, dtype=float)
    if omega.ndim != 2 or omega.shape[0] != omega.shape[1]:
        raise ValueError("omega_covariance must be a square matrix.")
    if omega.shape[0] < 1:
        raise ValueError("omega_covariance must have at least one ETA dimension.")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega_covariance must be finite.")
    if np.any(np.diag(omega) <= 0.0):
        raise ValueError("omega_covariance must be positive definite.")
    # Check asymmetry in correlation units so tiny variances cannot mask it.
    scale = np.sqrt(np.diag(omega))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = omega / scale[:, None] / scale[None, :]
    except FloatingPointError as exc:
        raise ValueError("omega_covariance has invalid standardized entries.") from exc
    if not np.allclose(standardized, standardized.T,
                       rtol=_OMEGA_SYMMETRY_RTOL, atol=_OMEGA_SYMMETRY_ATOL):
        raise ValueError("omega_covariance must be symmetric in correlation units (rtol=1e-10, atol=1e-12).")
    try:
        np.linalg.cholesky(omega)
    except np.linalg.LinAlgError as exc:
        raise ValueError("omega_covariance must be positive definite.") from exc
    return omega


def _omega_matrices(config: MultiEtaLaplaceConfig) -> tuple[np.ndarray, np.ndarray, float]:
    omega = _validated_omega(config.omega_covariance)
    # Keep the legacy arithmetic after validation for bitwise-compatible output.
    sign, logdet = np.linalg.slogdet(omega)
    if sign <= 0 or not math.isfinite(float(logdet)):
        raise ValueError("omega_covariance must be positive definite.")
    return omega, np.linalg.inv(omega), float(logdet)


def _validate_states(states: Sequence[DiscreteState]) -> None:
    if not states:
        raise ValueError("At least one discrete state is required.")
    state_ids = [state.state_id for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("Discrete state identifiers must be unique.")
    positive_count = 0
    for state in states:
        prior = float(state.prior)
        if not math.isfinite(prior):
            raise ValueError(f"Prior for state {state.state_id!r} must be finite.")
        if prior < 0.0:
            raise ValueError(f"Prior for state {state.state_id!r} must be non-negative.")
        if prior > 0.0:
            positive_count += 1
    if positive_count == 0:
        raise ValueError("At least one state must have positive prior mass.")


def _validate_prior_policy(config: MultiEtaLaplaceConfig) -> None:
    if config.prior_policy != "structural_zero":
        raise NotImplementedError("Only prior_policy='structural_zero' is currently implemented.")


def _log_prior(prior: float, *, policy: str) -> float:
    if policy != "structural_zero":
        raise NotImplementedError("Only prior_policy='structural_zero' is currently implemented.")
    value = float(prior)
    if value <= 0.0 or not math.isfinite(value):
        return -math.inf
    return math.log(value)

