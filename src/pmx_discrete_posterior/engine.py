from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .api import (
    BoundedMinimum,
    DiscretePosteriorResult,
    DiscreteState,
    LaplaceStateDiagnostics,
    OneEtaLaplaceConfig,
    OneEtaPredictionModel,
    StatePosterior,
)


try:  # pragma: no cover - runtime acceleration only
    from scipy.optimize import minimize_scalar as _scipy_minimize_scalar
except Exception:  # pragma: no cover - covered through the pure Python fallback
    _scipy_minimize_scalar = None


def logsumexp(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return -math.inf
    top = max(finite_values)
    return top + math.log(sum(math.exp(value - top) for value in finite_values))


def finite_second_derivative(
    func: Callable[[float], float],
    x: float,
    *,
    step_min: float = 1e-5,
    step_scale: float = 1e-4,
) -> float:
    step = max(float(step_min), abs(float(x)) * float(step_scale))
    return (func(x + step) - 2.0 * func(x) + func(x - step)) / (step * step)


def minimize_bounded(
    func: Callable[[float], float],
    *,
    lower: float = -5.0,
    upper: float = 5.0,
    xatol: float = 1e-10,
) -> BoundedMinimum:
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Bounds must be finite and lower must be less than upper.")

    if _scipy_minimize_scalar is not None:
        result = _scipy_minimize_scalar(
            func,
            bounds=(float(lower), float(upper)),
            method="bounded",
            options={"xatol": float(xatol)},
        )
        return BoundedMinimum(
            x=float(result.x),
            fun=float(result.fun),
            success=bool(getattr(result, "success", True)),
            method="scipy.optimize.minimize_scalar:bounded",
            iterations=getattr(result, "nit", None),
            message=str(getattr(result, "message", "")),
        )

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0
    a = float(lower)
    b = float(upper)
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    yc = float(func(c))
    yd = float(func(d))
    iterations = 0
    for iterations in range(1, 241):
        if abs(b - a) <= xatol:
            break
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = float(func(c))
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = float(func(d))
    x = (a + b) / 2.0
    return BoundedMinimum(
        x=float(x),
        fun=float(func(x)),
        success=True,
        method="golden_section_bounded",
        iterations=iterations,
    )


def laplace_state_diagnostics(
    observation: float,
    state: DiscreteState,
    model: OneEtaPredictionModel,
    config: OneEtaLaplaceConfig,
) -> LaplaceStateDiagnostics:
    sigma, omega_variance, warnings = _coerce_positive_config(config)
    _validate_prior_policy(config)
    y = _coerce_scalar_observation(observation)

    def objective(eta: float) -> float:
        pred = float(model.prediction(state.state_id, eta))
        return 0.5 * ((y - pred) / sigma) ** 2 + 0.5 * eta * eta / omega_variance

    result = minimize_bounded(
        objective,
        lower=float(config.eta_lower),
        upper=float(config.eta_upper),
        xatol=float(config.optimizer_xatol),
    )
    eta_hat = float(result.x)
    psi_hat = float(result.fun)
    raw_hessian = float(
        finite_second_derivative(
            objective,
            eta_hat,
            step_min=float(config.derivative_step_min),
            step_scale=float(config.derivative_step_scale),
        )
    )
    hessian = raw_hessian
    local_warnings = list(warnings)
    hessian_valid = math.isfinite(hessian) and hessian > 0.0
    if not hessian_valid:
        hessian = float(config.hessian_floor)
        local_warnings.append("nonpositive_or_nonfinite_hessian_clamped")
    else:
        hessian = max(hessian, float(config.hessian_floor))

    log_likelihood = (
        -0.5 * math.log(2.0 * math.pi)
        - math.log(sigma)
        - 0.5 * math.log(omega_variance)
        - psi_hat
        - 0.5 * math.log(hessian)
    )
    log_prior = _log_prior(state.prior, policy=config.prior_policy)
    log_weight = log_prior + log_likelihood if math.isfinite(log_prior) else -math.inf
    boundary_tol = max(float(config.optimizer_xatol) * 10.0, 1e-8)
    boundary_hit = (
        abs(eta_hat - float(config.eta_lower)) <= boundary_tol
        or abs(eta_hat - float(config.eta_upper)) <= boundary_tol
    )
    if boundary_hit:
        local_warnings.append("eta_hat_at_optimizer_boundary")
    if not result.success:
        local_warnings.append("optimizer_reported_unsuccessful")

    valid_laplace = bool(hessian_valid and result.success and not boundary_hit)
    indeterminate = not valid_laplace

    return LaplaceStateDiagnostics(
        state_id=state.state_id,
        eta_hat=eta_hat,
        psi_hat=psi_hat,
        hessian=float(hessian),
        pred_at_eta_hat=float(model.prediction(state.state_id, eta_hat)),
        laplace_log_likelihood=float(log_likelihood),
        log_prior=float(log_prior),
        log_weight=float(log_weight),
        omega_variance=float(omega_variance),
        residual_sd=float(sigma),
        eta_lower=float(config.eta_lower),
        eta_upper=float(config.eta_upper),
        optimizer_method=result.method,
        optimizer_success=bool(result.success),
        optimizer_iterations=result.iterations,
        boundary_hit=boundary_hit,
        valid_laplace=valid_laplace,
        indeterminate=indeterminate,
        warnings=tuple(local_warnings),
    )


def compute_laplace_posterior(
    observation: float,
    states: Sequence[DiscreteState],
    model: OneEtaPredictionModel,
    config: OneEtaLaplaceConfig,
) -> DiscretePosteriorResult:
    observation_value = _coerce_scalar_observation(observation)
    _validate_prior_policy(config)
    if not states:
        raise ValueError("At least one discrete state is required.")
    state_ids = [state.state_id for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("Discrete state identifiers must be unique.")
    for state in states:
        prior = float(state.prior)
        if not math.isfinite(prior):
            raise ValueError(f"Prior for state {state.state_id!r} must be finite.")
        if prior < 0.0:
            raise ValueError(f"Prior for state {state.state_id!r} must be non-negative.")
    if all(float(state.prior) <= 0.0 for state in states):
        raise ValueError("At least one state must have positive prior mass.")

    diagnostics = [
        laplace_state_diagnostics(observation_value, state, model, config)
        for state in states
    ]
    log_evidence = logsumexp(diag.log_weight for diag in diagnostics)
    state_results: list[StatePosterior] = []
    for state, diag in zip(states, diagnostics):
        probability = (
            math.exp(diag.log_weight - log_evidence)
            if math.isfinite(diag.log_weight) and math.isfinite(log_evidence)
            else 0.0
        )
        state_results.append(
            StatePosterior(
                state=state,
                posterior_probability=float(probability),
                diagnostics=diag,
            )
        )
    map_state_id = max(
        state_results,
        key=lambda item: item.posterior_probability,
    ).state.state_id
    return DiscretePosteriorResult(
        observation=observation_value,
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


def states_from_prior(
    prior: Mapping[str, float],
    *,
    labels: Mapping[str, str] | None = None,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[DiscreteState, ...]:
    labels = {} if labels is None else labels
    metadata = {} if metadata is None else metadata
    return tuple(
        DiscreteState(
            state_id=str(state_id),
            prior=float(prior_value),
            label=labels.get(str(state_id)),
            metadata=metadata.get(str(state_id), {}),
        )
        for state_id, prior_value in prior.items()
    )


def laplace_probabilities(
    observation: float,
    scenarios: Sequence[str],
    model: OneEtaPredictionModel,
    prior: Mapping[str, float],
    config: OneEtaLaplaceConfig,
) -> tuple[dict[str, float], dict[str, LaplaceStateDiagnostics]]:
    states = tuple(
        DiscreteState(
            state_id=str(scenario),
            prior=float(prior.get(scenario, 0.0)),
        )
        for scenario in scenarios
    )
    result = compute_laplace_posterior(observation, states, model, config)
    return result.posterior, result.diagnostics


def build_laplace_curve_payload(
    x_grid: Sequence[float],
    states: Sequence[DiscreteState],
    model: OneEtaPredictionModel,
    config: OneEtaLaplaceConfig,
) -> dict[str, Any]:
    curves = {
        state.state_id: {
            "mixture_submodel_id": state.state_id,
            "scenario_key": state.state_id,
            "label": state.display_label,
            "definition": str(state.metadata.get("definition", "discrete pharmacometric state")),
            "points": [],
        }
        for state in states
    }
    long_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for point_index, x_value in enumerate(x_grid, start=1):
        result = compute_laplace_posterior(float(x_value), states, model, config)
        post_max = max(result.posterior.values()) if result.posterior else 0.0
        for state_result in result.states:
            state = state_result.state
            diag = state_result.diagnostics
            probability = float(state_result.posterior_probability)
            point = {
                "point_index": int(point_index),
                "x": float(x_value),
                "DV": float(x_value),
                "mixture_submodel_id": state.state_id,
                "scenario_key": state.state_id,
                "posterior_probability": probability,
                "posterior_percent": probability * 100.0,
                "laplace_posterior": probability,
                "PMIX": probability,
                "prior": float(state.prior),
                "eta_hat": diag.eta_hat,
                "pred_at_eta_hat": diag.pred_at_eta_hat,
                "psi_hat": diag.psi_hat,
                "hessian": diag.hessian,
                "laplace_log_likelihood": diag.laplace_log_likelihood,
                "log_prior": diag.log_prior,
                "log_weight": diag.log_weight,
                "boundary_hit": diag.boundary_hit,
                "valid_laplace": diag.valid_laplace,
                "indeterminate": diag.indeterminate,
                "warnings": ";".join(diag.warnings),
                "optimizer_success": diag.optimizer_success,
                "MAP": result.map_state_id,
                "POST_MAX": post_max,
                "source": result.source,
            }
            curves[state.state_id]["points"].append(point)
            long_rows.append(dict(point))
            parameter_rows.append(
                {
                    "point_index": int(point_index),
                    "DV": float(x_value),
                    "scenario": state.state_id,
                    "mixture_submodel_id": state.state_id,
                    "scenario_key": state.state_id,
                    "prior": float(state.prior),
                    "eta_hat": diag.eta_hat,
                    "pred_at_eta_hat": diag.pred_at_eta_hat,
                    "psi_hat": diag.psi_hat,
                    "hessian": diag.hessian,
                    "laplace_log_likelihood": diag.laplace_log_likelihood,
                    "laplace_posterior": probability,
                    "PMIX": probability,
                    "MAP": result.map_state_id,
                    "POST_MAX": post_max,
                    "boundary_hit": diag.boundary_hit,
                    "valid_laplace": diag.valid_laplace,
                    "indeterminate": diag.indeterminate,
                    "warnings": ";".join(diag.warnings),
                    "optimizer_success": diag.optimizer_success,
                }
            )
    return {
        "curves": list(curves.values()),
        "long_rows": long_rows,
        "point_parameter_rows": parameter_rows,
    }


def _coerce_positive_config(config: OneEtaLaplaceConfig) -> tuple[float, float, tuple[str, ...]]:
    floor = max(float(config.positive_floor), 1e-300)
    sigma = float(config.residual_sd)
    omega_variance = float(config.omega_variance)
    warnings: list[str] = []
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = floor
        warnings.append("residual_sd_clamped")
    else:
        sigma = max(sigma, floor)
    if not math.isfinite(omega_variance) or omega_variance <= 0.0:
        omega_variance = floor
        warnings.append("omega_variance_clamped")
    else:
        omega_variance = max(omega_variance, floor)
    return sigma, omega_variance, tuple(warnings)


def _coerce_scalar_observation(observation: Any) -> float:
    if isinstance(observation, (str, bytes)):
        return float(observation)
    if isinstance(observation, Iterable):
        raise NotImplementedError(
            "This prototype engine supports a single scalar observation. "
            "Vector or repeated-observation likelihoods require a dedicated model adapter."
        )
    return float(observation)


def _validate_prior_policy(config: OneEtaLaplaceConfig) -> None:
    if config.prior_policy != "structural_zero":
        raise NotImplementedError("Only prior_policy='structural_zero' is currently implemented.")


def _log_prior(prior: float, *, policy: str) -> float:
    if policy != "structural_zero":
        raise NotImplementedError("Only prior_policy='structural_zero' is currently implemented.")
    prior_value = float(prior)
    if not math.isfinite(prior_value) or prior_value <= 0.0:
        return -math.inf
    return math.log(prior_value)
