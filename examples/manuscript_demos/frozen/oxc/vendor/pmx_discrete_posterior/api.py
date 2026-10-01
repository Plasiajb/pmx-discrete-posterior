from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class DiscreteState:
    """A candidate discrete pharmacometric state with an unnormalized prior."""

    state_id: str
    prior: float
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return self.state_id if self.label is None else self.label


@dataclass(frozen=True)
class OneEtaLaplaceConfig:
    """Numerical settings for the one-random-effect Laplace approximation."""

    omega_variance: float
    residual_sd: float
    prior_policy: str = "structural_zero"
    eta_lower: float = -5.0
    eta_upper: float = 5.0
    optimizer_xatol: float = 1e-10
    derivative_step_min: float = 1e-5
    derivative_step_scale: float = 1e-4
    positive_floor: float = 1e-12
    hessian_floor: float = 1e-300


@dataclass(frozen=True)
class BoundedMinimum:
    """Scalar bounded optimizer result."""

    x: float
    fun: float
    success: bool
    method: str
    iterations: int | None = None
    message: str = ""


@dataclass(frozen=True)
class LaplaceStateDiagnostics:
    """Per-state numerical diagnostics from the Laplace approximation."""

    state_id: str
    eta_hat: float
    psi_hat: float
    hessian: float
    pred_at_eta_hat: float
    laplace_log_likelihood: float
    log_prior: float
    log_weight: float
    omega_variance: float
    residual_sd: float
    eta_lower: float
    eta_upper: float
    optimizer_method: str
    optimizer_success: bool
    optimizer_iterations: int | None
    boundary_hit: bool
    valid_laplace: bool = True
    indeterminate: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatePosterior:
    """Posterior result for one discrete state."""

    state: DiscreteState
    posterior_probability: float
    diagnostics: LaplaceStateDiagnostics


@dataclass(frozen=True)
class DiscretePosteriorResult:
    """Posterior probabilities over a finite discrete state set."""

    observation: float
    states: tuple[StatePosterior, ...]
    log_evidence: float
    map_state_id: str | None
    source: str = "one_eta_laplace_approximation"
    indeterminate: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def posterior(self) -> dict[str, float]:
        return {
            state_result.state.state_id: state_result.posterior_probability
            for state_result in self.states
        }

    @property
    def diagnostics(self) -> dict[str, LaplaceStateDiagnostics]:
        return {
            state_result.state.state_id: state_result.diagnostics
            for state_result in self.states
        }


class OneEtaPredictionModel(Protocol):
    """Minimal model interface consumed by the one-eta Laplace engine."""

    def prediction(self, state_id: str, eta: float) -> float:
        """Return the model prediction for a discrete state and eta value."""
        ...
