from __future__ import annotations

from .api import (
    BoundedMinimum,
    DiscretePosteriorResult,
    DiscreteState,
    LaplaceStateDiagnostics,
    OneEtaLaplaceConfig,
    OneEtaPredictionModel,
    StatePosterior,
)
from .engine import (
    build_laplace_curve_payload,
    compute_laplace_posterior,
    finite_second_derivative,
    laplace_probabilities,
    laplace_state_diagnostics,
    logsumexp,
    minimize_bounded,
    states_from_prior,
)
from .multieta import (
    MultiEtaLaplaceConfig,
    MultiEtaPosteriorResult,
    MultiEtaPredictionModel,
    MultiEtaStateDiagnostics,
    MultiEtaStatePosterior,
    compute_multieta_posterior,
    finite_difference_hessian,
    gauss_hermite_posterior,
    multieta_state_diagnostics,
)
from .oxc import OxcLaplacePk, OxcOneEtaDoseHistoryModel, build_oxc_dose_history_states
from .reporting import (
    DEFAULT_NO_CALL_RULE,
    NoCallRule,
    PosteriorEvidenceReport,
    StateEvidenceRow,
    apply_no_call_rule,
    build_case_card,
    build_provenance_manifest,
    normalized_entropy,
    posterior_margin,
    posterior_to_long_rows,
    posterior_to_wide_row,
    reweight_prior_sensitivity,
    summarize_posterior_evidence,
    threshold_sensitivity_panel,
)

LaplaceConfig = OneEtaLaplaceConfig
OxcSingleEtaModel = OxcOneEtaDoseHistoryModel
compute_posterior = compute_laplace_posterior
compute_curve = build_laplace_curve_payload
__version__ = "0.1.4"

__all__ = [
    "__version__",
    "BoundedMinimum",
    "DiscretePosteriorResult",
    "DiscreteState",
    "LaplaceConfig",
    "LaplaceStateDiagnostics",
    "MultiEtaLaplaceConfig",
    "MultiEtaPosteriorResult",
    "MultiEtaPredictionModel",
    "MultiEtaStateDiagnostics",
    "MultiEtaStatePosterior",
    "OneEtaLaplaceConfig",
    "OneEtaPredictionModel",
    "OxcLaplacePk",
    "OxcOneEtaDoseHistoryModel",
    "OxcSingleEtaModel",
    "DEFAULT_NO_CALL_RULE",
    "NoCallRule",
    "PosteriorEvidenceReport",
    "StateEvidenceRow",
    "StatePosterior",
    "apply_no_call_rule",
    "build_case_card",
    "build_provenance_manifest",
    "compute_curve",
    "compute_posterior",
    "build_laplace_curve_payload",
    "build_oxc_dose_history_states",
    "compute_laplace_posterior",
    "compute_multieta_posterior",
    "finite_second_derivative",
    "finite_difference_hessian",
    "gauss_hermite_posterior",
    "laplace_probabilities",
    "laplace_state_diagnostics",
    "logsumexp",
    "minimize_bounded",
    "multieta_state_diagnostics",
    "normalized_entropy",
    "posterior_margin",
    "posterior_to_long_rows",
    "posterior_to_wide_row",
    "reweight_prior_sensitivity",
    "summarize_posterior_evidence",
    "states_from_prior",
    "threshold_sensitivity_panel",
]
