from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .engine import logsumexp


@dataclass(frozen=True)
class NoCallRule:
    """Prespecified rule for flagging weak posterior separation."""

    rule_id: str = "NC_DEFAULT"
    margin_threshold: float = 0.10
    entropy_threshold: float = 0.80
    include_indeterminate: bool = True


DEFAULT_NO_CALL_RULE = NoCallRule()
FSRS10_RULE_ID = "FSRS10_GEOMETRY_V1"


@dataclass(frozen=True)
class StateEvidenceRow:
    """Report-ready evidence for one candidate discrete state."""

    state_id: str
    label: str
    prior: float
    posterior_probability: float
    is_map: bool
    log_likelihood: float | None = None
    log_prior: float | None = None
    log_weight: float | None = None
    eta_hat: Any = None
    pred_at_eta_hat: Any = None
    psi_hat: float | None = None
    hessian: float | None = None
    hessian_logdet: float | None = None
    hessian_min_eigenvalue: float | None = None
    valid_laplace: bool | None = None
    indeterminate: bool | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PosteriorEvidenceReport:
    """Clinician-facing posterior evidence summary for prespecified states."""

    posterior: Mapping[str, float]
    state_rows: tuple[StateEvidenceRow, ...]
    map_state_id: str | None
    pmax: float
    second_state_id: str | None
    second_posterior: float
    margin: float
    normalized_entropy: float
    no_call: bool
    no_call_reasons: tuple[str, ...]
    no_call_rule: NoCallRule
    log_evidence: float | None = None
    source: str = "posterior_evidence_report"
    observation: Any = None
    case_metadata: Mapping[str, Any] = field(default_factory=dict)
    indeterminate: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FSRS10Report:
    """Finite-state reportability score on a prespecified 0-10 scale."""

    score: int
    pmax_points: int
    margin_points: int
    entropy_points: int
    reportable_ge7: bool
    diagnostic_gate_pass: bool | None = None
    reportable_ge7_diagnostic_gate: bool | None = None
    diagnostic_reasons: tuple[str, ...] = ()
    rule_id: str = FSRS10_RULE_ID
    threshold: int = 7


def posterior_margin(posterior: Mapping[str, float]) -> tuple[float, str | None, float, str | None]:
    """Return pmax, MAP state, second posterior, and second state."""

    if not posterior:
        return 0.0, None, 0.0, None
    ranked = sorted(
        ((str(state_id), float(probability)) for state_id, probability in posterior.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_state, top_value = ranked[0]
    if len(ranked) == 1:
        return top_value, top_state, 0.0, None
    second_state, second_value = ranked[1]
    return top_value, top_state, second_value, second_state


def normalized_entropy(posterior: Mapping[str, float]) -> float:
    """Compute entropy scaled to [0, 1] for K candidate states."""

    probabilities = [max(float(value), 0.0) for value in posterior.values()]
    k = len(probabilities)
    if k <= 1:
        return 0.0
    total = sum(probabilities)
    if total <= 0.0 or not math.isfinite(total):
        return 1.0
    normalized = [value / total for value in probabilities]
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0)
    return float(entropy / math.log(k))


def apply_no_call_rule(
    *,
    margin: float,
    entropy: float,
    indeterminate: bool = False,
    warnings: Sequence[str] = (),
    rule: NoCallRule = DEFAULT_NO_CALL_RULE,
) -> tuple[bool, tuple[str, ...]]:
    """Apply a prespecified no-call rule to posterior evidence metrics."""

    reasons: list[str] = []
    if float(margin) < float(rule.margin_threshold):
        reasons.append("low_margin")
    if float(entropy) > float(rule.entropy_threshold):
        reasons.append("high_entropy")
    if bool(rule.include_indeterminate) and bool(indeterminate):
        reasons.append("indeterminate_laplace")
    if any(str(warning).strip() for warning in warnings) and bool(rule.include_indeterminate):
        if any(_is_numerical_warning(str(warning)) for warning in warnings):
            reasons.append("numerical_warning")
    return bool(reasons), tuple(dict.fromkeys(reasons))


def compute_fsrs10(
    *,
    pmax: float,
    margin: float,
    normalized_entropy: float,
    valid_laplace: bool | None = None,
    indeterminate: bool = False,
    warnings: Sequence[str] = (),
    threshold: int = 7,
) -> FSRS10Report:
    """Compute the prespecified FSRS10 posterior-geometry score.

    Point allocation:
    - pmax: +1 each for >=0.50, >=0.60, >=0.70
    - margin: +1 each for >=0.05, >=0.10, >=0.20, >=0.30
    - normalized entropy: +1 each for <=0.90, <=0.80, <=0.70

    The default reporting threshold is FSRS10 >= 7. Numerical diagnostics are
    returned as a separate gate so downstream analyses can report
    geometry-only and diagnostic-gated denominators explicitly.
    """

    pmax_value = float(pmax)
    margin_value = float(margin)
    entropy_value = float(normalized_entropy)
    pmax_points = sum(1 for cutoff in (0.50, 0.60, 0.70) if pmax_value >= cutoff)
    margin_points = sum(1 for cutoff in (0.05, 0.10, 0.20, 0.30) if margin_value >= cutoff)
    entropy_points = sum(1 for cutoff in (0.90, 0.80, 0.70) if entropy_value <= cutoff)
    score = int(pmax_points + margin_points + entropy_points)

    diagnostic_reasons: list[str] = []
    if valid_laplace is False:
        diagnostic_reasons.append("invalid_laplace")
    if bool(indeterminate):
        diagnostic_reasons.append("indeterminate_laplace")
    if any(_is_numerical_warning(str(warning)) for warning in warnings):
        diagnostic_reasons.append("numerical_warning")

    diagnostic_gate_pass = None
    if valid_laplace is not None or bool(indeterminate) or any(str(w).strip() for w in warnings):
        diagnostic_gate_pass = not diagnostic_reasons

    reportable = score >= int(threshold)
    gated_reportable = None if diagnostic_gate_pass is None else bool(reportable and diagnostic_gate_pass)
    return FSRS10Report(
        score=score,
        pmax_points=int(pmax_points),
        margin_points=int(margin_points),
        entropy_points=int(entropy_points),
        reportable_ge7=bool(reportable),
        diagnostic_gate_pass=diagnostic_gate_pass,
        reportable_ge7_diagnostic_gate=gated_reportable,
        diagnostic_reasons=tuple(dict.fromkeys(diagnostic_reasons)),
        threshold=int(threshold),
    )


def compute_report_fsrs10(
    report: PosteriorEvidenceReport,
    *,
    threshold: int = 7,
) -> FSRS10Report:
    """Compute FSRS10 from a posterior evidence report."""

    valid_flags = [row.valid_laplace for row in report.state_rows if row.valid_laplace is not None]
    valid_laplace = all(valid_flags) if valid_flags else None
    warnings = tuple(report.warnings) + tuple(
        warning
        for row in report.state_rows
        for warning in row.warnings
    )
    return compute_fsrs10(
        pmax=report.pmax,
        margin=report.margin,
        normalized_entropy=report.normalized_entropy,
        valid_laplace=valid_laplace,
        indeterminate=report.indeterminate,
        warnings=warnings,
        threshold=threshold,
    )


def summarize_reportability_score(
    report: PosteriorEvidenceReport,
    *,
    threshold: int = 7,
) -> FSRS10Report:
    """Return the public reportability-score summary for a posterior report.

    This is a readability alias for :func:`compute_report_fsrs10`. It keeps the
    released API aligned with manuscript wording while preserving the explicit
    FSRS10 calculation function for reproducible numerical checks.
    """

    return compute_report_fsrs10(report, threshold=threshold)


def summarize_posterior_evidence(
    result: Any,
    *,
    rule: NoCallRule = DEFAULT_NO_CALL_RULE,
    case_metadata: Mapping[str, Any] | None = None,
    source: str | None = None,
) -> PosteriorEvidenceReport:
    """Convert a package posterior result into a stable reporting object."""

    posterior = _posterior_from_result(result)
    pmax, inferred_map, second_probability, second_state = posterior_margin(posterior)
    map_state_id = getattr(result, "map_state_id", None) or inferred_map
    entropy = normalized_entropy(posterior)
    margin = float(pmax - second_probability)
    warnings = tuple(str(item) for item in getattr(result, "warnings", ()))
    indeterminate = bool(getattr(result, "indeterminate", False))
    no_call, no_call_reasons = apply_no_call_rule(
        margin=margin,
        entropy=entropy,
        indeterminate=indeterminate,
        warnings=warnings,
        rule=rule,
    )
    state_rows = tuple(
        _state_evidence_row(state_result, map_state_id=str(map_state_id) if map_state_id is not None else None)
        for state_result in getattr(result, "states", ())
    )
    return PosteriorEvidenceReport(
        posterior=posterior,
        state_rows=state_rows,
        map_state_id=str(map_state_id) if map_state_id is not None else None,
        pmax=float(pmax),
        second_state_id=second_state,
        second_posterior=float(second_probability),
        margin=margin,
        normalized_entropy=float(entropy),
        no_call=bool(no_call),
        no_call_reasons=no_call_reasons,
        no_call_rule=rule,
        log_evidence=_optional_float(getattr(result, "log_evidence", None)),
        source=str(source or getattr(result, "source", "posterior_evidence_report")),
        observation=_observation_from_result(result),
        case_metadata={} if case_metadata is None else dict(case_metadata),
        indeterminate=indeterminate,
        warnings=warnings,
    )


def posterior_to_long_rows(report: PosteriorEvidenceReport) -> list[dict[str, Any]]:
    """Return one row per state with shared posterior evidence fields."""

    rows: list[dict[str, Any]] = []
    for state in report.state_rows:
        row = _shared_report_fields(report)
        row.update(
            {
                "state_id": state.state_id,
                "state_label": state.label,
                "prior": state.prior,
                "posterior_probability": state.posterior_probability,
                "posterior_percent": state.posterior_probability * 100.0,
                "is_map": state.is_map,
                "log_likelihood": state.log_likelihood,
                "log_prior": state.log_prior,
                "log_weight": state.log_weight,
                "eta_hat": state.eta_hat,
                "pred_at_eta_hat": state.pred_at_eta_hat,
                "psi_hat": state.psi_hat,
                "hessian": state.hessian,
                "hessian_logdet": state.hessian_logdet,
                "hessian_min_eigenvalue": state.hessian_min_eigenvalue,
                "valid_laplace": state.valid_laplace,
                "state_indeterminate": state.indeterminate,
                "state_warnings": ";".join(state.warnings),
            }
        )
        rows.append(row)
    return rows


def posterior_to_wide_row(report: PosteriorEvidenceReport) -> dict[str, Any]:
    """Return one row per case/profile with posterior probabilities as columns."""

    row = _shared_report_fields(report)
    row.update(
        {
            "posterior_vector": ";".join(
                f"{state_id}={probability:.12g}" for state_id, probability in report.posterior.items()
            ),
            "state_count": len(report.posterior),
        }
    )
    for state in report.state_rows:
        row[f"posterior_{state.state_id}"] = state.posterior_probability
        row[f"prior_{state.state_id}"] = state.prior
        row[f"label_{state.state_id}"] = state.label
    return row


def build_case_card(
    report: PosteriorEvidenceReport,
    *,
    wording_policy: str = "model_conditioned",
) -> dict[str, Any]:
    """Return compact manuscript/report fields without clinical diagnostic wording."""

    if wording_policy != "model_conditioned":
        raise NotImplementedError("Only wording_policy='model_conditioned' is currently implemented.")
    map_label = _map_label(report)
    if report.no_call:
        interpretation = (
            "Under the specified model and prespecified candidate states, the observations do not "
            "provide sufficient posterior separation for a single-state report."
        )
    else:
        interpretation = (
            "Under the specified model and prespecified candidate states, posterior evidence favors "
            f"{map_label}."
        )
    boundary = (
        "This is model-conditioned support among supplied states; it is not a clinical diagnosis, "
        "adherence proof, genotype test, dosing recommendation, or clinical decision-support output."
    )
    return {
        **_shared_report_fields(report),
        "map_label": map_label,
        "posterior_vector": "; ".join(
            f"{_label_for_state(report, state_id)}: {probability:.3f}"
            for state_id, probability in report.posterior.items()
        ),
        "interpretation": interpretation,
        "boundary": boundary,
        "warnings": ";".join(report.warnings),
    }


def threshold_sensitivity_panel(
    reports: Sequence[PosteriorEvidenceReport],
    rules: Sequence[NoCallRule],
) -> list[dict[str, Any]]:
    """Reclassify existing reports under several no-call rules."""

    rows: list[dict[str, Any]] = []
    denominator = len(reports)
    for rule in rules:
        no_call_count = 0
        reason_counts: dict[str, int] = {}
        for report in reports:
            no_call, reasons = apply_no_call_rule(
                margin=report.margin,
                entropy=report.normalized_entropy,
                indeterminate=report.indeterminate,
                warnings=report.warnings,
                rule=rule,
            )
            if no_call:
                no_call_count += 1
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rows.append(
            {
                "rule_id": rule.rule_id,
                "margin_threshold": rule.margin_threshold,
                "entropy_threshold": rule.entropy_threshold,
                "include_indeterminate": rule.include_indeterminate,
                "denominator": denominator,
                "called": denominator - no_call_count,
                "no_call": no_call_count,
                "no_call_fraction": no_call_count / denominator if denominator else math.nan,
                "reason_counts": dict(sorted(reason_counts.items())),
            }
        )
    return rows


def reweight_prior_sensitivity(
    report: PosteriorEvidenceReport,
    prior_grid: Mapping[str, Mapping[str, float]],
    *,
    rule: NoCallRule = DEFAULT_NO_CALL_RULE,
) -> list[PosteriorEvidenceReport]:
    """Recompute posterior reports from stored state log-likelihoods and new priors."""

    state_log_likelihoods = {
        row.state_id: row.log_likelihood for row in report.state_rows if row.log_likelihood is not None
    }
    if set(state_log_likelihoods) != set(report.posterior):
        raise ValueError("All state log-likelihoods are required for prior reweighting.")

    reports: list[PosteriorEvidenceReport] = []
    for prior_id, prior in prior_grid.items():
        log_weights: dict[str, float] = {}
        for state_id, log_likelihood in state_log_likelihoods.items():
            prior_value = float(prior.get(state_id, 0.0))
            log_weights[state_id] = (
                math.log(prior_value) + float(log_likelihood)
                if math.isfinite(prior_value) and prior_value > 0.0 and log_likelihood is not None
                else -math.inf
            )
        evidence = logsumexp(log_weights.values())
        posterior = {
            state_id: (
                math.exp(log_weight - evidence)
                if math.isfinite(log_weight) and math.isfinite(evidence)
                else 0.0
            )
            for state_id, log_weight in log_weights.items()
        }
        state_rows = tuple(
            _replace_state_prior_and_posterior(
                row,
                prior=float(prior.get(row.state_id, 0.0)),
                posterior_probability=float(posterior[row.state_id]),
                log_weight=log_weights[row.state_id],
            )
            for row in report.state_rows
        )
        reports.append(
            _report_from_rows(
                state_rows,
                rule=rule,
                log_evidence=evidence,
                observation=report.observation,
                source=f"{report.source}:prior_reweighted",
                case_metadata={**dict(report.case_metadata), "prior_id": str(prior_id)},
                indeterminate=report.indeterminate,
                warnings=report.warnings,
            )
        )
    return reports


def build_provenance_manifest(
    *,
    analysis_id: str,
    package_version: str,
    model_id: str | None = None,
    state_order: Sequence[str] = (),
    prior_id: str | None = None,
    no_call_rule: NoCallRule = DEFAULT_NO_CALL_RULE,
    source_paths: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a small reproducibility manifest for posterior evidence reports."""

    return {
        "analysis_id": str(analysis_id),
        "package_version": str(package_version),
        "model_id": model_id,
        "state_order": [str(item) for item in state_order],
        "prior_id": prior_id,
        "no_call_rule": {
            "rule_id": no_call_rule.rule_id,
            "margin_threshold": no_call_rule.margin_threshold,
            "entropy_threshold": no_call_rule.entropy_threshold,
            "include_indeterminate": no_call_rule.include_indeterminate,
        },
        "source_paths": [str(path) for path in source_paths],
        "metadata": {} if metadata is None else dict(metadata),
    }


def _posterior_from_result(result: Any) -> dict[str, float]:
    posterior = getattr(result, "posterior", None)
    if isinstance(posterior, Mapping):
        return {str(state_id): float(probability) for state_id, probability in posterior.items()}
    state_results = getattr(result, "states", ())
    return {
        str(state_result.state.state_id): float(state_result.posterior_probability)
        for state_result in state_results
    }


def _state_evidence_row(state_result: Any, *, map_state_id: str | None) -> StateEvidenceRow:
    state = state_result.state
    diagnostics = state_result.diagnostics
    state_id = str(state.state_id)
    warnings = tuple(str(item) for item in getattr(diagnostics, "warnings", ()))
    return StateEvidenceRow(
        state_id=state_id,
        label=str(getattr(state, "display_label", state_id)),
        prior=float(getattr(state, "prior", math.nan)),
        posterior_probability=float(state_result.posterior_probability),
        is_map=bool(map_state_id is not None and state_id == map_state_id),
        log_likelihood=_optional_float(getattr(diagnostics, "laplace_log_likelihood", None)),
        log_prior=_optional_float(getattr(diagnostics, "log_prior", None)),
        log_weight=_optional_float(getattr(diagnostics, "log_weight", None)),
        eta_hat=getattr(diagnostics, "eta_hat", None),
        pred_at_eta_hat=getattr(diagnostics, "pred_at_eta_hat", None),
        psi_hat=_optional_float(getattr(diagnostics, "psi_hat", None)),
        hessian=_optional_float(getattr(diagnostics, "hessian", None)),
        hessian_logdet=_optional_float(getattr(diagnostics, "hessian_logdet", None)),
        hessian_min_eigenvalue=_optional_float(getattr(diagnostics, "hessian_min_eigenvalue", None)),
        valid_laplace=_optional_bool(getattr(diagnostics, "valid_laplace", None)),
        indeterminate=_optional_bool(getattr(diagnostics, "indeterminate", None)),
        warnings=warnings,
    )


def _report_from_rows(
    state_rows: Sequence[StateEvidenceRow],
    *,
    rule: NoCallRule,
    log_evidence: float | None,
    observation: Any,
    source: str,
    case_metadata: Mapping[str, Any],
    indeterminate: bool,
    warnings: Sequence[str],
) -> PosteriorEvidenceReport:
    posterior = {row.state_id: row.posterior_probability for row in state_rows}
    pmax, map_state_id, second_probability, second_state = posterior_margin(posterior)
    entropy = normalized_entropy(posterior)
    margin = float(pmax - second_probability)
    no_call, no_call_reasons = apply_no_call_rule(
        margin=margin,
        entropy=entropy,
        indeterminate=indeterminate,
        warnings=warnings,
        rule=rule,
    )
    rows = tuple(
        StateEvidenceRow(
            **{
                **row.__dict__,
                "is_map": bool(map_state_id is not None and row.state_id == map_state_id),
            }
        )
        for row in state_rows
    )
    return PosteriorEvidenceReport(
        posterior=posterior,
        state_rows=rows,
        map_state_id=map_state_id,
        pmax=float(pmax),
        second_state_id=second_state,
        second_posterior=float(second_probability),
        margin=margin,
        normalized_entropy=float(entropy),
        no_call=bool(no_call),
        no_call_reasons=no_call_reasons,
        no_call_rule=rule,
        log_evidence=log_evidence,
        source=source,
        observation=observation,
        case_metadata=dict(case_metadata),
        indeterminate=indeterminate,
        warnings=tuple(str(item) for item in warnings),
    )


def _replace_state_prior_and_posterior(
    row: StateEvidenceRow,
    *,
    prior: float,
    posterior_probability: float,
    log_weight: float,
) -> StateEvidenceRow:
    return StateEvidenceRow(
        state_id=row.state_id,
        label=row.label,
        prior=float(prior),
        posterior_probability=float(posterior_probability),
        is_map=row.is_map,
        log_likelihood=row.log_likelihood,
        log_prior=math.log(prior) if prior > 0.0 and math.isfinite(prior) else -math.inf,
        log_weight=float(log_weight),
        eta_hat=row.eta_hat,
        pred_at_eta_hat=row.pred_at_eta_hat,
        psi_hat=row.psi_hat,
        hessian=row.hessian,
        hessian_logdet=row.hessian_logdet,
        hessian_min_eigenvalue=row.hessian_min_eigenvalue,
        valid_laplace=row.valid_laplace,
        indeterminate=row.indeterminate,
        warnings=row.warnings,
    )


def _shared_report_fields(report: PosteriorEvidenceReport) -> dict[str, Any]:
    fsrs10 = compute_report_fsrs10(report)
    row: dict[str, Any] = {
        "source": report.source,
        "observation": report.observation,
        "map_state_id": report.map_state_id,
        "pmax": report.pmax,
        "second_state_id": report.second_state_id,
        "second_posterior": report.second_posterior,
        "margin": report.margin,
        "normalized_entropy": report.normalized_entropy,
        "no_call": report.no_call,
        "no_call_reasons": ";".join(report.no_call_reasons),
        "no_call_rule_id": report.no_call_rule.rule_id,
        "log_evidence": report.log_evidence,
        "indeterminate": report.indeterminate,
        "warning_count": len(report.warnings),
        "fsrs10_rule_id": fsrs10.rule_id,
        "fsrs10": fsrs10.score,
        "fsrs10_pmax_points": fsrs10.pmax_points,
        "fsrs10_margin_points": fsrs10.margin_points,
        "fsrs10_entropy_points": fsrs10.entropy_points,
        "fsrs10_reportable_ge7": fsrs10.reportable_ge7,
        "fsrs10_diagnostic_gate_pass": fsrs10.diagnostic_gate_pass,
        "fsrs10_reportable_ge7_diagnostic_gate": fsrs10.reportable_ge7_diagnostic_gate,
        "fsrs10_diagnostic_reasons": ";".join(fsrs10.diagnostic_reasons),
    }
    for key, value in report.case_metadata.items():
        row[f"case_{key}"] = value
    return row


def _observation_from_result(result: Any) -> Any:
    if hasattr(result, "observations"):
        return tuple(float(value) for value in getattr(result, "observations"))
    if hasattr(result, "observation"):
        return float(getattr(result, "observation"))
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _is_numerical_warning(warning: str) -> bool:
    tokens = (
        "hessian",
        "optimizer",
        "boundary",
        "clamped",
        "laplace",
        "nonfinite",
        "nonpositive",
    )
    lowered = warning.lower()
    return any(token in lowered for token in tokens)


def _label_for_state(report: PosteriorEvidenceReport, state_id: str) -> str:
    for row in report.state_rows:
        if row.state_id == state_id:
            return row.label
    return str(state_id)


def _map_label(report: PosteriorEvidenceReport) -> str:
    if report.map_state_id is None:
        return "no MAP state"
    return _label_for_state(report, report.map_state_id)
