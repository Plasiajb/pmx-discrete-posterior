"""Project-authored executed numerical definitions; portable source projection."""
from __future__ import annotations
from dataclasses import asdict
import math
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from adapter import public_package

POSTERIOR_TOL = 1e-7
LOG_EVIDENCE_TOL = 1e-6

def validate_config(config):
    omega = np.asarray(config.omega_covariance, dtype=float)
    if omega.shape != (2, 2) or not np.all(np.isfinite(omega)):
        raise ValueError("Omega must be a finite 2x2 matrix")
    if not np.allclose(omega, omega.T, rtol=0, atol=1e-14):
        raise ValueError("Omega must be symmetric; no projection is permitted")
    chol = np.linalg.cholesky(omega)
    for name in ("additive_sd", "proportional_sd"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be nonnegative finite")
    if config.additive_sd == 0 and config.proportional_sd == 0:
        raise ValueError("Degenerate zero residual model is outside this protocol")
    if config.prior_policy != "structural_zero" or not math.isfinite(config.positive_floor) or config.positive_floor <= 0:
        raise ValueError("Invalid prior policy or residual floor")
    if not math.isfinite(config.eta_lower) or not math.isfinite(config.eta_upper) or config.eta_lower >= config.eta_upper:
        raise ValueError("Invalid optimizer bounds")
    return omega, chol


def observation_loglik(predictions, observations, config):
    pred, y = np.asarray(predictions, dtype=float), np.asarray(observations, dtype=float)
    if pred.ndim != 2 or y.ndim != 1 or len(y) == 0 or pred.shape[1] != len(y):
        raise ValueError("Prediction and observation dimensions must match")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(y)) or np.any(pred < 0):
        raise FloatingPointError("Nonfinite/negative prediction or nonfinite DV")
    floor = max(float(config.positive_floor), 1e-300)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        sd = np.maximum(np.hypot(max(config.additive_sd, floor), config.proportional_sd * pred), floor)
        values = np.sum(-0.5 * math.log(2 * math.pi) - np.log(sd)
                        - 0.5 * ((y[None, :] - pred) / sd) ** 2, axis=1)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Nonfinite GHQ log integrand; no node may be discarded")
    return values


def vector_valid(vector, ids):
    if list(vector) != list(ids):
        return False
    p = np.array(list(vector.values()), dtype=float)
    return bool(np.all(np.isfinite(p)) and np.all(p >= 0) and np.all(p <= 1)
                and abs(math.fsum(p) - 1.0) <= 1e-10)


def prior_ghq(case, nodes, *, chunk_size=4096):
    """Integrate likelihood under N(0,Omega), never accepting a mode/Hessian."""
    if isinstance(nodes, bool) or int(nodes) != nodes or nodes < 2 or chunk_size < 1:
        raise ValueError("At least two integer nodes and a positive chunk size required")
    model, cfg = case.model(), case.error_config.public_config()
    _, chol = validate_config(cfg)
    x, w = hermgauss(int(nodes))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise FloatingPointError("Invalid quadrature nodes/weights")
    xx, yy = np.meshgrid(x, x, indexing="ij")
    eta = math.sqrt(2.0) * np.column_stack((xx.ravel(), yy.ravel())) @ chol.T
    logw = (np.log(w)[:, None] + np.log(w)[None, :]).ravel() - math.log(math.pi)
    log_evidence = {}
    for state_id in case.state_ids:
        terms = []
        for start in range(0, len(eta), chunk_size):
            stop = min(start + chunk_size, len(eta))
            pred = model.batch_eta(state_id, eta[start:stop])
            ll = observation_loglik(pred, case.dv_ng_ml, cfg)
            terms.append(logsumexp(logw[start:stop] + ll))
        value = float(logsumexp(terms))
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite evidence for state {state_id}")
        log_evidence[state_id] = value
    weights = np.array([log_evidence[s] + math.log(p) if p > 0 else -np.inf
                        for s, p in zip(case.state_ids, case.priors)])
    posterior = dict(zip(case.state_ids, map(float, np.exp(weights - logsumexp(weights)))))
    if not vector_valid(posterior, case.state_ids):
        raise FloatingPointError("Invalid GHQ full posterior vector")
    logs = np.array(list(log_evidence.values()))
    return {"nodes_per_dimension": nodes, "node_count": nodes ** 2,
            "centering": "N(0,Omega) prior", "valid": True,
            "posterior": posterior, "state_log_evidence": log_evidence,
            "state_log_evidence_pairwise": (logs[:, None] - logs[None, :]).tolist()}


def compare_orders(previous, current):
    ids = list(previous["posterior"])
    if ids != list(current["posterior"]) or ids != list(current["state_log_evidence"]):
        raise ValueError("Quadrature state-order mismatch")
    p = max(abs(previous["posterior"][s] - current["posterior"][s]) for s in ids)
    e = max(abs(previous["state_log_evidence"][s] - current["state_log_evidence"][s]) for s in ids)
    old, new = (np.array(list(r["state_log_evidence"].values())) for r in (previous, current))
    pair = float(np.max(np.abs((new[:, None] - new[None, :]) - (old[:, None] - old[None, :]))))
    return {"from_order": previous["nodes_per_dimension"], "to_order": current["nodes_per_dimension"],
            "max_abs_posterior_change": p, "max_abs_state_log_evidence_change": e,
            "max_abs_pairwise_log_evidence_change": pair,
            "passes": bool(math.isfinite(p) and math.isfinite(e) and p <= POSTERIOR_TOL and e <= LOG_EVIDENCE_TOL)}


def laplace_gate(result, case):
    ids = case.state_ids
    if tuple(s.state.state_id for s in result.states) != ids or not vector_valid(result.posterior, ids):
        return False
    for state in result.states:
        d = state.diagnostics
        expected_prior = case.priors[ids.index(state.state.state_id)]
        if not math.isclose(state.state.prior, expected_prior, rel_tol=0, abs_tol=1e-14):
            return False
        if state.state.prior > 0 and (not d.valid_laplace or d.indeterminate
                or not d.optimizer_success or d.boundary_hit or d.hessian_min_eigenvalue <= 0
                or not all(math.isfinite(x) for x in (*d.eta_hat, d.psi_hat,
                           d.laplace_log_likelihood, d.hessian_logdet, d.hessian_min_eigenvalue))):
            return False
    return True


def laplace_route(case):
    pmx, model, cfg = public_package(), case.model(), case.error_config.public_config()
    omega, _ = validate_config(cfg)
    result = pmx.compute_multieta_posterior(case.dv_ng_ml, model.public_states(), model, cfg)
    diagnostics = {}
    inv, logdet = np.linalg.inv(omega), np.linalg.slogdet(omega)[1]
    for state in result.states:
        d = state.diagnostics
        item = asdict(d)
        # Supplement the public diagnostic summaries without changing its calculation.
        def objective(eta):
            v = np.asarray(eta)
            ll = observation_loglik(np.array([model.predictions(state.state.state_id, v)]), case.dv_ng_ml, cfg)[0]
            return float(-ll + math.log(2 * math.pi) + 0.5 * logdet + 0.5 * v @ inv @ v)
        try:
            h = pmx.finite_difference_hessian(objective, d.eta_hat,
                    step_min=cfg.derivative_step_min, step_scale=cfg.derivative_step_scale)
            item["diagnostic_reconstructed_hessian"] = h.tolist()
            item["diagnostic_reconstructed_eigenvalues"] = np.linalg.eigvalsh(h).tolist()
            item["hessian_reconstruction_status"] = "finite" if np.all(np.isfinite(h)) else "nonfinite"
        except Exception as exc:
            item["hessian_reconstruction_status"] = "failed"
            item["hessian_reconstruction_error"] = f"{type(exc).__name__}: {exc}"
        diagnostics[state.state.state_id] = item
    valid = laplace_gate(result, case)
    return {"status": "valid" if valid else "invalid", "valid_for_comparison": valid,
            "posterior": result.posterior, "state_diagnostics": diagnostics,
            "raw_map_state_id": result.map_state_id,
            "accepted_map_state_id": result.map_state_id if valid else None,
            "mixture_log_evidence": result.log_evidence,
            "raw_indeterminate": result.indeterminate, "warnings": list(result.warnings)}
