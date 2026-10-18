"""Portable functions extracted from the frozen October 2026 experiment."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
from .tac_model import covariance, predict


def run(case):
    import pmx_discrete_posterior as pmx
    import pmx_discrete_posterior.multieta as multieta
    omega = covariance(case)
    inverse, logdet = np.linalg.inv(omega), np.linalg.slogdet(omega)[1]
    dim = case.neta
    simplex = np.zeros((dim+1, dim))
    simplex[1:] = np.diag(.15*np.sqrt(np.diag(omega)))
    config = pmx.MultiEtaLaplaceConfig(
        omega_covariance=omega, additive_sd=0., proportional_sd=case.residual_sd,
        residual_mode='proportional_only', eta_lower=-math.inf, eta_upper=math.inf,
        optimizer_method='Nelder-Mead', optimizer_maxiter=6000, optimizer_start=np.zeros(dim),
        optimizer_options=dict(maxfev=10000, adaptive=True,
                               xatol=1e-13, fatol=1e-11, initial_simplex=simplex))

    class Model:
        def predictions(self, state_id, eta):
            return predict(case, state_id, eta)

    states = [pmx.DiscreteState(s, p) for s, p in zip(case.state_ids, case.priors)]
    result = pmx.compute_multieta_posterior(case.observations, states, Model(), config)
    checks = []
    for state, diag in result.diagnostics.items():
        def objective(eta):
            eta = np.asarray(eta)
            pred = predict(case, state, eta)
            sd = case.residual_sd*pred
            return float(.5*np.sum(np.log(2*math.pi*sd**2)+((case.observations-pred)/sd)**2)
                         + .5*(dim*math.log(2*math.pi)+logdet+eta@inverse@eta))
        point = np.asarray(diag.eta_hat)
        steps = np.eye(dim)*1e-4
        gradient = np.array([(objective(point+s)-objective(point-s))/2e-4 for s in steps])
        fine = multieta.finite_difference_hessian(objective, point, step_min=5e-5, step_scale=5e-5)
        eigmin = float(np.linalg.eigvalsh(fine).min())
        sign, fine_logdet = np.linalg.slogdet(fine)
        change = abs(float(fine_logdet)-diag.hessian_logdet)
        passed = bool(diag.valid_laplace and np.abs(gradient).max() <= 1e-3 and
                      sign > 0 and eigmin > 0 and change <= 1e-3)
        checks.append(dict(state_id=state, passed=passed, gradient_inf=float(np.abs(gradient).max()),
                           fine_hessian_min=eigmin, logdet_step_delta=change))
    probabilities = np.array(list(result.posterior.values()))
    complete = (set(result.posterior) == set(case.state_ids) == set(result.diagnostics)
                and len(checks) == len(case.states))
    eligible = bool(complete and not result.indeterminate and all(c['passed'] for c in checks) and
                    np.isfinite(probabilities).all() and np.all(probabilities >= 0) and
                    abs(probabilities.sum()-1) <= 1e-10)
    return dict(status='PASS_LAPLACE' if eligible else 'REJECTED_LAPLACE_DIAGNOSTICS', eligible=eligible,
                posterior=result.posterior if eligible else None, diagnostic_posterior=result.posterior,
                diagnostics={s: asdict(d) for s, d in result.diagnostics.items()}, independent_checks=checks,
                package_version=pmx.__version__, package_source=str(Path(pmx.__file__).resolve().parent), optimizer='Nelder-Mead',
                full_latent_dimension=dim, reduced_dimension_used=False, log_evidence=result.log_evidence,
                precision_policy='unified_source_ruv_20261006',
                optimizer_tolerances=dict(xatol=1e-13, fatol=1e-11))

