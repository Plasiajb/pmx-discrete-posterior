"""Portable functions extracted from the frozen October 2026 experiment."""
from __future__ import annotations

from dataclasses import asdict
import math
import mpmath as mp
import numpy as np
from . import oxc_model as a


def laplace(inp, pmx):
    model = a.Model(inp)
    variance = inp['variability']['joint_omega'][0][0]
    config = pmx.MultiEtaLaplaceConfig(
        omega_covariance=[[variance]], additive_sd=inp['residual']['sd'], residual_mode='additive_only',
        eta_lower=-math.inf, eta_upper=math.inf, optimizer_method='Nelder-Mead', optimizer_maxiter=6000,
        optimizer_start=[0.], optimizer_options={'maxfev': 10000, 'adaptive': True, 'xatol': 1e-13,
        'fatol': 1e-11, 'initial_simplex': np.array([[0.], [.15*math.sqrt(variance)]])})
    states = [pmx.DiscreteState(p['state_event_id'], p['probability']) for p in inp['priors']]
    result = pmx.compute_multieta_posterior([o['concentration_mg_L'] for o in inp['observations']], states, model, config)
    checks = []
    with mp.workdps(60):
        for state, d in result.diagnostics.items():
            point = mp.mpf(d.eta_hat[0])
            objective = lambda eta: model.objective_mp(state, eta)
            gradient, curvature = mp.diff(objective, point), mp.diff(objective, point, 2)
            scaled_gradient = float(abs(gradient)/mp.sqrt(abs(curvature))) if curvature else math.inf
            log_delta = float(abs(mp.log(abs(curvature))-d.hessian_logdet)) if curvature else math.inf
            objective_delta = float(abs(objective(point)-d.psi_hat))
            checks.append(dict(state=state, gradient_over_sqrt_curvature=scaled_gradient,
                               independent_curvature=float(curvature), hessian_logdet_delta=log_delta,
                               objective_delta=objective_delta,
                               passed=bool(d.valid_laplace and curvature > 0 and scaled_gradient <= 1e-4
                                           and log_delta <= 1e-3 and objective_delta <= 1e-3)))
    qualified = (not result.indeterminate and set(result.diagnostics) == model.states.keys()
                 and set(result.posterior) == model.states.keys() and all(x['passed'] for x in checks))
    return dict(route='package_0.1.4_public_multieta_Nelder_Mead', qualified=qualified,
                status='PASS' if qualified else 'INVALID_LAPLACE', raw_posterior=result.posterior,
                posterior=result.posterior if qualified else None,
                diagnostics={s: asdict(d) for s, d in result.diagnostics.items()}, independent_checks=checks,
                policy=dict(start=0, maxiter=6000, maxfev=10000, xatol=1e-13, fatol=1e-11,
                            simplex_step=.15*math.sqrt(variance), derivative_step_min=config.derivative_step_min,
                            derivative_step_scale=config.derivative_step_scale))

