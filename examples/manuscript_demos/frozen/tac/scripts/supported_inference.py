"""Supported study reproduction API; raw frozen package calls are not protected."""
from __future__ import annotations

import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def covariance_gate(covariance, eta_dimension):
    import numpy as np
    require(isinstance(eta_dimension,int) and not isinstance(eta_dimension,bool) and eta_dimension > 0,
            "Expected ETA dimension must be a positive integer")
    omega = np.asarray(covariance,dtype=float)
    require(omega.shape == (eta_dimension,eta_dimension), "Covariance/model ETA dimension mismatch")
    require(bool(np.all(np.isfinite(omega))), "Covariance must be finite")
    require(bool(np.allclose(omega,omega.T,rtol=0,atol=1e-14)), "Covariance must be symmetric")
    try:
        np.linalg.cholesky(omega)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance must be positive definite") from exc
    return omega


def inference_entry(route, *, config, eta_dimension, observations=None, states=None,
                    model=None, case=None, nodes=None):
    """All documented study Laplace/GHQ paths enter here before the frozen core.

    eta_dimension is the model contract: one for OXC and two for tacrolimus.
    Generic analytic fixtures declare their own model dimension. No covariance
    clipping, replacement or symmetrization is performed.
    """
    require(route in ("laplace","adaptive_ghq","tacrolimus_laplace","tacrolimus_prior_ghq"),
            "Unsupported study inference route")
    covariance_gate(config.omega_covariance,eta_dimension)
    if route.startswith("tacrolimus_"):
        require(eta_dimension == 2 and case is not None, "Tacrolimus requires the two-ETA study model")
        require(case.error_config.public_config() == config, "Tacrolimus case/config mismatch")
        from routes import laplace_route, prior_ghq
        result = laplace_route(case) if route == "tacrolimus_laplace" else prior_ghq(case,nodes)
        require(result.get("valid_for_comparison",result.get("valid",False)), "Study route diagnostic gate failed")
        return result
    from adapter import public_package
    pmx = public_package()
    if route == "laplace":
        result = pmx.compute_multieta_posterior(observations,states,model,config)
        require(not result.indeterminate and all(
            s.diagnostics.valid_laplace and s.diagnostics.optimizer_success
            and not s.diagnostics.boundary_hit and s.diagnostics.hessian_min_eigenvalue > 0
            and math.isfinite(s.diagnostics.laplace_log_likelihood)
            for s in result.states if s.state.prior > 0), "Laplace diagnostic gate failed")
        return result
    require(eta_dimension == 1, "The executed adaptive GHQ adapter is one-dimensional")
    from oxc_ghq import adaptive_ghq_posterior
    result = adaptive_ghq_posterior(pmx=pmx,observations=observations,states=states,model=model,
        config=config,nodes=nodes,mode_optimizer_method=config.optimizer_method)
    require(all(d["valid_mode"] and d["optimizer_success"] for d in result[2].values()),
            "Adaptive GHQ diagnostic gate failed")
    return result
