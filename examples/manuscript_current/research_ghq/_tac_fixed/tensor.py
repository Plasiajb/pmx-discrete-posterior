"""Streaming positive-weight, anisotropic Hermite-normal tensor quadrature."""
import math
import operator
import time
import numpy as np
from scipy.special import logsumexp, roots_hermitenorm


def checked_orders(orders):
    try:
        orders = tuple(operator.index(n) for n in orders)
    except TypeError as exc:
        raise ValueError('Integer Hermite orders required') from exc
    if not orders or not all(1 <= n <= 64 for n in orders) or len(orders) > 12:
        raise ValueError('Unsupported tensor orders/dimension')
    return orders


def node_count(orders):
    return math.prod(checked_orders(orders))


def order_ladder(dimension):
    if dimension == 6:
        return tuple((n,)+(w,)*5 for n,w in ((11,5),(15,5),(15,7),(19,7),(19,9)))
    if dimension == 8:
        return tuple((n,)+(w,)*7 for n,w in ((11,5),(15,5),(15,7),(19,7),(19,9)))
    if dimension == 10:
        return tuple((n,)+(w,)*9 for n,w in ((11,3),(15,3),(15,4),(19,4),(19,5)))
    raise ValueError('Only frozen 6D, 8D and 10D positive rules are defined')


def integrate(log_joint, mode, hessian, orders, max_nodes=550_000_000,
              deadline=None, batch_size=8192, rotation=None):
    orders = checked_orders(orders)
    dimension, total_nodes = len(orders), node_count(orders)
    if total_nodes > max_nodes or not 1 <= batch_size <= 8192:
        raise ValueError('Tensor node or batch budget exceeded')
    mode, curvature = np.asarray(mode, dtype=float), np.asarray(hessian, dtype=float)
    if (mode.shape != (dimension,) or curvature.shape != (dimension, dimension)
            or not np.isfinite(mode).all() or not np.isfinite(curvature).all()
            or not np.allclose(curvature, curvature.T, rtol=0, atol=1e-12)):
        raise ValueError('Invalid full geometry')
    chol = np.linalg.cholesky(curvature)
    scale = np.linalg.solve(chol.T, np.eye(dimension))
    log_jacobian = -float(np.log(np.diag(chol)).sum())
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if (rotation.shape != (dimension, dimension) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation, np.eye(dimension), rtol=0, atol=1e-12)):
            raise ValueError('Nonorthogonal proposal; no projection repair')
        scale = scale@rotation
    rules = {n:roots_hermitenorm(n) for n in set(orders)}
    if any(not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0) for x,w in rules.values()):
        raise ValueError('Nonpositive/nonfinite Hermite rule weights')
    strides = np.cumprod((1,)+orders[:-1], dtype=np.int64)
    total, evaluated = -math.inf, 0
    started = time.monotonic()
    for offset in range(0, total_nodes, batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Positive tensor deadline')
        indices = np.arange(offset, min(offset+batch_size, total_nodes), dtype=np.int64)
        digits = (indices[:,None]//strides) % np.array(orders)
        z = np.column_stack([rules[n][0][digits[:,i]] for i,n in enumerate(orders)])
        weights = sum(np.log(rules[n][1][digits[:,i]]) for i,n in enumerate(orders))
        density = np.asarray(log_joint(mode+z@scale.T))
        if density.shape != (len(z),) or not np.isfinite(density).all():
            raise ValueError('Nonfinite tensor node density; no discarded or repaired nodes')
        terms = weights+density+.5*(z*z).sum(axis=1)+log_jacobian
        if not np.isfinite(terms).all():
            raise ValueError('Nonfinite transformed node terms')
        total = float(np.logaddexp(total, logsumexp(terms)))
        evaluated += len(z)
    if evaluated != total_nodes or not math.isfinite(total):
        raise ValueError('Incomplete or nonfinite positive integral')
    return dict(orders=list(orders), dimension=dimension, nodes=evaluated, log_evidence=total,
                positive_log_mass=total, negative_log_mass=None, negative_weights=0,
                cancellation_condition=1., log_jacobian=log_jacobian,
                elapsed_seconds=time.monotonic()-started, discarded_nodes=0, cache_hits=0,
                algorithm='positive anisotropic Hermite-normal tensor in own posterior eigenbasis')
