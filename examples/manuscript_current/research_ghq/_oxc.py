"""Exact formal OXC GHQ functions; shared evaluation-model adapter."""
from __future__ import annotations
import math
import mpmath as mp
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp, roots_hermitenorm
from model_adapters import oxc_model as a
from model_adapters import _common as c

GHQ_ORDERS = (15, 31, 61, 121)


SEARCH = (-12., 8.)


def normalize(logs, priors):
    c.require(set(logs) == set(priors), 'Full candidate identity required before normalization')
    values = np.array([logs[s]+math.log(priors[s]) for s in logs])
    return dict(zip(logs, map(float, np.exp(values-logsumexp(values)))))


def independent_center(model, state):
    """Fixed search and root seeds use observations, never a comparator fit."""
    inp = model.input
    variance, sd = inp['variability']['joint_omega'][0][0], inp['residual']['sd']
    y = np.array([o['concentration_mg_L'] for o in inp['observations']])
    def objective_float(eta):
        error = (y-np.array(model.values(state, eta)))/sd
        return float(error@error/2 + eta**2/(2*variance))
    grid = np.linspace(*SEARCH, 161)
    values = [objective_float(float(x)) for x in grid]
    seeds = [0.]
    for j in range(1, len(grid)-1):
        if values[j] <= values[j-1] and values[j] <= values[j+1]:
            fit = minimize_scalar(objective_float, bounds=(grid[j-1], grid[j+1]), method='bounded',
                                  options={'xatol': 1e-13, 'maxiter': 200})
            seeds.append(float(fit.x))
    for j, observation in enumerate(y):
        fn = lambda x: float(model.values(state, x)[j])-observation
        if fn(SEARCH[0])*fn(SEARCH[1]) < 0:
            seeds.append(brentq(fn, *SEARCH, xtol=1e-14))
    objective = lambda eta: model.objective_mp(state, eta)
    minima, failures = [], []
    for seed in seeds:
        try:
            center = mp.findroot(lambda x: mp.diff(objective, x),
                                 (mp.mpf(seed)-mp.mpf('1e-7'), mp.mpf(seed)+mp.mpf('1e-7')),
                                 solver='secant', tol=mp.mpf('1e-45'), maxsteps=80)
            if not SEARCH[0] < center < SEARCH[1]:
                continue
            curvature = mp.diff(objective, center, 2)
            if curvature <= 0 or abs(mp.diff(objective, center))/mp.sqrt(curvature) > mp.mpf('1e-20'):
                continue
            if all(abs(center-old[0]) > mp.mpf('1e-15') for old in minima):
                minima.append((center, curvature))
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            failures.append(type(exc).__name__)
    c.require(len(minima) == 1, 'Independent GHQ requires exactly one identified minimum; no comparator fallback')
    center, curvature = minima[0]
    gap = min(objective(mp.mpf(SEARCH[0])), objective(mp.mpf(SEARCH[1])))-objective(center)
    c.require(gap >= 40, 'Independent center search boundary too close')
    return center, curvature, dict(seed_count=len(seeds), seed_failures=failures, minima_found=len(minima),
                                   search=list(SEARCH), boundary_objective_gap=float(gap), global_mode_certified=False)


def ghq(inp):
    model = a.Model(inp)
    priors = {p['state_event_id']: p['probability'] for p in inp['priors']}
    logs = {n: {} for n in GHQ_ORDERS}
    centers = {}
    with mp.workdps(60):
        for state in priors:
            center, curvature, audit = independent_center(model, state)
            scale = 1/mp.sqrt(curvature)
            centers[state] = dict(eta=mp.nstr(center, 50), curvature=mp.nstr(curvature, 50),
                                  scale=mp.nstr(scale, 50), audit=audit)
            for order in GHQ_ORDERS:
                nodes, weights = roots_hermitenorm(order)
                terms = [float(-model.objective_mp(state, center+scale*mp.mpf(float(x)))
                               + mp.mpf(float(x))**2/2 + mp.log(float(w))) for x, w in zip(nodes, weights)]
                logs[order][state] = float(logsumexp(terms)+mp.log(scale))
    levels = [dict(order=n, log_marginals=logs[n], posterior=normalize(logs[n], priors)) for n in GHQ_ORDERS]
    differences = [dict(from_order=x['order'], to_order=y['order'],
                        max_probability_delta=max(abs(x['posterior'][s]-y['posterior'][s]) for s in priors),
                        max_log_marginal_delta=max(abs(x['log_marginals'][s]-y['log_marginals'][s]) for s in priors))
                   for x, y in zip(levels, levels[1:])]
    valid = all(d['max_probability_delta'] <= 1e-6 and d['max_log_marginal_delta'] <= 1e-5 for d in differences[-2:])
    return dict(route='independent_1D_adaptive_GHQ', qualified=valid, status='PASS' if valid else 'NOT_CONVERGED',
                posterior=levels[-1]['posterior'] if valid else None, raw_posterior=levels[-1]['posterior'],
                levels=levels, convergence=differences, independent_centers=centers, comparator_modes_used=False,
                prior_application='full_vector_log_prior_added_once_then_one_logsumexp', orders=list(GHQ_ORDERS))
