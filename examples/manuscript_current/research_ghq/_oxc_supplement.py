"""Portable GHQ02 checks around the unchanged formal OXC numerical functions.

Profile/input authorization belongs to the calling API. No files, saved answers,
or integration caches are read here. Use a fresh single-thread process as for
the base OXC route; mpmath precision contexts are not thread-local.
"""
from __future__ import annotations

from collections import Counter
import math
import sys
import time
from types import FunctionType

import numpy as np
from scipy.special import roots_hermitenorm

from model_adapters import _common as c
from . import _oxc

ORDERS = (121, 241, 361)


def weight_check(nodes, weights, order):
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    c.require(nodes.shape == weights.shape == (order,), 'Quadrature node/weight count')
    c.require(np.all(np.isfinite(nodes)) and np.all(np.diff(nodes) > 0), 'Invalid nodes')
    c.require(np.all(np.isfinite(weights)) and np.all(weights >= sys.float_info.min), 'Zero/subnormal/nonfinite/negative quadrature weight')
    c.require(np.allclose(nodes, -nodes[::-1], rtol=0, atol=1e-12)
              and np.allclose(weights, weights[::-1], rtol=1e-11, atol=0), 'Quadrature symmetry')
    moments = []
    for power, normal in ((0, 1.), (2, 1.), (4, 3.), (6, 15.)):
        value = math.fsum(float(w)*float(x)**power for x, w in zip(nodes, weights))
        target = math.sqrt(2*math.pi)*normal
        c.require(math.isclose(value, target, rel_tol=1e-11, abs_tol=1e-12), 'Gaussian moment failed')
        moments.append(dict(power=power, value=value, target=target, absolute_delta=abs(value-target)))
    return dict(order=order, status='PASS', node_count=order, zero_weights=0, subnormal_weights=0,
                minimum_weight=float(min(weights)), log_minimum_weight=math.log(float(min(weights))),
                maximum_absolute_node=float(max(abs(nodes))), moments=moments,
                nodes_sha256=c.digest(nodes.tolist()), weights_sha256=c.digest(weights.tolist()),
                weights_discarded=0, weights_modified=False)


def bridge_check(result, reference):
    c.require(reference['status'] == 'NOT_CONVERGED' and not reference['qualified'], 'Wrong original GHQ slot')
    c.require(result['independent_centers'] == reference['independent_centers'], 'Independent center bridge changed')
    c.require(result['levels'][0] == reference['levels'][-1], 'Fresh 121-node anchor changed')
    return dict(status='PASS', fresh_center_and_121_anchor_identical=True, archived_result_reused=False)


def checked_logsumexp(values, reducer, receipts):
    terms = np.asarray(values)
    c.require(terms.ndim == 1 and terms.size > 0 and np.all(np.isfinite(terms)),
              'Nonfinite transformed terms BEFORE logsumexp; no nodes may be discarded')
    result = reducer(values)
    c.require(math.isfinite(float(result)), 'Nonfinite logsumexp reduction')
    receipts.append(dict(status='PASS', count=int(terms.size), finite_terms=int(terms.size),
        minimum=float(min(terms)), maximum=float(max(terms)), terms_sha256=c.digest(terms.tolist()),
        terms_discarded=0, values_modified=False))
    return result


def refinements_pass(differences):
    c.require([(d['from_order'], d['to_order']) for d in differences] == [(121, 241), (241, 361)],
              'Exactly both predeclared refinement steps required')
    c.require(all(math.isfinite(d[k]) and d[k] >= 0 for d in differences
                  for k in ('max_probability_delta', 'max_log_marginal_delta')), 'Invalid refinement delta')
    return all(d['max_probability_delta'] <= 1e-6 and d['max_log_marginal_delta'] <= 1e-5 for d in differences)


def numerical_accounting(result, weights, reductions, state_count):
    c.require(result['orders'] == list(ORDERS) and [v['order'] for v in result['levels']] == list(ORDERS),
              'Predetermined orders missing or changed')
    c.require(Counter(r['order'] for r in weights) == Counter({n: state_count for n in ORDERS}),
              'Missing or extra node/weight accounting')
    expected = Counter({n: state_count for n in ORDERS})
    expected[state_count] += len(ORDERS)
    c.require(Counter(r['count'] for r in reductions) == expected, 'Missing or extra transformed-term reductions')
    c.require(result['qualified'] == refinements_pass(result['convergence']), 'Refinement qualification mismatch')


def _calculate(inp, orders, weights, reductions):
    def checked_roots(order):
        c.require(order in orders, 'Undeclared order')
        nodes, node_weights = roots_hermitenorm(order)
        weights.append(weight_check(nodes, node_weights, order))
        return nodes, node_weights

    # New function namespaces retain the exact source code objects without
    # patching _oxc globals, including normalization's logsumexp guard.
    bindings = dict(vars(_oxc))
    bindings.update(GHQ_ORDERS=orders, roots_hermitenorm=checked_roots,
                    logsumexp=lambda values: checked_logsumexp(values, _oxc.logsumexp, reductions))
    bindings['normalize'] = FunctionType(_oxc.normalize.__code__, bindings, 'normalize')
    kernel = FunctionType(_oxc.ghq.__code__, bindings, 'ghq')
    return kernel(inp)


def _finite_levels(result):
    c.require(all(math.isfinite(v) for level in result['levels']
                  for key in ('log_marginals', 'posterior') for v in level[key].values()),
              'Nonfinite GHQ result')


def _fresh_bridge_check(result, anchor, weights, reductions, state_count):
    c.require(anchor['orders'] == [121] and [v['order'] for v in anchor['levels']] == [121],
              'Fresh bridge requires exactly one 121-node anchor')
    c.require(Counter(r['order'] for r in weights) == Counter({121: state_count}),
              'Missing or extra fresh-anchor node/weight accounting')
    expected = Counter({121: state_count})
    expected[state_count] += 1
    c.require(Counter(r['count'] for r in reductions) == expected,
              'Missing or extra fresh-anchor transformed-term reductions')
    _finite_levels(anchor)
    c.require(result['independent_centers'] == anchor['independent_centers'], 'Independent center bridge changed')
    c.require(result['levels'][0] == anchor['levels'][-1], 'Fresh 121-node anchor changed')
    return dict(status='PASS', fresh_center_and_121_anchor_identical=True, archived_result_reused=False,
                reference_kind='fresh_independent_121_anchor', archival_comparison_performed=False,
                original_slot_status_checked=False, numerical_accounting='PASS')


def compare_archival(result, reference):
    """Optional post-calculation comparison only; never changes qualification.

    The caller supplies an archival outcome only after ghq() returns. The
    original source bridge checks its NOT_CONVERGED status and exact centers
    and 121-node level. No archival data can initialize or replace integration.
    """
    c.require(result.get('fresh_invocation') is True and result.get('bridge', {}).get('status') == 'PASS',
              'Complete fresh calculation required before archival comparison')
    c.require(result['input_sha256'] == reference['input_sha256'], 'Reference input mismatch')
    comparison = bridge_check(result, reference)
    return dict(**comparison, comparison_only=True, qualification_changed=False,
                reference_kind='archival_original_NOT_CONVERGED', original_slot_status_checked=True)


def ghq(inp):
    """Run actual 121/241/361 GHQ02 plus a separately calculated 121 anchor.

    Returns the base raw-result shape with SUPPLEMENT_PASS,
    SUPPLEMENT_NOT_CONVERGED, or SUPPLEMENT_REFUSAL. Refusals retain completed
    diagnostics but never a qualified posterior. This function neither pins a
    case/profile nor enforces a wall timeout; those are caller responsibilities.
    """
    started = time.monotonic()
    weights, reductions, anchor_weights, anchor_reductions = [], [], [], []
    result = dict(route='independent_1D_adaptive_GHQ', status='SUPPLEMENT_REFUSAL', qualified=False,
                  posterior=None, raw_posterior=None, levels=[], convergence=[], independent_centers={},
                  orders=list(ORDERS), comparator_modes_used=False,
                  prior_application='full_vector_log_prior_added_once_then_one_logsumexp')
    bridge, anchor, input_digest = dict(status='NOT_RUN'), None, None
    accounting = 'NOT_RUN'
    try:
        input_digest = c.digest(inp)
        result = _calculate(inp, ORDERS, weights, reductions)
        numerical_accounting(result, weights, reductions, len(inp['priors']))
        _finite_levels(result)
        accounting = 'PASS'
        anchor = _calculate(inp, (121,), anchor_weights, anchor_reductions)
        # A single fresh level is a bridge, never a convergence claim.
        anchor.update(status='ANCHOR_ONLY_NOT_CONVERGENCE_TEST', qualified=False, posterior=None)
        bridge = _fresh_bridge_check(result, anchor, anchor_weights, anchor_reductions, len(inp['priors']))
        result['status'] = 'SUPPLEMENT_PASS' if result['qualified'] else 'SUPPLEMENT_NOT_CONVERGED'
    except Exception as exc:
        result.update(status='SUPPLEMENT_REFUSAL', qualified=False, posterior=None, error=repr(exc))
    result.update(bridge=bridge, fresh_anchor=anchor, weight_checks=weights,
                  transformed_term_checks=reductions, anchor_weight_checks=anchor_weights,
                  anchor_transformed_term_checks=anchor_reductions, numerical_accounting=accounting,
                  input_sha256=input_digest, original_status_overridden=False,
                  fresh_invocation=True, cached_result_used=False, reference_answers_read=False,
                  native_launches=0, elapsed_compute_seconds=time.monotonic()-started,
                  formal_qualified=False, absolute_error_certified=False)
    return result
