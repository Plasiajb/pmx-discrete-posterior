"""Positive anisotropic GHQ: directional planning, then full-axis validation."""
from __future__ import annotations
import math
import numpy as np
from .tensor import integrate, node_count

POLICY = dict(initial_leading_order=9, initial_other_order=3,
              increment=2, maximum_order=31, directional_tolerance=5e-5,
              directional_iterations=8, axes_per_update=2,
              validation_rounds=2, validation_increments=(0, 2, 4),
              posterior_threshold=1e-4, state_logevidence_threshold=1e-3)


def policy_for_dimension(dimension):
    selected = dict(POLICY)
    if dimension == 10:
        selected.update(initial_other_order=2,validation_increments=(0,1,2))
    return selected


class TensorState:
    def __init__(self, state_id, joint, geometry, budget, records, trace, checkpoint):
        self.state_id, self.joint, self.geo = state_id, joint, geometry
        self.budget, self.records, self.trace, self.checkpoint = budget, records, trace, checkpoint
        self.memo = {}

    def evaluate(self, orders):
        orders = tuple(orders)
        if orders in self.memo:
            self.budget.counts['same_attempt_grid_cache_hits'] += 1
            return self.memo[orders]
        nodes = node_count(orders)
        if nodes > self.budget.limit-self.budget.counts['state_points']:
            raise RuntimeError('Remaining point budget cannot complete the next tensor grid')
        value = integrate(self.joint, self.geo['mode'], self.geo['hessian'], orders,
                          max_nodes=self.budget.limit, deadline=self.budget.deadline,
                          batch_size=8192, rotation=self.geo['rotation'])
        value.update(state_event_id=self.state_id, evaluation_id=len(self.records))
        self.records.append(value)
        self.memo[orders] = value
        self.checkpoint()
        return value

    def plan(self, initial, round_index):
        orders = tuple(initial)
        for iteration in range(POLICY['directional_iterations']):
            base = self.evaluate(orders)
            deltas = []
            for axis in range(len(orders)):
                probe = list(orders)
                probe[axis] += POLICY['increment']
                if probe[axis] > POLICY['maximum_order']:
                    raise RuntimeError('Directional GHQ order cap reached before validation')
                value = self.evaluate(probe)
                deltas.append(abs(value['log_evidence']-base['log_evidence']))
            ranked = sorted(range(len(orders)), key=lambda j: (-deltas[j], j))
            selected = [j for j in ranked if deltas[j] > POLICY['directional_tolerance']]
            selected = selected[:POLICY['axes_per_update']]
            self.trace.append(dict(state_event_id=self.state_id, round=round_index,
                                   iteration=iteration, orders=list(orders),
                                   directional_logevidence_changes=deltas,
                                   selected_axes=selected,
                                   directional_only_not_acceptance=True))
            self.checkpoint()
            if not selected:
                return orders
            updated = list(orders)
            for axis in selected:
                updated[axis] += POLICY['increment']
            orders = tuple(updated)
        return orders


def refinement(current, previous):
    if ([s['state_event_id'] for s in current['states']] !=
            [s['state_event_id'] for s in previous['states']]):
        raise ValueError('Candidate order changed between validation grids')
    dz = float(np.max(np.abs(np.array([s['log_evidence'] for s in current['states']])-
                            [s['log_evidence'] for s in previous['states']])))
    dp = float(np.max(np.abs(np.array(current['posterior'])-previous['posterior'])))
    return dict(max_state_logevidence_change=dz, max_posterior_change=dp,
                diagnostic_stable=dz <= POLICY['state_logevidence_threshold'] and
                                  dp <= POLICY['posterior_threshold'])


def qualifies(stages):
    if len(stages) != 3 or not all(r.get('complete') for r in stages):
        return False
    try:
        if [r['step'] for r in stages] != [0, 1, 2]:
            return False
        dimension = len(stages[0]['states'][0]['orders'])
        increment = policy_for_dimension(dimension)['validation_increments'][1]
        for i in (1, 2):
            for old, new in zip(stages[i-1]['states'], stages[i]['states'], strict=True):
                if (old['state_event_id'] != new['state_event_id'] or
                        not np.all(np.array(new['orders'])-old['orders'] == increment)):
                    return False
            if not refinement(stages[i], stages[i-1])['diagnostic_stable']:
                return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


def full_axis_validation(evaluators, orders, posterior, priors, stages, checkpoint):
    policy = policy_for_dimension(len(orders[0]))
    for step, increment in enumerate(policy['validation_increments']):
        row = dict(step=step, complete=False, states=[])
        stages.append(row)
        for evaluator, base in zip(evaluators, orders, strict=True):
            refined = tuple(n+increment for n in base)
            if max(refined) > POLICY['maximum_order']:
                raise RuntimeError('Full-axis validation order exceeds the prescribed cap')
            row['states'].append(evaluator.evaluate(refined))
            checkpoint()
        row['posterior'] = posterior([v['log_evidence'] for v in row['states']], priors).tolist()
        row['complete'] = True
        if step:
            row.update(refinement(row, stages[-2]))
        checkpoint()
    return qualifies(stages)
