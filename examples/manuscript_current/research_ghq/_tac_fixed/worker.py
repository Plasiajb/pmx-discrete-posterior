"""Fresh full-model high-dimensional GHQ with bounded anisotropic planning."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.special import logsumexp
from .fast_model import Model, LOG2PI
from .geometry import geometry, eigen_rotation
from .kernel import POLICY, TensorState, full_axis_validation, qualifies, policy_for_dimension
from .tensor import checked_orders
from .atomic_io import write_text_atomic

CASE_POINTS = 550_000_000
PASS = 'PASS_ADAPTIVE_FULL_AXIS_REFINEMENT_DIAGNOSTIC'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    write_text_atomic(path,json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


class BudgetExceeded(RuntimeError):
    pass


class CounterBudget:
    def __init__(self, deadline, limit):
        self.deadline, self.limit = deadline, limit
        self.counts = Counter(state_points=0, geometry_points=0, quadrature_points=0)

    def claim(self, size, stage):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Case wall deadline')
        if self.counts['state_points']+size > self.limit:
            raise BudgetExceeded('Case state-point budget')
        self.counts['state_points'] += size
        self.counts[stage+'_points'] += size


def posterior(logs, priors):
    logs, priors = np.asarray(logs,dtype=float), np.asarray(priors,dtype=float)
    if (logs.shape != priors.shape or not np.isfinite(logs).all() or not np.isfinite(priors).all()
            or np.any(priors < 0) or not np.isclose(priors.sum(),1,rtol=0,atol=1e-12)):
        raise ValueError('Invalid evidence/prior vector')
    weights = np.full_like(logs,-np.inf)
    positive = priors > 0
    weights[positive] = logs[positive]+np.log(priors[positive])
    return np.exp(weights-logsumexp(weights))


def case_attempt(case, output, seconds=1780, point_limit=CASE_POINTS, backend='numpy', integration_plan=None, run_identity=None):
    output = Path(output)
    started = time.monotonic()
    budget = CounterBudget(started+seconds,point_limit)
    result = dict(numeric_id=case.numeric_id,status='RUNNING',modes={},counters={},
                  validation_stages=[],validation_history=[],adaptive_trace=[],integration_records=[],
                  formal_qualified=False,absolute_error_certified=False,reused_old_evidence=False,
                  fresh_integration=True,backend=backend,policy=POLICY,stage='initialization')

    def checkpoint():
        result['counters'] = dict(budget.counts)
        result['elapsed_seconds'] = time.monotonic()-started
        dump(output/'checkpoint.json',result)

    try:
        if backend != 'numpy':
            raise ValueError('Unvalidated backend; only numpy is currently frozen')
        model = Model(case)
        policy = policy_for_dimension(model.dimension)
        planned_orders = None
        if integration_plan is not None:
            from .planned_grid import validate_plan
            planned_orders = validate_plan(integration_plan, case, POLICY['maximum_order'])
            if POLICY['validation_rounds'] != 1:
                raise ValueError('Planned-grid attempt requires one fixed three-level validation')
            if integration_plan['full_validation_nodes'] + 100_000 > point_limit:
                raise ValueError('Insufficient preallocated budget for complete planned validation')
            result.update(integration_plan=integration_plan,
                          planning_mode='predeclared_orders_fresh_full_axis_validation',
                          run_identity=run_identity)
        result.update(full_dimension=case.neta,quadrature_dimension=model.dimension,
                      transform=model.transform,covariance=model.covariance,
                      geometry_source='fresh own fixed two-start BFGS',
                      all_axis_refinement_required=True,positive_weights_only=True,policy=policy)
        stage = 'geometry'

        def joint(w, state):
            w = np.atleast_2d(w)
            budget.claim(len(w),stage)
            return model.log_likelihood(w@model.prior_cholesky.T,state)-.5*(model.dimension*LOG2PI+(w*w).sum(axis=1))

        for state_id in case.state_ids:
            records = []
            result['modes'][state_id] = dict(starts=records,qualified=False)
            checkpoint()
            geo = geometry(lambda w:-joint(w,state_id),model.dimension,budget.deadline,records)
            result['modes'][state_id] = geo
            if geo['qualified']:
                rotation = eigen_rotation(geo['hessian'])
                geo['rotation'] = rotation.tolist()
                geo['rotation_diagnostics'] = dict(method='full posterior Hessian eigenbasis',
                    dropped_axes=0,integrated_axes=model.dimension,comparator_results_used=False)
            checkpoint()
        if not all(g['qualified'] for g in result['modes'].values()):
            result['status'] = 'NOT_INTEGRATED_GEOMETRY_REJECTED'
        else:
            if planned_orders is not None:
                from .planned_grid import check_fresh_geometry
                check_fresh_geometry(result['modes'], integration_plan)
                result['fresh_geometry_matches_planning_basis'] = True
                if integration_plan['full_validation_nodes'] > budget.limit - budget.counts['state_points']:
                    raise BudgetExceeded('Fresh geometry left insufficient budget for all planned grids')
            stage = result['stage'] = 'quadrature'
            evaluators = [TensorState(s,lambda w,s=s:joint(w,s),result['modes'][s],budget,
                          result['integration_records'],result['adaptive_trace'],checkpoint) for s in case.state_ids]
            starts = [(policy['initial_leading_order'],)+(policy['initial_other_order'],)*(model.dimension-1)]*len(evaluators)
            result['status'] = 'NOT_CONVERGED'
            for round_index in range(POLICY['validation_rounds']):
                orders = planned_orders if planned_orders is not None else [
                    e.plan(initial,round_index) for e,initial in zip(evaluators,starts,strict=True)]
                stages = []
                result['validation_stages'] = stages
                result['validation_history'].append(dict(round=round_index,stages=stages))
                if full_axis_validation(evaluators,orders,posterior,case.priors,stages,checkpoint):
                    result['status'] = PASS
                    result['accepted_round'] = round_index
                    break
                starts = [tuple(n+policy['validation_increments'][1] for n in o) for o in orders]
    except BaseException as exc:
        from .planned_grid import PlanGeometryMismatch
        result['status'] = ('PLAN_GEOMETRY_MISMATCH' if isinstance(exc,PlanGeometryMismatch) else
                            'TIMEOUT' if isinstance(exc,TimeoutError) else
                            'NODE_BUDGET_EXHAUSTED' if isinstance(exc,BudgetExceeded) or
                            'point budget' in str(exc) else
                            'IO_FAILURE' if isinstance(exc,OSError) else 'NUMERICAL_FAILURE')
        result['error'],result['traceback'] = repr(exc),traceback.format_exc()
    try:
        checkpoint()
    except OSError as exc:
        result['status_before_final_checkpoint_io'] = result['status']
        result['status'] = 'IO_FAILURE'
        result['checkpoint_write_error'] = repr(exc)
        result['checkpoint_write_traceback'] = traceback.format_exc()
        if 'error' not in result:
            result['error'],result['traceback'] = repr(exc),result['checkpoint_write_traceback']
    dump(output/'result.json',result)
    return result


def validate_result(result, case, integration_plan=None):
    """Recompute acceptance from saved arrays, not the worker's status string."""
    try:
        if integration_plan is not None:
            from .planned_grid import validate_saved_plan
            if not validate_saved_plan(result, case, integration_plan, POLICY['maximum_order']):
                return False
        elif 'integration_plan' in result:
            return False
        if (result.get('status') != PASS or result.get('numeric_id') != case.numeric_id
                or result.get('fresh_integration') is not True or result.get('reused_old_evidence') is not False
                or result.get('formal_qualified') is not False or result.get('absolute_error_certified') is not False
                or result['full_dimension'] != case.neta or result['quadrature_dimension'] != case.neta-2
                or clean(result['policy']) != clean(policy_for_dimension(case.neta-2))
                or result.get('backend') != 'numpy'
                or set(result['modes']) != set(case.state_ids)):
            return False
        dimension = result['quadrature_dimension']
        accepted = result['accepted_round']
        history = result['validation_history']
        if (type(accepted) is not int or not 0 <= accepted < POLICY['validation_rounds']
                or len(history) != accepted+1
                or [r['round'] for r in history] != list(range(accepted+1))
                or result['validation_stages'] != history[-1]['stages']
                or any(len(r['stages']) != 3 for r in history)
                or any(qualifies(r['stages']) for r in history[:-1])):
            return False
        for mode in result['modes'].values():
            rotation = np.array(mode['rotation'])
            curvature = np.array(mode['hessian'])
            if (mode['qualified'] is not True or rotation.shape != (dimension,dimension)
                    or not np.allclose(rotation.T@rotation,np.eye(dimension),rtol=0,atol=1e-12)
                    or mode['rotation_diagnostics']['dropped_axes'] != 0
                    or mode['rotation_diagnostics']['integrated_axes'] != dimension
                    or mode['rotation_diagnostics']['comparator_results_used'] is not False
                    or curvature.shape != (dimension,dimension) or not np.isfinite(curvature).all()
                    or not np.allclose(curvature,curvature.T,atol=1e-12,rtol=0)
                    or np.linalg.eigvalsh(curvature).min() <= 0):
                return False
        records, nodes = result['integration_records'], 0
        for i, row in enumerate(records):
            checked_orders(row['orders'])
            if (row['evaluation_id'] != i or row['state_event_id'] not in case.state_ids
                    or len(row['orders']) != dimension or row['nodes'] != math.prod(row['orders'])
                    or any(type(n) is not int for n in row['orders'])
                    or max(row['orders']) > POLICY['maximum_order']
                    or not math.isfinite(row['log_evidence']) or row['negative_weights'] != 0
                    or row['discarded_nodes'] != 0 or row['cancellation_condition'] != 1):
                return False
            nodes += row['nodes']
        stages = result['validation_stages']
        for row in (s for h in history for s in h['stages']):
            if [s['state_event_id'] for s in row['states']] != list(case.state_ids):
                return False
            for state in row['states']:
                if state != records[state['evaluation_id']]:
                    return False
            values = np.asarray(row['posterior'])
            if not np.isfinite(values).all() or not np.allclose(values,
                    posterior([s['log_evidence'] for s in row['states']],case.priors),atol=1e-12,rtol=0):
                return False
        counts = result['counters']
        return (counts['quadrature_points'] == nodes and counts['state_points'] ==
                counts['geometry_points']+nodes and qualifies(stages))
    except (KeyError,TypeError,ValueError,IndexError,OverflowError):
        return False
