"""Validate exploratory order plans without importing old integral estimates."""
from __future__ import annotations

import math
import numpy as np


class PlanGeometryMismatch(ValueError):
    pass


def validate_plan(plan, case, maximum_order):
    dimension = case.neta - 2
    if (plan.get('schema') != 'tac_ghq_fixed_grid_plan_v1'
            or plan.get('numeric_id') != case.numeric_id
            or plan.get('state_ids') != list(case.state_ids)
            or plan.get('increments') != [0, 2, 4]
            or plan.get('old_integrals_imported') is not False
            or plan.get('geometry_tolerances') != dict(mode_atol=1e-8, mode_rtol=1e-8,
                hessian_relative_frobenius=1e-8, absolute_axis_overlap_atol=1e-8)
            or set(plan.get('geometry_reference', {})) != set(case.state_ids)
            or set(plan.get('orders_by_state', {})) != set(case.state_ids)):
        raise ValueError('Order plan identity, state mapping or scope mismatch')
    orders = []
    for state_id in case.state_ids:
        row = plan['orders_by_state'][state_id]
        if (len(row) != dimension or any(type(n) is not int or n < 1 for n in row)
                or max(row) + 4 > maximum_order):
            raise ValueError('Invalid planned tensor orders')
        reference = plan['geometry_reference'][state_id]
        for key, shape in (('mode', (dimension,)), ('hessian', (dimension, dimension)),
                           ('rotation', (dimension, dimension))):
            value = np.asarray(reference[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid planning geometry reference')
        hessian = np.asarray(reference['hessian'], dtype=float)
        rotation = np.asarray(reference['rotation'], dtype=float)
        if (not np.allclose(hessian, hessian.T, rtol=0, atol=1e-12)
                or np.linalg.eigvalsh(hessian).min() <= 0
                or not np.allclose(rotation.T @ rotation, np.eye(dimension), rtol=0, atol=1e-12)):
            raise ValueError('Planning curvature must be SPD and rotation orthogonal')
        orders.append(tuple(row))
    nodes = sum(math.prod(n + inc for n in row) for inc in (0, 2, 4) for row in orders)
    if plan.get('full_validation_nodes') != nodes:
        raise ValueError('Declared grid cost does not match all-state full-axis grids')
    return orders


def check_fresh_geometry(modes, plan):
    """Only compare after independent optimization; never initialize from reference."""
    for state_id, reference in plan['geometry_reference'].items():
        fresh = modes[state_id]
        if not np.allclose(fresh['mode'], reference['mode'], rtol=1e-8, atol=1e-8):
            raise PlanGeometryMismatch('Fresh mode differs from planning basis: ' + state_id)
        current, prior = np.asarray(fresh['hessian']), np.asarray(reference['hessian'])
        if np.linalg.norm(current - prior) > 1e-8 * np.linalg.norm(prior):
            raise PlanGeometryMismatch('Fresh curvature differs from planning basis: ' + state_id)
        axes = []
        for hessian, rotation in ((current, fresh['rotation']), (prior, reference['rotation'])):
            scale = np.linalg.solve(np.linalg.cholesky(hessian).T, np.asarray(rotation))
            axes.append(scale / np.linalg.norm(scale, axis=0))
        overlap = np.abs(np.sum(axes[0] * axes[1], axis=0))
        if not np.allclose(overlap, 1., rtol=0, atol=1e-8):
            raise PlanGeometryMismatch('Fresh axes differ from planning basis: ' + state_id)


def validate_saved_plan(result, case, plan, maximum_order):
    try:
        orders = validate_plan(plan, case, maximum_order)
        if (result.get('integration_plan') != plan or result.get('accepted_round') != 0
                or result.get('planning_mode') != 'predeclared_orders_fresh_full_axis_validation'
                or result.get('fresh_geometry_matches_planning_basis') is not True
                or result.get('adaptive_trace') != []
                or len(result.get('validation_history', [])) != 1):
            return False
        check_fresh_geometry(result['modes'], plan)
        stages = result['validation_stages']
        if len(stages) != 3 or len(result['integration_records']) != 3 * len(case.state_ids):
            return False
        for step, stage in enumerate(stages):
            if [s['state_event_id'] for s in stage['states']] != list(case.state_ids):
                return False
            for state, base in zip(stage['states'], orders, strict=True):
                if state['orders'] != [n + 2 * step for n in base]:
                    return False
        return result['counters']['quadrature_points'] == plan['full_validation_nodes']
    except (KeyError, TypeError, ValueError, IndexError):
        return False
