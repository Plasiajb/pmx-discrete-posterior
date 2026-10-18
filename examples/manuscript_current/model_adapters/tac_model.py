"""Portable functions extracted from the frozen October 2026 experiment."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import numpy as np
from scipy.linalg import expm


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def clock_add(a, b):
    return float(Decimal(str(a)) + Decimal(str(b)))


@dataclass
class ResolvedCase:
    numeric_id: str
    parameters: dict
    fixed: dict
    slots: tuple
    shared_last: float
    shared_dose: float
    tau: float
    states: tuple
    omega: np.ndarray
    eta_order: tuple
    residual_sd: float
    times: tuple
    observations: tuple

    @property
    def state_ids(self):
        return tuple(s['state_event_id'] for s in self.states)

    @property
    def priors(self):
        return tuple(s['prior'] for s in self.states)

    @property
    def neta(self):
        return len(self.omega)

    @property
    def nocc(self):
        return len(self.slots) + 1

    def to_dict(self):
        return dict(schema='tac_resolved_v1', numeric_id=self.numeric_id,
                    parameters=self.parameters, fixed=self.fixed, slots=list(self.slots),
                    shared_last=self.shared_last, shared_dose=self.shared_dose, tau=self.tau,
                    states=list(self.states), omega=self.omega.tolist(), eta_order=list(self.eta_order),
                    residual_sd=self.residual_sd, times=list(self.times), observations=list(self.observations))

    @classmethod
    def from_dict(cls, data):
        allowed = {'schema', 'numeric_id', 'parameters', 'fixed', 'slots', 'shared_last',
                   'shared_dose', 'tau', 'states', 'omega', 'eta_order', 'residual_sd', 'times', 'observations'}
        if set(data) != allowed or data['schema'] != 'tac_resolved_v1':
            raise ValueError('Unrecognized inference fields or schema; truth data prohibited')
        c = cls(data['numeric_id'], dict(data['parameters']), dict(data['fixed']), tuple(data['slots']),
                float(data['shared_last']), float(data['shared_dose']), float(data['tau']),
                tuple(data['states']), np.array(data['omega'], dtype=float), tuple(data['eta_order']),
                float(data['residual_sd']), tuple(data['times']), tuple(data['observations']))
        c.validate()
        return c

    def validate(self):
        if self.neta != 4 + 2*self.nocc or len(self.eta_order) != self.neta:
            raise ValueError('Full random-effect dimension required')
        order = ['Clearance IIV', 'Central-volume IIV', 'Peripheral-volume IIV', 'Absorption-rate IIV']
        order += [f'{name} IOV@occ_{t:g}' for t in (self.shared_last, *self.slots)
                  for name in ('Clearance', 'Central-volume')]
        if tuple(order) != self.eta_order:
            raise ValueError('Unsupported ETA ordering')
        if self.omega.shape != (self.neta, self.neta) or not np.isfinite(self.omega).all():
            raise ValueError('Invalid covariance')
        allowed = np.eye(self.neta, dtype=bool)
        allowed[1:4, 1:4] = True
        if np.any(self.omega[~allowed] != 0):
            raise ValueError('Unsupported off-block covariance; native target must not drop entries')
        sd = np.sqrt(np.diag(self.omega))
        corr = self.omega / np.outer(sd, sd)
        if not np.allclose(corr, corr.T, rtol=1e-10, atol=1e-12):
            raise ValueError('Asymmetric covariance')
        np.linalg.cholesky(self.omega)
        if len(set(self.state_ids)) != len(self.state_ids) or not self.states:
            raise ValueError('Duplicate/empty state identities')
        if not all(math.isfinite(p) and p >= 0 for p in self.priors) or not math.isclose(sum(self.priors), 1, abs_tol=1e-12):
            raise ValueError('Invalid bound priors')
        if not self.times or len(self.times) != len(self.observations):
            raise ValueError('Missing observations')
        if not np.isfinite([*self.times, *self.observations, self.residual_sd]).all() or self.residual_sd <= 0:
            raise ValueError('Invalid numeric observations/residual SD')
        if not 0 <= self.parameters['lag'] < self.tau or self.shared_dose <= 0:
            raise ValueError('Unsupported periodic lag/dose')
        if tuple(sorted(set(self.slots))) != self.slots or self.shared_last >= min(self.slots):
            raise ValueError('Invalid fixed schedule')
        for state in self.states:
            if set(state) != {'state_event_id', 'manuscript_id', 'machine_bitstring', 'prior', 'events'}:
                raise ValueError('Unexpected candidate field')
            for event in state['events']:
                if set(event) != {'nominal_slot_h', 'time_h', 'amount_mg'}:
                    raise ValueError('Unexpected event field')
                if event['nominal_slot_h'] not in self.slots or event['time_h'] < self.shared_last or event['amount_mg'] <= 0:
                    raise ValueError('Invalid candidate event')
        kinetics(self, np.zeros(self.neta), 1)


def covariance(case):
    return np.array(case.omega, copy=True)


def kinetics(case, eta, occasion):
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (case.neta,) or not np.isfinite(eta).all() or not 1 <= occasion <= case.nocc:
        raise ValueError('Complete finite random effects and valid occasion required')
    p, f = case.parameters, case.fixed
    k = 4 + 2*(occasion-1)
    cl = (p['CL'] * f['theta_CYP3A5']**f['X'] * (1+f['theta_HEM']*(f['HEM']-.33))
          * (p['WT']/f['normalization_weight'])**f['WT_exponent']
          * (1+f['theta_POD']*(min(f['POD'], f['POD_cap'])-22.7)) * math.exp(eta[0]+eta[k]))
    vc = p['V1']*(1+f['theta_Pred']*(f['Pred']-155.5))*math.exp(eta[1]+eta[k+1])
    vp, ka, q = p['V2']*math.exp(eta[2]), p['ka']*math.exp(eta[3]), p['Q']
    if not all(math.isfinite(v) and v > 0 for v in (cl, vc, vp, ka, q)):
        raise ValueError('Nonpositive kinetics; no clipping')
    matrix = np.array([[-ka, 0., 0.], [ka, -(cl+q)/vc, q/vp], [0., q/vc, -q/vp]])
    return matrix, dict(CL=cl, V2=vc, V3=vp, KA=ka)


def _transition(matrix, spectral):
    if not spectral:
        return lambda dt: expm(matrix*dt)
    values, vectors = np.linalg.eig(matrix)
    if np.iscomplexobj(values) or np.linalg.cond(vectors) > 1e7:
        return lambda dt: expm(matrix*dt)
    inverse = np.linalg.inv(vectors)
    return lambda dt: (vectors * np.exp(values*dt)) @ inverse


def periodic_initial(case, eta, *, spectral=True):
    matrix, _ = kinetics(case, eta, 1)
    transition = _transition(matrix, spectral)
    state = np.linalg.solve(np.eye(3)-transition(case.tau),
                            transition(case.tau-case.parameters['lag']) @ np.array([case.shared_dose, 0., 0.]))
    if not np.isfinite(state).all() or np.any(state < 0):
        raise ValueError('Invalid periodic initial state; no repair')
    return state


def _predict(case, events, eta, times, spectral=True):
    times = tuple(float(t) for t in times)
    if not times or not np.isfinite(times).all() or min(times) < case.shared_last:
        raise ValueError('Observation precedes conditional initialization')
    lag = case.parameters['lag']
    schedule = [(t, 10, 'switch', i+1) for i, t in enumerate(case.slots)]
    schedule += [(clock_add(case.shared_last, lag), 20, 'input', case.shared_dose)]
    schedule += [(clock_add(e['time_h'], lag), 20, 'input', e['amount_mg']) for e in events]
    schedule += [(t, 0, 'sample', i) for i, t in enumerate(times)]
    matrices = [kinetics(case, eta, i+1) for i in range(case.nocc)]
    transitions = [_transition(m[0], spectral) for m in matrices]
    state = np.linalg.solve(np.eye(3)-transitions[0](case.tau),
                            transitions[0](case.tau-lag) @ np.array([case.shared_dose, 0., 0.]))
    result, last, occ = np.empty(len(times)), case.shared_last, 0
    for t, _, kind, value in sorted(schedule):
        if t > max(times):
            break
        if t < last:
            raise ValueError('Input precedes periodic initial state')
        if t != last:
            state = transitions[occ](t-last) @ state
        last = t
        if kind == 'switch':
            occ = value
        elif kind == 'input':
            state[0] += value
        else:
            result[value] = 1000.*state[1]/matrices[occ][1]['V2']
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('Invalid concentration; no floor')
    return result


def predict(case, state_id, eta, times=None):
    state = case.states[case.state_ids.index(state_id)]
    return _predict(case, state['events'], eta, case.times if times is None else times)


def predict_expm(case, state_id, eta, times=None):
    state = case.states[case.state_ids.index(state_id)]
    return _predict(case, state['events'], eta, case.times if times is None else times, False)

