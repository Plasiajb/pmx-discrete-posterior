"""Batched, exact linear-compartment TAC likelihood for resolved numeric inputs.

The Gaussian pushforward integrates two null directions analytically. It does
not remove any IIV/IOV component or replace the nonlinear likelihood by a normal.
"""
from __future__ import annotations
from decimal import Decimal
import math
import numpy as np

LOG2PI = math.log(2 * math.pi)


def covariance_transform(case):
    dimension = 2 * case.nocc + 2
    transform = np.zeros((dimension, case.neta))
    for j in range(case.nocc):
        transform[2*j, 0] = transform[2*j, 4+2*j] = 1
        transform[2*j+1, 1] = transform[2*j+1, 5+2*j] = 1
    transform[-2, 2] = transform[-1, 3] = 1
    return transform, transform @ case.omega @ transform.T


def rates(case, value, occasion):
    p, f = case.parameters, case.fixed
    cl = (p['CL'] * f['theta_CYP3A5']**f['X'] * (1+f['theta_HEM']*(f['HEM']-.33))
          * (p['WT']/f['normalization_weight'])**f['WT_exponent']
          * (1+f['theta_POD']*(min(f['POD'], f['POD_cap'])-22.7)) * np.exp(value[:, 2*occasion]))
    vc = p['V1']*(1+f['theta_Pred']*(f['Pred']-155.5))*np.exp(value[:, 2*occasion+1])
    vp, ka = p['V2']*np.exp(value[:, -2]), p['ka']*np.exp(value[:, -1])
    if not np.isfinite(np.column_stack((cl, vc, vp, ka))).all() or np.any(np.column_stack((cl, vc, vp, ka)) <= 0):
        raise ValueError('Invalid physical parameters; no clipping')
    k10, k12, k21 = cl/vc, p['Q']/vc, p['Q']/vp
    delta = np.sqrt((k10+k12-k21)**2 + 4*k12*k21)
    alpha = .5*(k10+k12+k21+delta)
    beta = k10*k21/alpha
    return k10, k12, k21, ka, alpha, beta, delta, vc


def divided_exponential(rate, ka, duration):
    difference = np.abs(rate-ka)
    factor = np.full_like(difference, duration)
    np.divide(-np.expm1(-difference*duration), difference, out=factor, where=difference != 0)
    return np.exp(-np.minimum(rate, ka)*duration)*factor


def transition(kinetics, duration):
    if duration < 0:
        raise ValueError('Negative propagation time')
    k10, k12, k21, ka, alpha, beta, delta, _ = kinetics
    ea, eb, eg = np.exp(-alpha*duration), np.exp(-beta*duration), np.exp(-ka*duration)
    slow_c, slow_p = (k21-beta)/delta, (k10+k12-beta)/delta
    ja, jb = divided_exponential(alpha, ka, duration), divided_exponential(beta, ka, duration)
    matrix = np.zeros((len(ka), 3, 3))
    matrix[:, 0, 0] = eg
    matrix[:, 1, 0] = ka*(slow_p*ja+slow_c*jb)
    matrix[:, 2, 0] = ka*(k12/delta)*(jb-ja)
    matrix[:, 1, 1] = ea+(eb-ea)*slow_c
    matrix[:, 1, 2] = (eb-ea)*k21/delta
    matrix[:, 2, 1] = (eb-ea)*k12/delta
    matrix[:, 2, 2] = ea+(eb-ea)*slow_p
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError('Invalid propagator; no clipping')
    return matrix


def clock_add(a, b):
    return float(Decimal(str(a))+Decimal(str(b)))


class Model:
    def __init__(self, case):
        self.case = case
        self.transform, self.covariance = covariance_transform(case)
        self.dimension = len(self.covariance)
        self.prior_cholesky = np.linalg.cholesky(self.covariance)
        self.prior_inverse = np.linalg.inv(self.covariance)
        self.prior_logdet = float(2*np.log(np.diag(self.prior_cholesky)).sum())
        self.schedules = {}
        for state in case.states:
            schedule = [(t, 10, 'switch', i+1) for i, t in enumerate(case.slots)]
            schedule += [(clock_add(case.shared_last, case.parameters['lag']), 20, 'input', case.shared_dose)]
            schedule += [(clock_add(e['time_h'], case.parameters['lag']), 20, 'input', e['amount_mg']) for e in state['events']]
            schedule += [(t, 0, 'sample', i) for i, t in enumerate(case.times)]
            self.schedules[state['state_event_id']] = tuple(e for e in sorted(schedule) if e[0] <= max(case.times))

    def points(self, value):
        value = np.atleast_2d(np.asarray(value, dtype=float))
        if value.ndim != 2 or value.shape[1] != self.dimension or not np.isfinite(value).all():
            raise ValueError('Invalid full pushforward vector')
        return value

    def predict(self, value, state_id):
        value = self.points(value)
        case = self.case
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            kinetics = [rates(case, value, i) for i in range(case.nocc)]
            period = transition(kinetics[0], case.tau)
            input_phase = case.shared_dose*transition(kinetics[0], case.tau-case.parameters['lag'])[:, :, 0]
            amount = np.linalg.solve(np.eye(3)[None]-period, input_phase[..., None])[..., 0]
            if not np.isfinite(amount).all() or np.any(amount < 0):
                raise ValueError('Invalid conditional SS state')
            predictions = np.empty((len(value), len(case.times)))
            last, occasion = case.shared_last, 0
            for t, _, kind, argument in self.schedules[state_id]:
                if t != last:
                    amount = np.einsum('nij,nj->ni', transition(kinetics[occasion], t-last), amount)
                last = t
                if kind == 'switch':
                    occasion = argument
                elif kind == 'input':
                    amount[:, 0] += argument
                else:
                    predictions[:, argument] = 1000*amount[:, 1]/kinetics[occasion][-1]
            if not np.isfinite(predictions).all() or np.any(predictions <= 0):
                raise ValueError('Invalid concentration; no residual or prediction floor')
            return predictions

    def log_prior(self, value):
        value = self.points(value)
        return -.5*(self.dimension*LOG2PI+self.prior_logdet+np.einsum('ni,ij,nj->n', value, self.prior_inverse, value))

    def log_likelihood(self, value, state_id):
        prediction = self.predict(value, state_id)
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            sd = self.case.residual_sd*prediction
            result = (-.5*LOG2PI-np.log(sd)-.5*((np.array(self.case.observations)-prediction)/sd)**2).sum(axis=1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite log likelihood')
        return result

    def log_joint(self, value, state_id):
        return self.log_prior(value)+self.log_likelihood(value, state_id)
