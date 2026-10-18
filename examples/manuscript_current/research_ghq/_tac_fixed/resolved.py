"""Self-contained, inference-only reader for the existing TAC resolved DTO."""
from dataclasses import dataclass
import json
import math
import numpy as np


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

    @classmethod
    def from_dict(cls, data):
        fields = set(cls.__dataclass_fields__)
        if set(data) != fields | {'schema'} or data['schema'] != 'tac_resolved_v1':
            raise ValueError('Unrecognized inference fields/schema; truth data prohibited')
        args = {k: data[k] for k in fields}
        args['omega'] = np.array(args['omega'], dtype=float)
        for key in ('slots', 'states', 'eta_order', 'times', 'observations'):
            args[key] = tuple(args[key])
        result = cls(**args)
        result.validate()
        return result

    def validate(self):
        if self.neta != 4 + 2*self.nocc or self.neta not in (8, 10, 12):
            raise ValueError('Full 4 IIV + 2 IOV per occasion required')
        order = ['Clearance IIV', 'Central-volume IIV', 'Peripheral-volume IIV', 'Absorption-rate IIV']
        order += [f'{name} IOV@occ_{t:g}' for t in (self.shared_last, *self.slots)
                  for name in ('Clearance', 'Central-volume')]
        if tuple(order) != self.eta_order:
            raise ValueError('Unsupported ETA ordering')
        if self.omega.shape != (self.neta, self.neta) or not np.isfinite(self.omega).all():
            raise ValueError('Invalid covariance')
        mask = np.eye(self.neta, dtype=bool)
        mask[1:4, 1:4] = True
        if np.any(self.omega[~mask] != 0) or np.any(np.diag(self.omega) <= 0):
            raise ValueError('Unsupported covariance or removed effects')
        corr = self.omega / np.sqrt(np.outer(np.diag(self.omega), np.diag(self.omega)))
        if not np.allclose(corr, corr.T, rtol=1e-10, atol=1e-12):
            raise ValueError('Asymmetric covariance')
        np.linalg.cholesky(self.omega)
        if not self.states or len(set(self.state_ids)) != len(self.states):
            raise ValueError('Empty/duplicate states')
        if not all(math.isfinite(p) and p >= 0 for p in self.priors) or not math.isclose(sum(self.priors), 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Invalid priors; no renormalization')
        if not self.times or len(self.times) != len(self.observations):
            raise ValueError('Missing observations')
        if not np.isfinite([*self.times, *self.observations, self.residual_sd, self.shared_last, self.shared_dose, self.tau, *self.slots]).all():
            raise ValueError('Nonfinite numeric input')
        if self.residual_sd <= 0 or min(self.times) < self.shared_last:
            raise ValueError('Invalid residual or observation time')
        if tuple(sorted(set(self.slots))) != self.slots or self.shared_last >= min(self.slots):
            raise ValueError('Invalid occasion schedule')
        if set(self.parameters) != {'CL', 'V1', 'Q', 'V2', 'ka', 'lag', 'WT'}:
            raise ValueError('Unsupported model parameter fields')
        fixed = {'X', 'HEM', 'POD', 'Pred', 'theta_CYP3A5', 'theta_HEM', 'theta_POD', 'theta_Pred', 'POD_cap', 'WT_exponent', 'normalization_weight'}
        if set(self.fixed) != fixed or not np.isfinite([*self.parameters.values(), *self.fixed.values()]).all():
            raise ValueError('Unsupported/nonfinite model fields')
        if not 0 <= self.parameters['lag'] < self.tau or self.shared_dose <= 0:
            raise ValueError('Invalid periodic lag/dose')
        if any(v <= 0 for k, v in self.parameters.items() if k != 'lag') or self.fixed['normalization_weight'] <= 0:
            raise ValueError('Nonpositive structural parameter')
        for state in self.states:
            if set(state) != {'state_event_id', 'manuscript_id', 'machine_bitstring', 'prior', 'events'}:
                raise ValueError('Unexpected candidate field')
            for event in state['events']:
                if set(event) != {'nominal_slot_h', 'time_h', 'amount_mg'} or not np.isfinite(list(event.values())).all():
                    raise ValueError('Unexpected/nonfinite event field')
                if event['nominal_slot_h'] not in self.slots or event['time_h'] < self.shared_last or event['amount_mg'] <= 0:
                    raise ValueError('Invalid candidate event')


def read_case(path):
    return ResolvedCase.from_dict(json.loads(path.read_text(encoding='utf-8-sig')))
