"""Portable functions extracted from the frozen October 2026 experiment."""
from __future__ import annotations

import math
import mpmath as mp
from . import _common as c
SD = 2.992
CV = .111
VARIANCE = math.log1p(CV**2)
FIXED = dict(k0=.624, kmax=.233, k50=8.97, gamma=219, normalization_weight=70)
FIELDS = {'schema', 'model', 'initialization', 'variability', 'residual', 'candidate_states', 'priors', 'observations'}


def no_truth(value):
    if isinstance(value, dict):
        c.require(not set(value) & {'truth_metadata', 'actual_events', 'profile_events', 'design_history_id',
                                   'iiv_realizations', 'iov_realizations_by_occasion', 'generated_observation_id'},
                  'Truth/generation metadata in inference')
        for child in value.values():
            no_truth(child)
    elif isinstance(value, list):
        for child in value:
            no_truth(child)


def validate(inp):
    c.require(set(inp) == FIELDS and inp['schema'] == 'oxc_numeric_source_ruv_v1', 'Unexpected inference schema/fields')
    no_truth(inp)
    model, init, var, residual = (inp[k] for k in ('model', 'initialization', 'variability', 'residual'))
    c.require(model['model_id'] == 'Lin_OXC_MHD_Hill219' and model['fixed'] == FIXED, 'OXC structural definition changed')
    c.require(model['dose_unit'] == 'mg' and model['concentration_unit'] == 'mg/L', 'OXC unit mismatch')
    p = model['parameters']
    c.require(set(p) == {'CL_p', 'V_p', 'ka', 'BW'} and all(math.isfinite(v) and v > 0 for v in p.values()), 'Invalid parameters')
    slots = init['time_axes']['manuscript_candidate_slots_h']
    c.require(slots in ([12], [0, 12], [-12, 0, 12]), 'Unsupported candidate slots')
    last = -24 if len(slots) == 3 else -12
    c.require(init['method'] == 'analytic_steady_state' and init['conditioning'] == 'individual_eta_CL'
              and init['shared_last_dose_h'] == init['initial_state_time_h'] == last
              and init['initial_state_phase'] == 'immediately_before_shared_last_dose'
              and init['interval_h'] == 12 and not init['iov_present'], 'Initialization changed')
    c.require(init['initial_dose_event']['time_h'] == last and init['initial_dose_event']['amount_mg'] == 300
              and init['earlier_history'] == {'interval_h': 12, 'dose_mg': 300., 'last_implicit_dose_h': last-12, 'extends_to': 'minus_infinity'},
              'Shared history must contain last dose exactly once')
    c.require(init['no_0h_dose_in_one_dose_window'] == (slots == [12]), 'One-dose window mismatch')
    c.require(var['joint_order'] == var['iiv_order'] == ['Clearance IIV'] and var['iov_order'] == []
              and var['iov_omega_per_occasion'] == [] and var['iov_occasion_ids'] == []
              and var['joint_omega'] == var['iiv_omega'], 'Exactly one clearance effect required')
    omega = var['joint_omega']
    c.require(len(omega) == len(omega[0]) == 1 and math.isfinite(omega[0][0]) and omega[0][0] > 0, 'Invalid omega')
    c.require(abs(var['source_reference_log_variances']['Clearance IIV']-VARIANCE) <= 2e-17, 'Reference CV not exact log1p(0.111**2)')
    c.require(residual['model'] == 'additive' and residual['sd'] > 0 and math.isfinite(residual['sd'])
              and residual['variance'] == residual['sd']**2 and residual['residual_floor'] is None
              and residual['normal_zero_mean'] and residual['additive_component_present'], 'Invalid additive RUV')
    states, priors = inp['candidate_states'], inp['priors']
    c.require(2 <= len(states) <= 8 and len(priors) == len(states), 'Invalid state count')
    c.require(len({s['state_event_id'] for s in states}) == len(states)
              and len({s['manuscript_id'] for s in states}) == len(states), 'Duplicate state identity')
    for s, prior in zip(states, priors):
        c.require(s['state_event_id'] == 'state_'+c.digest({k: v for k, v in s.items() if k != 'state_event_id'}), 'State hash mismatch')
        c.require((prior['state_event_id'], prior['manuscript_id']) == (s['state_event_id'], s['manuscript_id']), 'Prior/state identity mismatch')
        c.require(math.isfinite(prior['probability']) and 0 < prior['probability'] <= 1, 'Positive full prior required')
        c.require(s['nominal_slots_h'] == slots, 'Candidate slot identity mismatch')
        bits = s['machine_chronological_bits']
        if bits is not None:
            c.require(len(bits) == len(slots) and all(b in (0, 1) for b in bits)
                      and s['manuscript_id'] == 'omega_'+''.join(map(str, reversed(bits)))
                      and s['machine_bitstring'] == ''.join(map(str, bits)), 'Chronological/manuscript bit reversal mismatch')
            c.require([e['nominal_slot_h'] for e in s['dose_events']] == [t for t, b in zip(slots, bits) if b], 'Candidate events do not match bits')
        else:
            c.require(s['manuscript_id'] == 'omega_add', 'Unknown nonbinary history')
        for e in s['dose_events']:
            c.require(e['nominal_slot_h'] in slots and last < e['time_h'] and math.isfinite(e['amount_mg']) and e['amount_mg'] > 0
                      and e['route'] == 'oral' and e['compartment'] == 'depot', 'Unsupported candidate event')
            offset = -min(slots)
            c.require(e['local_time_h'] == e['time_h']+offset and e['local_nominal_slot_h'] == e['nominal_slot_h']+offset, 'Local event axis mismatch')
    c.require(abs(math.fsum(p['probability'] for p in priors)-1) <= 1e-14, 'Priors must sum to one, no repair')
    obs = inp['observations']
    c.require(obs and all(set(o) == {'time_h', 'concentration_mg_L'} and math.isfinite(o['time_h']) and o['time_h'] >= last
                         and math.isfinite(o['concentration_mg_L']) and o['concentration_mg_L'] > 0 for o in obs), 'Invalid observations')
    c.require([o['time_h'] for o in obs] == sorted({o['time_h'] for o in obs}), 'Observation times must be unique and ordered')


def rate(model, eta=0., high_precision=False):
    lib, cast = (mp, mp.mpf) if high_precision else (math, float)
    p, f = model['parameters'], model['fixed']
    wt, ka, volume = (cast(p[k]) for k in ('BW', 'ka', 'V_p'))
    z = cast(f['gamma'])*lib.log(wt/cast(f['k50']))
    ratio = 1/(1+lib.exp(-z)) if z >= 0 else lib.exp(z)/(1+lib.exp(z))
    exponent = cast(f['k0'])-cast(f['kmax'])*ratio
    k = cast(p['CL_p'])*(wt/cast(f['normalization_weight']))**exponent*lib.exp(eta)/volume
    return k, ka, volume


def concentration(model, initialization, events, time, eta=0., high_precision=False):
    lib, cast = (mp, mp.mpf) if high_precision else (math, float)
    k, ka, volume = rate(model, eta, high_precision)
    tau, age = cast(initialization['interval_h']), cast(time)-cast(initialization['shared_last_dose_h'])
    c.require(age >= 0, 'Observation precedes initialization')
    def series(x):
        return lib.exp(-x*age)/(-lib.expm1(-x*tau))
    # The high-precision branch must remain differentiable for GHQ derivatives.
    if high_precision:
        if abs(k-ka) < mp.mpf('1e-40'):
            history = -mp.diff(series, ka)
        else:
            history = (series(k)-series(ka))/(ka-k)
    elif abs(k-ka) < 1e-8*max(k, ka):
        with mp.workdps(40):
            return float(concentration(model, initialization, events, time, mp.mpf(eta), True))
    else:
        history = (series(k)-series(ka))/(ka-k)
    result = cast(initialization['earlier_history']['dose_mg'])*ka/volume*history
    for event in events:
        dt = cast(time)-cast(event['time_h'])
        if dt <= 0:
            continue  # Observation before simultaneous input, never a time epsilon.
        term = dt*lib.exp(-ka*dt) if k == ka else (lib.exp(-k*dt)-lib.exp(-ka*dt))/(ka-k)
        result += cast(event['amount_mg'])*ka/volume*term
    return result


class Model:
    def __init__(self, inp):
        validate(inp)
        self.input = inp
        self.states = {s['state_event_id']: s for s in inp['candidate_states']}

    def values(self, state, eta=0., high_precision=False):
        i = self.input
        return [concentration(i['model'], i['initialization'], self.states[state]['dose_events'], o['time_h'], eta, high_precision)
                for o in i['observations']]

    def predictions(self, state_id, eta):
        c.require(len(eta) == 1, 'One ETA required')
        return list(map(float, self.values(state_id, float(eta[0]))))

    def objective_mp(self, state, eta):
        i = self.input
        sd, omega = mp.mpf(i['residual']['sd']), mp.mpf(i['variability']['joint_omega'][0][0])
        values = self.values(state, eta, True)
        return (sum((mp.mpf(o['concentration_mg_L'])-v)**2 for o, v in zip(i['observations'], values))/(2*sd**2)
                + len(values)*mp.log(2*mp.pi*sd**2)/2 + eta**2/(2*omega) + mp.log(2*mp.pi*omega)/2)

