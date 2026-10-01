from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Sequence
import pmx_discrete_posterior as pmx
from numpy.polynomial.hermite import hermgauss
SUM_TOL=1e-6

@dataclass(frozen=True)
class OxcRouteModel:
    model: dict[str, Any]
    patient: dict[str, Any]
    observations: list[dict[str, Any]]
    candidate_events: dict[str, list[dict[str, Any]]]
    dose_interval_h: float
    steady_state_at_time_zero: bool
    time_zero_epsilon_h: float

    @property
    def wt_exp(self) -> float:
        wt = float(self.patient['weight_kg'])
        hill = float(self.model['theta_wt_exp_hill'])
        wt_hill = wt ** hill
        half_hill = float(self.model['theta_wt_exp_half']) ** hill
        return float(self.model['theta_wt_exp_base']) - float(self.model['theta_wt_exp_max_dec']) * wt_hill / (half_hill + wt_hill)

    @property
    def cl_typical_l_h(self) -> float:
        return float(self.model['theta_cl_base_l_h']) * (float(self.patient['weight_kg']) / 70.0) ** self.wt_exp

    def predictions(self, state_id: str, eta: Sequence[float]) -> tuple[float, ...]:
        eta_cl = float(tuple(eta)[0])
        return tuple((self.prediction_at_time(str(state_id), float(obs['time_h']), eta_cl) for obs in self.observations if bool(obs.get('enabled', True))))

    def prediction_at_time(self, state_id: str, observation_time_h: float, eta_cl: float) -> float:
        concentration = 0.0
        if self.steady_state_at_time_zero:
            concentration += self._steady_state_pre_zero_contribution(observation_time_h, eta_cl)
        for event in self.candidate_events[str(state_id)]:
            dose_time = float(event.get('nonmem_time_h', event['time_h']))
            concentration += self._oral_contribution(dose_age_h=float(observation_time_h) - dose_time, dose_mg=float(event['dose_mg']), eta_cl=eta_cl)
        return max(float(concentration), 0.0)

    def _k_elim(self, eta_cl: float) -> float:
        return self.cl_typical_l_h * math.exp(float(eta_cl)) / float(self.model['v_l'])

    def _oral_contribution(self, *, dose_age_h: float, dose_mg: float, eta_cl: float) -> float:
        if dose_age_h < 0.0:
            return 0.0
        k = self._k_elim(eta_cl)
        ka = float(self.model['ka_h'])
        v_l = float(self.model['v_l'])
        if abs(ka - k) < 1e-08:
            return dose_mg * ka / v_l * dose_age_h * math.exp(-k * dose_age_h)
        return dose_mg * ka / (v_l * (ka - k)) * (math.exp(-k * dose_age_h) - math.exp(-ka * dose_age_h))

    def _steady_state_pre_zero_contribution(self, observation_time_h: float, eta_cl: float) -> float:
        tau = max(float(self.dose_interval_h), 1e-12)
        anchor_age_h = float(observation_time_h) + tau
        k = self._k_elim(eta_cl)
        ka = float(self.model['ka_h'])
        if abs(ka - k) < 1e-08:
            k = k * (1.0 + 1e-08)
        denom_k = max(1.0 - math.exp(-k * tau), 1e-300)
        denom_ka = max(1.0 - math.exp(-ka * tau), 1e-300)
        dose_mg = self._anchor_dose_mg()
        return dose_mg * ka / (float(self.model['v_l']) * (ka - k)) * (math.exp(-k * anchor_age_h) / denom_k - math.exp(-ka * anchor_age_h) / denom_ka)

    def _anchor_dose_mg(self) -> float:
        for events in self.candidate_events.values():
            for event in events:
                return float(event['dose_mg'])
        return 300.0

def build_candidate_events(normalized: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    events_by_id = {str(event['id']): dict(event) for event in normalized['dose_events_for_nonmem']}
    time_zero_epsilon_h = float(normalized['time_zero']['nonmem_dose_time_zero_epsilon_h'])
    out: dict[str, list[dict[str, Any]]] = {}
    candidate_map = normalized['candidate_definition']['candidate_event_map']
    for candidate_id, payload in candidate_map.items():
        rows = []
        for event in payload['given_dose_events']:
            event_id = str(event['id'])
            row = dict(events_by_id.get(event_id, event))
            time_h = float(row['time_h'])
            if 'input_time_h' not in row:
                row['input_time_h'] = time_h
            if 'nonmem_time_h' not in row:
                row['nonmem_time_h'] = time_zero_epsilon_h if time_h == 0.0 else time_h
            if 'nonmem_time_adjustment' not in row:
                row['nonmem_time_adjustment'] = 'time_zero_to_0.0001' if time_h == 0.0 else 'none'
            rows.append(row)
        out[str(candidate_id)] = rows
    return out
@dataclass(frozen=True)
class ExplicitHistoryOxcModel(OxcRouteModel):
    history_dose_mg: float
    history_interval_h: float
    history_last_dose_time_h: float
    candidate_window_start_h: float

    def __post_init__(self):
        if self.history_interval_h <= 0 or self.history_dose_mg < 0:
            raise ValueError('Invalid explicit history')
        if self.history_last_dose_time_h >= self.candidate_window_start_h:
            raise ValueError('Common history must precede the declared candidate window')
        if any((float(e['time_h']) < self.candidate_window_start_h for events in self.candidate_events.values() for e in events)):
            raise ValueError('Candidate event precedes the fixed window')

    def _anchor_dose_mg(self):
        return self.history_dose_mg

    def _steady_state_pre_zero_contribution(self, observation_time_h, eta_cl):
        if observation_time_h < self.candidate_window_start_h:
            raise ValueError('This bounded adapter requires observations at/after the candidate window')
        tau = self.history_interval_h
        age = float(observation_time_h) - self.history_last_dose_time_h
        k = self._k_elim(eta_cl)
        ka = float(self.model['ka_h'])
        if abs(ka - k) < 1e-08:
            k *= 1.0 + 1e-08
        return self.history_dose_mg * ka / (float(self.model['v_l']) * (ka - k)) * (math.exp(-k * age) / max(1.0 - math.exp(-k * tau), 1e-300) - math.exp(-ka * age) / max(1.0 - math.exp(-ka * tau), 1e-300))

def inference_inputs(n):
    order = n['candidate_definition']['candidate_order']
    states = tuple((pmx.DiscreteState(state_id=s, prior=float(n['prior_vector'][s]), label=s) for s in order))
    config = pmx.MultiEtaLaplaceConfig(omega_covariance=((float(n['model']['iiv']['omega_variance']),),), additive_sd=float(n['model']['ruv']['additive_sd_mg_l']), proportional_sd=float(n['model']['ruv']['proportional_sd']), optimizer_method='Nelder-Mead')
    observations = [float(o['dv_mg_l']) for o in n['observation_events'] if o.get('enabled', True)]
    return (states, config, observations)

def validity(order, pairs):
    keys = [k for k, _ in pairs]
    values = [float(v) for _, v in pairs]
    finite = all((math.isfinite(v) for v in values))
    total = math.fsum(values) if finite else None
    checks = {'count': len(keys) == len(order), 'unique': len(keys) == len(set(keys)), 'keyset': set(keys) == set(order), 'order': keys == list(order), 'finite': finite, 'range': finite and all((0.0 <= v <= 1.0 for v in values)), 'sum': total is not None and abs(total - 1.0) <= SUM_TOL}
    result = {'valid': all(checks.values()), 'checks': checks, 'raw_sum': total, 'sum_tolerance': SUM_TOL, 'raw_pairs': pairs}
    if result['valid']:
        norm = {k: v / total for k, v in pairs}
        result.update(normalized=norm, normalization_factor=1.0 / total, max_printed_correction=max((abs(norm[k] - v) for k, v in pairs)))
    return result
def logsumexp(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum((math.exp(value - m) for value in finite)))

def log_observation_likelihood(observations: list[float], state_id: str, eta: float, model: OxcRouteModel, config: Any) -> float:
    pred = list(model.predictions(state_id, (float(eta),)))
    if len(pred) != len(observations):
        raise ValueError(f'Model returned {len(pred)} predictions for {len(observations)} observations.')
    total = 0.0
    for y, p in zip(observations, pred):
        sd = math.sqrt(float(config.additive_sd) ** 2 + (float(config.proportional_sd) * max(float(p), 0.0)) ** 2)
        sd = max(sd, float(config.positive_floor), 1e-300)
        total += -0.5 * math.log(2.0 * math.pi) - math.log(sd) - 0.5 * ((float(y) - float(p)) / sd) ** 2
    return float(total)

def log_eta_prior(eta: float, omega: float) -> float:
    return -0.5 * math.log(2.0 * math.pi) - 0.5 * math.log(float(omega)) - 0.5 * float(eta) ** 2 / float(omega)

def adaptive_ghq_posterior(*, pmx: Any, observations: list[float], states: tuple[Any, ...], model: OxcRouteModel, config: Any, nodes: int, mode_optimizer_method: str) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
    mode_config = pmx.MultiEtaLaplaceConfig(omega_covariance=config.omega_covariance, additive_sd=config.additive_sd, proportional_sd=config.proportional_sd, eta_lower=config.eta_lower, eta_upper=config.eta_upper, optimizer_method=mode_optimizer_method, optimizer_maxiter=config.optimizer_maxiter, derivative_step_min=config.derivative_step_min, derivative_step_scale=config.derivative_step_scale, positive_floor=config.positive_floor, prior_policy=config.prior_policy)
    omega = float(config.omega_covariance[0][0])
    node_values, weights = hermgauss(int(nodes))
    log_marginals: dict[str, float] = {}
    mode_diagnostics: dict[str, dict[str, Any]] = {}
    for state in states:
        diag = pmx.multieta_state_diagnostics(observations, state, model, mode_config)
        eta_hat = float(diag.eta_hat[0])
        hessian = float(diag.hessian_min_eigenvalue)
        if hessian <= 0.0 or not math.isfinite(hessian):
            log_marginals[state.state_id] = -math.inf
            mode_diagnostics[state.state_id] = {'eta_hat': eta_hat, 'hessian': hessian, 'optimizer_success': bool(diag.optimizer_success), 'optimizer_message': str(diag.optimizer_message), 'valid_mode': False, 'warnings': list(diag.warnings) + ['invalid_mode_hessian']}
            continue
        scale = 1.0 / math.sqrt(hessian)
        terms = []
        for x, w in zip(node_values, weights):
            eta = eta_hat + math.sqrt(2.0) * scale * float(x)
            log_joint = log_observation_likelihood(observations, state.state_id, eta, model, config) + log_eta_prior(eta, omega)
            terms.append(math.log(float(w)) + log_joint + float(x) ** 2)
        log_marginal = math.log(math.sqrt(2.0) * scale) + logsumexp(terms)
        log_marginals[state.state_id] = float(log_marginal)
        mode_diagnostics[state.state_id] = {'eta_hat': eta_hat, 'hessian': hessian, 'optimizer_success': bool(diag.optimizer_success), 'optimizer_message': str(diag.optimizer_message), 'valid_mode': bool(diag.optimizer_success and diag.valid_laplace), 'warnings': list(diag.warnings)}
    log_weights = {state.state_id: (math.log(float(state.prior)) if float(state.prior) > 0 else -math.inf) + log_marginals[state.state_id] for state in states}
    norm = logsumexp(list(log_weights.values()))
    posterior = {state_id: math.exp(log_weight - norm) if math.isfinite(log_weight) and math.isfinite(norm) else 0.0 for state_id, log_weight in log_weights.items()}
    return (posterior, log_marginals, mode_diagnostics)
