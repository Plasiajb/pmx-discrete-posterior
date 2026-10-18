"""Current manuscript inputs and the formal Python-Laplace calculation path."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def catalog():
    """Return scenario descriptions, not posterior answers."""
    return json.loads((ROOT / 'catalog.json').read_text(encoding='utf-8'))['cases']


def load_case(case_id):
    """Load and verify all resolved scientific inputs for one published case."""
    entries = {entry['id']: entry for entry in catalog()}
    entry = entries[case_id.upper()]
    path = (ROOT / entry['input']).resolve()
    if not path.is_relative_to((ROOT / 'inputs').resolve()):
        raise ValueError('Input path must remain inside the published input directory')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['input_sha256']:
        raise ValueError('Published input hash mismatch; use compute() explicitly for edited inputs')
    return deepcopy(entry), json.loads(raw)


def _clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_tac_fields(data):
    from .oxc_model import no_truth
    no_truth(data)
    if set(data['parameters']) != {'CL', 'V1', 'Q', 'V2', 'ka', 'lag', 'WT'}:
        raise ValueError('Unexpected TAC model parameter fields')
    if set(data['fixed']) != {'X', 'HEM', 'POD', 'Pred', 'theta_CYP3A5', 'theta_HEM',
                             'theta_POD', 'theta_Pred', 'POD_cap', 'WT_exponent', 'normalization_weight'}:
        raise ValueError('Unexpected TAC fixed-covariate fields')
    for values in (data['parameters'], data['fixed']):
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values.values()):
            raise ValueError('Finite numerical model/covariate values required')


def compute(model, inputs):
    """Compute from supplied inputs; no catalog or expected result is consulted.

    Numerical settings and independent diagnostic gates match the formal runs.
    Scientific validity is conveyed by eligible, never by a process exit code.
    """
    import numpy as np
    import scipy
    import mpmath
    import pmx_discrete_posterior as pmx
    started = time.monotonic()
    model = model.upper()
    data = deepcopy(inputs)
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    try:
        if model == 'OXC':
            from .oxc_laplace import laplace
            if set(data['residual']) != {'model', 'sd', 'variance', 'additive_component_present',
                                        'residual_floor', 'normal_zero_mean'}:
                raise ValueError('Unexpected OXC residual fields')
            result = laplace(data, pmx)
            result['eligible'] = result['qualified']
            state_rows = [{'state_id': s['state_event_id'], 'label': s['manuscript_id'],
                           'events': s['dose_events'], 'prior': p['probability']}
                          for s, p in zip(data['candidate_states'], data['priors'], strict=True)]
            dimension = 1
        elif model == 'TAC':
            from .tac_model import ResolvedCase
            from .tac_laplace import run
            _validate_tac_fields(data)
            case = ResolvedCase.from_dict(data)
            result = run(case)
            state_rows = [{'state_id': s['state_event_id'], 'label': s['manuscript_id'],
                           'events': s['events'], 'prior': s['prior']} for s in data['states']]
            dimension = case.neta
        else:
            raise ValueError('Supported models: OXC, TAC')
        posterior = result.get('posterior')
        if result['eligible']:
            expected_ids = [s['state_id'] for s in state_rows]
            if (not isinstance(posterior, dict) or set(posterior) != set(expected_ids)
                    or not all(math.isfinite(p) and 0 <= p <= 1 for p in posterior.values())
                    or abs(math.fsum(posterior.values()) - 1) > 1e-10):
                raise ValueError('Incomplete or invalid posterior vector')
        result['states'] = [{**s, 'posterior_probability': posterior[s['state_id']] if posterior else None}
                            for s in state_rows]
        result['full_latent_dimension'] = dimension
    except Exception as exc:
        result = {'status': 'FAILED_COMPUTATION', 'eligible': False, 'posterior': None,
                  'error_type': type(exc).__name__, 'error': str(exc)}
    result.update(model=model, input_sha256=hashlib.sha256(raw).hexdigest(),
                  elapsed_seconds=time.monotonic() - started,
                  versions={'package': pmx.__version__, 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'mpmath': mpmath.__version__},
                  method='Python-Laplace; formal October 2026 settings')
    return _clean(result)


def run_case(case_id):
    entry, inputs = load_case(case_id)
    result = compute(entry['model'], inputs)
    result.update(case_id=entry['id'], source_case_id=entry['source_case_id'],
                  role=entry['role'], published_input_sha256=entry['input_sha256'])
    return result
