"""Pinned case profiles; no accepted probabilities or old integral evaluations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = '2026.10.18.2'
REGISTRY_SHA256 = '1126d67ae3b7053c48a13e3c8b00d3ef300e12fb276d4c99b231918842072fa0'
DATA = Path(__file__).resolve().parent / 'profile_data'


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _registry():
    raw = (DATA / 'registry.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRY_SHA256:
        raise ValueError('Pinned GHQ profile registry changed')
    data = json.loads(raw)
    if data['schema'] != 'research_ghq_profile_registry_v1' or data['version'] != VERSION:
        raise ValueError('Unsupported GHQ profile registry')
    return {row['profile_id']: row for row in data['profiles']}


def get_profile(profile_id):
    """Return an exportable copy of a frozen OXC_Cnnn/TAC_Cnnn declaration."""
    if not isinstance(profile_id, str):
        raise TypeError('profile_id must be a catalog case identifier')
    try:
        return deepcopy(_registry()[profile_id])
    except KeyError as exc:
        raise ValueError('Unknown GHQ profile identifier') from exc


def get_integration_plan(profile_id):
    """Return the pinned fixed grid; geometry is a post-optimization check only."""
    profile = get_profile(profile_id)
    if profile['variant'] != 'tac_fixed_grid':
        return None
    plan = json.loads((DATA / (profile_id + '.plan.json')).read_text(encoding='utf-8'))
    if canonical_digest(plan) != profile['integration_plan_sha256']:
        raise ValueError('Pinned integration plan changed')
    return plan


def validate_profile(model, inputs, profile, integration_plan=None):
    """Reject altered input, policy, plan, thresholds and cross-case identities."""
    supplied = get_profile(profile) if isinstance(profile, str) else deepcopy(profile)
    if not isinstance(supplied, dict) or not isinstance(supplied.get('profile_id'), str):
        raise ValueError('profile must be a pinned profile identifier or exported object')
    expected = get_profile(supplied['profile_id'])
    if canonical_digest(supplied) != canonical_digest(expected):
        raise ValueError('GHQ profile differs from its frozen declaration')
    if model != expected['model'] or canonical_digest(inputs) != expected['input_canonical_sha256']:
        raise ValueError('GHQ profile model/input identity mismatch')
    plan = get_integration_plan(expected['profile_id'])
    if integration_plan is not None:
        if plan is None or canonical_digest(integration_plan) != expected['integration_plan_sha256']:
            raise ValueError('Integration plan differs from the pinned case plan')
        plan = deepcopy(integration_plan)
    if plan is not None:
        if (plan['input_sha256'] != expected['input_file_sha256']
                or plan['numeric_id'] != expected['numeric_id']
                or plan['state_ids'] != expected['state_ids']):
            raise ValueError('Fixed-grid plan/input/state binding mismatch')
    return expected, plan
