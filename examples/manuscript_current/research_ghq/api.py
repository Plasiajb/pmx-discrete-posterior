"""Portable orchestration only; numerical rules live in the formal modules."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_load(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON key: ' + key)
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def _write(path, value):
    from ._tac.atomic_io import write_text_atomic
    write_text_atomic(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def _sources():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'source_manifest.json').read_text(encoding='utf-8'))
    files = {p.relative_to(here.parent).as_posix(): _sha(p.read_bytes())
             for p in sorted(here.rglob('*.py'))}
    for row in [*manifest['records'], *manifest.get('generated_files', [])]:
        path = here.parent / row['target']
        actual = _sha(path.read_bytes())
        if actual != row['target_sha256']:
            raise RuntimeError('Pinned formal/shared source changed: ' + row['target'])
        files[row['target']] = actual
    return dict(version=manifest['version'], source_manifest_sha256=_sha((here / 'source_manifest.json').read_bytes()),
                files=files)


def _environment():
    modules = {}
    for name in ('numpy', 'scipy', 'mpmath'):
        module = __import__(name)
        modules[name] = dict(runtime=module.__version__, distribution=importlib.metadata.version(name),
                             file=str(Path(module.__file__).resolve()))
    return dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
                prefix=sys.prefix, base_prefix=sys.base_prefix, dependencies=modules,
                thread_environment={key: os.environ.get(key) for key in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})


def _vector_ok(ids, values):
    return (values is not None and len(values) == len(ids) and len(set(ids)) == len(ids)
            and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in values)
            and math.isclose(math.fsum(values), 1., rel_tol=0, abs_tol=1e-12))


def run_ghq(model, inputs, output_dir, seconds=None, point_limit=None, *,
            profile=None, integration_plan=None):
    """Run exact formal GHQ in-process and retain evidence in a new directory.

    model: 'oxc' or 'tac'. inputs: inference-only dict or JSON path.
    Without a profile the original 1780s/550M budget and reference paths remain.
    Profiles are pinned case IDs or unchanged get_profile() exports. TAC profile
    budgets default to the declared worker allowance; a smaller positive seconds
    value may reproduce a scheduling receipt or bound a diagnostic timeout.
    Fixed-grid plans may be supplied explicitly or loaded from pinned plan assets.
    OXC does not enforce either resource budget. No subprocesses,
    saved answers, fallback estimators, old checkpoints or catalogs are used.
    Run in a fresh single-thread numerical process for reproduction; importing
    TAC sets its original thread environment before importing NumPy/SciPy.
    """
    if model not in ('oxc', 'tac'):
        raise ValueError("model must be 'oxc' or 'tac'")
    selected = None
    if profile is not None:
        from .profiles import get_profile
        profile_id = profile if isinstance(profile, str) else profile.get('profile_id') if isinstance(profile, dict) else None
        selected = get_profile(profile_id)
    elif integration_plan is not None:
        raise ValueError('An integration plan requires its pinned case profile')
    resources = selected['resources'] if selected is not None else None
    seconds = (resources['worker_seconds'] if resources else 1780) if seconds is None else seconds
    point_limit = (resources['point_limit'] if resources else 550000000) if point_limit is None else point_limit
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('seconds must be finite and positive')
    if type(point_limit) is not int or point_limit <= 0:
        raise ValueError('point_limit must be a positive integer')
    if resources and (point_limit != resources['point_limit'] or seconds > resources['worker_seconds']):
        raise ValueError('Profile point limit is fixed; seconds cannot exceed its declared worker allowance')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(schema='research_ghq_process_receipt_v1', model=model, pid=os.getpid(),
                   started_utc=_now(), execution_state='RUNNING', synchronous=True,
                   child_processes_created=False, numerical_libraries_preloaded=[name for name in
                   ('numpy', 'scipy') if name in sys.modules],
                   requested_budget=dict(seconds=seconds, point_limit=point_limit),
                   budget_enforced_by_native_worker=model == 'tac', fresh_output=True,
                   resumed_checkpoint=False, reference_answers_read=False)
    _write(output / 'process_receipt.json', receipt)
    try:
        if isinstance(inputs, dict):
            raw_input = (json.dumps(inputs, indent=2, allow_nan=False) + '\n').encode('utf-8')
        elif isinstance(inputs, (str, os.PathLike)):
            raw_input = Path(inputs).read_bytes()
        else:
            raise TypeError('inputs must be an inference dictionary or JSON path')
        payload = _json_load(raw_input.decode('utf-8-sig'))
        (output / 'input.json').write_bytes(raw_input)
        receipt.update(input_sha256=_sha(raw_input), source_identity=_sources())
        plan = None
        if selected is not None:
            from .profiles import validate_profile, canonical_digest
            if isinstance(integration_plan, (str, os.PathLike)):
                integration_plan = _json_load(Path(integration_plan).read_text(encoding='utf-8-sig'))
            selected, plan = validate_profile(model, payload, profile, integration_plan)
            receipt.update(profile=selected, profile_sha256=canonical_digest(selected),
                           profile_budget_overridden=bool(resources and seconds != resources['worker_seconds']))
            _write(output / 'profile.json', selected)
            if plan is not None:
                _write(output / 'integration_plan.json', plan)
        if model == 'tac':
            if plan is not None:
                from ._fixed_runtime import fixed_worker
                context = fixed_worker(selected['policy_overrides'])
            else:
                from ._tac import worker
                context = nullcontext(worker)
            with context as worker:
                from ._tac.resolved import ResolvedCase
                case = ResolvedCase.from_dict(payload)
                receipt['environment'] = _environment()
                _write(output / 'process_receipt.json', receipt)
                options = dict(integration_plan=plan, run_identity=dict(
                    profile_id=selected['profile_id'], profile_sha256=receipt['profile_sha256'])) if plan is not None else {}
                worker.case_attempt(case, output, seconds=seconds, point_limit=point_limit,
                                    backend='numpy', **options)
                # Match formal validation of persisted JSON, not ndarray equality.
                persisted = json.loads((output / 'result.json').read_text(encoding='utf-8'))
                valid = (worker.validate_result(persisted, case, plan) if plan is not None
                         else worker.validate_result(persisted, case))
                qualifies = worker.qualifies(persisted['validation_stages'])
            ids = list(case.state_ids)
            labels = [s['manuscript_id'] for s in case.states]
            complete = [s for s in persisted['validation_stages'] if s.get('complete')]
            latest = complete[-1] if complete else None
            raw_values = latest['posterior'] if latest else None
            order_ok = latest is not None and [s['state_event_id'] for s in latest['states']] == ids
            integrity = order_ok and _vector_ok(ids, raw_values)
            validation = dict(validate_result=valid, qualifies=qualifies,
                              state_order_matches=order_ok, posterior_integrity=integrity)
            accepted = valid and qualifies and integrity
            raw = persisted
        else:
            if selected is not None and selected['variant'] == 'oxc_ghq02':
                from . import _oxc_supplement as _oxc
                pass_status = 'SUPPLEMENT_PASS'
            else:
                from . import _oxc
                pass_status = 'PASS'
            receipt['environment'] = _environment()
            _write(output / 'process_receipt.json', receipt)
            raw = _oxc.ghq(payload)
            _write(output / 'result.json', raw)
            ids = [p['state_event_id'] for p in payload['priors']]
            labels = [p['manuscript_id'] for p in payload['priors']]
            native_posterior = raw.get('raw_posterior')
            order_ok = isinstance(native_posterior, dict) and list(native_posterior) == ids
            raw_values = [native_posterior[s] for s in ids] if order_ok else None
            integrity = order_ok and _vector_ok(ids, raw_values)
            valid = raw['status'] == pass_status and raw['qualified'] is True
            validation = dict(native_qualified=raw['qualified'], native_status_pass=raw['status'] == pass_status,
                              state_order_matches=order_ok, posterior_integrity=integrity)
            accepted = valid and integrity
        summary = dict(schema='research_ghq_summary_v1', model=model, status=raw['status'],
                       state_ids=ids, manuscript_ids=labels,
                       posterior=raw_values if accepted else None, raw_posterior=raw_values,
                       validation=validation, diagnostic_accepted=bool(accepted),
                       formal_qualified=False, absolute_error_certified=False,
                       input_sha256=receipt['input_sha256'], version=receipt['source_identity']['version'],
                       profile_id=selected['profile_id'] if selected else None,
                       profile_variant=selected['variant'] if selected else None,
                       profile_sha256=receipt.get('profile_sha256'),
                       artifacts=dict(input='input.json', native_result='result.json',
                                      native_checkpoint='checkpoint.json' if model == 'tac' else None,
                                      process_receipt='process_receipt.json'),
                       budget_enforced_by_native_worker=model == 'tac')
        _write(output / 'run_summary.json', summary)
        receipt.update(execution_state='COMPLETED', native_status=raw['status'],
                       diagnostic_accepted=bool(accepted), summary='run_summary.json')
        return summary
    except BaseException as exc:
        receipt.update(execution_state='EXCEPTION', exception_type=type(exc).__name__, error=str(exc))
        _write(output / 'exception.json', dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        receipt.update(finished_utc=_now(), elapsed_seconds=time.monotonic() - started)
        _write(output / 'process_receipt.json', receipt)
