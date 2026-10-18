"""Reader entry for the frozen manuscript GHQ reference and sensitivity cases."""
from __future__ import annotations

import os

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import importlib
import json
from pathlib import Path
import pprint
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)


def _modules():
    for prefix in ('research_ghq', 'model_adapters'):
        for name, module in tuple(sys.modules.items()):
            if name == prefix or name.startswith(prefix + '.'):
                source = getattr(module, '__file__', None)
                if source is None or not Path(source).resolve().is_relative_to(HERE / prefix):
                    raise RuntimeError('Foreign ' + name + ' loaded; start a fresh Python process')
    value = str(HERE)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
    return importlib.import_module('model_adapters'), importlib.import_module('research_ghq')


def _output_path(value):
    path = Path(value).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError('Output already exists; choose a new name: ' + str(path))
    path = path.resolve()
    for protected in (REPO / 'examples', REPO / 'src', REPO / 'tests', REPO / '.git'):
        if path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError('Output must be disjoint from repository source and Git metadata')
    if not path.parent.is_dir():
        raise ValueError('Output parent must already exist: ' + str(path.parent))
    return path


def _case(case_id):
    adapters, ghq = _modules()
    try:
        entry, inputs = adapters.load_case(case_id)
    except KeyError as exc:
        raise ValueError('Unknown case; use list to inspect available identifiers') from exc
    profile = ghq.get_profile(entry['id'])
    plan = ghq.get_integration_plan(entry['id'])
    from research_ghq.profiles import validate_profile
    validate_profile(entry['model'].lower(), inputs, profile, plan)
    return entry, inputs, profile, plan


def run_inputs(model, inputs, profile, integration_plan, output):
    """Full inference inputs only; no saved posterior is read or substituted."""
    output = _output_path(output)
    _, ghq = _modules()
    resources = profile['resources']
    if resources:
        print('GHQ profile {}: {} points, {} worker seconds (not a runtime estimate).'.format(
              profile['profile_id'], resources['point_limit'], resources['worker_seconds']),
              file=sys.stderr, flush=True)
    summary = ghq.run_ghq(model, inputs, output, profile=profile,
                          integration_plan=integration_plan)
    print(_json(summary), flush=True)
    return 0 if summary['diagnostic_accepted'] else 1


def export_python(case_id, output):
    entry, inputs, profile, plan = _case(case_id)
    path = _output_path(output)
    # Embed complete inputs and declarations, not author-local paths or answers.
    source = '''"""Complete frozen GHQ example. Keep the full public repository available."""
from pathlib import Path
import argparse
import importlib.util
import sys

# 1. Model, random effects, residual error, observations, candidates and priors.
MODEL = {model!r}
INPUTS = {inputs}

# 2. Frozen numerical policy and, if needed, the three-level integration plan.
PROFILE = {profile}
INTEGRATION_PLAN = {plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Complete public repository checkout')
    parser.add_argument('--output', type=Path, required=True, help='New directory; parent must exist')
    args = parser.parse_args()
    try:
        # 3. Load the reader entry from the selected checkout; it pins imports/threads.
        source = args.repo.resolve() / 'examples/manuscript_current/run_ghq_demo.py'
        if not source.is_file():
            raise ValueError('Missing reader entry; --repo must name the complete repository')
        spec = importlib.util.spec_from_file_location('_selected_ghq_demo', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # 4. Recompute all states and print the complete posterior and diagnostics.
        return module.run_inputs(MODEL, INPUTS, PROFILE, INTEGRATION_PLAN, args.output)
    except KeyboardInterrupt:
        print('Interrupted; retain any partial diagnostics.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
'''.format(inputs=pprint.pformat(inputs, sort_dicts=False, width=100),
           profile=pprint.pformat(profile, sort_dicts=False, width=100),
           plan=pprint.pformat(plan, sort_dicts=False, width=100), model=entry['model'].lower())
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)
    print(str(path))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='Inspect cases and budgets without computing')
    listing.add_argument('--model', choices=['OXC', 'TAC'])
    listing.add_argument('--role', choices=['reference', 'sensitivity'])
    listing.add_argument('--json', action='store_true')
    show = commands.add_parser('show', help='Print full input, profile and integration plan')
    show.add_argument('case_id')
    for name, help_text in [('run', 'Compute one explicit case'),
                            ('export-python', 'Export a complete callable example, without computing')]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument('case_id')
        command.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            adapters, ghq = _modules()
            rows = []
            for entry in adapters.catalog():
                if args.model and args.model != entry['model']:
                    continue
                if args.role and args.role != entry['role']:
                    continue
                profile = ghq.get_profile(entry['id'])
                rows.append(dict(case_id=entry['id'], model=entry['model'], role=entry['role'],
                                 variant=profile['variant'], resources=profile['resources'],
                                 historical_accepted=profile['accepted_configuration'],
                                 memberships=entry['memberships']))
            if args.json:
                print(_json(rows))
            else:
                for row in rows:
                    resources = row['resources']
                    budget = ('{} points / {} s'.format(resources['point_limit'], resources['worker_seconds'])
                              if resources else 'OXC fixed order sequence')
                    print('{case_id}\t{role}\t{variant}\t'.format(**row) + budget +
                          ('\thistorical failure' if not row['historical_accepted'] else ''))
            return 0
        if args.command == 'export-python':
            return export_python(args.case_id, args.output)
        entry, inputs, profile, plan = _case(args.case_id)
        if args.command == 'show':
            print(_json(dict(catalog=entry, inputs=inputs, profile=profile, integration_plan=plan)))
            return 0
        return run_inputs(entry['model'].lower(), inputs, profile, plan, args.output)
    except KeyboardInterrupt:
        print('Interrupted; retain any partial diagnostics.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
