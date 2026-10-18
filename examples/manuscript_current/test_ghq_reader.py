"""Portable reader contracts; no pytest or numerical dependencies by default.

Run: python -B -m unittest discover -s examples/manuscript_current -p test_ghq_reader.py -v
Set GHQ_TEST_TMPDIR to an existing artifact parent (workspace-local when required).
GHQ_TEST_KEEP=1 retains artifacts; GHQ_TEST_RUN_OXC=1 enables two real OXC smokes.
No TAC calculation or archived posterior/reference-answer loading is performed.
"""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location('_ghq_reader_test_target', HERE / 'run_ghq_demo.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
adapters, ghq = reader._modules()
from research_ghq.profiles import canonical_digest, validate_profile


def capture(call, *args):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = call(*args)
        except SystemExit as exc:
            code = exc.code
    return code, stdout.getvalue(), stderr.getvalue()


class GHQReaderTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('GHQ_TEST_TMPDIR', tempfile.gettempdir())).resolve()
        # Deep workspace artifact folders need the Windows extended-path form.
        if os.name == 'nt' and not str(base).startswith('\\\\?\\'):
            value = str(base)
            base = Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)
        self.tmp = Path(tempfile.mkdtemp(prefix='ghq_', dir=base)).resolve()
        if os.environ.get('GHQ_TEST_KEEP') != '1':
            assert self.tmp.is_relative_to(base) and self.tmp != base
            self.addCleanup(shutil.rmtree, self.tmp)

    def export(self, case_id):
        path = self.tmp / (case_id + '.py')
        code, _, error = capture(reader.main, ['export-python', case_id, '--output', str(path)])
        self.assertEqual(code, 0, error)
        return path

    def exported_main(self, path, repo=REPO):
        namespace = {'__name__': '_export_contract'}
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
        with patch.object(sys, 'argv', [str(path), '--repo', str(repo),
                                       '--output', str(self.tmp / 'run')]):
            return capture(namespace['main'])

    def process(self, *args, cwd=None):
        command = [sys.executable, '-B', *map(str, args)]
        start = time.monotonic()
        # This supervisor timeout is a failed/incomplete attempt, not a GHQ budget.
        result = subprocess.run(command, cwd=cwd or self.tmp, capture_output=True,
                                text=True, encoding='utf-8', timeout=90)
        (self.tmp / 'process.json').write_text(json.dumps(dict(
            command=command, cwd=str(cwd or self.tmp), seconds=time.monotonic() - start,
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
            indent=2), encoding='utf-8')
        return result

    def relocated_sources(self):
        root = self.tmp / 'checkout'
        target = root / 'examples' / 'manuscript_current'
        target.mkdir(parents=True)
        for name in ('research_ghq', 'model_adapters'):
            shutil.copytree(HERE / name, target / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(HERE / 'run_ghq_demo.py', target / 'run_ghq_demo.py')
        return root

    def test_all_218_input_profile_and_five_plan_bindings(self):
        entries = adapters.catalog()
        self.assertEqual(len(entries), 218)
        self.assertEqual(len({e['id'] for e in entries}), 218)
        counts, fixed, accepted = Counter(), [], 0
        for entry in entries:
            with self.subTest(case=entry['id']):
                loaded, inputs, profile, plan = reader._case(entry['id'])
                self.assertEqual(loaded, entry)
                digest = hashlib.sha256((HERE / entry['input']).read_bytes()).hexdigest()
                self.assertEqual(digest, entry['input_sha256'])
                self.assertEqual(digest, profile['input_file_sha256'])
                self.assertEqual(canonical_digest(inputs), profile['input_canonical_sha256'])
                states = inputs['priors'] if entry['model'] == 'OXC' else inputs['states']
                self.assertEqual([s['state_event_id'] for s in states], profile['state_ids'])
                self.assertEqual(validate_profile(entry['model'].lower(), inputs, profile, plan),
                                 (profile, plan))
                self.assertFalse(profile['formal_qualified'])
                self.assertFalse(profile['absolute_error_certified'])
                counts[profile['variant']] += 1
                accepted += profile['accepted_configuration']
                if plan is not None:
                    fixed.append(entry['id'])
                    self.assertEqual(canonical_digest(plan), profile['integration_plan_sha256'])
                    self.assertEqual(plan['state_ids'], profile['state_ids'])
                    self.assertEqual(plan['numeric_id'], profile['numeric_id'])
                    self.assertEqual(plan['input_sha256'], digest)
        self.assertEqual(counts, dict(oxc_base=81, oxc_ghq02=2,
                                      tac_adaptive_v22=130, tac_fixed_grid=5))
        self.assertEqual(accepted, 217)
        self.assertEqual(fixed, ['TAC_C010', 'TAC_C060', 'TAC_C079', 'TAC_C099', 'TAC_C111'])
        self.assertFalse(ghq.get_profile('TAC_C078')['accepted_configuration'])

    def test_source_hashes_without_original_sources(self):
        root = self.relocated_sources()
        target = root / 'examples' / 'manuscript_current'
        manifest_path = target / 'research_ghq' / 'source_manifest.json'
        self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                         '09ffc643db37504ed26052273c66413476855e51711f0a2407bd5ca116a25bd3')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for row in manifest['records'] + manifest['generated_files']:
            path = (target / row['target']).resolve()
            self.assertTrue(path.is_relative_to(target))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             row['target_sha256'], row['target'])
        probe = """import json, sys
from pathlib import Path
from unittest.mock import patch
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from research_ghq.api import _sources
original = Path.open
def guarded(path, *args, **kwargs):
    if not path.resolve().is_relative_to(root):
        raise AssertionError('Non-shipped source read: ' + str(path))
    return original(path, *args, **kwargs)
with patch.object(Path, 'open', guarded):
    print(json.dumps(_sources()))
"""
        result = self.process('-c', probe, target)
        self.assertEqual(result.returncode, 0, result.stderr)
        identity = json.loads(result.stdout)
        for row in manifest['records'] + manifest['generated_files']:
            self.assertEqual(identity['files'][row['target']], row['target_sha256'])

    def test_reject_changed_input_model_profile_and_plan(self):
        _, inputs, profile, plan = reader._case('TAC_C079')
        mutations = []
        changed = deepcopy(inputs)
        changed['observations'][0] += 1
        mutations.append(('tac', changed, profile, plan))
        changed = deepcopy(profile)
        changed['policy']['maximum_order'] += 2
        mutations.append(('tac', inputs, changed, plan))
        changed = deepcopy(plan)
        changed['state_ids'].reverse()
        mutations.append(('tac', inputs, profile, changed))
        mutations += [('oxc', inputs, profile, plan),
                      ('tac', inputs, ghq.get_profile('TAC_C003'), plan)]
        for args in mutations:
            with self.subTest(model=args[0], profile=args[2]['profile_id']):
                with self.assertRaises(ValueError):
                    validate_profile(*args)

    def test_help_and_required_arguments_never_dispatch(self):
        with patch.object(ghq, 'run_ghq') as run:
            for args in (['--help'], *[[name, '--help'] for name in
                                      ('list', 'show', 'run', 'export-python')]):
                self.assertEqual(capture(reader.main, args)[0], 0)
            for args in ([], ['show'], ['run'], ['run', 'OXC_C003'],
                         ['run', '--output', str(self.tmp / 'run')],
                         ['export-python', 'OXC_C003'],
                         ['run', 'TAC_C003', '--output', str(self.tmp / 'run'), '--seconds', '1']):
                code, stdout, stderr = capture(reader.main, args)
                self.assertEqual(code, 2)
                self.assertEqual(stdout, '')
                self.assertIn('usage:', stderr)
            run.assert_not_called()

    def test_list_show_and_export_are_inert(self):
        with patch.object(ghq, 'run_ghq') as run, patch.object(adapters, 'compute') as laplace:
            code, stdout, stderr = capture(reader.main, ['list', '--json'])
            self.assertEqual(code, 0, stderr)
            rows = json.loads(stdout)
            self.assertEqual(len(rows), 218)
            self.assertEqual(sum(r['historical_accepted'] for r in rows), 217)
            code, stdout, _ = capture(reader.main, ['list', '--model', 'TAC', '--role', 'reference', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual([r['case_id'] for r in json.loads(stdout)], ['TAC_C003'])
            for case_id in ('OXC_C024', 'TAC_C078', 'TAC_C079'):
                entry, inputs, profile, plan = reader._case(case_id)
                code, stdout, stderr = capture(reader.main, ['show', case_id])
                self.assertEqual(code, 0, stderr)
                self.assertEqual(json.loads(stdout), dict(catalog=entry, inputs=inputs,
                                                          profile=profile, integration_plan=plan))
                self.export(case_id)
            self.assertEqual(len(list(self.tmp.iterdir())), 3)
            run.assert_not_called()
            laplace.assert_not_called()

    def test_export_literals_roundtrip_all_branches_and_state_sizes(self):
        for case_id in ('OXC_C003', 'OXC_C024', 'OXC_C025', 'OXC_C082', 'OXC_C083',
                        'TAC_C003', 'TAC_C079', 'TAC_C134', 'TAC_C135'):
            with self.subTest(case=case_id), patch.object(ghq, 'run_ghq') as run:
                source = self.export(case_id).read_text(encoding='utf-8')
                tree = ast.parse(source)
                compile(tree, case_id, 'exec')
                literals = {node.targets[0].id: ast.literal_eval(node.value)
                            for node in tree.body if isinstance(node, ast.Assign)}
                entry, inputs, profile, plan = reader._case(case_id)
                self.assertEqual(literals, dict(MODEL=entry['model'].lower(), INPUTS=inputs,
                                                PROFILE=profile, INTEGRATION_PLAN=plan))
                self.assertNotIn(str(REPO), source)
                self.assertNotIn(REPO.as_posix(), source)
                for forbidden in ('truth_metadata', 'actual_events', 'expected_posterior',
                                  'accepted_posterior', 'raw_posterior'):
                    self.assertNotIn(forbidden, json.dumps(literals))
                run.assert_not_called()

    def test_unknown_case_and_output_protection(self):
        existing = self.tmp / 'existing'
        existing.mkdir()
        sentinel = self.tmp / 'keep.txt'
        sentinel.write_text('unchanged', encoding='utf-8')
        targets = [existing, sentinel, self.tmp / 'missing' / 'out']
        targets += [REPO / name / 'ghq_forbidden_output' for name in ('src', 'examples', 'tests', '.git')]
        with patch.object(ghq, 'run_ghq') as run:
            for command in ('run', 'export-python'):
                for output in targets:
                    with self.subTest(command=command, output=str(output)):
                        code, stdout, stderr = capture(reader.main, [command, 'OXC_C003', '--output', str(output)])
                        self.assertEqual(code, 2)
                        self.assertEqual(stdout, '')
                        self.assertIn('ValueError', stderr)
            for args in (['show', 'UNKNOWN'], ['run', 'UNKNOWN', '--output', str(self.tmp / 'new')],
                         ['export-python', 'UNKNOWN', '--output', str(self.tmp / 'new.py')]):
                code, stdout, stderr = capture(reader.main, args)
                self.assertEqual(code, 2)
                self.assertIn('use list', stderr)
                self.assertEqual(stdout, '')
            run.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'unchanged')
        self.assertEqual(list(existing.iterdir()), [])

    def test_dangling_symlink_output_is_not_overwritten(self):
        link = self.tmp / 'link'
        try:
            link.symlink_to(self.tmp / 'missing')
        except (OSError, NotImplementedError) as exc:
            self.skipTest('Symlink creation unavailable: ' + str(exc))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            reader._output_path(link)
        self.assertTrue(link.is_symlink())

    def test_mock_native_full_vector_acceptance_and_rejections(self):
        for case_id in ('OXC_C083', 'TAC_C135', 'TAC_C079', 'TAC_C078'):
            entry, inputs, profile, plan = reader._case(case_id)
            states = inputs['priors'] if entry['model'] == 'OXC' else inputs['states']
            values = [1 / len(states)] * len(states)  # Synthetic contract fixture, not an answer.
            path = self.export(case_id)
            for accepted, status in ((True, 'MOCK_ACCEPTED'), (False, 'TIMEOUT'),
                                     (False, 'NODE_BUDGET'), (False, 'NONCONVERGENCE'),
                                     (False, 'GEOMETRY_MISMATCH')):
                summary = dict(status=status, diagnostic_accepted=accepted,
                               formal_qualified=False, absolute_error_certified=False,
                               state_ids=profile['state_ids'], manuscript_ids=[s['manuscript_id'] for s in states],
                               posterior=values if accepted else None, raw_posterior=values,
                               validation={'synthetic_fixture': True})
                for exported in (False, True):
                    with self.subTest(case=case_id, status=status, exported=exported):
                        with patch.object(ghq, 'run_ghq', return_value=deepcopy(summary)) as run:
                            result = self.exported_main(path) if exported else capture(
                                reader.main, ['run', case_id, '--output', str(self.tmp / 'run')])
                            code, stdout, stderr = result
                            self.assertEqual(code, 0 if accepted else 1, stderr)
                            self.assertEqual(json.loads(stdout), summary)
                            run.assert_called_once_with(entry['model'].lower(), inputs,
                                self.tmp / 'run', profile=profile, integration_plan=plan)
                            if profile['resources']:
                                self.assertIn(str(profile['resources']['worker_seconds']), stderr)
                                self.assertIn(str(profile['resources']['point_limit']), stderr)

    def test_mock_exceptions_and_interrupt_exit_codes(self):
        path = self.export('TAC_C003')
        for exc, expected in ((OSError('fixture IO failure'), 2),
                              (ValueError('fixture numerical failure'), 2),
                              (ImportError('fixture dependency failure'), 2),
                              (KeyboardInterrupt(), 130)):
            for exported in (False, True):
                with self.subTest(error=type(exc).__name__, exported=exported):
                    with patch.object(ghq, 'run_ghq', side_effect=exc) as run:
                        result = self.exported_main(path) if exported else capture(
                            reader.main, ['run', 'TAC_C003', '--output', str(self.tmp / 'run')])
                        code, stdout, stderr = result
                        self.assertEqual(code, expected)
                        self.assertEqual(stdout, '')
                        self.assertIn('Interrupted' if expected == 130 else type(exc).__name__, stderr)
                        run.assert_called_once()
                    self.assertFalse((self.tmp / 'run').exists())

    def test_export_bootstrap_requires_repo_and_output(self):
        path = self.export('OXC_C024')
        with patch.object(ghq, 'run_ghq') as run:
            code, stdout, stderr = self.exported_main(path, self.tmp)
            self.assertEqual(code, 2)
            self.assertEqual(stdout, '')
            self.assertIn('--repo', stderr)
            run.assert_not_called()
        result = self.process(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--repo', result.stderr)
        self.assertIn('--output', result.stderr)

    def assert_real_oxc(self, result, output, case_id, status):
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary, json.loads((output / 'run_summary.json').read_text(encoding='utf-8')))
        self.assertEqual(summary['status'], status)
        self.assertTrue(summary['diagnostic_accepted'])
        self.assertFalse(summary['formal_qualified'])
        self.assertFalse(summary['absolute_error_certified'])
        _, inputs, profile, _ = reader._case(case_id)
        self.assertEqual(summary['state_ids'], profile['state_ids'])
        self.assertEqual(summary['manuscript_ids'], [s['manuscript_id'] for s in inputs['priors']])
        self.assertEqual(len(summary['posterior']), len(profile['state_ids']))
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in summary['posterior']))
        self.assertAlmostEqual(math.fsum(summary['posterior']), 1, places=12)
        receipt = json.loads((output / 'process_receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['execution_state'], 'COMPLETED')
        self.assertFalse(receipt['reference_answers_read'])
        self.assertFalse(receipt['resumed_checkpoint'])
        self.assertFalse(receipt['budget_enforced_by_native_worker'])
        self.assertEqual(set(receipt['environment']['thread_environment'].values()), {'1'})
        self.assertEqual(json.loads((output / 'input.json').read_text(encoding='utf-8')), inputs)
        self.assertEqual(json.loads((output / 'profile.json').read_text(encoding='utf-8')), profile)

    @unittest.skipUnless(os.environ.get('GHQ_TEST_RUN_OXC') == '1', 'Opt-in real OXC smoke')
    def test_actual_oxc_base_cli(self):
        output = self.tmp / 'base'
        result = self.process(HERE / 'run_ghq_demo.py', 'run', 'OXC_C003', '--output', output)
        self.assert_real_oxc(result, output, 'OXC_C003', 'PASS')

    @unittest.skipUnless(os.environ.get('GHQ_TEST_RUN_OXC') == '1', 'Opt-in real OXC smoke')
    def test_actual_oxc_supplement_export_relocated(self):
        path = self.export('OXC_C024')
        checkout = self.relocated_sources()
        output = self.tmp / 'supplement'
        result = self.process(path, '--repo', checkout, '--output', output)
        self.assert_real_oxc(result, output, 'OXC_C024', 'SUPPLEMENT_PASS')


if __name__ == '__main__':
    unittest.main()
