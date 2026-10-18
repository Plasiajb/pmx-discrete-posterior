# Numerical Replay

Verified on Windows with Python 3.12.0, NumPy 2.5.2, SciPy 1.16.2 and mpmath 1.3.0.
All 218 supplied input records were recomputed in fresh processes. Expected
results were inaccessible during inference and read only afterward for comparison.
Every accepted receipt was checked against the delivered source and input hashes.

| Python-Laplace | Designed cases | Valid vectors | Maximum absolute difference |
|---|---:|---:|---:|
| OXC | 83 | 83 | 1.11e-16 |
| TAC | 135 | 134 | 5.55e-17 |

Each count includes one reference case. Full vectors were aligned by state ID
and compared with the normalized means of the accepted five formal repeats.
The predeclared componentwise criterion was
`abs(actual - expected) <= 1e-15 + 1e-13 * abs(expected)`.
It was not changed after examining the outputs.

`TAC_C078` retained its numerical rejection under the near-zero proportional
residual setting. Its accepted posterior is null; raw diagnostics are retained.
It remains in the catalog and is not counted among the 134 valid TAC vectors.

The separate GHQ research extension recomputed both references, using the same
full-vector comparison criterion. Maximum absolute differences were `1.11e-16`
for OXC and `2.78e-17` for TAC. The full GHQ sensitivity matrix was not rerun for
this code delivery. Native convergence decisions and the separate
`formal_qualified=false` / `absolute_error_certified=false` fields remain unchanged.

Additional checks passed: 232 public-package regression tests; 13 adapter
contract tests and four numerical tests; and 11 GHQ contract tests, including
relocated execution without author-local source or expected-result access.
These checks concern numerical reproduction, not clinical validity or a proof
of exact integration. Other operating systems and dependency versions have not
been included in this replay.

## Public GHQ Sensitivity Sample

Research GHQ `2026.10.18.2` was published in commit
`30555f53ce0d4583c0eefdc7f65176ed04676634` and fetched into a new GitHub clone.
Six prespecified cases were computed from that public checkout with the pinned
environment above. No local source overlays, saved posterior answers or old
checkpoints were used. All 349 tracked files were unchanged after execution.

| Case | Coverage | States | Elapsed seconds | Max. absolute difference from formal raw R1 |
|---|---|---:|---:|---:|
| OXC_C003 | Reference | 4 | 0.969 | 0 |
| OXC_C024 | Clearance IIV; GHQ02 supplement | 4 | 1.594 | 0 |
| OXC_C025 | Clearance IIV; GHQ02 supplement | 4 | 1.594 | 0 |
| OXC_C083 | Three-dose candidate window | 8 | 1.266 | 0 |
| TAC_C134 | One-dose candidate window | 2 | 7.656 | 0 |
| TAC_C055 | Central-volume IOV | 4 | 108.406 | 0 |

All six attempts passed native diagnostic checks. All 26 unmodified posterior
components exactly matched the archived GHQ first repeat. Direct comparison of
the raw vector with the normalized five-repeat mean also passed the componentwise
criterion above; a separately normalized fresh-versus-mean comparison is retained
as an additional check. Formal comparators were read after each calculation exited.
Full vectors, state labels, differences, hashes, budgets and environment are in
[ghq_replay_sample.json](ghq_replay_sample.json), which inference does not read.

Five cases used `run_ghq_demo.py run`; TAC_C134 used the complete input/profile
script produced by `export-python`. For example, from the repository root:

```text
python -B examples/manuscript_current/run_ghq_demo.py run OXC_C024 --output new_oxc_c024
python -B examples/manuscript_current/run_ghq_demo.py export-python TAC_C134 --output tac_c134.py
python -B tac_c134.py --repo . --output new_tac_c134
```

The same public checkout's portable reader suite passed 12 tests; one symlink
test was skipped because this Windows account could not create symlinks.
It covers all 218 input/profile bindings and five fixed plans, inert inspection,
complete exports, output protection and mocked failure handling. Two opt-in
OXC smoke calculations are separate from the six-case comparison above.

```text
python -B -m unittest discover -s examples/manuscript_current -p test_ghq_reader.py -v
```

Set `GHQ_TEST_RUN_OXC=1` to include the two smoke calculations; otherwise those
are skipped. This is a sampled replay, not a new full-matrix or five-repeat run.
The five expensive TAC fixed-grid cases, including C079, are supplied and
source/profile-checked but were not recomputed in this public sample. TAC_C078
retains its historical failure declaration and was not among these six attempts.
Measured times describe this host/run, not runtime guarantees. Later documentation
and evidence commits do not change the tested numerical code.
