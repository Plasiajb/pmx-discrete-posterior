# Contributing

Use a source checkout and a fresh Python >=3.11 environment. Keep changes
focused, preserve the MIT copyright notice, and include a test for a changed
behavior. The [API](docs/api.md) and [model contract](docs/models.md) describe
the supported inputs; new scientific assumptions need an explicit proposal
and evidence before implementation.

## Develop And Test

With the environment activated, from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not slow"
python -m pytest -m slow
```

The standard `-m "not slow"` selection excludes slow tests. The portable suite covers input
validation, model behavior, analytic integrals, Hessian checks, failure
propagation and reference calculations. Slow cases, including full TAC
calculations, are selected separately. To run both groups explicitly:

```bash
python -m pytest -m "slow or not slow"
```

These commands use pytest directly; internal development histories and
author-specific paths are not required. Do not count deselected tests as
passed, or treat a successful budget-rejection test as integration accuracy.

For numerical changes, preserve failure cases and compare complete candidate
vectors and log marginal likelihoods, not only the highest-probability state.
Do not relax reference tolerances to obtain a pass. A reference input change
needs separate review and provenance, not a silent fixture replacement.

For c026 recovery changes, test both omitted and explicitly supplied
`optimization.stationarity_tolerance`, including explicit `1e-5`. Preserve the
failed optimizer/derivative diagnostics and recovery warnings; integral acceptance
must not imply optimizer convergence. Keep the unchanged-endpoint eligibility,
curvature-scale and local consistency checks, with no alternate optimizer or
partial posterior normalization. See the [API contract](docs/api.md#precision-limited-recovery).

The [prior c026 acceptance](docs/validation/c026_repair_20261021.md) covers five
targeted ten-case OXC passes, not a full matrix; TAC c078 still fails. Its source
regression result combines a full-suite run and an affected-module rerun, not
one all-green run. Preserve source snapshots, historical JSON receipts and
earlier failed-run records; do not replace them or add overlapping test counts.

## Build

From the source root:

```bash
python -m pip install build
python -m build --sdist --wheel
```

Build isolation installs the declared setuptools build dependency. The wheel
contains the runtime package; the source archive also supplies examples,
portable tests and reference inputs. Artifacts are written to `dist/`.

Check the wheel in a fresh environment from a directory outside the source
checkout so that the source tree cannot mask the installed package:

```bash
python -m pip install --force-reinstall --no-cache-dir /path/to/dist/pmx_discrete_posterior-0.2.0-py3-none-any.whl
python -c "from importlib.metadata import version; import pmx_discrete_posterior as p; print(version('pmx-discrete-posterior'), p.__file__)"
```

The reader distribution dated 2026-10-25 retains version `0.2.0` and
replaces the previous release assets. The previous five assets and tag bundle
have been backed up and hash-checked. Keep backups separate and retain their
original checksums and provenance; never relabel them as repaired builds.
Build the release from the exact clean commit used by its tag/main, recording
that commit and artifact hashes in the new release receipts. Version and
filename alone cannot distinguish the builds. Use a fresh environment, or
both `--force-reinstall` and `--no-cache-dir` when reusing one. Test each artifact
with its matching source/tests; an old local v0.2.0 tag is not proof of identity.
Replace the artifact path with your build location. Extract the matching source
archive into a separate directory, change to that directory, install test
dependencies and check the installed wheel without pytest's source-path setting:

```bash
python -m pip install pytest mpmath
python -m pytest -o pythonpath= --require-installed -m "not slow"
python -m pytest -o pythonpath= --require-installed -m slow
python examples/posterior_reference.py OXC
python examples/posterior_reference.py TAC
```

Keep `PYTHONPATH` unset. Repeat in another fresh environment with the source
archive installed instead of the wheel. Record the artifact hashes, Python
and dependency versions, commands and actual outcomes. The
[release validation page](docs/validation/README.md) distinguishes release receipts from prior acceptance;
local success is not a claim that CI or every supported environment passed.

Attach new `VALIDATION.json`, `VALIDATION.md` and `SHA256SUMS.txt` to the
replacement release after fresh wheel/source-distribution checks. Record the
actual outcomes and hashes; do not copy prior counts into them. The
[c026 installation receipt](docs/validation/c026_installation_evidence.json)
is prior acceptance for its recorded artifacts, not the rebuilt release.
The [pre-c026 installation receipt](docs/validation/repair_installation_evidence.json)
is historical evidence only. Preserve both records unchanged.

## Report Problems

Open a repository issue if you have access. Include the package version and
commit or artifact hash, environment, minimal synthetic inputs, command,
expected behavior, actual status and relevant diagnostics. For a numerical
failure, retain all candidate rows and priors. Exclude credentials, identifiable
patient data and machine-specific private paths. See [SECURITY.md](SECURITY.md)
for sensitive reports.

The manuscript runners require the [legacy tag and environment](docs/migration.md#reproduce-the-manuscript).
Do not update their expected results as a side effect of a v0.2 change.
