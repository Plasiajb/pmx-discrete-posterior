# pmx_discrete_posterior Package Status - updated 2026-06-04

## Status

`pmx_discrete_posterior` is a project-local Python package for Laplace posterior probabilities over predefined discrete states in pharmacometrics.

Use current manuscript wording for the main validation:

```text
Python-based Laplace framework/prototype package, with OXC same-assumption validation focused on the one-ETA case and a supplementary multi-ETA stress test.
```

Do not call it a mature general-purpose public package, an open-source package, a public GitHub/PyPI/DOI release, a NONMEM replacement, or a clinical decision-support tool.

## 2026-06-04 Release Archive Update

The package root is now a local git repository:

```text
path: D:/AI-ready PMX benchmark engine/pmx_discrete_posterior
branch: main
release tag: v0.1.1-manuscript-phase112
release commit: recorded in the Phase112 RELEASE_INFO.md and SHA256SUMS.txt files
previous tag: v0.1.1-manuscript
remote: none configured
version: 0.1.1
license metadata: LicenseRef-Proprietary-Research-Prototype
standalone LICENSE file: absent
```

A Phase112 controlled-access local manuscript/reviewer release archive was created:

```text
archive_root: JPKPD_IMA_Laplace_Manuscript_20260521/software_releases/phase112_pmx_discrete_posterior_v0.1.1_local_release_20260604/
release_info: RELEASE_INFO.md
sha256_manifest: SHA256SUMS.txt
wheel: dist/pmx_discrete_posterior-0.1.1-py3-none-any.whl
sdist: dist/pmx_discrete_posterior-0.1.1.tar.gz
git_bundle: source_archives/pmx_discrete_posterior_v0.1.1-manuscript-phase112.git.bundle
source_zip: source_archives/pmx_discrete_posterior_v0.1.1-manuscript-phase112_source.zip
```

Phase112 verification results:

```text
compileall: exit code 0
unittest discover: exit code 0; 24 tests OK
import check: exit code 0; __version__ == 0.1.1
wheel/sdist build: exit code 0
git bundle: exit code 0
git source archive: exit code 0
```

Key Phase112 SHA-256 values:

```text
wheel: see the current Phase112 SHA256SUMS.txt manifest
sdist: see the current Phase112 SHA256SUMS.txt manifest
git_bundle: see the current Phase112 SHA256SUMS.txt manifest
source_zip: see the current Phase112 SHA256SUMS.txt manifest
```

Boundary: Phase112 supports local controlled-access manuscript/reviewer release/archive wording. It does not support public release, open-source release, PyPI/TestPyPI release, Zenodo/DOI release, or external journal availability wording until the author-approved repository/archive route and license terms are finalized.

## 2026-06-04 GitHub Release Update

The package has been released through a private/controlled GitHub release for
manuscript review, collaborator audit, and reproducibility assessment. The
formal package tag for this GitHub release is `v0.1.1`, while the older
`v0.1.1-manuscript-phase112` tag remains the local manuscript/reviewer archive
tag.

Release files added:

```text
LICENSE
RELEASE_NOTES.md
CITATION.cff
MANIFEST.in
GITHUB_RELEASE_CHECKLIST.md
.github/workflows/ci.yml
```

Local release checks completed after these additions:

```text
compileall: exit code 0
unittest discover: exit code 0; 24 tests OK
wheel/sdist build: exit code 0
twine check dist/*: PASSED
sdist examples inclusion: confirmed
```

The included `LICENSE` is a controlled-access research prototype license, not
an open-source license. The GitHub release is private/controlled, not public.
Zenodo/DOI archiving, PyPI/TestPyPI publication, and any open-source relicensing
remain separate author decisions.

## Public API

- `DiscreteState`
- `LaplaceConfig` / `OneEtaLaplaceConfig`
- `compute_posterior` / `compute_laplace_posterior`
- `compute_curve` / `build_laplace_curve_payload`
- `laplace_state_diagnostics`
- `OxcSingleEtaModel` / `OxcOneEtaDoseHistoryModel`
- `build_oxc_dose_history_states`
- `MultiEtaLaplaceConfig`
- `compute_multieta_posterior`
- `gauss_hermite_posterior`

## Scientific Boundary

- Supported in the OXC main validation path: scalar observation, one ETA, additive residual SD, user-provided prediction model, finite discrete states, structural-zero prior policy.
- Supported experimentally for supplement stress testing: multiple ETAs, multiple observations, full covariance matrix, and combined additive/proportional residual SD.
- OXC example: native steady-state dose-history model with latest-to-older binary state semantics.
- Not supported now: automatic NONMEM/Pharmpy model import or high-dimensional adaptive integration.
- NONMEM remains an optional licensed reference path outside the package core.

## Diagnostics

The package exports per-state:

- `eta_hat`
- `pred_at_eta_hat`
- `psi_hat`
- `hessian`
- `laplace_log_likelihood`
- `laplace_posterior`
- `PMIX` as a plot-compatibility alias only
- `valid_laplace`
- `indeterminate`
- `warnings`

Non-positive/non-finite Hessian, optimizer-boundary hits, or optimizer failure mark the result as indeterminate.

## Verification

Latest Phase 6.5D verification executed on 2026-05-22 from `D:/AI-ready PMX benchmark engine/pmx_discrete_posterior`.

Current package metadata:

```text
name: pmx-discrete-posterior
version: 0.1.1
runtime __version__: 0.1.1
source_manifest_rows: 13
source_manifest_scope: non-cache source/test/example files, excluding mutable status notes
source_manifest_csv_sha256: 10e2014b6fe6ab1068bea3a936aaa86f922391ac13e0e27a5ac27d5e6b3fd860
```

```text
python -m unittest discover -s tests
```

Result: 13 tests passed.

Log: `JPKPD_IMA_Laplace_Manuscript_20260521/logs/phase65d_pmx_discrete_posterior_unittest_20260522.log`.

```text
python -m compileall -q src tests examples
```

Result: passed.

Log: `JPKPD_IMA_Laplace_Manuscript_20260521/logs/phase65d_pmx_discrete_posterior_compileall_20260522.log`.

```text
python examples/oxc4_pmix_dv_plot.py --output-dir examples/oxc4_pmix_dv_output --points 121
```

Result: generated CSV, PNG, and summary JSON.

Log: `JPKPD_IMA_Laplace_Manuscript_20260521/logs/phase65d_pmx_discrete_posterior_oxc_example_20260522.log`.

```text
python -m pip install -e . --no-deps
```

Result: installed editable package `pmx-discrete-posterior==0.1.1`.

Log: `JPKPD_IMA_Laplace_Manuscript_20260521/logs/phase65d_pmx_discrete_posterior_editable_install_20260522.log`.

```text
python -c "import pmx_discrete_posterior as p; print(p.__version__)"
```

Result: `0.1.1`.

Log: `JPKPD_IMA_Laplace_Manuscript_20260521/logs/phase65d_pmx_discrete_posterior_import_version_20260522.log`.

GUI4 old implementation equivalence probe:

```text
points: 8
max_posterior_delta: 0.0
max_eta_hat_delta: 0.0
max_loglik_delta: 0.0
any_indeterminate: False
```

`python -m pytest -q` was not available because the current Python environment has no `pytest` installed.

Supplement stress test:

```text
python examples/multieta_stress_test.py --gh-nodes 35
```

Result:

```text
output_dir: JPKPD_IMA_Laplace_Manuscript_20260521/supplement/experiments/multieta_stress_test_20260521
model: synthetic oral one-compartment PK with ETA(CL), ETA(V), four observations, combined additive/proportional RUV
reference: tensor-product Gauss-Hermite numerical integration, 35 nodes per dimension
Laplace MAP: state11
Quadrature MAP: state11
MAP agreement: true
max_abs_posterior_diff: 0.00016427367801286064
laplace_indeterminate: false
```

Log: `JPKPD_IMA_Laplace_Manuscript_20260521/logs/phase65d_pmx_discrete_posterior_multieta_example_20260522.log`.

Source manifest: `JPKPD_IMA_Laplace_Manuscript_20260521/tables/phase65d_package_source_manifest_20260522.csv`.

The historical Phase65D note above has been superseded by the Phase112 git and release-archive state. Use the Phase112 tag, archive root, `RELEASE_INFO.md`, and `SHA256SUMS.txt` manifest for current manuscript/reviewer reproducibility wording.

## Article Figure Rule

Future manuscript figures for the Laplace method should call `pmx_discrete_posterior` public APIs and should not copy the old GUI4 internal Laplace implementation.
