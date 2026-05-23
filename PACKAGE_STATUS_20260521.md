# pmx_discrete_posterior Package Status - 2026-05-21

## Status

`pmx_discrete_posterior` is now a project-local editable Python package for Laplace posterior probabilities over finite discrete pharmacometric states.

Use current manuscript wording for the main validation:

```text
Python-based Laplace framework/prototype package, with OXC same-assumption validation focused on the one-ETA case and a supplementary multi-ETA stress test.
```

Do not yet call it a mature general-purpose public package.

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

The package root is not currently a git repository. Use the source manifest hash above as a local reproducibility checksum until a repository tag, release archive, or DOI is created.

## Article Figure Rule

Future manuscript figures for the Laplace method should call `pmx_discrete_posterior` public APIs and should not copy the old GUI4 internal Laplace implementation.
