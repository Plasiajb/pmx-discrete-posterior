# Release Notes

## Current Source - TAC Observation Phase (2026-10-22)

- Optional per-record `event_phase` accepts `before_events` or `after_events`.
  The default remains parameter switch, due input, then concentration readout.
- `LinearPredictor` can read concentration before same-time switches and inputs,
  without shifting time or rescaling amounts. Undefined pre-switch readout at
  initialization is rejected. Custom predictors must honor explicit phases or
  reject unsupported requests.
- The TAC reference mapper explicitly uses `before_events`, matching this study's
  convention. OXC mapping, optimization, Hessian, Laplace, normalization and
  numerical thresholds are unchanged. No case-ID-specific branch is added.
- Package version remains 0.2.0. This is an additive source commit, not replacement
  of the existing v0.2.0 tag/assets or historical experiments. Install the patched
  checkout to use the new field. See [contract and scope](docs/observation_phase.md).

## v0.2.0 - c026 Repaired Distribution (2026-10-21)

This distribution replaces the original v0.2.0 release assets while retaining
package version `0.2.0`. The original five assets and tag bundle have been backed
up and hash-checked; they remain historical artifacts, not relabelled c026 builds.
Install from the repaired release or its matching tag/main source, not a
repair-only branch. Record the exact build commit and artifact SHA-256, not
version or filename alone. Use a fresh virtual environment, or reinstall with
both `--force-reinstall` and `--no-cache-dir`.

The replacement release carries fresh `VALIDATION.json`, `VALIDATION.md` and
`SHA256SUMS.txt` for the actual rebuilt wheel and source archive. These record
the tested source commit, artifact hashes and test results; prior acceptance
below is not a validation receipt for these new archives.

- Changes the Laplace acceptance path for narrowly eligible precision-limited
  search endpoints. Independent gradient schedules, strict Hessian checks,
  curvature-scale gradients and local objective probes must agree at the unchanged
  endpoint. This is not a claim that all acceptance rules are unchanged.
- An accepted integral has `status="success"` and
  `method_details.precision_recovery.status="accepted"`. Its nonconverged optimizer
  remains `optimization_status="failed"` in that record, with the original
  failure reason and warning. Failed derivative records are not relabelled successful.
- Explicit `optimization.stationarity_tolerance`, including explicit `1e-5`,
  is binding. Omission permits curvature-relative recovery only after strict
  eligibility checks; it is not equivalent to explicitly supplying the default.
- Keeps the prior repair's BFGS default `gtol=1e-6` under default settings and
  both `1e-3` Hessian/stationarity-impact thresholds. Scientific inputs, priors,
  full covariance handling and normalization are unchanged. No automatic alternate
  optimizer, multistart, endpoint move or partial normalization is added.

### Prior Acceptance And Limits

Prior c026 source validation records five targeted passes of ten originally failing OXC
cases, each with 10/10 cases and 40/40 candidate integrals successful. For c026,
maximum differences from an independent high-precision **local Laplace** reference
are `4.631482208861115e-8` in posterior probability and `2.357708501765643e-7` in
log marginal likelihood, not errors against exact nonlinear integration.
TAC c003/c036/c037 retain their baseline results; TAC c078 still fails with no
posterior normalization. These checks are not a full-matrix rerun.

The [prior c026 acceptance note](docs/validation/c026_repair_20261021.md),
[source receipt](docs/validation/c026_source_evidence.json) and
[installation receipt](docs/validation/c026_installation_evidence.json) retain
their original run identities, counts and hashes. The merged source result was
a full-suite run plus an affected-module rerun, not one all-green invocation;
overlapping installed suites must not be added together. Those records are
historical evidence, not the new release's artifact identities or final counts.
See the [validation index](docs/validation/README.md) for release-receipt scope.

## Historical 0.2.1.dev20261021 - Pre-c026 Numerical Repair

The following record describes the earlier development snapshot and its installed
artifacts. Its version label, c026 deferral and test counts are historical, not
the repaired distribution identity or its release acceptance.

Branch `repair/prior-coordinates-20261021` is an independent repair based on
v0.2.0. `0.2.1.dev20261021` is a development identifier only, not a new release,
tag or merge to the main branch. Published v0.2.0 artifacts remain unchanged;
full release acceptance is not claimed.

- Laplace optimization and full Hessian evaluation use covariance-standardized
  coordinates, retaining the original density callback and change-of-variables
  Jacobian. Initial points and returned modes remain in original effect coordinates.
- Optimization diagnostics retain SciPy termination details, the returned point
  and gradient, last/best primary objective evaluations and derivative failures.
- A fourth production change, in `interfaces/inputs/numerical_options.py`, sets
  the BFGS default to `min(1e-6, stationarity_tolerance / 10)`: `gtol=1e-6`
  with default `stationarity_tolerance=1e-5`. Explicit `gtol=1e-7` remains
  supported. The three earlier numerical repair files are unchanged by this update.
- Scientific models, inputs, priors, posterior normalization, the seven-group
  API and independent gradient, Hessian and stationarity acceptance gates are
  unchanged. The search stopping default changes as described above. No new
  Hessian or GHQ backend, R dependency or automatic retry is added.

Before the default change, a local TAC c036/c037 comparison recorded BFGS
precision loss at the former `gtol=1e-7` setting. Explicit `gtol=1e-6` returned
complete posteriors directly through `compute_posterior`: 4/4 candidates succeeded
in each case and passed the original independent checks. Each case was called
once per setting, four full calls total. This configuration evidence supports
the new default choice; it is not fresh whole-matrix or five-repeat validation
of the changed default.

Maximum posterior differences of `2.639003821e-9` (c036) and `5.871017728e-9`
(c037) are relative to previously independently validated external vectors,
not the historical `gtol=1e-7` calls, which returned no complete posterior.
Rankings and the historical 132/135 TAC first-pass count are unchanged.
OXC c026 and TAC c078 remain deferred.

The source default-update full regression recorded 1427 passes and two
Windows symlink-permission skips. Fresh isolated wheel and source-distribution
installations passed 1176 and 1165 tests respectively; both verified the OXC/TAC
reference examples. All 33 installed production files matched the repair source.
These overlapping test groups must not be added together. The installed checks
used Windows, Python 3.12.0, NumPy 2.5.2 and SciPy 1.16.2.
A fresh full-matrix external R `aghq` pass, followed after review by five Laplace
repeats, is planned, not completed. Full release acceptance remains outstanding.
See the [repair validation note](docs/validation/repair_20261021.md)
for artifact hashes, test identities and the scope of each check. Existing v0.2.0
documentation and legacy manuscript examples are retained.

## Original v0.2.0 - Unified Posterior API

The distribution name remains `pmx-discrete-posterior`, the Python import is
`pmx_discrete_posterior`, and the license remains MIT. The repository is private;
this version is distributed through GitHub and is not published on PyPI.

### Changes

- `compute_posterior` accepts seven keyword input groups and handles zero, one
  or multiple active random effects through one pipeline.
- Model inputs separate parameter relationships, prediction, IIV/IOV and
  residual error. Candidate histories use explicit dose events, not encoded IDs.
- Results retain all candidates, log marginal likelihoods and diagnostics.
  Positive-prior failures prevent posterior normalization; zero priors remain
  explicit skipped states with zero posterior probability.
- OXC and TAC reference examples compute complete Laplace posteriors from
  bundled inputs. The source archive includes the examples and portable tests;
  the wheel contains the runtime package.

### Compatibility And Scope

This is not API-compatible with 0.1.4. Old one-ETA/multi-ETA entry points,
reporting/no-call helpers and adapters are not exported. See the
[migration guide](docs/migration.md). Manuscript examples remain tied to the
`manuscript-20261018` source tag and a separate environment.

Python >=3.11, NumPy >=1.26 and SciPy >=1.16.2,<2 are required. SciPy is no
longer optional. Source builds require setuptools >=77.0.3.

Laplace is the maintained core. Prior-centered GHQ is retained with explicit
orders, tolerances and node budgets; external backends such as R `aghq` are
not integrated. TAC high-order GHQ accuracy has not passed acceptance.
Reference checks do not constitute a full sensitivity replay with v0.2.0.
Removing old interfaces isolates those paths; it does not certify repairs to
the old reporting or model implementations.

## Historical Releases

The records below describe their original releases, not the repository's
current visibility or the v0.2 API.

## v0.1.4 - Posterior evidence reporting layer release

Date: 2026-06-15

This GitHub release updates the open-source `pmx-discrete-posterior` research
package with a lightweight reporting layer for model-conditioned posterior
evidence summaries.

### Added

- `NoCallRule` and `DEFAULT_NO_CALL_RULE` for prespecified weak-separation
  reporting diagnostics.
- `summarize_posterior_evidence()` to convert posterior results into stable
  report objects containing MAP state, posterior margin, normalized entropy,
  no-call status, numerical warnings, state rows, and provenance-friendly
  metadata.
- `posterior_to_long_rows()` and `posterior_to_wide_row()` for audit tables.
- `build_case_card()` for manuscript/report wording that preserves the
  model-conditioned boundary.
- `threshold_sensitivity_panel()` and `reweight_prior_sensitivity()` for
  reporting-rule and prior-sensitivity analyses from existing posterior
  evidence.
- `build_provenance_manifest()` for compact reproducibility metadata.

### Changed

- The public API now exports the reporting-layer objects and helpers.
- The OXC PMIX-DV example test now writes to unique temporary directories to
  avoid stale Windows file-lock interference.
- README usage examples now include the reporting-layer workflow.

### Verification

- `python -m unittest discover -s tests -v`
- `python -m pip wheel . -w dist --no-deps --no-build-isolation`
- `python -c "import setuptools.build_meta as bm; print(bm.build_sdist('dist'))"`
- Wheel and sdist metadata inspected as version `0.1.4`.
- `dist/SHA256SUMS.txt` generated for release assets.

### Boundary

This release does not change the Laplace engine. It adds a reporting layer for
auditable posterior evidence summaries. It is not a clinical decision-support
tool, not a NONMEM replacement, not a public PyPI release, and does not include
private FOCE-like manuscript diagnostics that have not been promoted into the
public package API.

## v0.1.3 - Public open-source manuscript release

Date: 2026-06-04

This public GitHub release archives the open-source `pmx-discrete-posterior`
research package used for the manuscript validation analyses.

### Included

- One-ETA Laplace posterior calculation for finite predefined discrete states.
- OXC dose-history helper functions and PMIX-DV curve payload generation.
- Experimental multi-ETA posterior calculation API.
- Tacrolimus and multi-ETA stress-test examples used as supplementary evidence.
- Boundary-focused tests for posterior normalization, zero-prior handling,
  diagnostic flags, OXC state mapping, plotting examples, and stress-test
  reference agreement.

### Verification

- `python -m compileall -q src tests examples`
- `python -m unittest discover -s tests -v`
- `python -m build --sdist --wheel`
- `python -m twine check dist/*`

### Boundary

This is a public open-source research package release under the MIT License. It
is not a clinical decision-support tool, not a NONMEM replacement, and not a
public PyPI release.

### Known Limitations

- DOI/institutional archiving and PyPI publication remain manuscript-submission
  decisions.
- NONMEM runtime, compiler, and licensed local-environment artifacts are not
  redistributed.
- The package calculations are conditional on the supplied model, candidate
  states, priors, variability assumptions, observations, and reference-route
  provenance.
- Internal manuscript-status files are not included in the repository or source
  distribution for this release.
