# Release Notes

## v0.2.0 - Unified Posterior API

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
