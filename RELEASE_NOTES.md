# Release Notes

## v0.1.1 - Controlled-access manuscript release

Date: 2026-06-04

This private/controlled GitHub release archives the `pmx-discrete-posterior`
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

This is a controlled-access research prototype release for manuscript review
and collaborator audit under the included controlled-access license. It is not a
clinical decision-support tool, not a NONMEM replacement, and not an open-source
or public PyPI release.

### Known Limitations

- DOI/institutional archiving, PyPI publication, and any public/open-source
  relicensing remain manuscript-submission decisions.
- NONMEM runtime, compiler, and licensed local-environment artifacts are not
  redistributed.
- The package calculations are conditional on the supplied model, candidate
  states, priors, variability assumptions, observations, and reference-route
  provenance.
