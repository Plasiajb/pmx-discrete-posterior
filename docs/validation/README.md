# Validation evidence

## Reader Release (2026-10-25)

The reader distribution includes c026 recovery, the TAC observation-phase patch
and portable scenario entrypoints without changing package version `0.2.0`.
The preceding release assets and tag bundle have been backed up and hash-checked,
not relabelled. Version and filename alone cannot identify the build.

The [v0.2.0 release](https://github.com/Plasiajb/pmx-discrete-posterior/releases/tag/v0.2.0)
carries fresh `VALIDATION.json`, `VALIDATION.md` and `SHA256SUMS.txt` for
the rebuilt wheel and source archive. They identify the exact build/tag/main
commit, artifact hashes, Python/dependency versions, commands, test counts,
reference vectors and verification scope. Verify downloaded hashes and use a fresh virtual environment, or
both `--force-reinstall` and `--no-cache-dir` for a same-version reinstall.

## Checks And Reproduction

Prior c026 acceptance includes component tests and complete OXC/TAC reference
calculations. The tests cover input validation, model prediction, likelihoods,
optimization, gradients and Hessians, analytic integrals, posterior
normalization, and explicit numerical failure.

The reference checks compare all four labelled states, their events and
priors, log marginal likelihoods, and posterior probabilities. Absolute
tolerances are `1e-6` for OXC log marginal likelihoods, `1e-3` for TAC log
marginal likelihoods, and `5e-6` for posterior probabilities in both models.
Reference input JSON files are hash-protected; they are inputs, not cached
answers returned by the examples.

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for portable test, build, and
installed-package commands. Tests run from source and tests run against an
installed archive are distinct checks. Installation verification must remove
pytest's `pythonpath` override and confirm that the package comes from the
new environment's `site-packages`.

## Prior Acceptance Records

- [c026 acceptance note](c026_repair_20261021.md),
  [source evidence](c026_source_evidence.json) and
  [installation evidence](c026_installation_evidence.json): targeted repair and
  earlier clean-install checks for their recorded source/artifact identities.
  Their counts and hashes are not the new release receipts. The source regression
  combines a full-suite run with an affected-module rerun; overlapping counts
  must not be added together.
- [Pre-c026 repair note](repair_20261021.md) and
  [installation evidence](repair_installation_evidence.json): historical
  development-version checks, not c026 release validation.
- [Baseline installation evidence](baseline_installation_evidence.json):
  pre-packaging acceptance for the original distribution, not the repaired build.

Historical notes and JSON receipts retain their original scope, hashes and
results. Their repair-branch or unreleased wording describes that earlier
acceptance stage, not the current distribution. Do not overwrite these records
or reuse their hashes for rebuilt release archives. Local logs and recoverable
development snapshots are retained by the maintainers, not distributed here.

## Scope

Reference agreement is numerical verification, not a clinical evaluation.
The later formal manuscript experiments covered 83 OXC and 135 TAC inputs,
including their references, with five attempts per input. Their outcomes are
summarized below. These are previously completed experiments, not fresh tests of
the reader release. Their exact package and bridge snapshots remain distinct.

| Formal route | Attempts | Successful attempts | Recorded failure |
| --- | ---: | ---: | --- |
| Python Laplace, OXC | 415 | 415 | None |
| Python Laplace, TAC | 675 | 670 | C078, five attempts |
| External R aghq, OXC | 415 | 410 | C026, five attempts |
| External R aghq, TAC | 675 | 670 | C078, five attempts |

The OXC and retained TAC Laplace computations used the c026 snapshot;
TAC C107/C110 used the boundary-patched snapshot. External R TAC likewise
combines retained and boundary-patched attempts. No old outputs are relabelled
as calculations from the reader build. See [source and acceptance records](manuscript_sources.md).
Fresh reader-release tests and sampled reruns are listed separately in the
release's `VALIDATION.json`, with their actual source and artifact hashes.

Precision recovery retains failed-optimizer diagnostics and explicit warnings.
Integral success does not establish optimizer convergence, a global mode or
exact nonlinear integration. Explicit stationarity tolerances remain binding;
the normal BFGS default is `gtol=1e-6`. No automatic optimizer/integration
fallback or partial posterior normalization is introduced. See the
[acceptance contract](../api.md#precision-limited-recovery).

The previous Python GHQ implementations have been removed from the current
release. They are distinct from the manuscript's external R `aghq` route.
For external R results, order-change thresholds are diagnostic only; valid complete
outputs are not rejected for exceeding them. Actual numerical failures remain
failures. See [external R comparison](../manuscript_aghq.md).
