# Validation evidence

Version 0.2.0 is checked with component tests and complete OXC/TAC reference
calculations. The tests cover input validation, model prediction, likelihoods,
optimization, gradients and Hessians, analytic integrals, posterior
normalization, and explicit numerical failure.

The reference checks compare all four labelled states, their events and
priors, log marginal likelihoods, and posterior probabilities. Absolute
tolerances are `1e-6` for OXC log marginal likelihoods, `1e-3` for TAC log
marginal likelihoods, and `5e-6` for posterior probabilities in both models.
Reference input JSON files are hash-protected; they are inputs, not cached
answers returned by the examples.

## Reproduce the checks

The [baseline installation evidence](baseline_installation_evidence.json)
records the accepted source before GitHub packaging. Its original wheel and
sdist hashes are distinct from the rebuilt release archives. It verifies
239 component tests in each of two wheel environments, and both complete
references under all four wheel/sdist installations. This is the same test
suite across environments, not 478 different tests.

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for portable test, build, and
installed-package commands. Tests run from source and tests run against an
installed archive are distinct checks. Installation verification must remove
pytest's `pythonpath` override and confirm that the package comes from the
new environment's `site-packages`.

The versioned GitHub Release carries `VALIDATION.json`, `VALIDATION.md` and
`SHA256SUMS.txt` for its actual rebuilt wheel and sdist. Those records identify
the tested Python/dependency combinations, test counts, full reference
vectors, archive hashes and verification scope. Local logs and recoverable
development snapshots are retained by the maintainers, not distributed here.

## Scope

Reference agreement is numerical verification, not a clinical evaluation.
The complete manuscript sensitivity matrix has not been rerun with 0.2.0.
Its earlier implementation and evidence remain at `manuscript-20261018`.

The retained prior-centered GHQ implementation is not an external R
`aghq` backend. OXC has a reference check; a TAC low-order calculation or
budget-rejection test does not establish high-order TAC GHQ accuracy.
See [GHQ scope](../ghq.md).
