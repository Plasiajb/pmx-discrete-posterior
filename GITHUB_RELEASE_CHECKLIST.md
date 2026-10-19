# Release checklist

1. Confirm the target repository and its intended visibility. The 0.2.0
   delivery keeps this repository private; PyPI publication is not part of it.
2. Preserve the prior manuscript implementation at `manuscript-20261018`.
   Do not move existing tags or rewrite published commits.
3. Run the portable component tests and both reference examples. Verify
   the production modules and reference inputs against the accepted source.
4. Review the Git diff and distribution file lists for credentials, personal
   paths, internal records, caches and unrelated artifacts.
5. Build wheel and sdist from the release source. Install each into a clean
   environment, check import origin, run tests without source-path overrides,
   and verify all OXC/TAC state probabilities. Record failures as well as passes.
6. Calculate SHA-256 values for the actual tested archives. Do not reuse hashes
   from an earlier build, even when the version number is unchanged.
7. Commit and push normally, tag the tested commit, and attach the wheel,
   sdist, checksums and concise validation report to its GitHub Release.
8. Download the attachments and compare checksums. Clone the published tag
   into a new directory, install and rerun both reference examples.
9. Verify the remote commit, tag and repository visibility. Record any CI
   results as observed, distinguishing local verification from hosted CI.

See [CONTRIBUTING.md](CONTRIBUTING.md) for build and test commands and
[docs/validation/README.md](docs/validation/README.md) for release evidence.
