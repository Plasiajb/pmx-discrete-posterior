# Release checklist

1. Confirm the target repository and its intended visibility. The repository
   is public as of 2026-10-25; PyPI publication is not part of this delivery.
2. Preserve the prior manuscript implementation at `manuscript-20261018`.
   Normally, do not move existing tags or rewrite published commits. The
   maintainer-authorized 2026-10-25 reader republication retains version 0.2.0:
   back up the original tag, release metadata and all attachments before
   updating v0.2.0. Keep the old commit recoverable, use an exact expected-old
   tag lease, and disclose the replacement in the release notes.
3. Run portable component tests, both references and representative sensitivity
   cases from the installed package. Verify production modules and scientific
   inputs against the accepted source. Confirm old Python GHQ is absent and
   its method is rejected. External R `aghq` is documented, not bundled or
   tested as part of this package release.
4. Review the Git diff and distribution file lists for credentials, personal
   paths, internal records, caches and unrelated artifacts.
5. Build wheel and sdist from the release source. Install each into a clean
   environment, check import origin, run tests without source-path overrides,
   and verify all OXC/TAC state probabilities. Record failures as well as passes.
6. Calculate SHA-256 values for the actual tested archives. Do not reuse hashes
   from an earlier build, even when the version number is unchanged.
7. Commit and push normally, tag the tested commit, and attach the wheel,
   sdist, checksums and concise validation report to its GitHub Release.
   For the authorized same-version replacement, verify that the main branch,
   replacement tag, source archive and wheel identify the same tested source;
   remove any stale attachment only after its verified backup exists.
8. Download the attachments and compare checksums. Clone the published tag
   into a new directory, install and rerun both reference examples.
9. Verify the remote commit, tag and repository visibility. Record any CI
   results as observed, distinguishing local verification from hosted CI.

See [CONTRIBUTING.md](CONTRIBUTING.md) for build and test commands and
[docs/validation/README.md](docs/validation/README.md) for release evidence.
