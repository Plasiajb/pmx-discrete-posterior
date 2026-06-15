# GitHub Release Checklist

This checklist records the GitHub release procedure
for `pmx-discrete-posterior==0.1.4`.

## Pre-Release Checks

Run from the package root:

```bash
python -m unittest discover -s tests -v
python -m pip wheel . -w dist --no-deps --no-build-isolation
python -c "import setuptools.build_meta as bm; print(bm.build_sdist('dist'))"
```

Expected release assets:

```text
dist/pmx_discrete_posterior-0.1.4-py3-none-any.whl
dist/pmx_discrete_posterior-0.1.4.tar.gz
dist/SHA256SUMS.txt
dist/RELEASE_NOTES_v0.1.4.md
```

If `twine` is available locally, run:

```bash
python -m twine check dist/pmx_discrete_posterior-0.1.4-py3-none-any.whl dist/pmx_discrete_posterior-0.1.4.tar.gz
```

For the 2026-06-15 local release preparation, `twine` was not installed in the
active Python environment, so asset metadata was inspected directly instead.

## Local Tag

Use the formal package tag for the GitHub release:

```bash
git tag -a v0.1.4 -m "pmx-discrete-posterior v0.1.4"
```

## GitHub Remote

The intended repository URL is:

```text
https://github.com/Plasiajb/pmx-discrete-posterior
```

If the repository already exists:

```bash
git remote add origin https://github.com/Plasiajb/pmx-discrete-posterior.git
git push -u origin main
git push origin v0.1.4
```

## GitHub Release

Create the GitHub release from the pushed tag. With GitHub CLI:

```bash
gh release create v0.1.4 \
  dist/pmx_discrete_posterior-0.1.4-py3-none-any.whl \
  dist/pmx_discrete_posterior-0.1.4.tar.gz \
  dist/SHA256SUMS.txt \
  --title "pmx-discrete-posterior v0.1.4" \
  --notes-file dist/RELEASE_NOTES_v0.1.4.md \
  --verify-tag
```

If using the GitHub web UI, create the release from tag `v0.1.4`, paste
`dist/RELEASE_NOTES_v0.1.4.md`, and upload the wheel, source distribution, and
`dist/SHA256SUMS.txt`.

## Boundary

This release uses the included MIT License. Do not describe it as public PyPI,
Zenodo DOI, or clinical decision-support software unless those routes are
separately approved and completed.
