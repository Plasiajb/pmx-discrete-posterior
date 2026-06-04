# GitHub Release Checklist

This checklist records the intended controlled-access GitHub release procedure
for `pmx-discrete-posterior==0.1.1`.

## Pre-Release Checks

Run from the package root:

```bash
python -m pip install -e ".[scipy,plots,test]" build twine
python -m compileall -q src tests examples
python -m unittest discover -s tests -v
python -m build --sdist --wheel
python -m twine check dist/*
```

Expected release assets:

```text
dist/pmx_discrete_posterior-0.1.1-py3-none-any.whl
dist/pmx_discrete_posterior-0.1.1.tar.gz
dist/SHA256SUMS.txt
```

## Local Tag

Use the formal package tag for the GitHub release:

```bash
git tag -a v0.1.1 -m "pmx-discrete-posterior v0.1.1"
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
git push origin v0.1.1
```

## GitHub Release

Create a controlled-access GitHub release from the pushed tag. With GitHub CLI:

```bash
gh release create v0.1.1 dist/* \
  --title "pmx-discrete-posterior v0.1.1" \
  --notes-file RELEASE_NOTES.md \
  --verify-tag
```

If using the GitHub web UI, create the release from tag `v0.1.1`, paste
`RELEASE_NOTES.md`, and upload the two files in `dist/` plus
`dist/SHA256SUMS.txt`.

## Boundary

This release uses the included controlled-access research prototype license. Do
not describe it as open source, public PyPI, Zenodo DOI, or clinical
decision-support software unless those routes are separately approved and
completed.
