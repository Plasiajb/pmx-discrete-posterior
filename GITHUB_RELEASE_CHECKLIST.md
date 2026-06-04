# GitHub Release Checklist

This checklist records the GitHub release procedure
for `pmx-discrete-posterior==0.1.3`.

## Pre-Release Checks

Run from the package root:

```bash
python -m pip install -e ".[scipy,plots,test]" build twine
python -m compileall -q src tests examples
python -m unittest discover -s tests -v
python -m build --sdist --wheel
python -m twine check \
  dist/pmx_discrete_posterior-0.1.3-py3-none-any.whl \
  dist/pmx_discrete_posterior-0.1.3.tar.gz
```

Expected release assets:

```text
dist/pmx_discrete_posterior-0.1.3-py3-none-any.whl
dist/pmx_discrete_posterior-0.1.3.tar.gz
dist/SHA256SUMS.txt
```

## Local Tag

Use the formal package tag for the GitHub release:

```bash
git tag -a v0.1.3 -m "pmx-discrete-posterior v0.1.3"
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
git push origin v0.1.3
```

## GitHub Release

Create the GitHub release from the pushed tag. With GitHub CLI:

```bash
gh release create v0.1.3 dist/* \
  --title "pmx-discrete-posterior v0.1.3" \
  --notes-file RELEASE_NOTES.md \
  --verify-tag
```

If using the GitHub web UI, create the release from tag `v0.1.3`, paste
`RELEASE_NOTES.md`, and upload the two files in `dist/` plus
`dist/SHA256SUMS.txt`.

## Boundary

This release uses the included MIT License. Do not describe it as public PyPI,
Zenodo DOI, or clinical decision-support software unless those routes are
separately approved and completed.
