# Examples by package version

For the **2026-10-25 reader release of 0.2.0**:

- `posterior_reference.py`: OXC/TAC reference posteriors using the seven-group API.
- `posterior_scenario.py`: select a reference or sensitivity case from the
  218-input catalog, assemble its inputs and call the installed Laplace package.
  See [commands and inputs](../docs/scenario_examples.md).

The external R `aghq` comparison is described in
[dependencies, settings and acceptance](../docs/manuscript_aghq.md);
no R execution code is bundled here. Previous Python GHQ implementations are
available only in history/backups, not in the current package.

The catalog and input JSONs under `manuscript_current/` are reused as scientific
inputs. The executable scripts in that directory retain the old API and must
not be used as current-package entrypoints.

The old coupled Laplace/GHQ demo bundle has been removed from the current
`manuscript_demos/` directory. Reproduce its results in a separate historical
checkout and environment:

```bash
git clone --branch manuscript-20261018 https://github.com/Plasiajb/pmx-discrete-posterior.git manuscript-replay
cd manuscript-replay
```

Then follow `examples/manuscript_current/README.md` there, using Python 3.12
and its pinned requirements. The tag resolves to
`408f28f4812040db5ad553c6295041e3bd2ecd66`.
The repository is public.

That tag's outputs are historical evidence, not validation receipts for the
current reader release. Preserve the source identities of each experiment.
