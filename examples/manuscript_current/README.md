# Manuscript Input Catalog

This directory retains 218 hash-protected synthetic input records: 83 OXC and
135 TAC, including the C003 reference for each model. The current package
entrypoint is [posterior_scenario.py](../posterior_scenario.py):

```bash
python examples/posterior_scenario.py list --model TAC
python examples/posterior_scenario.py run OXC C003 --output oxc-reference.json
python examples/posterior_scenario.py run TAC C107 --output tac-c107.json
```

Run these commands from the source root with the matching 0.2.0 package
installed. See the [reader guide](../../docs/scenario_examples.md) for input
units, mapping, numerical options and failure handling. The reader calls the
installed package; it does not return saved results.

## Historical Files

`run_demo.py`, `model_adapters/` and their validation records describe the
older Laplace API. They are not current-package callers. Use the
[historical checkout](../../docs/migration.md#historical-october-18-snapshot)
to inspect or run that implementation.

The former `research_ghq/` and `run_ghq_demo.py` have been removed from the
current tree. Historical validation files retain their original claims and
source identities; they do not certify this release. The current manuscript's
[external R aghq comparison](../../docs/manuscript_aghq.md) is documentation
only here, not a package backend.
