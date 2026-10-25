# Migration From 0.1.4

Version 0.2.0 changes the API while retaining the distribution name
`pmx-discrete-posterior` and MIT license. Use separate virtual environments
for the new package and legacy manuscript work. Installing one version over
the other does not migrate callers.

## API Changes

| Old API or behavior | v0.2.0 |
|---|---|
| `compute_laplace_posterior`, `OneEtaLaplaceConfig` | `compute_posterior` with the seven keyword groups below. |
| `compute_multieta_posterior`, `MultiEtaLaplaceConfig` | The same entry point; one pipeline handles all active-effect dimensions. |
| `OxcOneEtaDoseHistoryModel`, `build_oxc_dose_history_states` | Explicit model, individual data and candidate event mappings; no label-to-dose parser. |
| Configured covariance and residual fields | Parameter values/units plus named covariance references and residual SD references in `model`. |
| `result.posterior` and old diagnostics objects | Dictionary `result["state_results"]`; check overall `result["status"]` first. |
| Reporting, `NoCallRule`, case cards, curve payloads and offline reweighting helpers | Not exported or replaced in this version. There is no built-in no-call or clinical classification rule. |
| Optional SciPy installation | SciPy >=1.16.2,<2 is required, with Python >=3.11 and NumPy >=1.26. |

The new call is:

```python
result = compute_posterior(
    model=model,
    individual_data=individual_data,
    observations=observations,
    candidate_states=candidate_states,
    priors=priors,
    integration_method="laplace",
    numerical_options=None,
)
```

See the [runnable example](../README.md#minimal-call), [API](api.md) and
[model fields](models.md). There is no compatibility wrapper.

## Migrate A Model

1. Put parameter values and units in `model.parameters`. Supply parameter
   relationships and a deterministic predictor separately.
2. Declare IIV and IOV covariance references, or `None` for absent effects.
   Declare `random_effects_dependence="independent_blocks"` and residual
   `dependence="conditionally_independent"` explicitly.
3. Move covariates, occasions, shared dosing and initialization to
   `individual_data`. Create observation records with IDs and units.
4. Translate each candidate to explicit dose events. Keep IDs, events and
   priors paired; never infer doses by slicing `omega_01` or other labels.
5. Supply normalized priors for every candidate. Choose the integration
   method and supported numerical settings explicitly.
6. Check complete vectors and diagnostics against an appropriate reference
   before adopting the migrated model.

Not every old configuration is supported. v0.2 has normal additive or
proportional residual error, not combined error. Its built-in linear predictor
defaults to parameter switches, doses, then concentration readout at coincident
times. The boundary-patched build also supports explicit `before_events`, used
by the manuscript TAC mapper. Check boundary samples, initialization, units and
covariance ordering, not only a version string.

A numerical failure leaves positive-prior posteriors unavailable. It must not
be converted into a zero likelihood or a normalized subset. Removal of old
reporting or parsing code is not a claim that those historical implementations
were repaired.

## Reproduce The Manuscript

For the current manuscript inputs with the repaired 0.2.0 package, use the
[scenario reader](scenario_examples.md). The current external comparison uses
[R aghq](manuscript_aghq.md), documented separately and not bundled as a package
backend. The former Python `ghq` method and its settings are no longer supported.
The catalog JSON inputs are shared; their old Python runners
are not current-API callers.

### Historical October 18 Snapshot

The tag `manuscript-20261018` preserves the pre-v0.2 source snapshot
(commit prefix `408f28f`). Use it for **both**
`examples/manuscript_current` and `examples/manuscript_demos` in that historical
checkout. Do not run them against the v0.2 source or in its
virtual environment.

From a parent directory, create a separate checkout and environment:

```bash
git clone --branch manuscript-20261018 --single-branch https://github.com/Plasiajb/pmx-discrete-posterior.git pmx-manuscript
cd pmx-manuscript
python -m venv .venv-manuscript
```

Use Python 3.12 for this environment. Activate with
`.venv-manuscript\Scripts\Activate.ps1` in PowerShell or
`source .venv-manuscript/bin/activate` on POSIX. For the historical manuscript
catalog:

```bash
python -m pip install -r examples/manuscript_current/requirements.txt
python -B examples/manuscript_current/run_demo.py list
python -B examples/manuscript_current/run_demo.py show OXC_C003
python -B examples/manuscript_current/run_demo.py run OXC_C003 --output reviewer_oxc_r01
```

The output directory must be new. Follow that checkout's
`examples/manuscript_current/README.md` for Laplace and its
`research_ghq/README.md` for the separate GHQ workflow. For the historical
October 1 two-case demos, follow `examples/manuscript_demos/README.md` and
its own requirements in a dedicated environment.

The original 0.1.4 wheel is not interchangeable with this manuscript source
tag. Results from that workflow retain their historical identity. Later c026
and TAC-boundary experiments use separately recorded sources; the current
reader release does not relabel them. Saved NONMEM comparators in legacy demos
are not fresh NONMEM executions.
