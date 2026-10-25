# Manuscript Laplace Scenario Reader

This experimental reader maps the manuscript's 218 frozen scientific inputs
(83 OXC, 135 TAC) to the installed `pmx-discrete-posterior` 0.2.0 API. It computes
one complete Laplace posterior per invocation. C003 is the reference in each
model; the other cases are sensitivity experiments, not different patients.

## Run

Use Python 3.11+ in an environment with the current package installed. From the
repository root or extracted source distribution:

```console
python -m pip install .
python examples/posterior_scenario.py list
python examples/posterior_scenario.py list --model TAC
python examples/posterior_scenario.py run OXC C003
python examples/posterior_scenario.py run OXC 3 --output oxc-c003-result.json
python examples/posterior_scenario.py run TAC C107 --output tac-c107-result.json
```

`run OXC` and `run TAC` select C003. Numbers and C-prefixed identifiers are
equivalent and case-insensitive. `list` reads only the catalog and never
integrates. TAC computations can be substantially more expensive than OXC;
there is no batch matrix or repeat mode. Output filenames must be new and their
parent directories must already exist. Omit `--output` to print JSON to stdout.
The module form `python -m examples.posterior_scenario ...` also works from the
source root.

The wheel supplies the framework, not these experimental data attachments. The
reader needs `examples/posterior_scenario.py`, `examples/scenario_inputs.py`,
`examples/reference_models.py`, and `examples/manuscript_current/catalog.json`
plus its `inputs/*.json`, supplied by the source distribution/repository. Copy
those files with the same relative layout for an isolated wheel replay. No old
`manuscript_current` executable code or test fixtures are required.

## Custom Inputs And Options

`--input` accepts a local JSON in `oxc_numeric_source_ruv_v1` or
`tac_resolved_v1` format. A catalog input is a complete schema example. For an
edited experiment, use a separate file and update all duplicated declarations
consistently (for example SD/variance, or both covariance declarations in OXC).
Do not edit the frozen catalog inputs. These are raw study schemas, not arbitrary
current-API model objects, NONMEM control streams, or a general model importer.

This runnable custom-input example reads the existing file without modifying it:

```console
python examples/posterior_scenario.py run OXC --input examples/manuscript_current/inputs/OXC_C003.json --output custom-oxc-result.json
```

For a modified file named `custom.json`:

```console
python examples/posterior_scenario.py run OXC --input custom.json --output custom-result.json
python examples/posterior_scenario.py run OXC C003 --numerical-options options.json --output configured-result.json
```

`options.json` may contain `null` or an explicit current-API Laplace options
object, for example `{"optimization":{"backend_options":{"maxiter":500}}}`.
Unsupported fields and GHQ options fail. Omitting the file passes
`numerical_options=None` exactly, as in the formal Laplace run. The reader never
expands validated defaults and passes them back as explicit settings. In
particular, explicitly supplying `stationarity_tolerance` is a user override,
not equivalent provenance to an omitted setting. Requested options are
recorded verbatim, and actual numerical diagnostics remain in the API result.

Custom mode cannot be combined with a case number. It records the actual file
SHA256 and `role: custom`, without claiming catalog verification. A copied TAC
`numeric_id` is source metadata, not a newly computed experiment identity; use
the actual byte hash to distinguish edits.

## Scientific Mapping

The mapper ports the validated boundary-study
`rerun_tac_boundary_20261022/laplace/shared/scenario_inputs.py` logic and reuses
the matching callbacks in `examples/reference_models.py`. The original study
directory is provenance only and is not a runtime dependency.

- Preserve source candidate order, explicit doses, observation order, covariates,
  units, priors (including exact zeros), and independent IIV/IOV blocks. No doses
  are inferred from manuscript labels or bitstrings; no priors are normalized.
- OXC uses source manuscript times, checks the redundant local-time metadata,
  retains clearance IIV and additive error, and distinguishes the shared dose
  from the earlier periodic dose. Its omitted observation phase retains the API
  default. There is no OXC IOV in this supported schema.
- TAC retains four IIV effects and CL/V1 IOV for every declared occasion, with
  proportional error. Every observation explicitly uses `before_events`, even
  away from boundaries. Occasion boundaries are the shared dose, nominal slots,
  and `max(48, max(times)+1, last_slot+1)`. Conditional steady state uses the
  first occasion's effects before the shared dose.
- Doses are mg and time is h. OXC concentrations are mg/L; TAC concentrations
  are ng/mL, using the source model's `1000/V1` observation coefficient. Source
  concentrations are not converted or relabeled.
- Covariance parameters preserve the source upper triangle. Only nonzero,
  same-sign unequal pairs within two ULP may be copied to the lower triangle;
  each changed pair and both original values are reported. Direct `build_inputs`
  emits `CovarianceRoundoffWarning`; `load_scenario` records these warnings in
  `mapping_adjustments`. Genuine asymmetry, cross-block dependence, unequal IOV
  blocks, and unresolved rank deficiency fail. No jitter or variance repair is
  performed. Strict zero-variance coordinates remain deterministic.

The output identifies the catalog case, original source/numeric IDs, memberships,
input SHA256 and catalog SHA256, requested options, dependency versions, and
mapping adjustments. It preserves every state's events, prior, log marginal
likelihood, posterior, status, failure reason, and full method diagnostics.
Nonfinite diagnostic numbers are represented as explicit JSON strings
`"Infinity"`, `"-Infinity"`, or `"NaN"`; invalid posterior probabilities are
rejected before serialization.

Exit codes: **0** complete valid posterior (or successful listing); **1** failed
computation/result contract with a complete candidate vector; **2** invalid
CLI/input/options-file/output request, reported on stderr. Failed calculations
never drop candidates or partially renormalize. All posterior probabilities are
unavailable on failure, except exact zero-prior skipped states retain zero.

## Framework And Experimental Sources

`src/pmx_discrete_posterior` is the general integration/model framework. These
example adapters and catalog/inputs are experiment-specific scientific sources;
they do not change the native mathematical implementation. No cached posterior
is read to compute results. This entrypoint is **Laplace only**, not NONMEM PMIX,
not legacy research GHQ, and not the separate formal external R `aghq` workflow.
The reference CLI's compact Laplace diagnostic output is unchanged.

Python callers can use `build_inputs(raw, model_name)` for the five scientific
keyword arguments, or `load_scenario(model_name, case_number=None,
input_json=None)` for `identity`, `inputs`, and `mapping_adjustments` (pass
`input_json` by keyword). `run_scenario(scenario, numerical_options=None)` runs
the latter through the API; pass `numerical_options` by keyword.

## Tests

From the checkout, the focused suite validates all input contracts/hashes, TAC
boundary phase, unsupported covariance/semantics, source immutability, custom
input and CLI behavior, defaults provenance, mocked failures, and one bounded
real OXC C003 calculation. It does not execute the full numerical matrix or
formal five-repeat studies.

```console
python -B -m pytest -c pyproject.toml -q -p no:cacheprovider tests/test_scenario_reader.py
```

That command explicitly selects checkout configuration. Clean-wheel validation
must instead use the isolated installed package and copied example/data files,
without a checkout `src` path override.
