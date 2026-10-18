# Formal GHQ Research Extension

This importable research namespace packages the frozen October 2026 OXC/TAC
GHQ computation. It is separate from `pmx_discrete_posterior`'s public Laplace
API. The old public `gauss_hermite_posterior` dimension limit is unchanged.
It is not a general arbitrary-model GHQ interface or an absolute-error certificate.

## Setup And Reader CLI

Use the complete source checkout and a clean Python 3.12 environment. From the
repository root, install the [pinned dependencies](../requirements.txt):

```text
python -m pip install -r examples/manuscript_current/requirements.txt
```

The pins are NumPy 2.5.2, SciPy 1.16.2 and mpmath 1.3.0; Python 3.12.0 is the
recorded reproduction interpreter. Neither NONMEM, Pharmpy, a compiler nor
installation of `pmx_discrete_posterior` is needed. Research GHQ `2026.10.18.2`
is separate from package `0.1.4`; the old release wheel/tag is not this snapshot.

**Cost warning:** GHQ is not the fast/default Laplace route. Inspect each case's
actual resource policy before computing. TAC worker allowances are cooperative,
not hard elapsed-time guarantees; fixed grids allow about 3-6 hours and billions
of points. OXC has no enforced seconds/point budget, and this wrapper imposes no
memory cap. The [resource table](#pinned-case-profiles) gives exact allowances.

Inspect and export without computing:

```text
python -B examples/manuscript_current/run_ghq_demo.py list
python -B examples/manuscript_current/run_ghq_demo.py show OXC_C024
python -B examples/manuscript_current/run_ghq_demo.py export-python OXC_C024 --output reviewer_oxc_c024.py
```

`list` covers all 218 designs; `show` prints the complete input dictionary,
unchanged pinned profile and applicable integration plan (`null` for adaptive
cases). `export-python` embeds those same complete inputs/profile/plan, with
`INTEGRATION_PLAN = None` where no fixed plan applies. Neither command reads
expected answers or computes GHQ. The exported script calls this checkout's
engine; it is not an independent implementation.

Two small OXC examples, each an explicit single-case computation:

```text
python -B examples/manuscript_current/run_ghq_demo.py run OXC_C024 --output reviewer_oxc_c024_r01
python -B examples/manuscript_current/run_ghq_demo.py run OXC_C025 --output reviewer_oxc_c025_r01
```

Alternatively execute the exported example against the selected checkout:

```text
python -B reviewer_oxc_c024.py --repo . --output NEWDIR
```

Both routes print the full `run_summary` JSON, including all state IDs, labels,
accepted probabilities and diagnostics, and write the native artifacts listed
below. `--output` is required: its parent must exist and the target must be new,
even if an existing target is empty. Paths resolve from the working directory;
source directories are protected. For export, the target is a new `.py` file;
for execution, it is a new directory. There is no GHQ batch command.

For an optional adaptive TAC example, `TAC_C134` allows 550,000,000 points and
1780 worker seconds (about 30 minutes), **not a runtime estimate or guarantee**:

```text
python -B examples/manuscript_current/run_ghq_demo.py show TAC_C134
python -B examples/manuscript_current/run_ghq_demo.py run TAC_C134 --output reviewer_tac_c134_r01
```

Exit `0` means successful inspection/export or diagnostic acceptance; `1` means
native nonacceptance; `2` means an argument/input/execution error; `130` means
interruption. Retain the native status and partial artifacts; no cached answer
or Laplace result replaces a failed GHQ computation.

## Pinned Case Profiles

Version `2026.10.18.2` covers 218 designed cases: 83 OXC and 135 TAC. Of these,
217 have historically accepted configurations; `TAC_C078` is retained without
one. These are historical counts, not outcomes of fresh calls. `TAC_C078` is
computed normally, neither skipped nor assigned an artificial failure or
substitute vector; retain its actual status, diagnostics and null accepted
posterior on failure.

A profile
binds the complete inference input to its numerical settings; it is not a saved
answer. The five TAC fixed-grid profiles also contain the declared three-level
orders and geometry tolerances. Geometry is freshly optimized before comparison
with the planning basis. No stored posterior or log evidence is consumed.

For direct API use, run from `examples/manuscript_current` in a fresh process
with the thread limits described below:

```python
import json
from pathlib import Path
from research_ghq import get_profile, get_integration_plan, run_ghq

inputs = json.loads(Path("inputs/OXC_C024.json").read_text(encoding="utf-8"))
profile = get_profile("OXC_C024")
plan = get_integration_plan("OXC_C024")  # None unless a fixed-grid plan applies
summary = run_ghq(
    "oxc", inputs, "new_profile_run",
    profile=profile, integration_plan=plan,
)
print(json.dumps(summary, indent=2))
```

`profile` may instead be the exact catalog identifier. `integration_plan` may
be a dictionary or JSON path; omission loads the pinned plan where required.
Exported profiles must be unchanged. Altered model inputs, state identities,
plans, thresholds or policy settings are rejected.

| Profile | Computation and resource defaults |
|---|---|
| OXC except C024/C025 | Original 15/31/61/121 orders |
| OXC C024/C025 | GHQ02 121/241/361, node/finite-term checks and both successive changes |
| Ordinary TAC, quadrature dimensions 6/8/10 | 550M/1.5B/5B points; 1780/3580/7180 worker seconds (about 30/60/120 minutes) |
| TAC_C122 | Original adaptive path, 3B points and 7180 worker seconds (about 2 hours) |
| TAC_C010/C060/C099/C111 | Declared fixed three-level plan, case-specific points, 10780 worker seconds (about 3 hours), maximum order 47 |
| TAC_C079 | Declared fixed three-level plan, 7,337,759,416 points, 21580 worker seconds (about 6 hours), maximum order 49 |

For TAC, the profile fixes the point limit. Worker seconds default to the declared
case allowance minus the original controller's 20-second margin. Pass the precise
smaller scheduling allowance from a particular run receipt when reproducing that
run. Smaller positive values also permit diagnostic timeouts; they do not certify
successful reproduction. Fixed-grid `validation_rounds=1` still evaluates all
three levels and checks both changes. The profile's private module instance is
discarded after each call, so its order cap cannot change a later adaptive call.

These additions are source- and contract-validated. Six selected cases were also
recomputed from a fresh public GitHub checkout; all 26 posterior components
matched the accepted formal GHQ first-repeat vectors exactly. See
[VALIDATION.md](../VALIDATION.md#public-ghq-sensitivity-sample)
for the tested commit, cases and full-vector evidence. This is not a new full
sensitivity, five-repeat or expensive fixed-grid replay. Earlier reference
checks and the 217 historically accepted configurations are separate evidence.

### Custom / Compatibility Use

The unprofiled API remains available for custom inputs, but changing only an
input filename does not select that case's frozen numerical profile. Such calls
are not frozen case-profile replays. Use complete inference-only dictionaries or
JSON paths, not expected-result files. Model selectors are lowercase `oxc`/`tac`;
schemas are `oxc_numeric_source_ruv_v1` and `tac_resolved_v1`, respectively.

## Budget And Output Contract

`run_ghq(model, inputs, output_dir, seconds=None, point_limit=None, *,
profile=None, integration_plan=None)` is
synchronous and launches no processes. Use a fresh single-thread process for
reproduction. TAC's original worker sets thread-count environment variables
before importing NumPy/SciPy. If those libraries were already loaded, changing
environment variables does not reliably reconfigure their thread pools; the
receipt records preloaded libraries. Set OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS to
1 before starting Python when calling either model.

The TAC budgets are passed unchanged to the formal cooperative checks. Without a profile,
omitted budgets preserve the original 1780 seconds and 550M points.
A caller needing a hard wall limit must supervise the entire Python process
tree and join/reap it. No memory cap is imposed by this synchronous wrapper.
OXC has no native seconds/point limit or intermediate checkpoint: it always
uses the selected original order sequence. Its receipt explicitly says budgets are not
enforced. Invalid wrapper arguments raise before creating an output directory.

Each run creates a new directory and records:

- `input.json`: actual consumed input, preserving path-supplied bytes.
- `result.json`: native result, with original status/qualification/diagnostics.
- `checkpoint.json`: TAC's native checkpoint, retained even on budget failure.
- `run_summary.json`: state IDs and manuscript labels in input order; complete
  accepted posterior or null; raw latest-complete-stage posterior separately.
- `process_receipt.json`: PID, start/end timestamps, interpreter/dependencies,
  supplied budgets, input/source hashes, completion or exception. Completion
  here means the function returned, not an assertion of OS process termination.
- `exception.json`: traceback if input loading/validation or orchestration raises.
- `profile.json` and, for fixed grids, `integration_plan.json`: the exact selected
  declarations, separately identified from the consumed pharmacometric input.

Never retry into an existing run directory. There is no resume or old-result
fallback. Native TAC failures remain TIMEOUT, NODE_BUDGET_EXHAUSTED,
NOT_INTEGRATED_GEOMETRY_REJECTED, NOT_CONVERGED, PLAN_GEOMETRY_MISMATCH,
NUMERICAL_FAILURE or IO_FAILURE,
as determined by the unchanged worker. The wrapper does not reinterpret them
as PASS. An exception is re-raised after recording available evidence.

TAC acceptance calls both the original `validate_result` and `qualifies` on the
persisted JSON. OXC retains its exact `qualified` and status decisions. Vector
identity/integrity is checked separately, without renormalizing probabilities.
The wrapper's `diagnostic_accepted` is not formal qualification. In particular,
`formal_qualified=false` and `absolute_error_certified=false` are preserved.
OXC's native `qualified=true` refers only to its convergence criterion; it does
not override these separate wrapper boundaries.

## Scientific Identity

`source_manifest.json` records exact original and delivered SHA256 values and
the extraction audit. Both the default and fixed-grid TAC module sets are included,
with only package-relative import changes. OXC numerical functions/constants use the
shared `model_adapters.oxc_model` and `_common` definitions, AST-verified against
the formal source. The wrapper checks these pinned delivered source hashes.
Source paths in the manifest are provenance strings, not runtime dependencies.

The TAC reference retains the full 10-dimensional ETA covariance. Its exact
Gaussian pushforward integrates two likelihood-null directions analytically,
then performs 8-dimensional numerical quadrature. No IIV/IOV terms are dropped.
All-axis refinement PASS remains a diagnostic, not an absolute error bound.

The optional profile registry contains input identities and numerical declarations,
not probability results. Paths inside original plan provenance are descriptive
strings, not files opened during inference. The extension does not load comparator results, stored posterior
answers or historical formal output folders. Compare completed outputs in a
separate caller/test, preserving exact state order. Reference replay results and
the tested environment are recorded in [../VALIDATION.md](../VALIDATION.md).
