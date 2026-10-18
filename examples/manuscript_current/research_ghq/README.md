# Formal GHQ Research Extension

This importable research namespace packages the frozen October 2026 OXC/TAC
GHQ computation. It is separate from `pmx_discrete_posterior`'s public Laplace
API. The old public `gauss_hermite_posterior` dimension limit is unchanged.
It is not a general arbitrary-model GHQ interface or an absolute-error certificate.

## Setup And Call

Keep `research_ghq/` beside the supplied `model_adapters/` directory. Put their
parent, `examples/manuscript_current`, on the Python import path, or use that
directory as the working directory. Install the pinned dependencies in a clean
Python 3.12 environment:

```text
python -m pip install -r requirements.txt
```

The parent directory's `requirements.txt` pins NumPy 2.5.2, SciPy 1.16.2 and
mpmath 1.3.0. Python 3.12.0 is the recorded reproduction interpreter; this is not
a claim of testing every supported Python/platform version. Neither NONMEM,
Pharmpy, a compiler nor the public posterior package is needed for this route.

```python
from pathlib import Path
from research_ghq import run_ghq

# Complete inference-only JSON, not a case label or an expected-result file.
summary = run_ghq(
    model="tac",
    inputs=Path("inputs/TAC_C003.json"),
    output_dir=Path("my_new_tac_ghq_run"),  # must not already exist
    seconds=3579.9839999999967,
    point_limit=1500000000,
)
print(summary["status"], summary["validation"])
if summary["diagnostic_accepted"]:
    for state, label, probability in zip(
        summary["state_ids"], summary["manuscript_ids"], summary["posterior"],
        strict=True,
    ):
        print(state, label, probability)
```

Use `model="oxc"` with `inputs/OXC_C003.json`, or the complete OXC inference dictionary.
Only the model selector is lowercase; preserve input filenames and identifiers
exactly as supplied. The available catalog may use different input filenames;
pass the actual JSON path, not a reconstructed name. The input schemas are
`tac_resolved_v1` and `oxc_numeric_source_ruv_v1`, respectively.

## Budget And Output Contract

`run_ghq(model, inputs, output_dir, seconds=1780, point_limit=550000000)` is
synchronous and launches no processes. Use a fresh single-thread process for
reproduction. TAC's original worker sets thread-count environment variables
before importing NumPy/SciPy. If those libraries were already loaded, changing
environment variables does not reliably reconfigure their thread pools; the
receipt records preloaded libraries. Set OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS to
1 before starting Python when calling either model.

The TAC budgets are passed unchanged to the formal cooperative checks. The
reference run uses the explicit larger budget in the example, not the defaults.
A caller needing a hard wall limit must supervise the entire Python process
tree and join/reap it. No memory cap is imposed by this synchronous wrapper.
OXC has no native seconds/point limit or intermediate checkpoint: it always
uses its original four orders. Its receipt explicitly says budgets are not
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

Never retry into an existing run directory. There is no resume or old-result
fallback. Native TAC failures remain TIMEOUT, NODE_BUDGET_EXHAUSTED,
NOT_INTEGRATED_GEOMETRY_REJECTED, NOT_CONVERGED, NUMERICAL_FAILURE or IO_FAILURE,
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
the extraction audit. The seven TAC modules have only package-relative import
changes. OXC GHQ functions/constants are verbatim extractions and use the
shared `model_adapters.oxc_model` and `_common` definitions, AST-verified against
the formal source. The wrapper checks these pinned delivered source hashes.
Source paths in the manifest are provenance strings, not runtime dependencies.

The TAC reference retains the full 10-dimensional ETA covariance. Its exact
Gaussian pushforward integrates two likelihood-null directions analytically,
then performs 8-dimensional numerical quadrature. No IIV/IOV terms are dropped.
All-axis refinement PASS remains a diagnostic, not an absolute error bound.

The extension does not load catalogs, comparator results, stored posterior
answers or historical formal output folders. Compare completed outputs in a
separate caller/test, preserving exact state order. Reference replay results and
the tested environment are recorded in [../VALIDATION.md](../VALIDATION.md).
