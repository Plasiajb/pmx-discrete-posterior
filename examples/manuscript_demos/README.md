# Two Manuscript Reviewer Demos

Self-contained, fixed-input Python replays of **OXC SCN003 (current Hill=219)**
and **tacrolimus TAC001**. Each runs Laplace and GHQ, checks diagnostics, and
compares its full posterior with saved manuscript references. No repository
installation, author files, network access at runtime, or NONMEM installation
is needed. These are two bounded examples, not a full-grid reproduction.

## Run

Use Python 3.12 and the NumPy/SciPy versions in `requirements.txt`. From the
repository root, in an environment with those dependencies:

```text
python -B examples/manuscript_demos/run_demo.py verify
python -B examples/manuscript_demos/run_demo.py oxc --output demo_outputs/oxc
python -B examples/manuscript_demos/run_demo.py tac --output demo_outputs/tac
```

For a new environment, install the dependencies first:

```text
python -m pip install -r examples/manuscript_demos/requirements.txt
```

No dependencies are installed by the runner. `--output` must name a new directory
outside `manuscript_demos`; existing output and source directories are rejected.
Each run writes `report.json` (complete inputs, diagnostics, comparisons, hashes)
and `posterior.csv` (all states, source labels, eligibility). Exit 0 means the
requested routes passed; a comparison failure exits 1 with its report retained.
Input, dependency, or integrity errors exit nonzero before creating output.

The default `--route both` executes both Python routes. Use `--route laplace` or
`--route ghq` for separate checks. The only supported cases are `oxc --case SCN003`
and `tac --case TAC001`; arbitrary inputs and quadrature orders are not accepted.
The entire `manuscript_demos` directory can be copied elsewhere and run with
`python -B run_demo.py ...`; keep outputs outside that copied directory.

## What Is Calculated

| | OXC SCN003 | Tacrolimus TAC001 |
|---|---|---|
| Model | One compartment, oral, stable Hill=219 weight relation | Reduced two-compartment oral model, lag 0.44 h |
| States | omega00, omega01, omega10, omega11 | 00, 01, 10, 11 |
| Prior | 0.25 per state | 0.25 per state |
| Recent doses | 300 mg at 0 and 12 h | 5 mg at 0 and 12 h |
| Common history | Analytic steady-state history through -12 h | 120 known doses, -1440 through -12 h; no SS/ADDL |
| Observation | 24 h, 12 mg/L | 24 h, 12 ng/mL |
| ETA covariance | ETA(CL): 0.012245714245884425 | ETA(CL), ETA(V1): diag(0.0834446069546344, 0.19805053857439608) |
| Residual SD | Additive 2.992 mg/L; proportional 0 | Additive 0.05 ng/mL; proportional 0.183 |
| GHQ | Mode/Hessian-adaptive, 41 nodes | Prior-centered tensor, 241 per dimension, 58,081 per state |

Bits run **latest to older**: `01` includes only the 0 h recent dose; `10`
includes only the 12 h recent dose. OXC computes the nominal 0 h dose at
0.0001 h, matching the frozen time-zero convention. Residual variance is
`additive_sd^2 + (proportional_sd * prediction)^2`; ETA variances are not SDs.
Structural parameters, covariates, events, priors, units and optimizer settings
are printed and retained in the JSON report. OXC and TAC concentration units differ.

TAC uses the original `supported_inference.inference_entry` before either route.
It rejects dimension errors, nonfinite, nonsymmetric and non-positive-definite
covariance without projection. Frozen core v0.1.4 itself is **not repaired or
universally guarded**. Run the included original guard tests separately:

```text
python -B examples/manuscript_demos/frozen/tac/scripts/test_inference_domain.py
```

## References And Interpretation

NONMEM values are always labelled **SAVED NONMEM PMIX - NOT FRESH**. The wrapper
reads bundled `.phm` values and checks them against saved references. `.mod`,
data, `.shm`, and saved diagnostic records are included for inspection. No
native run or fresh native diagnostic review occurs. The small subset does not
include the complete historical command/listing chain; original diagnostic
records can refer to those full-ESM artifacts. Python results are not live NONMEM.

Fresh-vs-saved checks use absolute tolerances:

| Replay | Posterior | State log evidence |
|---|---:|---:|
| OXC, each route | 1e-10 | 1e-10 |
| TAC Laplace | 1e-7 | Not compared: no selected saved Laplace evidence fixture |
| TAC GHQ241 | 1e-7 | 1e-6 |

OXC thresholds are this wrapper's regression policy, not a new scientific
accuracy criterion. TAC thresholds retain the frozen smoke-test policy.
Different routes need not agree with each other to pass same-route replay.
Eligibility screens optimizer/Hessian/vector failures; it does not certify
global optimality, exact integration, clinical accuracy or a true dosing label.
TAC001 posterior mass is distributed across all four histories, not a definitive
adherence classification. Do not run Python with `-O`.

## Source Identity

Source: public `ESM_2.zip`, submission v1.2 / V26 (2026-10-01), SHA-256:

```text
a7bdaef0e80e50ee931f1895bf1f6c0dcb388e22301171bb162ec99718781b88
```

- OXC code/input: `oxc_current_hill219`. SCN003 has unchanged numeric input in
  V26. Its current registry explicitly selects `verified_reuse` historical
  results; `frozen/oxc/selected_records.json` preserves the selected case and
  repeat-1 bindings. Current and parent input JSON are equal after parsing.
  References are single saved runs, not the five-repeat normalized means.
- TAC code/input: nested `historical/v1_1/archive/ESM_2_v16.zip`, protected study
  entry, not the seven-case high-order wrapper. The four TAC data tables are
  byte-identical to their current outer-ESM counterparts. They retain other
  historical rows for byte preservation, without expanding this demo's scope.
- `subset_manifest.json` records each public-ESM member path, source SHA-256,
  subset SHA-256 and size. Forty-six members are byte-exact; the selected OXC
  provenance record is a lossless selection with both parent file hashes.
  This verifies the subset only, not the original complete ESM bundle.
- `run_demo.py` is a new wrapper. Every scientific source and core file under
  `frozen/` is unmodified. The two source trees remain separate; relative
  import-path checks reject a preloaded foreign project package.
- Both frozen packages declare version 0.1.4, but V26's OXC `reporting.py` and
  `__init__.py` include pre-existing FSRS10 reporting exports absent from the
  GitHub tag/TAC copy. These are preserved, not used to score these demos.
  The four inference/model files are identical between the frozen trees and
  match the tag after checkout line-ending normalization. Version strings
  alone are not a source-identity check; use the manifest hashes.

See `NOTICE` for licensing and scope, and `VALIDATION.json` for test evidence.
A fresh Windows Python 3.12.0 virtual environment, without system site packages,
installed NumPy 2.5.2 and SciPy 1.16.2 from PyPI wheels with pip's cache disabled.
Both demos and subset verification passed from a copied path containing spaces;
an audit hook blocked network, child processes and external project files.
The copied source was unchanged. This is a Python audit-hook test, not an OS
sandbox. Linux/macOS remain unverified. Strict replay can fail on another
numerical environment; retain the report and investigate rather than silently
relaxing tolerances.
