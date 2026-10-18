# Current Manuscript Laplace And GHQ Examples

This entry computes the frozen October 2026 evaluation models from complete
synthetic inputs through [model_adapters](model_adapters). It contains 218 design
records: 83 OXC and 135 TAC. The references are `OXC_C003` and `TAC_C003`.
The formal record has 217 valid cases and one numerical failure, `TAC_C078`.
These are expected coverage counts, not a claim that a new execution has passed.
The historical October 1 two-case entry remains in [manuscript_demos](../manuscript_demos).

Choose the [Python-Laplace commands](#python-laplace-commands-and-outputs) below
or the separate [profile-bound GHQ workflow](research_ghq/README.md).
`run_demo.py` remains Laplace-only; `run_ghq_demo.py` provides GHQ
`list`, `show`, `run` and `export-python`, with no GHQ batch command.
Research GHQ `2026.10.18.2` requires this checkout, not the old package `0.1.4`
wheel; no package installation is needed. Its 218 profiles include 217
historically accepted configurations and the historical `TAC_C078` failure,
not 218 newly accepted runs. Fresh public sampled replay remains pending;
earlier Laplace and GHQ reference evidence is in [VALIDATION.md](VALIDATION.md).

## Setup

1. Keep the complete repository, including `src`, this directory and its input
   files. The runner imports this checkout's `src`, never an old installed
   `pmx_discrete_posterior`. No package installation or NONMEM installation is
   needed. See [NOTICE](NOTICE) and the repository [MIT license](../../LICENSE).
2. Select Python 3.12 and create a virtual environment at a **short path**.
   Deep Windows paths can exceed the 260-character limit during installation.
3. Install the formal pins from [requirements.txt](requirements.txt):
   NumPy 2.5.2, SciPy 1.16.2 and mpmath 1.3.0. Installation of these pins on
   Python 3.12 and all-case numerical replay are recorded in
   [VALIDATION.md](VALIDATION.md).
4. From a short repository checkout path, list and inspect cases before computing.
   The PowerShell example below creates the environment inside that checkout.

```powershell
py -3.12 -m venv .venv
$py = './.venv/Scripts/python.exe'
& $py -m pip install -r examples/manuscript_current/requirements.txt
& $py -B examples/manuscript_current/run_demo.py list
& $py -B examples/manuscript_current/run_demo.py show OXC_C003
& $py -B examples/manuscript_current/run_demo.py run OXC_C003 --output reviewer_oxc_r01
& $py -B examples/manuscript_current/run_demo.py run TAC_C003 --output reviewer_tac_r01
```

Reuse an already prepared environment by skipping creation and installation.
On POSIX, use `python3.12 -m venv /short/path/.venv` and that environment's
`bin/python`; the runner commands are otherwise identical. No runtime network
access is needed. BLAS/OpenMP thread limits are set to 1 before NumPy is imported.
Use a fresh Python process, not a notebook with another checkout already loaded.

## Python-Laplace Commands And Outputs

```powershell
& $py -B examples/manuscript_current/run_demo.py --help
& $py -B examples/manuscript_current/run_demo.py list --model TAC --role reference --json
& $py -B examples/manuscript_current/run_demo.py show TAC_C078
& $py -B examples/manuscript_current/run_demo.py export-python TAC_C003 --output reviewer_tac_input.py
& $py -B reviewer_tac_input.py --repo .
& $py -B examples/manuscript_current/run_demo.py batch --role reference --output reviewer_refs_r01
& $py -B examples/manuscript_current/run_demo.py batch --model OXC --case 'OXC_C00*' --output reviewer_subset_r01
& $py -B examples/manuscript_current/run_demo.py batch --case OXC_C003,TAC_C003,TAC_C078 --output reviewer_selected_r01
& $py -B examples/manuscript_current/run_demo.py batch --output reviewer_all_r01
```

`list` reads the catalog only. `show` verifies the input file's published SHA-256
and prints **all** input fields plus separate catalog provenance; neither command
performs inference or verifies posterior agreement. `export-python` embeds the
entire input dictionary, not expected answers; it calls the same adapter through
the repository runner. It is not a standalone scientific engine. The raw callable
interface is `model_adapters.compute(model, inputs)` after establishing the same
repository import path and thread limits.

`run` and `batch` always compute Python-Laplace, with no GHQ or native NONMEM
default. Batch runs sequentially, includes all 218 cases by default, and accepts
`--model`, `--role` and repeatable/comma-separated `--case` IDs or quoted globs.
An unmatched filter is an error. Catalog order is preserved and duplicates are
not run twice. High-dimensional TAC cases take longer than the OXC reference.

`--output` is required for computations. Relative paths resolve from the current
working directory. Its parent must exist; the target must not exist, even if
empty. Source directories are protected. Choose a new name such as `_r02` for a
repeat; there is no overwrite, cleanup or implicit output destination.

Each output directory contains:

- `CASE_ID.json`: complete inputs, ordered input states, published input hash,
  the unmodified adapter `result` (status, eligibility, full posterior,
  diagnostics, settings, computed-input hash and versions), and wrapper `outcome`.
- `posterior.csv`: one row per candidate state, preserving input order, full
  precision probabilities, label, prior, events, status, hashes and JSON locator.
  Failed cases retain their state rows with blank valid probabilities. No failed
  vector is normalized or replaced by a cached or uniform answer.
- `run_started.json`: selected IDs, source hashes, actual import path, Python
  version and thread configuration. It is an initial receipt, not success.
- `status.json`: completed/selected counts, actual outcomes and per-case locators.
  A missing final status means the run did not finish normally. Ctrl+C retains
  completed case files and reports `INTERRUPTED` when finalization is possible.

Exit `0` means successful inspection/export or all computed cases eligible;
`1` means at least one numerical rejection; `2` means command, dependency,
input, execution, output or adapter-contract error; `130` means interrupted.
Batch continues after an individual case error. Read both `outcome` and the
adapter's `result.status`/`result.eligible`, not only the process exit code.
`TAC_C078` is computed normally, expected to retain a null valid posterior with
its actual diagnostics. It is neither skipped nor forced to fail by its ID.
An unexpected eligible result needs investigation. A 217-eligible/1-rejected
full batch therefore exits `1`, not an all-case pass. Comparison with accepted
vectors is a separate test harness; the CLI never reads expected vectors.

## Complete Inputs And Units

Read the full [OXC reference](inputs/OXC_C003.json) and
[TAC reference](inputs/TAC_C003.json), or use `show`. The catalog retains original
case IDs, reference/sensitivity roles, recipe memberships and input byte hashes.
Sensitivity families cover structure/weight, variability, actual histories,
observations, priors and candidate definitions. Use each case's actual fields,
not reference constants or a fixed four-state denominator.

| Quantity | OXC | TAC |
|---|---|---|
| Observed concentration | MHD, mg/L | TAC, ng/mL |
| Dose / time | mg / h | mg / h, including absorption lag |
| Clearance / volume / absorption rate | `CL_p`: L/h; `V_p`: L; `ka`: 1/h | `CL`, `Q`: L/h; `V1` central and `V2` peripheral: L; `ka`: 1/h |
| Weight | `BW`: kg | `WT`: kg |
| Other covariates | Hill relationship and fixed coefficients in `model` | `X`: binary source covariate; `HEM`: hematocrit L/L; `POD`: days; `Pred`: free prednisolone peak concentration nmol/L, not dose |
| Random effects | One clearance log-effect; full covariance in `variability` | Full `omega` matrix in `eta_order`; correlated four-effect IIV and two IOV effects per occasion |
| Residual | Additive Normal SD in mg/L | Proportional Normal SD coefficient, dimensionless; no additive floor |

TAC has 10 latent effects at reference and 8/12 for other dosing windows; it is
not the historical two-effect example. Full covariance entries are log-effect
variances/covariances, not percentages. Keep their precision and order intact.
OXC uses analytic steady-state prehistory; TAC uses conditional-periodic
initialization recomputed at each effect vector with the initial occasion's IOV.
Shared prehistory, nominal occasion boundaries, actual event times and dose
amounts are part of the complete inputs. Missed/shifted doses do not remove/move
nominal occasions. Coincident observations precede occasion switches and inputs.

Display labels read **latest to oldest**; machine bitstrings read chronological
slot order. At reference slots `[0,12]`, `omega_01` means dose at 0 h only
(machine `10`), while `omega_10` means dose at 12 h only (machine `01`). Keep
labels, `state_event_id`, events and priors together; never sort hashes to create
state order. Other windows/candidate sets have their own explicit definitions.

Numerical settings live in [oxc_laplace.py](model_adapters/oxc_laplace.py) and
[tac_laplace.py](model_adapters/tac_laplace.py): zero-start Nelder-Mead,
`maxiter=6000`, `maxfev=10000`, `xatol=1e-13`, `fatol=1e-11`, full covariance,
and the formal independent diagnostic gates, including OXC's 60-digit mpmath
curvature audit. These are optimizer settings, not replay-comparison tolerances.
Source identities are in
[extraction_manifest.json](model_adapters/extraction_manifest.json).

The separate [GHQ research extension](research_ghq/README.md) provides pinned
OXC/TAC reference and sensitivity configurations. Inspect its resource warnings
before running: fixed-grid TAC allowances reach about six hours and billions of
points, without a hard runtime guarantee. It does not replace the older public GHQ
API or remove that API's dimension limit.

These examples reproduce the specified evaluation models and numerical settings.
They do not run NONMEM or establish clinical classification accuracy or dose guidance.
