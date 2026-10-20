# pmx-discrete-posterior

**0.2.0 | UNRELEASED repair branch | Python >=3.11 | MIT**

Compute model-conditional posterior probabilities over a finite set of
predefined pharmacometric states, such as alternative dosing histories.
Supply a model, individual data, observations, candidate histories and their
prior probabilities. The package integrates over random effects and returns
each candidate's log marginal likelihood, posterior probability and diagnostics.

Laplace is the maintained numerical route. A retained prior-centered
Gauss-Hermite quadrature (GHQ) route is also available. Neither NONMEM nor R is
required. These are research calculations conditional on the supplied model
and assumptions, not direct evidence of adherence or standalone clinical decisions.

## Unreleased Numerical Repair

Branch `repair/prior-coordinates-20261021` retains package version `0.2.0`.
The c026 integration extends commit `be4619b` (the pre-c026 repair baseline);
identify a repair build by its exact commit and artifact hashes, not version alone.
This is UNRELEASED work, not a new tag, release or merge to main. Immutable
published v0.2.0 assets are unchanged and do not contain this repair.

The branch retains covariance-standardized Laplace coordinates, the full Hessian
and Jacobian. The c026 repair **changes the acceptance path**: a narrowly eligible
precision-limited endpoint can yield an accepted integral after independent
curvature-relative and local consistency checks. The nonconverged optimizer
remains `failed` in `method_details.precision_recovery`, with its original reason
and a warning. No fallback optimizer, endpoint move or partial normalization is
introduced; scientific inputs and priors are unchanged.

Explicit `optimization.stationarity_tolerance`, including an explicitly supplied
default `1e-5`, remains binding. Omitting it permits curvature-relative recovery
only after strict eligibility and consistency checks. The BFGS default
`min(1e-6, stationarity_tolerance / 10)` (normally `gtol=1e-6`) and both `1e-3`
Hessian/stationarity-impact thresholds are unchanged by c026. See the
[acceptance contract](docs/api.md#precision-limited-recovery).

Source evidence records five targeted passes of ten originally failing OXC cases,
including c026, not a full sensitivity-matrix rerun. TAC c078 still fails.
The **1463 passed, 2 skipped, 0 remaining failures** result combines a full-suite
run with an affected-module rerun; it is not one all-green invocation.
**Fresh c026 wheel and source-distribution installation results: PENDING.**
See the [c026 validation note](docs/validation/c026_repair_20261021.md) and
[compact source receipt](docs/validation/c026_source_evidence.json).
Earlier development-version and installation results remain
[historical evidence](docs/validation/repair_20261021.md).

## Install

The [GitHub repository](https://github.com/Plasiajb/pmx-discrete-posterior) is
private; repository and release downloads require access. This package is not
published on PyPI. Runtime dependencies are NumPy >=1.26 and SciPy >=1.16.2,<2.

For the unreleased repair, use a separate source checkout of its branch and
record the exact commit (the branch can advance):

```bash
git clone https://github.com/Plasiajb/pmx-discrete-posterior.git
cd pmx-discrete-posterior
git switch repair/prior-coordinates-20261021
git rev-parse HEAD
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell, then:

```bash
python -m pip install .
python examples/posterior_reference.py OXC
```

Repair builds retain `0.2.0` filenames; keep them separate from published assets
and record checksums. Fresh installation acceptance is pending.

For the unchanged v0.2.0 release archives, download the wheel, source archive and checksum file
from the [GitHub releases](https://github.com/Plasiajb/pmx-discrete-posterior/releases).
In a fresh virtual environment, install either local artifact:

```bash
python -m pip install ./pmx_discrete_posterior-0.2.0-py3-none-any.whl
# Alternative: build and install from the source archive.
python -m pip install ./pmx_discrete_posterior-0.2.0.tar.gz
```

The wheel contains the runtime package. To run examples after a wheel install,
extract the matching source archive and work from its root. Keep `examples/`
and `tests/fixtures/` together. Initial installation needs access to dependencies;
the examples do not need network access at runtime.

## Minimal Call

Run this Python code from the source root in the environment above. The
reference loader supplies a complete OXC model and input data; it does not
supply a precomputed posterior.

```python
from pmx_discrete_posterior import compute_posterior
from examples.reference_models import load_reference

case = load_reference("OXC")
model = case["model"]
model["random_effects_dependence"] = "independent_blocks"
model["residual_model"]["dependence"] = "conditionally_independent"
states = case["candidate_states"]

result = compute_posterior(
    model=model,
    individual_data=case["individual_data"],
    observations=case["observations"],
    candidate_states=states,
    priors={state["state_id"]: 1.0 / len(states) for state in states},
    integration_method="laplace",
    numerical_options=None,
)
if result["status"] != "success":
    raise RuntimeError(result["failure_reason"])
for row in result["state_results"]:
    print(row["state_id"], row["posterior_probability"])
```

All seven input groups are keyword arguments. Custom models use the same
[API](docs/api.md) and [model contract](docs/models.md); the reference loader is
an example helper, not an installed package API.

## Read Results

Always check the overall `status` before using probabilities. On success,
`state_results` preserves candidate order and contains a complete normalized
posterior. Invalid inputs raise `InputError`. Numerical or model-execution
failure returns `status="failed"` with a reason and all candidate rows;
positive-prior posterior values remain `None`, even for successful integrals.
The package never drops failed candidates and renormalizes the remainder.

An accepted precision recovery is integral success, not optimizer convergence.
Retain the per-candidate `method_details.precision_recovery` record and its
`warning` even when the complete posterior succeeds.

An exactly zero prior skips integration, keeps the candidate with
`status="skipped_zero_prior"`, and returns posterior `0.0`. Inputs for that
candidate must still be valid. See [result fields and failure handling](docs/api.md#results).

## Examples And Scope

```bash
python examples/posterior_reference.py OXC
python examples/posterior_reference.py TAC
```

These compute the four-state OXC and TAC Laplace reference cases. TAC has ten
active random effects and takes longer. The CLI prints JSON and exits `0`
on calculation success or `1` on numerical failure. See
[reference examples](docs/reference_examples.md) for units, checks and output details.

For published v0.2.0, OXC and TAC Laplace references have been checked; **the full manuscript
sensitivity matrix has not been replayed with v0.2.0**. The retained
[GHQ route](docs/ghq.md) has an OXC reference check, but high-order TAC accuracy
has not passed acceptance. It is not R `aghq`; no external GHQ backend is integrated.

The directories `examples/manuscript_current` and `examples/manuscript_demos`
preserve earlier manuscript work. Run them only from the
`manuscript-20261018` tag in a separate checkout and environment, following the
[migration guide](docs/migration.md#reproduce-the-manuscript).
They are not v0.2 examples, and the original 0.1.4 wheel is not a substitute
for that source snapshot.

## Test, Build And Contribute

From a source checkout:

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not slow"
python -m pytest -m slow
```

The standard `-m "not slow"` selection excludes slow tests. The second command above runs the
portable component suite; the third runs the separately marked slow cases,
including full TAC reference calculations. See [CONTRIBUTING.md](CONTRIBUTING.md)
for source builds and installed-package checks.

[Release notes](RELEASE_NOTES.md) | [Migration](docs/migration.md) |
[Validation](docs/validation/README.md) |
[Security](SECURITY.md) | [Citation](CITATION.cff)

Copyright (c) 2026 AI-ready PMX benchmark engine contributors.
Distributed under the [MIT License](LICENSE).
