# pmx-discrete-posterior

**0.2.0 | reader release, 2026-10-25 | Python >=3.11 | MIT**

Compute posterior probabilities for predefined discrete pharmacometric states,
such as alternative dosing histories. Supply a model, individual data,
observations, candidate states and prior probabilities. The package integrates
over random effects and returns the complete posterior vector, state-specific
log marginal likelihoods and numerical diagnostics.

The Python package uses Laplace approximation. The manuscript's R `aghq`
comparison is external software, not a package backend. These research outputs
are conditional on the supplied inputs and model.

## Install

The repository and [release downloads](https://github.com/Plasiajb/pmx-discrete-posterior/releases/tag/v0.2.0)
are public. The package is not published on PyPI.

For the package and matching examples, clone the release into a new directory:

```bash
git clone --branch v0.2.0 --single-branch https://github.com/Plasiajb/pmx-discrete-posterior.git
cd pmx-discrete-posterior
python -m venv .venv
```

Use Python 3.12 for the recorded manuscript dependency versions. Activate the
environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install --no-cache-dir "numpy==2.5.2" "scipy==1.16.2" .
git rev-parse HEAD
```

Alternatively, download the wheel and matching source archive from the release.
Install the wheel with `python -m pip install ./pmx_discrete_posterior-0.2.0-py3-none-any.whl`
and run the examples from the extracted source archive. The wheel contains the
runtime package; the source archive contains the examples and their input files.

**Version 0.2.0 has been republished.** Use the assets dated 2026-10-25 and their
`SHA256SUMS.txt`, `VALIDATION.json` and `VALIDATION.md`. Record the commit and
artifact hash, not the version alone. In an existing environment, add
`--force-reinstall --no-cache-dir` to the installation command. Earlier builds
and experimental results retain their original identities; see [release history](RELEASE_NOTES.md).

## Reference Scenarios

From the source root in the installed environment:

```bash
python examples/posterior_reference.py OXC
python examples/posterior_reference.py TAC
```

Both commands calculate the four-state C003 reference scenario and print JSON.
TAC has ten active random effects and takes longer. See the
[reference guide](docs/reference_examples.md) for units and output fields.

## Sensitivity Scenarios

Use the current-API reader for both references and the manuscript input catalog:

```bash
python examples/posterior_scenario.py list --model TAC
python examples/posterior_scenario.py run OXC C026 --output oxc-c026.json
python examples/posterior_scenario.py run TAC C107 --output tac-c107.json
```

The catalog contains 83 OXC and 135 TAC inputs, including one reference per
model. The reader assembles the inputs and calls the installed package; it does
not return saved experimental answers. Numerical failures remain failures.
See [scenario inputs and commands](docs/scenario_examples.md) for custom inputs,
covariance-roundoff warnings, event ordering and the recorded validation scope.

For the manuscript's external R `aghq` route, see the
[comparison documentation](docs/manuscript_aghq.md). This repository does not
bundle R `aghq` or its execution bridge. The previous Python GHQ implementations
have been removed from the current release; `integration_method="ghq"` is no
longer supported. Earlier code remains in Git history and release backups.

## Minimal Call

The example loader provides model inputs, not a cached posterior. Custom models
use the same [API](docs/api.md) and [model contract](docs/models.md).

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

`examples.reference_models` is an example helper, not an installed package API.
All seven input groups are keyword arguments. Omitted numerical options are not
equivalent to explicitly supplying every default; see [precision recovery](docs/api.md#precision-limited-recovery).

## Read Results

Check the overall `status` before using probabilities. On success, `state_results`
contains the complete normalized posterior in candidate order. A numerical
failure does not discard failed candidates and renormalize the remainder.
Invalid inputs raise `InputError`. An exactly zero prior keeps the state with
posterior `0.0` and `status="skipped_zero_prior"`.

Retain per-candidate diagnostics, including `method_details.precision_recovery`
and its warning when present. Accepted integral recovery is not optimizer
convergence. The reference CLI retains its existing compact output format;
use the Python result for full method details. See [result fields](docs/api.md#results).

TAC observations use `before_events`, including observations at occasion
boundaries. The package's omitted-phase default is unchanged. See
[observation ordering](docs/observation_phase.md).

## Tests And Contributions

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not slow"
python -m pytest -m slow
```

The slow selection includes full TAC reference calculations. CI additionally
builds and tests installed distributions without importing the checkout's `src`
directory. See [contributor instructions](CONTRIBUTING.md) and
[validation scope](docs/validation/README.md). Release receipts distinguish new
distribution checks from historical formal experiments.

[Examples](examples/README.md) | [API](docs/api.md) | [Migration](docs/migration.md) |
[Release notes](RELEASE_NOTES.md) | [Security](SECURITY.md) | [Citation](CITATION.cff)

Copyright (c) 2026 AI-ready PMX benchmark engine contributors.
Distributed under the [MIT License](LICENSE).
