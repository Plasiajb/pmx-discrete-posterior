# pmx-discrete-posterior

Research package for model-conditional posterior probabilities over a finite set of predefined discrete states in pharmacometric models. It supports single- and multi-random-effect Laplace calculations. The current manuscript examples include OXC and TAC models with complete reference and sensitivity inputs.

The probabilities are conditional on the model, candidate states, priors, IIV/RUV assumptions, covariates, dosing history, and observation. They are not direct evidence of true adherence and should not be used as standalone clinical decisions.

## Method Boundary

For the one-random-effect, additive-error example, each discrete state `k` has a model-conditional likelihood contribution computed by minimizing

```text
psi_k(eta) = 0.5 * ((y - f_k(eta)) / sigma)^2 + 0.5 * eta^2 / omega_variance
```

and applying the one-dimensional Laplace approximation

```text
log L_k ~= -0.5 * log(2*pi) - log(sigma) - 0.5 * log(omega_variance)
           - psi_k(eta_hat_k) - 0.5 * log(H_k)
```

The posterior is then

```text
Pr(k | y) = prior_k * L_k / sum_j(prior_j * L_j)
```

Zero prior mass is treated as a structural zero by default. Non-positive Hessian, optimizer-boundary, and optimizer-failure cases are marked as indeterminate in diagnostics rather than silently treated as fully valid posteriors.

The multi-random-effect interface evaluates the joint observation density with
the supplied covariance and residual model, and uses the full Hessian in its
Laplace approximation. The current manuscript adapters preserve the formal
optimizer settings and independent numerical checks.

The package does not bundle NONMEM. NONMEM `$MIX`/`$MIXTURE` PMIX outputs are an optional licensed reference path outside the core Python calculation.

## Release Status

Version `0.1.4` is available as a public open-source GitHub repository and
release for manuscript review, collaborator audit, and reproducibility
assessment. The package is distributed under the MIT License. It is not a public
PyPI release, not a NONMEM replacement, and not a clinical decision-support
tool. DOI archiving and PyPI publication remain separate author decisions.

The current manuscript examples require this updated source checkout. The
original `0.1.4` release wheel does not contain the new explicit numerical
configuration or current scenario catalog; it is not an interchangeable snapshot.
The separate research GHQ module is version `2026.10.18.2`, not package `0.1.4`.
It needs the source checkout and pinned dependencies, but no package installation.

## Installation

From a checked-out repository:

```bash
python -m pip install -e ".[scipy,plots,test]"
```

From a local release wheel:

```bash
python -m pip install pmx_discrete_posterior-0.1.4-py3-none-any.whl
```

## Manuscript Review Demos

The [current manuscript examples](examples/manuscript_current/README.md)
contain the complete October 2026 design: 83 oxcarbazepine and 135 tacrolimus
cases, including both reference cases and all sensitivity settings. Each case
includes resolved model, variability, observation, candidate-history and prior
inputs. The Python-Laplace entry computes the full posterior vector and retains
numerical diagnostics. TAC uses correlated IIV and occasion-specific IOV; the
near-zero residual case remains in the catalog with its numerical failure.

The separate [current GHQ reference and sensitivity examples](examples/manuscript_current/research_ghq/README.md)
use `run_ghq_demo.py` to inspect, export or compute one pinned case. All 218
designs are configured; 217 were historically accepted, with `TAC_C078` retained
as a historical failure. These counts are not fresh public-replay results.
Inspect resource allowances first: some TAC fixed grids allow about six hours
and billions of points, without a hard runtime guarantee. See the
[validation record](examples/manuscript_current/VALIDATION.md) for measured replay scope.

The [historical two-case demos](examples/manuscript_demos/README.md) retain the
October 1 analysis and its original validation records. They are not substitutes
for the current scenario set. Each demo directory has its own dependency and
source instructions. No NONMEM installation is needed for the Python examples;
saved NONMEM comparator values in historical examples are not live executions.

## Minimal Usage

```python
from pmx_discrete_posterior import (
    OneEtaLaplaceConfig,
    OxcOneEtaDoseHistoryModel,
    build_oxc_dose_history_states,
    build_laplace_curve_payload,
    compute_laplace_posterior,
)

scenarios = ("omega00", "omega01", "omega10", "omega11")
prior = {scenario: 0.25 for scenario in scenarios}

model = OxcOneEtaDoseHistoryModel(
    dose_mg=300.0,
    interval_h=12.0,
    obs_time_h=36.0,
    weight_kg=25.0,
    steady_state_anchor_age_h=36.0,
    candidate_dose_ages_h=(12.0, 24.0),
    candidate_dose_mg=(300.0, 300.0),
)
states = build_oxc_dose_history_states(scenarios, prior)
config = OneEtaLaplaceConfig(
    omega_variance=0.012245714245884425,
    residual_sd=1e-6,
)

observation = model.prediction("omega11", eta=0.0)
result = compute_laplace_posterior(observation, states, model, config)
print(result.posterior)

curve = build_laplace_curve_payload([2.5, 6.3, 11.7, 15.5], states, model, config)
print(curve["long_rows"][0])
```

## Posterior Evidence Reporting Layer

The package also exposes a lightweight reporting layer for manuscript and audit
workflows. It does not change the Laplace calculation; it summarizes an existing
posterior result into report-ready fields:

```python
from pmx_discrete_posterior import (
    NoCallRule,
    build_case_card,
    posterior_to_long_rows,
    posterior_to_wide_row,
    summarize_posterior_evidence,
)

rule = NoCallRule(rule_id="NC_DEFAULT", margin_threshold=0.10, entropy_threshold=0.80)
report = summarize_posterior_evidence(result, rule=rule, case_metadata={"case_id": "example"})
print(report.map_state_id, report.margin, report.normalized_entropy, report.no_call)
print(build_case_card(report)["interpretation"])
rows = posterior_to_long_rows(report)
wide = posterior_to_wide_row(report)
```

The reporting layer is intended to support model-conditioned posterior evidence
summaries: posterior vector, MAP state, posterior margin, normalized entropy,
no-call status, numerical warnings, and provenance fields. The default no-call
rule is a prespecified reporting diagnostic for weak posterior separation, not a
validated clinical abstention rule.

## OXC4 PMIX-DV Plot

```bash
python examples/oxc4_pmix_dv_plot.py --output-dir examples/oxc4_pmix_dv_output
```

Outputs:

- `oxc4_pmix_dv_curve.csv`
- `oxc4_pmix_dv_curve.png`
- `summary.json`

The script uses the public package API and writes only under the selected output directory.

## Historical Multi-ETA Stress Test

The multi-ETA, multi-observation interface is also used by the historical
combined additive/proportional residual-error example:

```python
from pmx_discrete_posterior import MultiEtaLaplaceConfig, compute_multieta_posterior
```

That stress test uses a synthetic oral one-compartment PK model with ETA(CL), ETA(V), four observations, and combined RUV. It compares the package's multivariate Laplace posterior against a tensor-product Gauss-Hermite numerical integration reference:

```bash
python examples/multieta_stress_test.py
```

Default outputs are written to:

```text
../JPKPD_IMA_Laplace_Manuscript_20260521/supplement/experiments/multieta_stress_test_20260521/
```

This is supplementary generality evidence only. It is not a replacement for the OXC same-assumption NONMEM IMA/PMIX reference validation.

## Tests

From this directory:

```bash
python -m pytest
python -m unittest discover -s tests
```

The tests are written with `unittest` and run cleanly under `pytest` when pytest is installed. They cover posterior normalization, zero-prior exclusion, OXC four-state MAP behavior, GUI4-compatible PMIX-DV row fields, multi-ETA stress-test agreement with Gauss-Hermite reference, and plotting examples.
