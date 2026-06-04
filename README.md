# pmx-discrete-posterior

Small research package for model-conditional posterior probabilities over a finite set of predefined discrete states in pharmacometric models. The current implementation supports a one-ETA OXC validation path and an experimental multi-ETA stress-test path.

The probabilities are conditional on the model, candidate states, priors, IIV/RUV assumptions, covariates, dosing history, and observation. They are not direct evidence of true adherence and should not be used as standalone clinical decisions.

## Method Boundary

For each discrete state `k`, the package computes a model-conditional likelihood contribution by minimizing

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

The package does not bundle NONMEM. NONMEM `$MIX`/`$MIXTURE` PMIX outputs are an optional licensed reference path outside the core Python calculation.

## Release Status

Version `0.1.1` is released through a private/controlled GitHub release for
manuscript review, collaborator audit, and reproducibility assessment. It is not
an open-source release, not a public PyPI release, and not a clinical
decision-support tool. This snapshot uses the included controlled-access
research prototype license. Public/open-source relicensing, DOI archiving,
PyPI publication, and final journal-facing availability wording remain author
decisions before external submission.

## Installation

From a checked-out repository:

```bash
python -m pip install -e ".[scipy,plots,test]"
```

From a local release wheel:

```bash
python -m pip install pmx_discrete_posterior-0.1.1-py3-none-any.whl
```

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

## OXC4 PMIX-DV Plot

```bash
python examples/oxc4_pmix_dv_plot.py --output-dir examples/oxc4_pmix_dv_output
```

Outputs:

- `oxc4_pmix_dv_curve.csv`
- `oxc4_pmix_dv_curve.png`
- `summary.json`

The script uses the public package API and writes only under the selected output directory.

## Multi-ETA Supplement Stress Test

An experimental extension supports multi-ETA, multi-observation Laplace calculations with combined additive/proportional residual error:

```python
from pmx_discrete_posterior import MultiEtaLaplaceConfig, compute_multieta_posterior
```

The current stress test uses a synthetic oral one-compartment PK model with ETA(CL), ETA(V), four observations, and combined RUV. It compares the package's multivariate Laplace posterior against a tensor-product Gauss-Hermite numerical integration reference:

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
