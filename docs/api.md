# API Reference

The package entry point returns a plain dictionary:

```python
from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.interfaces.validation import InputError

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

All inputs are keyword-only. Only `numerical_options` has a default.
See the [runnable README example](../README.md#minimal-call).

## Inputs

| Group | Fields and meaning |
|---|---|
| `model` | `parameters`, `parameter_model`, `iiv`, `iov`, `residual_model`, `predict`, and explicit `random_effects_dependence="independent_blocks"`. See the [model contract](models.md). |
| `individual_data` | `covariates`, `occasions`, `dosing_prehistory`, `initialization`. |
| `observations` | Nonempty ordered list of records with `record_id`, `time`, `time_unit`, `value`, `value_unit`. |
| `candidate_states` | Nonempty ordered list of `{"state_id": ..., "events": [...]}` mappings with unique IDs. |
| `priors` | Mapping from every candidate ID to a finite probability in [0, 1], summing to one. Zero is allowed; unnormalized weights are rejected. |
| `integration_method` | Exactly `"laplace"` or `"ghq"`; no automatic fallback. |
| `numerical_options` | `None` or supported method-specific settings. GHQ requires explicit settings. |

Unknown fields are rejected. Values and units must be explicit; the package
does not sort records, infer missing doses, convert units or decode state IDs.
Record IDs are unique; repeated observation times are allowed. Observations
share one time unit and one concentration unit. Finite negative observations
are permitted under the Gaussian observation model, without implied censoring
or BLQ treatment.

Each dose event has `time`, `amount`, `amount_unit`, `route`, `compartment`.
Amounts must be finite and nonnegative. `events=[]` is a valid no-dose history
within the candidate window. Shared doses belong in
`individual_data.dosing_prehistory.events`, not both shared and candidate lists.
All explicit events and observations must be at or after initialization.

The prior keys must match the candidate IDs exactly, including zero-prior
states. The sum tolerance is eight float64 ULPs at one. Priors are validated,
not silently normalized.

## Numerical Options

For Laplace, `None` selects the implementation defaults. Supported groups:

| Group | Supported fields |
|---|---|
| `optimization` | `method` (only `"BFGS"`), `initial_point`, `stationarity_tolerance`, `backend_options`. |
| `optimization.backend_options` | `maxiter`, `gtol`, `xrtol`, `c1`, `c2`. |
| `derivatives` | `backend` (only `"scipy"`), `atol`, `rtol`, `backend_options`. |
| `derivatives.backend_options` | `initial_step`, `step_factor`, `maxiter`, `order` (2, 4, 6 or 8). |

The initial point is a vector with one entry per active random effect; the
default is the zero-mean prior point. BFGS is unconstrained. Default
stationarity tolerance is `1e-5`; derivative absolute and relative tolerances
are `1e-6`. In this unreleased repair, the BFGS default is
`gtol=min(1e-6, stationarity_tolerance / 10)`; with the default stationarity
tolerance it is `1e-6`. An explicit `optimization.backend_options.gtol`
override remains supported, including the former `1e-7` setting.
A successful optimizer alone is not a successful Laplace integral:
the calculation also checks stationarity and curvature. These local checks do
not prove a global mode or exact integration.

On the unreleased repair branch (package version `0.2.0`), Laplace optimization
and differentiation use covariance-standardized coordinates, `eta = mean + L z`. Optimization tolerances
and derivative steps (including `initial_step`) apply in these standardized
coordinates. The supplied `initial_point` and returned `conditional_mode`
remain in original random-effect coordinates. Diagnostics label the coordinate
system and retain both modes when optimization or integral recovery succeeds.
A failed search's labelled endpoint is not accepted solely because it is retained.
Explicit numerical settings from published v0.2.0 therefore need rechecking in
the new coordinates. The c026 acceptance path below differs from published
v0.2.0; the Hessian and stationarity-impact thresholds remain `1e-3`.

GHQ accepts only a `ghq` group, with `variant`, `orders`,
`log_integral_atol`, `posterior_atol`, `max_nodes`.
See [GHQ settings](ghq.md). Mixing GHQ and Laplace settings is an input error.
For complete validation rules, see
[numerical_options.py](../src/pmx_discrete_posterior/interfaces/inputs/numerical_options.py).

### Precision-Limited Recovery

This is an explicit acceptance-path change, not optimizer convergence and not
blanket acceptance of failed searches. Recovery requires finite gradient/error
evidence from recognized precision-limited derivative statuses during the search.
The candidate endpoint must match both the last and best primary objective
evaluation, and repeated evaluation of the original objective must agree.
Independent, nonnested gradient step schedules, the strict full Hessian checks,
curvature-relative stationarity, a curvature-scale gradient recheck and symmetric
local objective probes must all pass. Model exceptions, nonfinite evidence and
ineligible failures do not qualify.

| `optimization.stationarity_tolerance` | Recovery behavior |
|---|---|
| Omitted, including `numerical_options=None` | The optimizer still uses its default `1e-5` absolute check. An eligible failed search may instead pass the independent curvature-relative recovery checks. |
| Explicitly supplied, including `1e-5` | The requested absolute gradient-plus-error limit in prior-standardized coordinates is binding in recovery, including after mapping the curvature-scale gradient back to those coordinates. |

Thus `{"optimization": {"stationarity_tolerance": 1e-5}}` is intentionally
not equivalent to omission for recovery. The public posterior call preserves
this distinction through option validation. The BFGS default remains
`min(1e-6, stationarity_tolerance / 10)`, normally `gtol=1e-6`; both Hessian and
stationarity-impact screens remain `1e-3`. These are local numerical screens,
not rigorous bounds on Laplace approximation error.

Re-submitting a fully normalized options mapping also makes its filled-in
`stationarity_tolerance` explicit. To reproduce omission-based recovery, retain
the original requested options rather than replacing them with expanded defaults.

Recovery rechecks the same endpoint; it does not move it, select a new starting
point, launch another optimizer or switch integration methods. Scientific inputs,
priors and posterior normalization are unchanged. See the
[c026 evidence and limits](validation/c026_repair_20261021.md).

## Results

Top-level fields are `status`, `integration_method`, `state_results`,
`failure_reason`, and optional `method_details`. Each ordered state row has:

| Field | Meaning |
|---|---|
| `state_id` | Original candidate identifier. |
| `prior_probability` | Supplied prior probability. |
| `posterior_probability` | Probability on overall success; otherwise `None`, except zero-prior skips remain `0.0`. |
| `status` | `"success"`, `"failed"` or `"skipped_zero_prior"`; describes this candidate's integration, not overall posterior availability. |
| `log_marginal_likelihood` | Successful log integral; `None` for failed or skipped integration. |
| `failure_reason` | `None` on candidate success/skip, otherwise a diagnostic string. |
| `method_details` | Optional numerical diagnostics. |

An accepted precision-limited integral has candidate `status="success"` and
`method_details.precision_recovery.status="accepted"`. The recovery record keeps
`optimization_status="failed"`, the original `failure_reason`, gradient/probe
diagnostics and a `warning`; it does not rewrite the nonconverged optimizer as
successful. Failed derivative records likewise remain `failed`, with their
public `gradient` field `None`; finite candidates are diagnostic evidence only.
Preserve these warnings even on overall success. Acceptance establishes local
numerical checks, not a global mode or equality with an exact integral.

Invalid static inputs raise `InputError` before candidate calculations.
Model callbacks or numerical operations that fail during integration are
reported in the result. If any positive-prior candidate fails, or normalization
or GHQ posterior checks fail, the overall status is `"failed"`. All candidates
and completed integrals remain available, but non-skipped posterior values
are `None`. Do not renormalize a surviving subset.

An exactly zero prior produces `"skipped_zero_prior"`: no integration,
log marginal likelihood `None`, posterior `0.0`, failure reason `None`.
A true zero likelihood may be represented by a successful log integral of
`-inf`; numerical failure is not zero likelihood. If all positive-prior
evidence is zero, normalization fails. A positive posterior that would
underflow to floating-point zero also causes failure.

`method_details` may contain arrays or nonfinite diagnostic settings and
is not guaranteed to serialize directly with strict JSON. The
[reference CLI](../examples/posterior_reference.py) selects serializable output
fields; preserve full diagnostics separately when needed.

Multiple observations are evaluated jointly under the same random-effect
vector. The engine integrates the continuous-effect density once, then applies
discrete priors once in log-space normalization:

```text
posterior[s] = prior[s] * marginal_likelihood[s]
               / sum(prior[j] * marginal_likelihood[j] for j in candidates)
```

See [results.py](../src/pmx_discrete_posterior/interfaces/results.py) for result
validation and [computation.py](../src/pmx_discrete_posterior/posterior/computation.py)
for the calculation.
