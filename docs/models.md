# Model Inputs

A model is a mapping, not a subclass or a model-file parser. It supplies the
parameter values, their relationships, random-effect and residual definitions,
and a prediction callable. The [OXC/TAC adapter](../examples/reference_models.py)
is a complete example of these fields.

## Model Fields

| Field | Required content |
|---|---|
| `parameters` | `values` and `units` mappings with exactly the same parameter names. Values are finite scalars. |
| `parameter_model` | `covariate_names` (ordered names; empty allowed) and callable `relations`. |
| `iiv` | `None`, or `names`, `distribution="normal"`, `covariance_parameters`. |
| `iov` | `None`, or the same fields as IIV plus `occasion_key="occasion_id"`. |
| `random_effects_dependence` | `"independent_blocks"`, required by `compute_posterior`. |
| `residual_model` | `distribution="normal"`, `error_type`, `parameter_names`, `dependence="conditionally_independent"`. |
| `predict` | Callable returning structural concentration predictions. |

`covariance_parameters` is a square matrix of names in `parameters.values`;
`None` explicitly denotes a zero entry. These values are variances and
covariances in the zero-mean normal effect space, with unit `"1"`, not SDs
or percentages. Strictly zero-variance coordinates are removed from numerical
integration and expanded back as zeros for parameter relationships. Invalid
covariance is rejected rather than repaired with a diagonal offset.

`independent_blocks` means IIV is independent of IOV and different occasions'
IOV blocks are independent. Correlation within an IIV or IOV block is retained.
IOV names repeat for each explicitly declared occasion.

Residual types are `"additive"`, using
`parameter_names={"sd": "parameter_name"}`, and `"proportional"`, using
`parameter_names={"coefficient_sd": "parameter_name"}`. The referenced SD
must be positive. Additive SD has the concentration unit; proportional SD
is a dimensionless fraction, so `0.183` means 18.3%. A pure proportional
model with zero prediction has a degenerate scale and fails; no residual floor
is added. Combined error and correlated residuals are not supported.

## Individual Data

`individual_data` has exactly four groups:

| Group | Fields |
|---|---|
| `covariates` | Finite values keyed by exactly `parameter_model.covariate_names`; use `{}` if none. |
| `occasions` | Ordered records with `occasion_id`, `start`, `end`; use `[]` if none. |
| `dosing_prehistory` | `events` for shared explicit doses; `regimen` is required only for steady-state initialization. |
| `initialization` | `mode`, `start_time`, `event_phase="before_events"`, plus the mode-specific group below. |

Occasions have unique IDs, positive widths and no overlap. With IOV, they
must cover the evaluation interval without gaps. Intervals are half-open,
except the final endpoint is accepted.

| Initialization mode | Additional content |
|---|---|
| `"zero"` | No additional group. |
| `"provided"` | `provided_state` with named `values`, named `units`, and `pending_inputs`. Each pending record has `time`, `values`, `units` using the same state names. |
| `"steady_state"` | `steady_state={"parameter_rule": "fixed_reference", "reference_occasion_id": ...}`; use `None` for the reference ID when no occasions exist. |

Steady-state `dosing_prehistory.regimen` contains `period`, `anchor_time`,
`time_unit`, `events`. Each periodic event uses `offset` in [0, period)
instead of `time`, together with `amount`, `amount_unit`, `route`,
`compartment`. It defines history before initialization, not automatic
future dosing. List doses at and after the boundary explicitly.

## Callbacks

The parameter-relationship signature is:

```python
def relations(*, parameters, covariates, iiv_values, iov_values,
              occasions, evaluation_interval):
    ...
```

`iiv_values` maps effect names to current values. `iov_values` maps
occasion IDs to effect-name/value mappings. Return
`{"units": {...}, "segments": [...]}`; each segment has `start`, `end`,
`occasion_id`, `values`. Segments cover the evaluation interval without
gaps. Apply covariates and random effects here, once; do not sample effects.

The predictor signature is:

```python
def predict(*, individual_parameters, candidate_state,
            prediction_records, individual_data):
    ...
```

It receives resolved parameters, explicit candidate events, records without
observed `value`, and context containing only `occasions`,
`dosing_prehistory`, `initialization`. It does not receive raw covariates,
raw random effects, observations' values or priors. Return
`{"record_ids": [...], "values": [...], "units": [...]}` in request order,
with one finite nonnegative concentration and unit per record. Do not
reapply random effects or sample residual noise.

Callbacks are Python code executed locally, not sandboxed data. They should
be deterministic for fixed inputs and must not read observed values or
expected answers from external state. See [SECURITY.md](../SECURITY.md).

## Linear Prediction Helper

`from pmx_discrete_posterior.models.prediction.linear import LinearPredictor`
provides an optional matrix-exponential compartment predictor. Construct it as
`LinearPredictor(system)`, where `system(*, values, units)` returns:

- `state_names`, `amount_unit`, `time_unit`, `concentration_unit`.
- `matrix` for the linear amount dynamics and `observation_vector`.
- `inputs`, keyed by `(route, compartment)`, each with `fractions` and `lag`.

This helper supports piecewise parameters, lagged inputs and the three
initialization modes above. At coincident times, it switches occasion
parameters, applies inputs, then samples concentration. This ordering differs
from the legacy TAC adapter; boundary observations need explicit migration
checks. Steady-state initialization is recomputed at the current random effects
using fixed reference-occasion parameters.

Units are matched as strings, not converted. The helper does not provide
an infusion interface, changing compartment topology or a general nonlinear
ODE steady-state solver. A custom predictor supplies its own structural
implementation; using this interface does not validate that model.

See the [input validators](../src/pmx_discrete_posterior/interfaces/inputs/model_inputs.py)
and [callback protocols](../src/pmx_discrete_posterior/interfaces/protocols.py)
for exact contracts.
