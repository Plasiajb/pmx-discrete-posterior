# Reference Examples

The v0.2 examples compute two complete reference cases from bundled synthetic
inputs: `OXC_C003` and `TAC_C003`. They are not a replay of the full manuscript
sensitivity catalog.

## Run

Use a source checkout or extracted source archive with the matching package
installed. From that source root:

```bash
python examples/posterior_reference.py OXC
python examples/posterior_reference.py TAC
```

| Reference | Model | Active random effects | Residual error | Concentration |
|---|---|---:|---|---|
| OXC | One-compartment oral model with weight-dependent clearance | 1 | Additive normal SD | mg/L |
| TAC | Two-compartment oral model with lag, covariates, correlated IIV and occasion-specific IOV | 10 | Proportional normal SD coefficient | ng/mL |

Both references use doses in mg, time in hours, four candidate histories and
equal priors of 0.25. TAC takes longer because all ten active effects are
integrated. State IDs are opaque identifiers; the printed `events` define
the actual history.

The CLI prints JSON with `model`, overall `status`, `integration_method`,
`state_results` and `failure_reason`. State rows include their events,
prior, log marginal likelihood and posterior. Exit `0` means calculation
success; `1` means a returned numerical failure. Invalid CLI arguments exit
`2`; invalid input or execution exceptions also exit nonzero.

The Laplace CLI omits per-state `method_details` from its JSON output.
Use the Python [seven-group call](../README.md#minimal-call) to retain the full
result and diagnostics. Successful execution is distinct from comparison with
a saved reference.

## Inputs And Checks

[reference_models.py](../examples/reference_models.py) constructs the model
relationships, linear predictors, initialization and event mappings from
JSON under [tests/fixtures](../tests/fixtures/). Keep both directories
when extracting the source archive. The loader verifies input hashes; modify
the returned explicit inputs for a new experiment rather than editing the
frozen JSON.

The posterior calculation evaluates the model and integrates random effects;
it does not return saved posterior vectors. Expected results are used by
separate tests. The [validation record](validation/README.md) identifies the
tested release artifacts, environments, commands and comparison criteria.

To inspect only predictions and conditional log likelihoods at zero random
effects:

```bash
python examples/reference_models.py
```

Those are conditional model evaluations, not posterior probabilities.

For retained OXC quadrature, see the [GHQ guide](ghq.md). High-order TAC GHQ
accuracy is not accepted. For the older manuscript examples, use the
[separate legacy checkout](migration.md#reproduce-the-manuscript).
