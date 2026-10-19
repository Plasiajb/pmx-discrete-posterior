# Retained GHQ Route

Version 0.2.0 retains prior-centered tensor-product Gauss-Hermite quadrature
alongside Laplace. It integrates over the declared normal random-effect
distribution. This is the package's existing implementation, not adaptive
R `aghq` or another external backend. No external GHQ backend is integrated;
R is not a dependency.

## Run OXC

From the source root in the installed environment:

```bash
python examples/posterior_reference.py OXC --method ghq --ghq-orders 12 20 32 --log-integral-atol 1e-6 --posterior-atol 1e-6 --max-nodes 1000
```

For the seven-group Python call, select `integration_method="ghq"` and:

```python
numerical_options = {
    "ghq": {
        "variant": "prior_centered",
        "orders": [12, 20, 32],
        "log_integral_atol": 1e-6,
        "posterior_atol": 1e-6,
        "max_nodes": 1000,
    }
}
```

All five fields are required. Orders must be at least three strictly
increasing positive integers. Both tolerances must be positive, and
`posterior_atol` must be less than one. Laplace optimization/derivative
groups cannot be combined with GHQ settings.

## Acceptance And Cost

The final two changes in each candidate's log integral and in the same-order
posterior vector must meet their respective absolute tolerances. Diagnostics
include order results and posterior stability checks. Stability across selected
orders is not a rigorous bound on integration error.

`max_nodes` caps cumulative likelihood evaluations **per candidate**, including
one repeated first-node check per order. In `d` active dimensions, order `n`
uses `n**d` tensor nodes before that additional check. This grows rapidly for
the ten-dimensional TAC reference.

Budget exhaustion, failed evaluations or failed stability checks return an
explicit failure. The engine does not reduce dimension, drop candidates,
switch methods or return a partially normalized posterior.

## Evidence Scope

The OXC reference has been checked against retained GHQ results and scalar
quadrature. TAC full-dimensional low-order execution and budget-stop behavior
have been tested, but **high-order TAC accuracy and stability have not passed
acceptance**. A passing budget-rejection test is not an accepted TAC integral.

The older manuscript GHQ workflow uses its own pinned source and configuration.
Follow the [legacy checkout instructions](migration.md#reproduce-the-manuscript);
do not transfer its acceptance claims to the v0.2 route.
