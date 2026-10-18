# Numerical Replay

Verified on Windows with Python 3.12.0, NumPy 2.5.2, SciPy 1.16.2 and mpmath 1.3.0.
All 218 supplied input records were recomputed in fresh processes. Expected
results were inaccessible during inference and read only afterward for comparison.
Every accepted receipt was checked against the delivered source and input hashes.

| Python-Laplace | Designed cases | Valid vectors | Maximum absolute difference |
|---|---:|---:|---:|
| OXC | 83 | 83 | 1.11e-16 |
| TAC | 135 | 134 | 5.55e-17 |

Each count includes one reference case. Full vectors were aligned by state ID
and compared with the normalized means of the accepted five formal repeats.
The predeclared componentwise criterion was
`abs(actual - expected) <= 1e-15 + 1e-13 * abs(expected)`.
It was not changed after examining the outputs.

`TAC_C078` retained its numerical rejection under the near-zero proportional
residual setting. Its accepted posterior is null; raw diagnostics are retained.
It remains in the catalog and is not counted among the 134 valid TAC vectors.

The separate GHQ research extension recomputed both references, using the same
full-vector comparison criterion. Maximum absolute differences were `1.11e-16`
for OXC and `2.78e-17` for TAC. The full GHQ sensitivity matrix was not rerun for
this code delivery. Native convergence decisions and the separate
`formal_qualified=false` / `absolute_error_certified=false` fields remain unchanged.

Additional checks passed: 232 public-package regression tests; 13 adapter
contract tests and four numerical tests; and 11 GHQ contract tests, including
relocated execution without author-local source or expected-result access.
These checks concern numerical reproduction, not clinical validity or a proof
of exact integration. Other operating systems and dependency versions have not
been included in this replay.
