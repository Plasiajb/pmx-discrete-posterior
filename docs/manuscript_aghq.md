# External R aghq Comparison

The manuscript compares Python and NONMEM Laplace calculations with adaptive
Gauss-Hermite quadrature from the external R package `aghq`. The R route shares
the pharmacokinetic prediction and likelihood functions with the Python route;
it compares integration methods, not independently implemented PK models.

R `aghq` is not a package feature. This Python release includes neither an R
runtime nor an executable R bridge. R and its dependencies are separate
software. The former Python GHQ implementation and dispatch have been removed.

## Recorded Settings

OXC includes reference C003 and 82 sensitivity cases; TAC includes reference
C003 and 134 sensitivity cases. Each model uses the settings below throughout.

| Setting | OXC | TAC |
| --- | --- | --- |
| Acceptance mode | `external_comparison` | `external_comparison` |
| Orders | 7, 15, 31, 61 | 4, 5, 6 |
| Construction / rule | product / `GHe` | sparse / `nHe` |
| Optimizer | `aghq_BFGS` | `R_BFGS_scaled` |
| Derivative policy | `library_default` | `floored_richardson` |
| Coordinates | `prior_standardized` | `prior_standardized` |
| Maximum nodes / evaluations | 1,000 / 20,000 | 50,000 / 70,000 |
| Log-integral / posterior order-change tolerances | `1e-6` / `1e-6` | `1e-6` / `1e-6` |
| Curvature / stationarity tolerances | `0.001` / `0.001` | `0.001` / `0.001` |

The last prescribed order supplies the output. TAC uses the recorded exact
Gaussian projection (ten source coordinates to eight integration coordinates
in C003); the repaired mapper specifies `before_events` observations.

## Outcomes and Diagnostics

| External R route | Formal attempts | Successful | Failed |
| --- | ---: | ---: | --- |
| OXC | 415 (83 x 5) | 410 | C026, five attempts |
| TAC | 675 (135 x 5) | 670 | C078, five attempts |

OXC Python-Laplace separately completed 415/415 successful attempts.
Order-change thresholds are diagnostic only: exceeding them does not reject a
complete, finite normalized external posterior. Order-specific results,
`stability_passed`, warnings and actual failures remain recorded. Failed
candidates are not dropped before normalization. Acceptance does not require
agreement with Laplace or NONMEM and does not certify absolute integral accuracy.

## Recorded Dependencies

The runs used R 4.4.3 and Python 3.12.0 on Windows, with:
- R: `aghq` 0.4.1, `reticulate` 1.46.0, `numDeriv` 2016.8-1.1,
  `mvQuad` 1.0-10, `jsonlite` 2.0.0 and `data.table` 1.18.2.1.
- Python: `pmx-discrete-posterior` 0.2.0, NumPy 2.5.2 and SciPy 1.16.2.

R reports the two hyphenated versions as 2016.8.1.1 and 1.0.10.
Per-run environment records contain complete session and library information.

## Source Provenance

The following hashes identify archived experiment sources, not the current
release, whose package source changed when Python GHQ was removed.

| Archived engine | Git commit | Bridge-recorded package-tree SHA256 |
| --- | --- | --- |
| Original c026 source | `90abfe43a5728b8d34725dc8ff96a49a1a9d6e03` | `5d65cd8f75f1e457a6e326f499702d41d3905690989d2cfd4a38b58f192b33d5` |
| Boundary-repaired source | `dd570509cca6f678641677d8a0cff08007333655` | `2e553f124d5d27e1b95be3f4dd2d28c145f408f22c59fc7ac23f2c5f49e52f8e` |

All 415 OXC attempts used the original source group. TAC retained 135 attempts
from that group and used the repaired group for 540 attempts; C003 repeats 1-2
and 3-5 respectively retain those identities. See
[Manuscript Sources and Acceptance](validation/manuscript_sources.md) for
experimental mappings and acceptance-record hashes.

In the separate submission supplement, `Online_Resource_4/materials/` contains
`SOURCE_GROUP_INDEX.csv` (source seals), `RUN_INDEX.csv` (individual attempts)
and `FILE_INVENTORY.csv` (file identities). These materials are not bundled in
the Python release.
