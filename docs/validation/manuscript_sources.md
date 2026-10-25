# Manuscript Sources And Acceptance

The reader release keeps package version 0.2.0. Its Laplace integrator,
optimizer, derivatives, model calculations and normalization are unchanged
from Git commit `dd570509cca6f678641677d8a0cff08007333655`. The former Python
GHQ implementation and dispatch have been removed; this is a new source build.
This does not make every earlier experiment an execution of that snapshot.

## Experimental Snapshots

- OXC Laplace and retained TAC Laplace attempts used the c026 package source,
  corresponding to Git commit `90abfe43a5728b8d34725dc8ff96a49a1a9d6e03`.
- TAC C107/C110 Laplace repeats used the boundary-repaired source corresponding
  to `dd570509cca6f678641677d8a0cff08007333655`.
- External R aghq OXC used the original bridge and c026 source. TAC retained
  135 attempts from that source group and used the repaired mapping/source for
  540 attempts. The accepted set contains 675 attempts, not 675 executions of
  one source snapshot.
- The `manuscript-20261018` tag preserves the older self-written GHQ and
  pre-current-API workflow. It is not the current external R comparison.

The publication supplement retains complete source groups, input hashes,
attempt receipts and failure records. The identifiers below support matching
those records; local author-machine paths are not reader dependencies.

## Acceptance Record Identifiers

| Record | SHA-256 |
| --- | --- |
| OXC R aghq formal acceptance, 2026-10-22 | `844c0d4b84b317e03888332db1ca1e464d48dd91093750a5206932aa13429bdc` |
| Combined TAC Laplace acceptance, 2026-10-22 | `9aaf004b2041ad5e87888429cb12b13c7e400a3ae4829c6d74d6f82234220f68` |
| Combined TAC R aghq acceptance, 2026-10-23 | `a25c8189b4c89d3975ee3370fe58bf524e22c8fc01330e986ca0e16a872e0af2` |

OXC Laplace completed 415 successful attempts. OXC R aghq completed 415
attempts, with five C026 failures. Each TAC route completed 675 attempts,
with five C078 failures. Each count includes the reference scenario.

## External R Acceptance

The order-change thresholds added by the experimental bridge are diagnostics,
not acceptance criteria supplied by R aghq. Under `external_comparison`, retain
the order-specific outputs, change estimates, `stability_passed` flag and
warnings. Exceeding these thresholds does not reject an otherwise complete,
finite normalized posterior. Actual computation failures and incomplete or
invalid vectors remain failures; successful candidates are never renormalized
after dropping a failed candidate.

Acceptance is not conditional on agreement with Laplace or NONMEM. Repeated
agreement and order diagnostics do not certify absolute integration accuracy.
