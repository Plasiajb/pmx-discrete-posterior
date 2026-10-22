# Observation Phase

Each observation or prediction record may specify `event_phase`:

| Value | Order at the record's timestamp |
| --- | --- |
| Omitted or `after_events` | Switch parameters, apply due inputs, read concentration. |
| `before_events` | Propagate to the timestamp, read concentration, then switch parameters and apply due inputs. |

No timestamp shift is applied. A volume switch preserves compartment amounts,
but concentration can change because the observation map uses a different volume.
All same-time due inputs, including pending lagged inputs, follow a before-event
observation. Dose fractions and lag are still resolved at administration time
using the existing parameter rule. A missed dose does not cancel an IOV switch.

At initialization, readout uses the initialized reference system. If initialization
is exactly an interior parameter switch with a preceding segment, an explicit
before-event observation there is rejected: initialize earlier to define the
transition. No unprovided previous system is inferred.

Same-time records may use different phases and must have distinct record IDs.
Output preserves input record order. User-defined predictors must honor an
explicit phase or reject it; arbitrary callback internals cannot be verified
automatically by the package.

## TAC Mapping

The TAC study uses pre-event observations. `examples/reference_models.py`
therefore includes `"event_phase": "before_events"` in every TAC observation
dictionary. This is a study input convention, not a universal TAC rule or a
case-ID exception. Other TAC mappers implementing that same convention must
pass the same field. OXC records are unchanged.

## Scope Of Evidence

The repair development checks covered 135 TAC and 83 OXC designs with 2,610
fixed-random-effect prediction probes. Only TAC c107/c110 changed. Direct
repaired-package posterior calculations for these two cases differed from the
current NONMEM `.phm` probabilities by at most `6.294945675833219e-9` and
`3.474084964150137e-9`, respectively; complete rankings agreed. OXC/TAC references
and OXC c026 retained their recorded Python probabilities. These are bounded
development checks, not new five-repeat experiments or a full posterior rerun.

Portable regression tests are in `tests/test_observation_phase.py` and
`tests/test_tac_observation_phase_mapping.py`. Local experiment archives and R
integration code are not dependencies of these public tests. Optimization,
Hessian, Laplace, normalization and numerical thresholds were not modified.

This source patch keeps package version 0.2.0. The existing v0.2.0 release tag and
attachments remain unchanged. Install the patched main checkout and record its
commit; the version string alone does not identify this patch.
