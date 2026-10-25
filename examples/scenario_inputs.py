"""Experimental OXC/TAC source adapter for the installed current package API.

Scientific mapping ported from the validated boundary-study shared/scenario_inputs.py.
No integration, cached results, unit conversion, prior normalization, covariance
rank repair, or C003 substitution occurs here. Only same-sign, nonzero covariance
pairs within two ULP are canonicalized to the source upper triangle, with a warning
for every changed pair. See docs/scenario_examples.md for the mapping contract.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np

from pmx_discrete_posterior.interfaces.inputs.candidate_states import (
    validate_candidate_history,
    validate_candidate_states,
)
from pmx_discrete_posterior.interfaces.inputs.individual_data import validate_individual_data
from pmx_discrete_posterior.interfaces.inputs.model_inputs import validate_model
from pmx_discrete_posterior.interfaces.inputs.observations import validate_observations
from pmx_discrete_posterior.interfaces.inputs.priors import validate_priors
from pmx_discrete_posterior.interfaces.validation import (
    InputError, fields, identifier, numeric_array, scalar, validate_covariance,
)
from pmx_discrete_posterior.models.prediction.linear import LinearPredictor
from pmx_discrete_posterior.models.random_effects import resolve_random_effects

if __package__:
    from .reference_models import (
        _covariance_refs, _event, _oxc_relations, _oxc_system, _tac_relations, _tac_system,
    )
else:
    from reference_models import (
        _covariance_refs, _event, _oxc_relations, _oxc_system, _tac_relations, _tac_system,
    )


_OXC_IIV = ["Clearance IIV"]
_TAC_IIV = ["Clearance IIV", "Central-volume IIV", "Peripheral-volume IIV", "Absorption-rate IIV"]
_OXC_RELATIONSHIPS = {
    "CL": "CL_p*(BW/70)**(k0-kmax*BW**gamma/(k50**gamma+BW**gamma))*exp(eta_CL)",
    "V": "V_p", "ka": "ka", "concentration": "A_central/V_p",
}


class CovarianceRoundoffWarning(UserWarning):
    """Report archive covariance roundoff canonicalization."""


def _keys(value, required, path):
    """Reject unknown or missing fields."""
    return fields(value, required.split(), name=path)


def _equal(actual, expected, path):
    """Check fixed semantics and duplicate declarations."""
    if (isinstance(expected, bool) and type(actual) is not bool) or actual != expected:
        raise InputError(f"{path}: unsupported or inconsistent value; expected {expected!r}")


def _sequence(value, path, *, nonempty=False):
    """Validate a raw sequence."""
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise InputError(f"{path} must be a {'nonempty ' if nonempty else ''}sequence")
    return value


def _number_map(raw, keys, path, *, positive=(), nonnegative=()):
    """Validate and retain source parameters."""
    _keys(raw, keys, path)
    return {k: scalar(v, name=f"{path}.{k}", positive=k in positive,
                      minimum=0 if k in nonnegative else None) for k, v in raw.items()}


def _matrix(raw, dimension, path):
    """Validate the complete covariance."""
    matrix = numeric_array(raw, ndim=2, name=path)
    if matrix.shape != (dimension, dimension):
        raise InputError(f"{path} must have shape ({dimension}, {dimension})")
    changes = []
    for i in range(dimension):
        for j in range(i+1, dimension):
            upper, lower = float(matrix[i, j]), float(matrix[j, i])
            if upper == lower:
                continue
            if (upper == 0 or lower == 0 or (upper > 0) != (lower > 0)
                    or abs(upper-lower) > 2*max(math.ulp(upper), math.ulp(lower))):
                raise InputError(f"{path} is asymmetric at ({i}, {j}); outside two-ULP roundoff policy")
            changes.append(f"({i},{j}): upper={upper!r}, lower={lower!r}")
            matrix[j, i] = upper
    if changes:
        warnings.warn(f"{path}: canonicalized lower to source upper triangle (<=2 ULP); " + "; ".join(changes),
                      CovarianceRoundoffWarning, stacklevel=3)
    validate_covariance(matrix)
    return matrix


def _slots(raw, shared):
    """Validate nominal window boundaries."""
    slots = [scalar(t, name="slot") for t in _sequence(raw, "slots", nonempty=True)]
    if any(b <= a for a, b in zip([shared, *slots], slots)):
        raise InputError("slots must be strictly increasing after shared_last")
    return slots


def _raw_event(raw, *, oxc, offset=0):
    """Map actual dose time and amount."""
    keys = "time_h amount_mg nominal_slot_h"
    if oxc:
        keys += " event_id local_time_h local_nominal_slot_h route compartment"
    _keys(raw, keys, "dose event")
    time = scalar(raw["time_h"], name="time_h")
    nominal = scalar(raw["nominal_slot_h"], name="nominal_slot_h")
    amount = scalar(raw["amount_mg"], name="amount_mg", minimum=0)
    if oxc:
        identifier(raw["event_id"], name="event_id")
        _equal(raw["route"], "oral", "event.route")
        _equal(raw["compartment"], "depot", "event.compartment")
        _equal(scalar(raw["local_time_h"]), time + offset, "event.local_time_h")
        _equal(scalar(raw["local_nominal_slot_h"]), nominal + offset, "event.local_nominal_slot_h")
    return _event(time, amount), nominal


def _states(raw_states, *, oxc, slots, offset=0):
    """Preserve candidate order and explicit events."""
    result = []
    for state in _sequence(raw_states, "states", nonempty=True):
        keys = "state_event_id manuscript_id machine_bitstring "
        keys += ("machine_chronological_bits nominal_slots_h taken_by_slot dose_events"
                 if oxc else "prior events")
        _keys(state, keys, "state")
        identifier(state["manuscript_id"], name="manuscript_id")
        if oxc:
            _equal(state["nominal_slots_h"], slots, "state.nominal_slots_h")
            for taken in _sequence(state["taken_by_slot"], "taken_by_slot"):
                _keys(taken, "nominal_time_h taken", "taken_by_slot")
                if scalar(taken["nominal_time_h"]) not in slots or type(taken["taken"]) is not bool:
                    raise InputError("invalid taken_by_slot metadata")
        events = []
        for event in _sequence(state["dose_events" if oxc else "events"], "events"):
            mapped, nominal = _raw_event(event, oxc=oxc, offset=offset)
            if nominal not in slots:
                raise InputError("event nominal_slot_h is not a declared slot")
            events.append(mapped)
        result.append({"state_id": state["state_event_id"], "events": events})
    return validate_candidate_states(result)


def _history(shared, dose, period, occasions, *, historical_dose=None):
    """Build conditional steady state and common history."""
    event = _event(shared, dose)
    periodic = _event(shared, dose if historical_dose is None else historical_dose)
    periodic["offset"] = 0
    del periodic["time"]
    return {
        "dosing_prehistory": {
            "events": [event],
            "regimen": {"period": period, "anchor_time": shared, "time_unit": "h", "events": [periodic]},
        },
        "initialization": {
            "mode": "steady_state", "start_time": shared, "event_phase": "before_events",
            "steady_state": {"parameter_rule": "fixed_reference",
                             "reference_occasion_id": occasions[0]["occasion_id"] if occasions else None},
        },
    }


def _oxc_initialization(raw):
    """Validate OXC steady history and time axes."""
    _keys(raw, "definition_status interval_h shared_last_dose_h initial_state_time_h initial_state_phase "
          "initial_dose_event earlier_history explicit_occasion_slots_h no_0h_dose_in_one_dose_window "
          "numerical_initial_state time_axes method conditioning iov_present", "initialization")
    for key, expected in {
        "definition_status": "author_approved", "method": "analytic_steady_state",
        "conditioning": "individual_eta_CL", "iov_present": False,
        "initial_state_phase": "immediately_before_shared_last_dose",
        "numerical_initial_state": None, "explicit_occasion_slots_h": [],
    }.items():
        _equal(raw[key], expected, f"initialization.{key}")
    shared = scalar(raw["shared_last_dose_h"])
    period = scalar(raw["interval_h"], positive=True)
    _equal(scalar(raw["initial_state_time_h"]), shared, "initial_state_time_h")
    earlier = _keys(raw["earlier_history"], "interval_h dose_mg last_implicit_dose_h extends_to", "earlier_history")
    _equal(scalar(earlier["interval_h"]), period, "earlier_history.interval_h")
    _equal(scalar(earlier["last_implicit_dose_h"]), shared-period, "last_implicit_dose_h")
    _equal(earlier["extends_to"], "minus_infinity", "earlier_history.extends_to")
    historical_dose = scalar(earlier["dose_mg"], minimum=0)
    axes = _keys(raw["time_axes"], "time_h_axis local_axis manuscript_to_local_offset_h "
                 "manuscript_candidate_slots_h local_candidate_slots_h manuscript_shared_last_h local_shared_last_h "
                 "manuscript_last_implicit_dose_h local_last_implicit_dose_h "
                 "all_events_and_observations_shift_equally native_runtime_axis_adapter_verified", "time_axes")
    _equal(axes["time_h_axis"], "manuscript_study", "time_h_axis")
    _equal(axes["local_axis"], "earliest_candidate_slot_is_zero", "local_axis")
    _equal(axes["all_events_and_observations_shift_equally"], True, "all_events_and_observations_shift_equally")
    if type(axes["native_runtime_axis_adapter_verified"]) is not bool:
        raise InputError("native_runtime_axis_adapter_verified must be boolean")
    slots = _slots(axes["manuscript_candidate_slots_h"], shared)
    offset = scalar(axes["manuscript_to_local_offset_h"])
    _equal(offset, -slots[0], "manuscript_to_local_offset_h")
    _equal(axes["local_candidate_slots_h"], [t+offset for t in slots], "local_candidate_slots_h")
    for key, expected in {
        "manuscript_shared_last_h": shared, "local_shared_last_h": shared+offset,
        "manuscript_last_implicit_dose_h": shared-period, "local_last_implicit_dose_h": shared-period+offset,
    }.items():
        _equal(scalar(axes[key]), expected, f"time_axes.{key}")
    _equal(raw["no_0h_dose_in_one_dose_window"], len(slots) == 1 and 0 not in slots,
           "no_0h_dose_in_one_dose_window")
    event, nominal = _raw_event(raw["initial_dose_event"], oxc=True, offset=offset)
    _equal(event["time"], shared, "initial_dose_event.time_h")
    _equal(nominal, shared, "initial_dose_event.nominal_slot_h")
    return shared, event["amount"], period, historical_dose, slots, offset


def _assemble(values, units, iiv_names, iiv_matrix, iov_matrix, relations, system, error_type):
    """Reuse reference callbacks with explicit independence."""
    iov = None
    if iov_matrix is not None:
        iov = {"names": ["CL", "V1"], "distribution": "normal", "occasion_key": "occasion_id",
               "covariance_parameters": _covariance_refs(iov_matrix, "pi", values, units)}
    iiv = {"names": iiv_names, "distribution": "normal",
           "covariance_parameters": _covariance_refs(iiv_matrix, "omega", values, units)}
    role = "sd" if error_type == "additive" else "coefficient_sd"
    return {
        "parameters": {"values": values, "units": units},
        "parameter_model": {"covariate_names": [], "relations": relations},
        "iiv": iiv, "iov": iov, "random_effects_dependence": "independent_blocks",
        "residual_model": {"distribution": "normal", "error_type": error_type,
                           "parameter_names": {role: "sigma"}, "dependence": "conditionally_independent"},
        "predict": LinearPredictor(system),
    }


def _oxc(raw):
    """Map an OXC experiment directly."""
    _keys(raw, "model initialization variability residual candidate_states priors schema observations", "OXC")
    _equal(raw["schema"], "oxc_numeric_source_ruv_v1", "schema")
    source = _keys(raw["model"], "model_id compartments analyte dose_unit concentration_unit parameters fixed "
                   "relationships covariates_constant_over_record", "model")
    for key, expected in {
        "model_id": "Lin_OXC_MHD_Hill219", "compartments": ["depot", "central"], "analyte": "MHD",
        "dose_unit": "mg", "concentration_unit": "mg/L", "covariates_constant_over_record": True,
        "relationships": _OXC_RELATIONSHIPS,
    }.items():
        _equal(source[key], expected, f"model.{key}")
    parameters = _number_map(source["parameters"], "CL_p V_p ka BW", "model.parameters",
                             positive=("CL_p", "V_p", "ka", "BW"))
    fixed = _number_map(source["fixed"], "k0 kmax k50 gamma normalization_weight", "model.fixed",
                       positive=("k50", "normalization_weight"))
    _equal(fixed["normalization_weight"], 70, "fixed.normalization_weight (source relationship uses BW/70)")
    covariates = {"BW": parameters.pop("BW")}
    variability = _keys(raw["variability"], "iiv_order iiv_omega iiv_correlation iov_order iov_omega_per_occasion "
                        "iov_occasion_ids joint_order joint_omega source_reference_log_variances distribution", "variability")
    for key, expected in {
        "iiv_order": _OXC_IIV, "joint_order": _OXC_IIV, "iiv_correlation": [[1]],
        "iov_order": [], "iov_omega_per_occasion": [], "iov_occasion_ids": [],
        "distribution": "zero_mean_multivariate_normal_log_effects",
    }.items():
        _equal(variability[key], expected, f"variability.{key}")
    source_variances = fields(variability["source_reference_log_variances"], _OXC_IIV,
                             name="source_reference_log_variances")
    scalar(source_variances["Clearance IIV"], minimum=0)
    omega = _matrix(variability["joint_omega"], 1, "joint_omega")
    if not np.array_equal(omega, _matrix(variability["iiv_omega"], 1, "iiv_omega")):
        raise InputError("iiv_omega and joint_omega disagree")
    residual = _keys(raw["residual"], "model sd variance additive_component_present residual_floor normal_zero_mean", "residual")
    for key, expected in {"model": "additive", "additive_component_present": True,
                          "residual_floor": None, "normal_zero_mean": True}.items():
        _equal(residual[key], expected, f"residual.{key}")
    sigma = scalar(residual["sd"], positive=True)
    variance = scalar(residual["variance"], minimum=0)
    if not math.isclose(variance, sigma*sigma, rel_tol=8*np.finfo(float).eps, abs_tol=0):
        raise InputError("residual.variance must equal residual.sd squared")
    values = {**fixed, **parameters, "sigma": sigma}
    units = {k: "1" for k in values}
    units.update(CL_p="L/h", V_p="L", ka="1/h", k50="kg", normalization_weight="kg", sigma="mg/L")
    model = _assemble(values, units, ["CL"], omega, None, _oxc_relations, _oxc_system, "additive")
    shared, dose, period, historical_dose, slots, offset = _oxc_initialization(raw["initialization"])
    states = _states(raw["candidate_states"], oxc=True, slots=slots, offset=offset)
    priors = {}
    labels = {s["state_event_id"]: s["manuscript_id"] for s in raw["candidate_states"]}
    for prior in _sequence(raw["priors"], "priors", nonempty=True):
        _keys(prior, "state_event_id manuscript_id probability", "prior")
        key = identifier(prior["state_event_id"])
        if key in priors:
            raise InputError("duplicate prior state_event_id")
        if key not in labels or labels[key] != prior["manuscript_id"]:
            raise InputError("prior labels do not match candidate states")
        priors[key] = prior["probability"]
    observations = []
    for i, record in enumerate(_sequence(raw["observations"], "observations", nonempty=True)):
        _keys(record, "time_h concentration_mg_L", "observation")
        observations.append({"record_id": str(i), "time": record["time_h"], "value": record["concentration_mg_L"],
                             "time_unit": "h", "value_unit": "mg/L"})
    data = {"covariates": covariates, "occasions": [],
            **_history(shared, dose, period, [], historical_dose=historical_dose)}
    return dict(model=model, individual_data=data, observations=observations, candidate_states=states, priors=priors)


def _tac(raw):
    """Map full TAC block structure and occasions."""
    _keys(raw, "schema numeric_id parameters fixed slots shared_last shared_dose tau states omega eta_order "
          "residual_sd times observations", "TAC")
    _equal(raw["schema"], "tac_resolved_v1", "schema")
    identifier(raw["numeric_id"], name="numeric_id")
    parameters = _number_map(raw["parameters"], "CL V1 Q V2 ka lag WT", "parameters",
                             positive=("CL", "V1", "Q", "V2", "ka", "WT"), nonnegative=("lag",))
    fixed = _number_map(raw["fixed"], "X HEM POD Pred theta_CYP3A5 theta_HEM theta_POD theta_Pred "
                        "POD_cap WT_exponent normalization_weight", "fixed",
                        positive=("theta_CYP3A5", "normalization_weight"))
    covariates = {k: fixed.pop(k) for k in ("X", "HEM", "POD", "Pred")}
    covariates["WT"] = parameters.pop("WT")
    values = {**fixed, **parameters, "sigma": scalar(raw["residual_sd"], positive=True)}
    units = {k: "1" for k in values}
    units.update(CL="L/h", V1="L", V2="L", Q="L/h", ka="1/h", lag="h", normalization_weight="kg",
                 theta_POD="1/day", POD_cap="day", theta_Pred="mL/ng")
    shared = scalar(raw["shared_last"])
    dose, period = scalar(raw["shared_dose"], minimum=0), scalar(raw["tau"], positive=True)
    slots = _slots(raw["slots"], shared)
    times, concentrations = numeric_array(raw["times"], name="times"), numeric_array(raw["observations"], name="observations")
    if not len(times) or times.shape != concentrations.shape:
        raise InputError("times and observations must be nonempty and have equal lengths")
    boundaries = [shared, *slots, max(48, float(max(times))+1, slots[-1]+1)]
    occasions = [{"occasion_id": f"occ_{i}", "start": a, "end": b}
                 for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
    order = [*_TAC_IIV]
    for start in boundaries[:-1]:
        order.extend([f"Clearance IOV@occ_{start:g}", f"Central-volume IOV@occ_{start:g}"])
    _equal(raw["eta_order"], order, "eta_order")
    omega = _matrix(raw["omega"], 4+2*len(occasions), "omega")
    represented = np.zeros_like(omega)
    represented[:4, :4] = omega[:4, :4]
    for i in range(4, len(omega), 2):
        represented[i:i+2, i:i+2] = omega[4:6, 4:6]
    if not np.array_equal(omega, represented):
        raise InputError("omega cannot be represented by 0.2.0 independent_blocks: "
                         "cross-block covariance or unequal occasion-specific IOV blocks")
    model = _assemble(values, units, ["CL", "V1", "V2", "ka"], omega[:4, :4], omega[4:6, 4:6],
                      _tac_relations, _tac_system, "proportional")
    states = _states(raw["states"], oxc=False, slots=slots)
    observations = [{"record_id": str(i), "time": float(t), "value": float(y), "time_unit": "h", "value_unit": "ng/mL",
                     "event_phase": "before_events"}
                    for i, (t, y) in enumerate(zip(times, concentrations))]
    data = {"covariates": covariates, "occasions": occasions, **_history(shared, dose, period, occasions)}
    return dict(model=model, individual_data=data, observations=observations, candidate_states=states,
                priors={s["state_event_id"]: s["prior"] for s in raw["states"]})


def build_inputs(raw, model_name):
    """Adapt raw experiments to public posterior inputs."""
    if not isinstance(model_name, str) or model_name.upper() not in ("OXC", "TAC"):
        raise InputError("model_name must be OXC or TAC")
    result = _oxc(raw) if model_name.upper() == "OXC" else _tac(raw)
    result["model"]["parameter_model"]["covariate_names"] = list(result["individual_data"]["covariates"])
    model = validate_model(result["model"])
    observations = validate_observations(result["observations"])
    data = validate_individual_data(result["individual_data"], model=model, observations=observations)
    states = [validate_candidate_history(s, initialization_start=data["initialization"]["start_time"],
                                         common_events=data["dosing_prehistory"]["events"])
              for s in result["candidate_states"]]
    resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
                          occasions=data["occasions"], dependence=model["random_effects_dependence"])
    return dict(model=model, individual_data=data, observations=observations, candidate_states=states,
                priors=validate_priors(result["priors"], candidate_states=states))


DATA_ROOT = Path(__file__).resolve().parent / "manuscript_current"


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise InputError(f"nonfinite JSON constant: {value}")


def read_json(path):
    """Read strict JSON and its original-byte SHA256, without changing the file."""
    payload = Path(path).read_bytes()
    value = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_json_object,
                       parse_constant=_invalid_constant)
    return value, hashlib.sha256(payload).hexdigest()


def _model_name(value):
    if not isinstance(value, str) or value.upper() not in ("OXC", "TAC"):
        raise InputError("model_name must be OXC or TAC")
    return value.upper()


def _catalog():
    catalog, digest = read_json(DATA_ROOT / "catalog.json")
    _keys(catalog, "schema date designed_counts source_manifests cases", "catalog")
    _equal(catalog["schema"], "manuscript_current_catalog_v1", "catalog.schema")
    _equal(catalog["designed_counts"], {"OXC": 83, "TAC": 135}, "catalog.designed_counts")
    seen, counts = set(), {"OXC": 0, "TAC": 0}
    for entry in _sequence(catalog["cases"], "catalog.cases", nonempty=True):
        _keys(entry, "id model source_case_id numeric_id role input input_sha256 memberships", "catalog case")
        model = _model_name(entry["model"])
        number = entry["source_case_id"]
        if not isinstance(number, str) or not re.fullmatch(r"[Cc][0-9]{3}", number):
            raise InputError("invalid catalog source_case_id")
        _equal(entry["id"], f"{model}_{number.upper()}", "catalog.id")
        _equal(entry["input"], f"inputs/{entry['id']}.json", "catalog.input")
        if entry["id"] in seen:
            raise InputError("duplicate catalog case")
        if not isinstance(entry["input_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["input_sha256"]):
            raise InputError("invalid catalog input_sha256")
        identifier(entry["numeric_id"], name="catalog.numeric_id")
        _equal(entry["role"], "reference" if number.upper() == "C003" else "sensitivity", "catalog.role")
        seen.add(entry["id"])
        counts[model] += 1
    _equal(counts, catalog["designed_counts"], "catalog case counts")
    return catalog, digest


def list_scenarios(model_name=None):
    """List catalog identities and scientific memberships; never run integration."""
    model = None if model_name is None else _model_name(model_name)
    catalog, _ = _catalog()
    return [entry for entry in catalog["cases"] if model is None or entry["model"] == model]


def _mapping_adjustments(raw, model, inputs, caught):
    data = inputs["individual_data"]
    adjustments = [
        {"kind": "source_to_api", "detail": "Source fields mapped to five current-API input groups; labels are provenance, not dose recipes."},
        {"kind": "preserved", "detail": "Source candidate order, explicit events, observations, priors and strict zero variances retained; no prior normalization or rank repair."},
        {"kind": "dependence", "random_effects": "independent_blocks", "residuals": "conditionally_independent"},
        {"kind": "units", "dose": "mg", "time": "h", "concentration": "mg/L" if model == "OXC" else "ng/mL",
         "detail": "No source value conversion; TAC observation callback uses 1000/V1 to predict ng/mL from mg." if model == "TAC" else "No source value conversion; OXC observation callback uses 1/V."},
        {"kind": "conditional_steady_state", "initialization": data["initialization"],
         "dosing_prehistory": data["dosing_prehistory"]},
        {"kind": "covariance_parameter_references", "iiv_names": inputs["model"]["iiv"]["names"],
         "iov_names": inputs["model"]["iov"]["names"] if inputs["model"]["iov"] else [],
         "detail": "Upper-triangle values stored once as named covariance parameters; independent IOV blocks repeated per occasion."},
    ]
    if model == "TAC":
        adjustments.extend([
            {"kind": "observation_phase", "event_phase": "before_events", "record_ids": [r["record_id"] for r in inputs["observations"]],
             "detail": "All TAC concentrations read before same-time parameter switches and inputs, not only boundary records."},
            {"kind": "occasion_mapping", "occasions": data["occasions"], "source_eta_order": raw["eta_order"],
             "detail": "Boundaries are shared_last, slots, and max(48, max(times)+1, last_slot+1); IOV blocks are identical and independent."},
        ])
    else:
        adjustments.append({"kind": "time_axis", "axis": "manuscript_study",
                            "local_offset_h_checked_not_applied": raw["initialization"]["time_axes"]["manuscript_to_local_offset_h"],
                            "detail": "Local-axis metadata checked; event and observation times remain on source manuscript axis. OXC uses the API observation-phase default."})
    adjustments.extend({"kind": "covariance_roundoff" if issubclass(w.category, CovarianceRoundoffWarning) else "mapping_warning",
                        "category": w.category.__name__, "warning": str(w.message)} for w in caught)
    return adjustments


def load_scenario(model_name, case_number=None, *, input_json=None):
    """Return identity, inputs and mapping_adjustments for a catalog or custom JSON.

    Case numbers accept 3, '3' or 'C003'; omitted means C003. Custom files must
    use one supported raw schema and cannot be combined with a case selector.
    Direct build_inputs calls emit roundoff warnings; this loader records them.
    """
    model = _model_name(model_name)
    if input_json is not None:
        if case_number is not None:
            raise InputError("choose a catalog case or --input, not both")
        source = Path(input_json).resolve()
        identity = {"id": f"{model}_custom", "model": model, "role": "custom",
                    "source_case_id": None, "numeric_id": None, "catalog_sha256": None,
                    "input": str(source), "hash_verified_against_catalog": False}
    else:
        number = "3" if case_number is None else str(case_number).upper()
        if not re.fullmatch(r"C?[0-9]{1,3}", number) or int(number.removeprefix("C")) < 1:
            raise InputError("case number must be a positive integer or Cnnn")
        case_id = f"{model}_C{int(number.removeprefix('C')):03d}"
        catalog, catalog_hash = _catalog()
        entries = [entry for entry in catalog["cases"] if entry["id"] == case_id]
        if not entries:
            raise InputError(f"unknown catalog case: {case_id}")
        identity = {**entries[0], "catalog_sha256": catalog_hash, "hash_verified_against_catalog": True}
        source = DATA_ROOT / identity["input"]
    raw, input_hash = read_json(source)
    if identity["hash_verified_against_catalog"] and input_hash != identity["input_sha256"]:
        raise InputError(f"input SHA256 mismatch for {identity['id']}; use --input for a modified experiment")
    identity["input_sha256"] = input_hash
    if isinstance(raw, dict) and "numeric_id" in raw:
        if identity["hash_verified_against_catalog"]:
            _equal(raw["numeric_id"], identity["numeric_id"], "source numeric_id")
        else:
            identity["numeric_id"] = raw["numeric_id"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", CovarianceRoundoffWarning)
        try:
            inputs = build_inputs(raw, model)
        except InputError as exc:
            if caught:
                raise InputError(f"{exc}; mapping warnings before rejection: " + "; ".join(str(w.message) for w in caught)) from exc
            raise
    return {"identity": identity, "inputs": inputs,
            "mapping_adjustments": _mapping_adjustments(raw, model, inputs, caught)}

