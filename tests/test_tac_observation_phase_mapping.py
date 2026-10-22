"""Portable reference mappings and exact TAC occasion-boundary regressions."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from pmx_discrete_posterior.models.prediction import prepare_prediction_records

from fixtures.reference_adapters import tac_model
from test_reference_models import ROOT, examples


def test_tac_loader_marks_every_observation_before_events():
    case = examples.load_reference("TAC")
    raw = json.loads((ROOT / "tests/fixtures/TAC_C003.json").read_text())
    expected = [
        {"record_id": str(i), "time": time, "value": value,
         "time_unit": "h", "value_unit": "ng/mL", "event_phase": "before_events"}
        for i, (time, value) in enumerate(zip(raw["times"], raw["observations"]))
    ]
    assert case["observations"] == expected
    prepared = prepare_prediction_records(case["observations"], case["individual_data"])
    assert prepared["prediction_records"] == [
        {key: value for key, value in item.items() if key != "value"}
        for item in expected
    ]


def test_oxc_loader_preserves_observations_without_adding_phase():
    case = examples.load_reference("OXC")
    raw = json.loads((ROOT / "tests/fixtures/OXC_C003.json").read_text())
    expected = [
        {"record_id": str(i), "time": item["time_h"],
         "value": item["concentration_mg_L"], "time_unit": "h", "value_unit": "mg/L"}
        for i, item in enumerate(raw["observations"])
    ]
    assert case["observations"] == expected
    prepared = prepare_prediction_records(case["observations"], case["individual_data"])
    assert prepared["prediction_records"] == [
        {key: value for key, value in item.items() if key != "value"}
        for item in expected
    ]


@pytest.fixture
def tac_boundary_case():
    case = examples.load_reference("TAC")
    raw = json.loads((ROOT / "tests/fixtures/TAC_C003.json").read_text())
    reference = tac_model.ResolvedCase.from_dict(raw)
    eta = np.zeros(reference.neta)
    # Distinct volume IOV makes both switches observable without changing doses.
    eta[[5, 7, 9]] = np.log([1.25, 2.5, 7.5])
    return case, reference, eta


@pytest.mark.parametrize("state_index", range(4))
def test_tac_mapped_observations_at_zero_and_twelve_match_predose_reference(
    tac_boundary_case, state_index,
):
    case, reference, eta = tac_boundary_case
    template = case["observations"][0]
    case["observations"] = [
        dict(template, record_id=f"boundary-{i}", time=time)
        for i, time in enumerate((0, 12))
    ]
    candidate = case["candidate_states"][state_index]
    expected = tac_model.predict_expm(reference, candidate["state_id"], eta, [0, 12])
    predictions, likelihood = examples.evaluate_reference(case, candidate, eta)
    assert predictions["record_ids"] == ("boundary-0", "boundary-1")
    assert predictions["units"] == ("ng/mL", "ng/mL")
    assert predictions["values"] == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert math.isfinite(likelihood)


@pytest.mark.parametrize("state_index", range(4))
def test_tac_mixed_boundary_phases_keep_record_order_and_legacy_default(
    tac_boundary_case, state_index,
):
    case, reference, eta = tac_boundary_case
    requests = [
        (12, "after_events"), (0, "before_events"), (12, None),
        (0, "after_events"), (12, "before_events"), (0, None),
        (12, "before_events"),
    ]
    template = case["observations"][0]
    observations = []
    for i, (time, phase) in enumerate(requests):
        item = dict(template, record_id=f"mixed-{i}", time=time)
        if phase is None:
            item.pop("event_phase", None)
        else:
            item["event_phase"] = phase
        observations.append(item)
    case["observations"] = observations
    original = deepcopy(observations)
    candidate = case["candidate_states"][state_index]
    before_zero, before_twelve = tac_model.predict_expm(
        reference, candidate["state_id"], eta, [0, 12],
    )
    # Oral inputs do not jump central amount; only the observation volume changes.
    after_zero, after_twelve = before_zero / 2, before_twelve / 3
    expected = [after_twelve, before_zero, after_twelve, after_zero,
                before_twelve, after_zero, before_twelve]
    predictions, likelihood = examples.evaluate_reference(case, candidate, eta)
    assert predictions["record_ids"] == tuple(item["record_id"] for item in observations)
    assert predictions["units"] == ("ng/mL",) * len(observations)
    assert predictions["values"] == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert math.isfinite(likelihood)
    assert observations == original
