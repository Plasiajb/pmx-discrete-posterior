import importlib.util
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from pmx_discrete_posterior.interfaces.validation import InputError

from fixtures.reference_adapters import oxc_model, tac_model

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("new_reference_examples", ROOT / "examples/reference_models.py")
examples = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(examples)


def test_reference_loader_does_not_silently_reinterpret_modified_scientific_input(monkeypatch):
    real_read = Path.read_bytes
    def altered(path):
        payload = real_read(path)
        if path.name == "OXC_C003.json":
            return payload.replace(b"zero_mean_multivariate_normal_log_effects", b"uniform_log_effects")
        return payload
    monkeypatch.setattr(Path, "read_bytes", altered)
    with pytest.raises(InputError):
        examples.load_reference("OXC")


@pytest.mark.parametrize("eta", [-.8, 0, .7, 1.2, 2., 3.])
@pytest.mark.parametrize("state_index", range(4))
def test_oxc_reference_matches_high_precision_closed_form(eta, state_index):
    case = examples.load_reference("OXC")
    times = [-12, -6, 0, 6, 12, 18, 24]
    case["observations"] = [{"record_id": str(i), "time": t, "value": 12, "time_unit": "h", "value_unit": "mg/L"} for i, t in enumerate(times)]
    raw = json.loads((ROOT / "tests/fixtures/OXC_C003.json").read_text())
    candidate = case["candidate_states"][state_index]
    predictions, likelihood = examples.evaluate_reference(case, candidate, [eta])
    with mp.workdps(60):
        expected = [float(oxc_model.concentration(raw["model"], raw["initialization"], raw["candidate_states"][state_index]["dose_events"], t, mp.mpf(eta), True)) for t in times]
    assert predictions["values"] == pytest.approx(expected, rel=1e-10, abs=1e-11)
    expected_ll = sum(-math.log(math.sqrt(2*math.pi)*2.992) - .5*((12-value)/2.992)**2 for value in expected)
    assert likelihood == pytest.approx(expected_ll, rel=1e-10, abs=1e-10)


def test_oxc_nearly_equal_absorption_elimination_rates():
    case = examples.load_reference("OXC")
    raw = json.loads((ROOT / "tests/fixtures/OXC_C003.json").read_text())
    k, ka, _ = oxc_model.rate(raw["model"])
    eta = math.log(ka/k)
    predictions, _ = examples.evaluate_reference(case, case["candidate_states"][3], [eta])
    with mp.workdps(60):
        expected = float(oxc_model.concentration(raw["model"], raw["initialization"], raw["candidate_states"][3]["dose_events"], 24, mp.mpf(eta), True))
    assert predictions["values"][0] == pytest.approx(expected, rel=1e-10, abs=1e-11)


@pytest.mark.parametrize("eta", [np.zeros(10), np.linspace(-.3, .4, 10), np.full(10, .2), np.full(10, -.2)])
@pytest.mark.parametrize("state_index", range(4))
def test_tac_reference_matches_existing_adapter_away_from_switches(eta, state_index):
    case = examples.load_reference("TAC")
    times = [1, 6, 13, 18, 24]
    case["observations"] = [{"record_id": str(i), "time": t, "value": 12, "time_unit": "h", "value_unit": "ng/mL"} for i, t in enumerate(times)]
    raw = json.loads((ROOT / "tests/fixtures/TAC_C003.json").read_text())
    old = tac_model.ResolvedCase.from_dict(raw)
    candidate = case["candidate_states"][state_index]
    predictions, likelihood = examples.evaluate_reference(case, candidate, eta)
    expected = tac_model.predict_expm(old, candidate["state_id"], eta, times)
    assert predictions["values"] == pytest.approx(expected, rel=1e-10, abs=1e-10)
    expected_ll = sum(-math.log(math.sqrt(2*math.pi)*.183*value) - .5*((12-value)/(.183*value))**2 for value in expected)
    assert likelihood == pytest.approx(expected_ll, rel=1e-10, abs=1e-10)


def test_tac_explicit_after_events_uses_new_occasion():
    case = examples.load_reference("TAC")
    case["observations"][0]["time"] = 12
    case["observations"][0]["event_phase"] = "after_events"
    raw = json.loads((ROOT / "tests/fixtures/TAC_C003.json").read_text())
    old = tac_model.ResolvedCase.from_dict(raw)
    eta = np.zeros(10)
    eta[9] = math.log(2)
    candidate = case["candidate_states"][3]
    new, _ = examples.evaluate_reference(case, candidate, eta)
    old_value = tac_model.predict_expm(old, candidate["state_id"], eta, [12])[0]
    assert new["values"][0] == pytest.approx(old_value/2, rel=1e-10)
