"""Portable reader contracts; only one bounded OXC posterior is computed."""

from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest
import pmx_discrete_posterior

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import posterior_scenario as runner
import scenario_inputs as reader
from reference_models import evaluate_reference
from pmx_discrete_posterior.interfaces.inputs import validate_inputs
from pmx_discrete_posterior.interfaces.validation import InputError
from pmx_discrete_posterior.models.random_effects import resolve_random_effects


CATALOG = reader.list_scenarios()


def raw_case(model="TAC", number=3):
    return reader.read_json(reader.DATA_ROOT / "inputs" / f"{model}_C{number:03d}.json")[0]


def effects(inputs):
    model = inputs["model"]
    return resolve_random_effects(iiv=model["iiv"], iov=model["iov"], parameters=model["parameters"],
                                  occasions=inputs["individual_data"]["occasions"], dependence="independent_blocks")


def fake_result(inputs):
    return {"status": "success", "integration_method": "laplace", "failure_reason": None,
            "state_results": [{"state_id": state["state_id"], "prior_probability": inputs["priors"][state["state_id"]],
                               "posterior_probability": inputs["priors"][state["state_id"]],
                               "log_marginal_likelihood": 0.0, "status": "success", "failure_reason": None}
                              for state in inputs["candidate_states"]]}


def test_catalog_identity_and_frozen_manifest():
    assert Counter(entry["model"] for entry in CATALOG) == {"OXC": 83, "TAC": 135}
    assert hashlib.sha256((reader.DATA_ROOT / "catalog.json").read_bytes()).hexdigest() == (
        "67a42a8a2053985664e7e68292d50dc9c62c746ac326f84e93e48edee4ba43ae")
    for model, count in (("OXC", 83), ("TAC", 135)):
        entries = reader.list_scenarios(model)
        assert [e["id"] for e in entries] == [f"{model}_C{i:03d}" for i in range(1, count + 1)]
        assert [e["id"] for e in entries if e["role"] == "reference"] == [f"{model}_C003"]


@pytest.mark.parametrize("entry", CATALOG, ids=lambda entry: entry["id"])
def test_all_218_contracts_hashes_and_scientific_values(entry):
    path = reader.DATA_ROOT / entry["input"]
    before = path.read_bytes()
    raw = json.loads(before)
    original = deepcopy(raw)
    scenario = reader.load_scenario(entry["model"], entry["source_case_id"])
    inputs = scenario["inputs"]
    assert scenario["identity"]["input_sha256"] == hashlib.sha256(before).hexdigest() == entry["input_sha256"]
    assert scenario["identity"]["numeric_id"] == entry["numeric_id"]
    validate_inputs(**inputs, integration_method="laplace", numerical_options=None)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        reader.build_inputs(raw, entry["model"])
    assert raw == original and path.read_bytes() == before
    oxc = entry["model"] == "OXC"
    states = raw["candidate_states" if oxc else "states"]
    assert [state["state_id"] for state in inputs["candidate_states"]] == [s["state_event_id"] for s in states]
    for mapped, source in zip(inputs["candidate_states"], states):
        assert [(e["time"], e["amount"]) for e in mapped["events"]] == [
            (e["time_h"], e["amount_mg"]) for e in source["dose_events" if oxc else "events"]]
        assert all(e["amount_unit"] == "mg" and e["route"] == "oral" and e["compartment"] == "depot" for e in mapped["events"])
    priors = {s["state_event_id"]: s["probability"] for s in raw["priors"]} if oxc else {
        s["state_event_id"]: s["prior"] for s in states}
    assert inputs["priors"] == priors
    source_matrix = np.array(raw["variability"]["joint_omega"] if oxc else raw["omega"])
    canonical = np.triu(source_matrix) + np.triu(source_matrix, 1).T
    active = np.flatnonzero(np.diag(canonical) != 0)
    np.testing.assert_array_equal(effects(inputs).covariance, canonical[np.ix_(active, active)])
    roundoff = [a for a in scenario["mapping_adjustments"] if a["kind"] == "covariance_roundoff"]
    assert bool(roundoff) == (not np.array_equal(source_matrix, source_matrix.T))
    if roundoff:
        for i, j in zip(*np.where(np.triu(source_matrix != source_matrix.T, 1))):
            assert f"({i},{j})" in roundoff[0]["warning"]
    data = inputs["individual_data"]
    init = raw["initialization"] if oxc else raw
    shared = init["shared_last_dose_h"] if oxc else raw["shared_last"]
    assert data["initialization"]["start_time"] == shared
    assert data["initialization"]["event_phase"] == "before_events"
    assert data["dosing_prehistory"]["regimen"]["period"] == (init["interval_h"] if oxc else raw["tau"])
    assert data["dosing_prehistory"]["events"][0]["amount"] == (init["initial_dose_event"]["amount_mg"] if oxc else raw["shared_dose"])
    assert data["dosing_prehistory"]["regimen"]["events"][0]["amount"] == (init["earlier_history"]["dose_mg"] if oxc else raw["shared_dose"])
    expected = [(r["time_h"], r["concentration_mg_L"]) for r in raw["observations"]] if oxc else list(zip(raw["times"], raw["observations"]))
    assert [(r["time"], r["value"]) for r in inputs["observations"]] == expected
    assert all(r["time_unit"] == "h" and r["value_unit"] == ("mg/L" if oxc else "ng/mL") for r in inputs["observations"])
    if oxc:
        assert inputs["model"]["iov"] is None
        assert inputs["model"]["parameters"]["values"]["sigma"] == raw["residual"]["sd"]
    else:
        assert all(r["event_phase"] == "before_events" for r in inputs["observations"])
        assert inputs["model"]["parameters"]["values"]["sigma"] == raw["residual_sd"]
        assert len(data["occasions"]) == len(raw["slots"]) + 1


@pytest.mark.parametrize("number", [107, 110])
def test_tac_boundary_phase_with_nonzero_volume_iov(number):
    inputs = reader.load_scenario("TAC", number)["inputs"]
    distribution = effects(inputs)
    point = np.zeros(len(distribution.mean))
    point[distribution.effect_keys.index(("iov", "occ_1", "V1"))] = math.log(2)
    state = inputs["candidate_states"][0]
    before, _ = evaluate_reference(inputs, state, point)
    changed = deepcopy(inputs)
    for record in changed["observations"]:
        record["event_phase"] = "after_events"
    after, _ = evaluate_reference(changed, state, point)
    assert inputs["observations"][0]["time"] == 0
    assert before["values"][0] == pytest.approx(2 * after["values"][0], rel=1e-12)
    np.testing.assert_allclose(before["values"][1:], after["values"][1:], rtol=1e-12)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(unrecognized=True), "unsupported"),
    (lambda r: r.update(schema="unknown"), "schema"),
    (lambda r: r["eta_order"].reverse(), "eta_order"),
    (lambda r: r["omega"][0].__setitem__(1, 1e-3), "asymmetric"),
    (lambda r: (r["omega"][0].__setitem__(4, 1e-3), r["omega"][4].__setitem__(0, 1e-3)), "cross-block"),
    (lambda r: r["omega"][6].__setitem__(6, r["omega"][6][6] * 2), "unequal"),
    (lambda r: r["states"][0].update(prior=.7), "sum"),
])
def test_unsupported_semantics_are_not_repaired(mutation, match):
    raw = raw_case()
    mutation(raw)
    before = deepcopy(raw)
    with pytest.raises(InputError, match=match):
        reader.build_inputs(raw, "TAC")
    assert raw == before


def test_roundoff_is_warned_and_original_not_mutated():
    raw = raw_case()
    upper = .01
    raw["omega"][0][1] = upper
    raw["omega"][1][0] = math.nextafter(upper, math.inf)
    before = deepcopy(raw)
    with pytest.warns(reader.CovarianceRoundoffWarning, match=r"\(0,1\).*upper=.*lower="):
        inputs = reader.build_inputs(raw, "TAC")
    assert raw == before
    assert effects(inputs).covariance[1, 0] == upper
    raw["omega"][1][0] = upper + 3 * math.ulp(upper)
    with pytest.raises(InputError, match="two-ULP"):
        reader.build_inputs(raw, "TAC")


@pytest.mark.parametrize("lower", [0., -.01])
def test_covariance_roundoff_does_not_repair_zero_or_sign(lower):
    raw = raw_case()
    raw["omega"][0][1], raw["omega"][1][0] = .01, lower
    with pytest.raises(InputError, match="asymmetric"):
        reader.build_inputs(raw, "TAC")


def test_catalog_hash_mismatch_rejected_custom_input_allowed(monkeypatch):
    source = reader.DATA_ROOT / "inputs" / "OXC_C003.json"
    raw = raw_case("OXC")
    raw["model"]["parameters"]["CL_p"] *= 1.2
    payload = json.dumps(raw).encode()
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: payload if path == source else original_read(path))
    with pytest.raises(InputError, match="SHA256 mismatch"):
        reader.load_scenario("OXC", 3)
    scenario = reader.load_scenario("OXC", input_json=source)
    assert scenario["identity"]["role"] == "custom"
    assert scenario["identity"]["input_sha256"] == hashlib.sha256(payload).hexdigest()
    assert scenario["identity"]["hash_verified_against_catalog"] is False
    assert scenario["inputs"]["model"]["parameters"]["values"]["CL_p"] == raw["model"]["parameters"]["CL_p"]


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_ambiguous_json_rejected(monkeypatch, payload):
    monkeypatch.setattr(Path, "read_bytes", lambda path: payload)
    with pytest.raises(InputError):
        reader.read_json("unused.json")


@pytest.mark.parametrize("number", [3, "3", "003", "C003", "c003", None])
def test_case_selectors(number):
    assert reader.load_scenario("oxc", number)["identity"]["id"] == "OXC_C003"


@pytest.mark.parametrize("number", [0, True, -1, 3.1, "C000", "C999", "../C003"])
def test_invalid_selectors(number):
    with pytest.raises(InputError):
        reader.load_scenario("OXC", number)


@pytest.mark.parametrize("options", [None, {}, {"optimization": {"backend_options": {"gtol": 1e-6}}}])
def test_dispatch_does_not_materialize_defaults(monkeypatch, options):
    scenario = reader.load_scenario("TAC", 26)
    before = deepcopy(options)
    calls = []
    def compute(**kwargs):
        calls.append(kwargs)
        return fake_result(scenario["inputs"])
    monkeypatch.setattr(runner, "compute_posterior", compute)
    result = runner.run_scenario(scenario, numerical_options=options)
    assert len(calls) == 1 and calls[0]["integration_method"] == "laplace"
    assert calls[0]["numerical_options"] == options == before
    assert result["numerical_options_requested"] == options
    assert result["status"] == "success"
    assert result["identity"]["id"] == "TAC_C026"


@pytest.mark.parametrize("failure", ["exception", "missing", "nan", "partial", "failed", "wrong_method"])
def test_dispatch_failure_keeps_complete_unavailable_vector(monkeypatch, failure):
    scenario = reader.load_scenario("OXC", 3)
    def compute(**kwargs):
        if failure == "exception":
            raise RuntimeError("mock integration failed")
        result = fake_result(scenario["inputs"])
        if failure == "missing":
            result["state_results"].pop()
        elif failure == "nan":
            result["state_results"][0]["posterior_probability"] = float("nan")
        elif failure in ("partial", "failed"):
            result.update(status="failed", failure_reason="state failed")
            result["state_results"][0].update(status="failed", failure_reason="failed curvature",
                                              log_marginal_likelihood=None, posterior_probability=None)
            if failure == "failed":
                for row in result["state_results"]:
                    row["posterior_probability"] = None
        elif failure == "wrong_method":
            result["integration_method"] = "ghq"
        return result
    monkeypatch.setattr(runner, "compute_posterior", compute)
    result = runner.run_scenario(scenario)
    assert result["status"] == "failed" and result["failure_reason"]
    assert [r["state_id"] for r in result["state_results"]] == [s["state_id"] for s in scenario["inputs"]["candidate_states"]]
    assert all(r["posterior_probability"] is None for r in result["state_results"])
    assert result["identity"]["input_sha256"] == scenario["identity"]["input_sha256"]
    if failure == "failed":
        assert result["state_results"][1]["log_marginal_likelihood"] == 0.0


def test_ghq_not_dispatched(monkeypatch):
    monkeypatch.setattr(runner, "compute_posterior", lambda **kw: pytest.fail("must not dispatch GHQ"))
    result = runner.run_scenario(reader.load_scenario("OXC"), numerical_options={"ghq": {}})
    assert result["status"] == "failed" and "Laplace" in result["failure_reason"]


def test_zero_prior_skip_survives_failure(monkeypatch):
    scenario = reader.load_scenario("OXC")
    ids = [s["state_id"] for s in scenario["inputs"]["candidate_states"]]
    scenario["inputs"]["priors"] = dict(zip(ids, [0., .2, .3, .5]))
    monkeypatch.setattr(runner, "compute_posterior", lambda **kw: (_ for _ in ()).throw(RuntimeError("failed")))
    result = runner.run_scenario(scenario)
    assert result["status"] == "failed"
    assert result["state_results"][0]["status"] == "skipped_zero_prior"
    assert [r["posterior_probability"] for r in result["state_results"]] == [0., None, None, None]


@pytest.mark.parametrize("options", [None, {"optimization": {"backend_options": {"maxiter": 500}}}])
def test_cli_options_file_passes_only_user_fields(monkeypatch, capsys, options):
    original_read = reader.read_json
    option_path = Path("options.json")
    monkeypatch.setattr(runner, "read_json", lambda path: (options, "unused") if path == option_path else original_read(path))
    calls = []
    def compute(**kwargs):
        calls.append(kwargs["numerical_options"])
        return fake_result(kwargs)
    monkeypatch.setattr(runner, "compute_posterior", compute)
    assert runner.main(["run", "OXC", "--numerical-options", str(option_path)]) == 0
    assert calls == [options]
    assert json.loads(capsys.readouterr().out)["numerical_options_requested"] == options


def test_cli_list_and_mocked_run(monkeypatch, capsys):
    monkeypatch.setattr(runner, "compute_posterior", lambda **kw: fake_result(kw))
    assert runner.main(["list"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 218
    assert runner.main(["list", "--model", "TAC"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 135
    assert runner.main(["run", "TAC", "C107"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["identity"]["id"] == "TAC_C107" and len(output["state_results"]) == 4
    assert runner.main(["run", "OXC", "--input", str(reader.DATA_ROOT / "inputs" / "OXC_C003.json")]) == 0
    assert json.loads(capsys.readouterr().out)["identity"]["role"] == "custom"
    monkeypatch.setattr(runner, "compute_posterior", lambda **kw: (_ for _ in ()).throw(RuntimeError("failed")))
    assert runner.main(["run", "OXC"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert runner.main(["run", "OXC", "999"]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "failed"


def test_output_exclusive_creation_and_diagnostics(monkeypatch):
    class Buffer(io.StringIO):
        def close(self):
            pass
    buffer = Buffer()
    opened = []
    def open_output(path, mode, **kwargs):
        opened.append(mode)
        return buffer
    monkeypatch.setattr(Path, "open", open_output)
    runner._emit({"diagnostics": np.array([math.inf, -math.inf, math.nan])}, "new-result.json")
    assert opened == ["x"]
    assert json.loads(buffer.getvalue()) == {"diagnostics": ["Infinity", "-Infinity", "NaN"]}


def test_existing_output_and_ambiguous_custom_selection_never_compute(monkeypatch, capsys):
    monkeypatch.setattr(runner, "compute_posterior", lambda **kw: pytest.fail("must not run"))
    source = reader.DATA_ROOT / "inputs" / "OXC_C003.json"
    assert runner.main(["run", "OXC", "--output", str(source)]) == 2
    assert "already exists" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        runner.main(["run", "OXC", "3", "--input", str(source)])
    assert exc.value.code == 2


def test_bounded_real_oxc_cli():
    # The harness may target a checkout or a clean installed wheel; the runner
    # itself never prepends src, loads legacy executables, or reads cached results.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": str(Path(pmx_discrete_posterior.__file__).resolve().parent.parent)}
    run = subprocess.run([sys.executable, "-B", "examples/posterior_scenario.py", "run", "OXC", "C003"],
                         cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert run.returncode == 0, run.stderr + run.stdout
    result = json.loads(run.stdout)
    assert result["status"] == "success" and result["numerical_options_requested"] is None
    assert result["identity"]["input_sha256"] == "67d268cbc4e7268474753403ea195ec52a5ce95986e40772e2e1a8ab03db3d5a"
    probabilities = [row["posterior_probability"] for row in result["state_results"]]
    assert probabilities == pytest.approx([.006296560090573311, .11928606261935508, .5230728861787735, .35134449111129823], abs=5e-6)
    assert sum(probabilities) == pytest.approx(1.)
    assert all("method_details" in row for row in result["state_results"])
