import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(*arguments):
    return subprocess.run([sys.executable, "examples/posterior_reference.py", *arguments],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONPATH": "", "PYTHONIOENCODING": "utf-8"})


def test_ghq_example_outputs_verified_oxc_probabilities():
    run = cli("OXC", "--method", "ghq", "--ghq-orders", "12", "20", "32", "--log-integral-atol", "1e-6",
              "--posterior-atol", "1e-6", "--max-nodes", "1000")
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["integration_method"] == "ghq" and result["status"] == "success"
    assert result["method_details"]["posterior_stability_passed"] is True
    assert len(result["state_results"]) == 4
    assert result["state_results"][2]["posterior_probability"] == pytest.approx(.5228029819563068, abs=1e-8)


@pytest.mark.parametrize("arguments", [("--method", "ghq"), ("--max-nodes", "1000")])
def test_cli_requires_explicit_matching_ghq_settings(arguments):
    run = cli(*arguments)
    assert run.returncode == 2
    assert "GHQ" in run.stderr


def test_tac_budget_failure_is_visible_and_nonzero_exit():
    run = cli("TAC", "--method", "ghq", "--ghq-orders", "3", "5", "7", "--log-integral-atol", "1e-3",
              "--posterior-atol", "1e-4", "--max-nodes", "100")
    assert run.returncode == 1, run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "failed"
    assert all(row["posterior_probability"] is None for row in result["state_results"])
    assert all(row["method_details"]["stage"] == "node_budget" for row in result["state_results"])
