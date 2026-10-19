import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pmx_discrete_posterior


ROOT = Path(__file__).resolve().parents[1]


def test_reference_example_runs_without_old_package_import():
    result = subprocess.run([sys.executable, str(ROOT / "examples/reference_models.py")],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                            env={**os.environ, "PYTHONPATH": "", "PYTHONIOENCODING": "utf-8"}, check=True)
    output = json.loads(result.stdout)
    assert set(output) == {"OXC", "TAC"}
    assert all(len(rows) == 4 for rows in output.values())
    assert all(row["predictions"][0] > 0 and "conditional_log_likelihood" in row
               for rows in output.values() for row in rows)
    print(result.stdout)


def test_source_function_blocks_have_bilingual_contract_docstrings():
    missing = []
    for path in Path(pmx_discrete_posterior.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node) or ""
                if not all(label in doc for label in ("功能 / Function", "输入 / Inputs", "输出 / Outputs")):
                    missing.append(f"{path.name}:{node.lineno} {node.name}")
    assert not missing, "Missing bilingual contracts: " + "; ".join(missing)


def test_complete_oxc_posterior_example_runs_from_new_package():
    result = subprocess.run([sys.executable, str(ROOT / "examples/posterior_reference.py")],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                            env={**os.environ, "PYTHONPATH": "", "PYTHONIOENCODING": "utf-8"}, check=True)
    output = json.loads(result.stdout)
    assert output["model"] == "OXC" and output["status"] == "success"
    assert len(output["state_results"]) == 4
    expected = [.006296560121209, .119286062925454, .523072886625936, .351344490327401]
    for row, probability in zip(output["state_results"], expected):
        assert abs(row["posterior_probability"] - probability) <= 5e-7
        assert row["status"] == "success" and row["prior_probability"] == .25
        assert "events" in row
    print(result.stdout)
