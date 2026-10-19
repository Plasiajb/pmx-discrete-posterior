import ast
from pathlib import Path

import pmx_discrete_posterior

from pmx_discrete_posterior.interfaces.inputs import validate_model
from pmx_discrete_posterior.interfaces.inputs.model_inputs import validate_model as implementation


PACKAGE = Path(pmx_discrete_posterior.__file__).resolve().parent


def test_model_input_validator_has_explicit_source_name_and_same_public_export():
    assert validate_model is implementation
    assert validate_model.__module__.endswith(".inputs.model_inputs")
    assert not (PACKAGE / "interfaces/inputs/model.py").exists()


def test_production_imports_do_not_use_old_model_input_module():
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "model", path
                assert not (node.module or "").endswith("inputs.model"), path


def test_input_entry_is_exported_from_named_implementation():
    from pmx_discrete_posterior.interfaces.inputs import validate_inputs
    from pmx_discrete_posterior.interfaces.inputs.validate import validate_inputs as combined

    assert validate_inputs is combined
    assert validate_inputs.__module__.endswith(".inputs.validate")
    path = PACKAGE / "interfaces/inputs/__init__.py"
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))))


def test_prediction_entry_is_exported_from_named_implementation():
    from pmx_discrete_posterior.models import prediction
    from pmx_discrete_posterior.models.prediction import evaluation

    for name in ("evaluate_prediction", "prepare_prediction_records", "validate_predictions"):
        exported = getattr(prediction, name)
        assert exported is getattr(evaluation, name)
        assert exported.__module__.endswith(".prediction.evaluation")
    path = PACKAGE / "models/prediction/__init__.py"
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))))
