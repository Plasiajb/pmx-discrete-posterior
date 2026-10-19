"""Portable core inventory, reference-byte and import-boundary contracts."""

import ast
import hashlib
from pathlib import Path

import pytest
import pmx_discrete_posterior

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(pmx_discrete_posterior.__file__).resolve().parent
CORE_FILES = (
    "__init__.py",
    "interfaces/__init__.py",
    "interfaces/protocols.py",
    "interfaces/results.py",
    "interfaces/validation.py",
    "interfaces/inputs/__init__.py",
    "interfaces/inputs/candidate_states.py",
    "interfaces/inputs/individual_data.py",
    "interfaces/inputs/integration_method.py",
    "interfaces/inputs/model_inputs.py",
    "interfaces/inputs/numerical_options.py",
    "interfaces/inputs/observations.py",
    "interfaces/inputs/priors.py",
    "interfaces/inputs/validate.py",
    "models/__init__.py",
    "models/likelihood.py",
    "models/parameter_resolution.py",
    "models/random_effects.py",
    "models/residuals.py",
    "models/prediction/__init__.py",
    "models/prediction/evaluation.py",
    "models/prediction/initialization.py",
    "models/prediction/linear.py",
    "numerics/__init__.py",
    "numerics/derivatives.py",
    "numerics/ghq.py",
    "numerics/integration.py",
    "numerics/laplace.py",
    "numerics/optimization.py",
    "posterior/__init__.py",
    "posterior/computation.py",
    "posterior/normalization.py",
)
EXAMPLE_FILES = ("examples/posterior_reference.py", "examples/reference_models.py")
REFERENCE_HASHES = {
    "OXC_C003.json": "67d268cbc4e7268474753403ea195ec52a5ce95986e40772e2e1a8ab03db3d5a",
    "TAC_C003.json": "3aa3ae1408c8c8707e3644f1924217350847b3a3555d9dcec90b89a574ab14e6",
}


def test_current_core_source_inventory_is_complete():
    """功能 / Function: 核对当前核心文件并打印审计清单 / Check core files and print an audit inventory.
    输入 / Inputs: src下的Python文件 / Python files under src.
    输出 / Outputs: 必需文件及语法检查，不约束未来附加文件 / Required-file and syntax checks, allowing additions.
    """
    actual = {path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")}
    assert set(CORE_FILES) <= actual, sorted(set(CORE_FILES) - actual)
    for relative in sorted(actual):
        source = PACKAGE / relative
        payload = source.read_bytes()
        ast.parse(payload, filename=str(source))
        print(f"CORE pmx_discrete_posterior/{relative} sha256={hashlib.sha256(payload).hexdigest()}")


@pytest.mark.parametrize("filename", REFERENCE_HASHES)
def test_frozen_reference_json_keeps_original_bytes(filename):
    """功能 / Function: 保留标准参考原字节 / Preserve standard reference bytes.
    输入 / Inputs: 已确认SHA256与本地JSON / Approved SHA256 and local JSON.
    输出 / Outputs: 原始字节哈希断言，不重新序列化 / Raw-byte digest assertion, without reserialization.
    """
    source = ROOT / "tests" / "fixtures" / filename
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == REFERENCE_HASHES[filename]
    print(f"REFERENCE tests/fixtures/{filename} sha256={digest}")


def test_example_source_inventory_is_available():
    """功能 / Function: 核对示例附件所需脚本源文件 / Check source scripts needed by the example delivery.
    输入 / Inputs: 当前两份示例脚本 / The two current example scripts.
    输出 / Outputs: 存在性、语法与清单，不要求放入wheel / Existence, syntax and inventory, without requiring wheel placement.
    """
    for relative in EXAMPLE_FILES:
        source = ROOT / relative
        payload = source.read_bytes()
        ast.parse(payload, filename=str(source))
        print(f"EXAMPLE {relative} sha256={hashlib.sha256(payload).hexdigest()}")


def test_core_and_examples_do_not_statically_import_test_adapters():
    """功能 / Function: 防止核心与示例静态依赖旧测试适配器 / Reject static test-adapter imports in core and examples.
    输入 / Inputs: Python抽象语法树 / Python syntax trees.
    输出 / Outputs: 静态导入边界断言；不替代安装隔离检查 / Static import assertions, not installed isolation checks.
    """
    sources = [*PACKAGE.rglob("*.py"), *(ROOT / relative for relative in EXAMPLE_FILES)]
    forbidden = {"tests", "fixtures", "reference_adapters"}
    violations = []
    for source in sources:
        tree = ast.parse(source.read_bytes(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            else:
                continue
            if any(forbidden.intersection(name.split(".")) for name in names):
                violations.append(f"{source.as_posix()}:{node.lineno}")
    assert not violations, violations
