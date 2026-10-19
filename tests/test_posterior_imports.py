import os
from pathlib import Path
import subprocess
import sys


def test_lazy_package_export_preserves_interface_isolation_and_function_identity():
    root = Path(__file__).resolve().parents[1]
    command = """
import sys
import pmx_discrete_posterior as package
from pmx_discrete_posterior.interfaces import protocols, results
assert not any(n.startswith(('pmx_discrete_posterior.models', 'pmx_discrete_posterior.numerics')) for n in sys.modules)
assert not hasattr(package, 'unknown_function')
from pmx_discrete_posterior import compute_posterior
from pmx_discrete_posterior.posterior.computation import compute_posterior as implementation
assert compute_posterior is implementation is package.compute_posterior
assert package.__all__ == ['compute_posterior']
"""
    result = subprocess.run([sys.executable, "-c", command], cwd=root, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": ""})
    assert result.returncode == 0, result.stderr
