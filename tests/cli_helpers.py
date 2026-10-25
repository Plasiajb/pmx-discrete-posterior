"""Shared subprocess helper for the reference CLI, independent of any integrator."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def cli(*arguments):
    return subprocess.run([sys.executable, "-B", "examples/posterior_reference.py", *arguments],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90,
                          env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1",
                               "PYTHONIOENCODING": "utf-8"})
