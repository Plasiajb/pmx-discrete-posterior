"""Portable test selection and optional installed-package provenance checks."""

from importlib.metadata import version
from pathlib import Path
import sysconfig

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-installed", action="store_true",
        help="Fail unless pmx-discrete-posterior 0.2.1.dev20261021 is imported from this environment's site-packages.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "quick: portable component, analytic and fast reference regressions")
    config.addinivalue_line("markers", "slow: full TAC Laplace/curvature or bounded ten-dimensional GHQ")
    if config.getoption("--require-installed"):
        import pmx_discrete_posterior

        package = Path(pmx_discrete_posterior.__file__).resolve()
        site_packages = Path(sysconfig.get_path("purelib")).resolve()
        if not package.is_relative_to(site_packages):
            raise pytest.UsageError(f"Installed-package check imported {package}, outside {site_packages}")
        if version("pmx-discrete-posterior") != "0.2.1.dev20261021":
            raise pytest.UsageError("Installed-package tests require version 0.2.1.dev20261021")


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("slow") is None:
            item.add_marker(pytest.mark.quick)
