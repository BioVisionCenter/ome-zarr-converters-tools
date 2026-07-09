"""Pytest plugin shared by the OME-Zarr converter test suites.

Loaded explicitly via `pytest_plugins = ["ome_zarr_converters_tools.testing.plugin"]`
in each consumer's `tests/conftest.py` — deliberately NOT a `pytest11` entry
point (an entry point imports the package during pytest's plugin bootstrap,
before pytest-cov starts, which marks the whole package "module-not-measured"
and deflates coverage). It provides the `--update-snapshots` and `--extended`
options, the `extended` marker and its skip behaviour, and the
`update_snapshots` fixture consumed by `run_converter_test`.
"""

import pytest


def pytest_addoption(parser):
    """Register the snapshot and extended-dataset command-line options."""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate assertion snapshot JSON files",
    )
    parser.addoption(
        "--extended",
        action="store_true",
        default=False,
        help="Run extended tests requiring large test datasets",
    )


def pytest_configure(config):
    """Register the `extended` marker."""
    config.addinivalue_line(
        "markers", "extended: mark test as requiring the extended test datasets"
    )


def pytest_collection_modifyitems(config, items):
    """Skip `extended` tests unless `--extended` was passed."""
    if config.getoption("--extended"):
        return
    skip_marker = pytest.mark.skip(reason="Pass --extended to run extended tests")
    for item in items:
        if "extended" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def update_snapshots(request):
    """Whether `--update-snapshots` was passed for this run."""
    return request.config.getoption("--update-snapshots")
