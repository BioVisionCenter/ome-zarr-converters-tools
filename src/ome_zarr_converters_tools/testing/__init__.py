"""Snapshot testing helpers for OME-Zarr converters.

The heavy `_snapshot` module (numpy/ngio/pydantic) is imported lazily so that
loading the pytest plugin (`ome_zarr_converters_tools.testing.plugin`, wired via
`pytest_plugins` in each consumer's conftest, not a `pytest11` entry point) does
not pull it in — this keeps plugin load cheap and lets coverage instrument
`_snapshot` when tests first import it.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ome_zarr_converters_tools.testing._snapshot import (
        FingerprintModel,
        ImageAssertionModel,
        MultiPlateAssertionModel,
        MultiSingleImageAssertionModel,
        PlateAssertionModel,
        RoiAssertionModel,
        TableAssertionModel,
        build_snapshot,
        compare_snapshots,
        run_converter_test,
    )

__all__ = [
    "FingerprintModel",
    "ImageAssertionModel",
    "MultiPlateAssertionModel",
    "MultiSingleImageAssertionModel",
    "PlateAssertionModel",
    "RoiAssertionModel",
    "TableAssertionModel",
    "build_snapshot",
    "compare_snapshots",
    "run_converter_test",
]


def __getattr__(name: str):
    """Lazily resolve public names from `_snapshot` on first access."""
    if name in __all__:
        module = importlib.import_module("ome_zarr_converters_tools.testing._snapshot")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
