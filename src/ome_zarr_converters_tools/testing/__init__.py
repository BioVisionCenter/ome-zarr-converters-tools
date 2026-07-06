"""Snapshot testing helpers for OME-Zarr converters.

Importing this subpackage does not import pytest; the pytest plugin
(`ome_zarr_converters_tools.testing.plugin`) is loaded separately via a
`pytest11` entry point.
"""

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
