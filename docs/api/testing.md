---
description: API reference for the snapshot-testing helpers downstream converters use.
---

# Testing API reference

Snapshot testing for downstream converters: build a fingerprint of a converted store,
commit it, and assert later runs still match.

The pytest plugin (`--update-snapshots`, `--extended`, the `extended` marker and the
`update_snapshots` fixture) is loaded explicitly with
`pytest_plugins = ["ome_zarr_converters_tools.testing.plugin"]` in a consumer's
`tests/conftest.py`, not through an entry point.

## Running a test

::: ome_zarr_converters_tools.testing.run_converter_test
::: ome_zarr_converters_tools.testing.build_snapshot
::: ome_zarr_converters_tools.testing.compare_snapshots

## Assertion models

::: ome_zarr_converters_tools.testing.FingerprintModel
::: ome_zarr_converters_tools.testing.ImageAssertionModel
::: ome_zarr_converters_tools.testing.RoiAssertionModel
::: ome_zarr_converters_tools.testing.TableAssertionModel
::: ome_zarr_converters_tools.testing.PlateAssertionModel
::: ome_zarr_converters_tools.testing.MultiPlateAssertionModel
::: ome_zarr_converters_tools.testing.MultiSingleImageAssertionModel
