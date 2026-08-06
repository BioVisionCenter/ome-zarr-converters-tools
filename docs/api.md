# API Reference

## Core

Core data models and tile operations. This module contains the fundamental building blocks: `Tile` (a single image tile with position, size, and loader), `TiledImage` (a collection of tiles forming one image), and functions for parsing tiles from DataFrames or building them programmatically.

Key exports: `Tile`, `TiledImage`, `TileSlice`, `TileFOVGroup`, `hcs_images_from_dataframe`, `single_images_from_dataframe`, `tiled_image_from_tiles`, `build_dummy_tile`.

::: ome_zarr_converters_tools.core

## Models

Configuration models, collection types, and image loaders. This module defines the Pydantic models used to configure the conversion pipeline (`ConverterOptions`, `AcquisitionDetails`), the collection types that determine output structure (`ImageInPlate`, `SingleImage`), and the image loader interface for custom formats.

Key exports: `ConverterOptions`, `AcquisitionDetails`, `ChannelInfo`, `ImageInPlate`, `SingleImage`, `ImageLoaderInterface`, `DefaultImageLoader`, `Grouping`, `MosaicGrouping`, `PerFovGrouping`, `TilingStrategy`, `AutoTiling`, `SnapToGridTiling`, `SnapToCornersTiling`, `InplaceTiling`, `WriterMode`, `OverwriteMode`, `StagePositionCorrections`, `OmeZarrOptions`, `UserFacingModel`.

::: ome_zarr_converters_tools.models

## URL & Path Utilities

Protocol-aware path helpers that work transparently for local paths and remote `s3://` URLs, and are robust to Windows backslash separators (they always emit POSIX `/`). Use these instead of `os.path` / `pathlib` whenever a location may point at object storage -- they are the URL equivalents of `os.path.join` / `dirname` / `basename` / `isabs` / `glob`, plus fsspec filesystem resolution.

!!! note "`url` vs `path` naming"
    Fields and helpers named `*_url` (e.g. `filesystem_for_url`, `join_url_paths`) take an fsspec-style location that may carry a protocol (`s3://…`) or be a local path. Fields named `*_path` (e.g. `SingleImage.image_path`, `DefaultImageLoader.file_path`) are collection-relative output locations resolved against the zarr directory or a `resource` base. `s3://` support requires the optional `s3` extra: `pip install ome-zarr-converters-tools[s3]`.

Key exports: `join_url_paths`, `parent_url`, `basename_url`, `is_absolute_url`, `glob_url_paths`, `filesystem_for_url`, `find_url_type`, `local_url_to_path`, `UrlType`.

::: ome_zarr_converters_tools.models
    options:
      members:
        - join_url_paths
        - parent_url
        - basename_url
        - is_absolute_url
        - glob_url_paths
        - filesystem_for_url
        - find_url_type
        - local_url_to_path
        - UrlType

## Pipelines

Pipeline functions for aggregation, registration, filtering, validation, and writing. This module orchestrates the full conversion flow: aggregating tiles into images, running registration steps, applying filters, and writing the final OME-Zarr datasets. It also provides extension points for custom filters, validators, and registration steps.

Key exports: `tiles_aggregation_pipeline`, `tiled_image_creation_pipeline`, `build_default_registration_pipeline`, `apply_registration_pipeline`, `apply_filter_pipeline`, `add_filter`, `add_registration_func`, `add_validator`.

::: ome_zarr_converters_tools.pipelines

## Fractal Integration

Utilities for building [Fractal platform](https://fractal-analytics-platform.github.io/fractal-server/) tasks. This module provides `setup_images_for_conversion()` (init task) and `generic_compute_task()` (compute task factory) for parallelizing conversions across a Fractal cluster.

Key exports: `setup_images_for_conversion`, `generic_compute_task`, `ConvertParallelInitArgs`, `AcquisitionOptions`. Lower-level JSON plumbing (`tiled_image_from_json`, `dump_to_json`, …) is also exported from this module for task authors who need to serialize `TiledImage`s across the init/compute boundary; it is intentionally namespaced under `fractal` rather than the package root.

::: ome_zarr_converters_tools.fractal

## Testing

Snapshot-testing helpers shipped for downstream converter test suites. Load the pytest plugin from a consumer's `tests/conftest.py` with `pytest_plugins = ["ome_zarr_converters_tools.testing.plugin"]`; it provides the `--update-snapshots` / `--extended` options, the `extended` marker, and the `update_snapshots` fixture.

::: ome_zarr_converters_tools.testing
