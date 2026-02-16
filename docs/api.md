# API Reference

## Core

Core data models and tile operations. This module contains the fundamental building blocks: `Tile` (a single image tile with position, size, and loader), `TiledImage` (a collection of tiles forming one image), and functions for parsing tiles from DataFrames or building them programmatically.

Key exports: `Tile`, `TiledImage`, `TileSlice`, `TileFOVGroup`, `hcs_images_from_dataframe`, `single_images_from_dataframe`, `tiled_image_from_tiles`, `build_dummy_tile`.

::: ome_zarr_converters_tools.core

## Models

Configuration models, collection types, and image loaders. This module defines the Pydantic models used to configure the conversion pipeline (`ConverterOptions`, `AcquisitionDetails`), the collection types that determine output structure (`ImageInPlate`, `SingleImage`), and the image loader interface for custom formats.

Key exports: `ConverterOptions`, `AcquisitionDetails`, `ChannelInfo`, `ImageInPlate`, `SingleImage`, `ImageLoaderInterface`, `DefaultImageLoader`, `TilingMode`, `WriterMode`, `OverwriteMode`, `AlignmentCorrections`, `OmeZarrOptions`.

::: ome_zarr_converters_tools.models

## Pipelines

Pipeline functions for aggregation, registration, filtering, validation, and writing. This module orchestrates the full conversion flow: aggregating tiles into images, running registration steps, applying filters, and writing the final OME-Zarr datasets. It also provides extension points for custom filters, validators, and registration steps.

Key exports: `tiles_aggregation_pipeline`, `tiled_image_creation_pipeline`, `build_default_registration_pipeline`, `apply_registration_pipeline`, `apply_filter_pipeline`, `add_filter`, `add_registration_func`, `add_validator`.

::: ome_zarr_converters_tools.pipelines

## Fractal Integration

Utilities for building [Fractal platform](https://fractal-analytics-platform.github.io/fractal-server/) tasks. This module provides `setup_images_for_conversion()` (init task) and `generic_compute_task()` (compute task factory) for parallelizing conversions across a Fractal cluster.

Key exports: `setup_images_for_conversion`, `generic_compute_task`, `ConvertParallelInitArgs`, `AcquisitionOptions`.

::: ome_zarr_converters_tools.fractal
