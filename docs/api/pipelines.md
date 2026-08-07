---
description: API reference for the aggregation, filtering, validation, registration and writing pipelines.
---

# Pipelines API reference

The stages a conversion runs through, and the extension points for adding your own. See
[pipeline configuration](../guides/pipeline.md) for the prose version.

## Aggregation and creation

::: ome_zarr_converters_tools.tiles_aggregation_pipeline
::: ome_zarr_converters_tools.tiled_image_creation_pipeline

## Filters

Every filter takes a `mode` of `"Include"` or `"Exclude"`. Register a custom one with
`add_filter()`.

::: ome_zarr_converters_tools.pipelines.FilterModel
::: ome_zarr_converters_tools.pipelines.RegexFilter
::: ome_zarr_converters_tools.pipelines.WellFilter
::: ome_zarr_converters_tools.pipelines.FovNameFilter
::: ome_zarr_converters_tools.pipelines.AcquisitionFilter
::: ome_zarr_converters_tools.pipelines.AttributeFilter
::: ome_zarr_converters_tools.pipelines.ChannelFilter
::: ome_zarr_converters_tools.pipelines.ZRangeFilter
::: ome_zarr_converters_tools.pipelines.TRangeFilter
::: ome_zarr_converters_tools.pipelines.apply_filter_pipeline
::: ome_zarr_converters_tools.pipelines.add_filter
::: ome_zarr_converters_tools.pipelines.ImplementedFilters

## Validators

Validators are opt-in because they cost I/O — a probe reads real data.

::: ome_zarr_converters_tools.pipelines.ValidatorModel
::: ome_zarr_converters_tools.pipelines.ShapeDtypeProbeValidator
::: ome_zarr_converters_tools.pipelines.apply_validator_pipeline
::: ome_zarr_converters_tools.pipelines.add_validator
::: ome_zarr_converters_tools.pipelines.ImplementedValidators

## Registration

::: ome_zarr_converters_tools.build_default_registration_pipeline
::: ome_zarr_converters_tools.apply_registration_pipeline
::: ome_zarr_converters_tools.add_registration_func
::: ome_zarr_converters_tools.RegistrationStep

## Collection setup

::: ome_zarr_converters_tools.setup_ome_zarr_collection
::: ome_zarr_converters_tools.setup_singleimage
::: ome_zarr_converters_tools.add_collection_handler

## Writing

::: ome_zarr_converters_tools.write_tiled_image_as_zarr
