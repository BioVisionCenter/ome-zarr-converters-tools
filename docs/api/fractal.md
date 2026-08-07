---
description: API reference for packaging a converter as a Fractal init/compute task pair.
---

# Fractal API reference

Utilities for exposing a converter as a [Fractal](https://fractal-analytics-platform.github.io/)
task. See [converters as Fractal tasks](../guides/fractal_tasks.md) for the walkthrough.

## Task entry points

::: ome_zarr_converters_tools.setup_images_for_conversion
::: ome_zarr_converters_tools.generic_compute_task
::: ome_zarr_converters_tools.exec_compound_task
::: ome_zarr_converters_tools.ConvertParallelInitArgs
::: ome_zarr_converters_tools.ImageListUpdateDict

## UI-facing models

These are the models Fractal renders argument forms from, so their JSON schemas are part
of the public contract.

::: ome_zarr_converters_tools.AcquisitionOptions
::: ome_zarr_converters_tools.ChannelInfoUI
::: ome_zarr_converters_tools.fractal.PixelSizeModel
::: ome_zarr_converters_tools.converters_tools_models

## Runners

::: ome_zarr_converters_tools.RunnerType
::: ome_zarr_converters_tools.SequentialRunner
::: ome_zarr_converters_tools.ThreadedRunner
::: ome_zarr_converters_tools.MultiprocessingRunner

## JSON plumbing

`TiledImage` objects are handed between the init and compute tasks as JSON on disk.

::: ome_zarr_converters_tools.fractal.dump_to_json
::: ome_zarr_converters_tools.fractal.dump_json_str
::: ome_zarr_converters_tools.fractal.tiled_image_from_json
::: ome_zarr_converters_tools.fractal.tiled_image_from_json_str
::: ome_zarr_converters_tools.fractal.remove_json
::: ome_zarr_converters_tools.fractal.cleanup_if_exists
