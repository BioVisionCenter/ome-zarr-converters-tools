---
description: API reference for tiles, tiled images, and building them from a table.
---

# Core API reference

**Tiles, tiled images, and building them from a table.**

The objects the whole pipeline is expressed in. A `Tile` is one region of one file; a
`TiledImage` is the composite that becomes one image in the output store.

## Build tiles from a table

`tiles_aggregation_pipeline()` on the [Pipelines page](pipelines.md) is what converters
call: it applies filters and validators around the bare grouping step,
`tiled_image_from_tiles()`, listed here.

::: ome_zarr_converters_tools.hcs_images_from_dataframe
::: ome_zarr_converters_tools.single_images_from_dataframe
::: ome_zarr_converters_tools.tiled_image_from_tiles

## Tiles

::: ome_zarr_converters_tools.Tile
::: ome_zarr_converters_tools.TileSlice
::: ome_zarr_converters_tools.TileFOVGroup
::: ome_zarr_converters_tools.AttributeType
::: ome_zarr_converters_tools.CollectionInterfaceType
::: ome_zarr_converters_tools.ImageLoaderInterfaceType

## Tiled images

::: ome_zarr_converters_tools.TiledImage

## Fixtures

Synthetic tiles for tests: each renders its FOV name as text, so a wrong placement is
visible in the output.

::: ome_zarr_converters_tools.build_dummy_tile
::: ome_zarr_converters_tools.StartPosition
::: ome_zarr_converters_tools.TileShape
