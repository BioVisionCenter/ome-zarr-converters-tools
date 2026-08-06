---
description: API reference for acquisition details, converter options, collections, loaders and runtime settings.
---

# Models API reference

Everything that configures a conversion. See [pipeline
configuration](../guides/pipeline.md) for how these fit together in prose.

## Acquisition details

::: ome_zarr_converters_tools.AcquisitionDetails
::: ome_zarr_converters_tools.ChannelInfo
::: ome_zarr_converters_tools.StageOrientation
::: ome_zarr_converters_tools.DataTypeEnum

## Converter options

::: ome_zarr_converters_tools.ConverterOptions
::: ome_zarr_converters_tools.OmeZarrOptions

## Grouping and tiling

::: ome_zarr_converters_tools.Grouping
::: ome_zarr_converters_tools.MosaicGrouping
::: ome_zarr_converters_tools.PerFovGrouping
::: ome_zarr_converters_tools.TilingStrategy
::: ome_zarr_converters_tools.AutoTiling
::: ome_zarr_converters_tools.InplaceTiling
::: ome_zarr_converters_tools.SnapToCornersTiling
::: ome_zarr_converters_tools.SnapToGridTiling
::: ome_zarr_converters_tools.StagePositionCorrections

## Pyramids and chunking

::: ome_zarr_converters_tools.PyramidLevels
::: ome_zarr_converters_tools.NumberOfLevels
::: ome_zarr_converters_tools.NamedLevels
::: ome_zarr_converters_tools.Scalings
::: ome_zarr_converters_tools.ChunkingStrategy
::: ome_zarr_converters_tools.FixedSizeChunking
::: ome_zarr_converters_tools.FovBasedChunking

## Writing

::: ome_zarr_converters_tools.WriterMode
::: ome_zarr_converters_tools.OverwriteMode
::: ome_zarr_converters_tools.BackendType
::: ome_zarr_converters_tools.NgffVersions

## Collections

::: ome_zarr_converters_tools.CollectionInterface
::: ome_zarr_converters_tools.ImageInPlate
::: ome_zarr_converters_tools.SingleImage

## Image loaders

::: ome_zarr_converters_tools.ImageLoaderInterface
::: ome_zarr_converters_tools.DefaultImageLoader

## Runtime settings

::: ome_zarr_converters_tools.RuntimeSettings
::: ome_zarr_converters_tools.TempJsonOptions
::: ome_zarr_converters_tools.DaskScheduler
::: ome_zarr_converters_tools.DefaultScheduler
::: ome_zarr_converters_tools.SynchronousScheduler
::: ome_zarr_converters_tools.ThreadScheduler
::: ome_zarr_converters_tools.ProcessScheduler

## URL helpers

Paths and `s3://` URLs are handled through the same helpers, so a converter does not
branch on storage backend.

::: ome_zarr_converters_tools.models
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - UrlType
        - find_url_type
        - is_absolute_url
        - join_url_paths
        - parent_url
        - basename_url
        - glob_url_paths
        - filesystem_for_url
        - local_url_to_path
