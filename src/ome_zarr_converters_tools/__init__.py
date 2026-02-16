"""Shared utilities for building OME-Zarr image converters."""

from importlib.metadata import version

from ome_zarr_converters_tools.core import (
    AttributeType,
    Tile,
    TiledImage,
)
from ome_zarr_converters_tools.fractal import (
    AcquisitionOptions,
    ConvertParallelInitArgs,
    ImageListUpdateDict,
    converters_tools_models,
    generic_compute_task,
    setup_images_for_conversion,
)
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    ChunkingStrategy,
    CollectionInterface,
    CollectionInterfaceType,
    ConverterOptions,
    DataTypeEnum,
    DefaultImageLoader,
    FixedSizeChunking,
    FovBasedChunking,
    ImageInPlate,
    ImageLoaderInterfaceType,
    OmeZarrOptions,
    OverwriteMode,
    SingleImage,
    StageCorrections,
    default_axes_builder,
    join_url_paths,
)
from ome_zarr_converters_tools.pipelines import (
    tiles_aggregation_pipeline,
)

__version__ = version("ome-zarr-converters-tools")
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

__all__ = [
    "AcquisitionDetails",
    "AcquisitionOptions",
    "AttributeType",
    "ChannelInfo",
    "ChunkingStrategy",
    "CollectionInterface",
    "CollectionInterfaceType",
    "ConvertParallelInitArgs",
    "ConverterOptions",
    "DataTypeEnum",
    "DefaultImageLoader",
    "FixedSizeChunking",
    "FovBasedChunking",
    "ImageInPlate",
    "ImageListUpdateDict",
    "ImageLoaderInterfaceType",
    "OmeZarrOptions",
    "OverwriteMode",
    "SingleImage",
    "StageCorrections",
    "Tile",
    "TiledImage",
    "converters_tools_models",
    "default_axes_builder",
    "generic_compute_task",
    "join_url_paths",
    "setup_images_for_conversion",
    "tiles_aggregation_pipeline",
]
