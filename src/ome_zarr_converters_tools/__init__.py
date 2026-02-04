"""Tooling to build OME-Zarr HCS plate converters for the Fractal platform."""

from importlib.metadata import version

from ome_zarr_converters_tools.core import (
    Tile,
    TiledImage,
    tiles_aggregation_pipeline,
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

__version__ = version("ome-zarr-converters-tools")
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

__all__ = [
    "AcquisitionDetails",
    "AcquisitionOptions",
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
