"""Tooling to build OME-Zarr HCS plate converters for the Fractal platform."""

from importlib.metadata import version

from ome_zarr_converters_tools.core import (
    Tile,
    TiledImage,
    tiles_preprocessing_pipeline,
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
    CollectionInterface,
    CollectionInterfaceType,
    ConverterOptions,
    DefaultImageLoader,
    ImageInPlate,
    ImageLoaderInterfaceType,
    OverwriteMode,
    SingleImage,
    StageCorrections,
)

__version__ = version("ome-zarr-converters-tools")
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

__all__ = [
    "AcquisitionDetails",
    "AcquisitionOptions",
    "CollectionInterface",
    "CollectionInterfaceType",
    "ConvertParallelInitArgs",
    "ConverterOptions",
    "DefaultImageLoader",
    "ImageInPlate",
    "ImageListUpdateDict",
    "ImageLoaderInterfaceType",
    "OverwriteMode",
    "SingleImage",
    "StageCorrections",
    "Tile",
    "TiledImage",
    "converters_tools_models",
    "generic_compute_task",
    "setup_images_for_conversion",
    "tiles_preprocessing_pipeline",
]
