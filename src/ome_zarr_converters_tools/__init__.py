"""This tooling will be removed before v07 release."""

from importlib.metadata import version

from ome_zarr_converters_tools.core import (
    Tile,
    TiledImage,
    tiles_preprocessing_pipeline,
)
from ome_zarr_converters_tools.fractal import (
    ImageListUpdateDict,
    generic_compute_task,
    setup_images_for_conversion,
)
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    AcquisitionOptions,
    CollectionInterface,
    CollectionInterfaceType,
    ConverterOptions,
    ConvertParallelInitArgs,
    DefaultImageLoader,
    ImageInPlate,
    ImageLoaderInterfaceType,
    OverwriteMode,
    SingleImage,
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
    "Tile",
    "TiledImage",
    "generic_compute_task",
    "setup_images_for_conversion",
    "tiles_preprocessing_pipeline",
]
