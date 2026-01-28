"""API for building OME-Zarr converters tasks for Fractal."""

from ome_zarr_converters_tools.fractal._compute_task import (
    ImageListUpdateDict,
    generic_compute_task,
)
from ome_zarr_converters_tools.fractal._init_task import (
    setup_images_for_conversion,
)
from ome_zarr_converters_tools.fractal._models import (
    AcquisitionOptions,
    ChannelInfo,
    ConvertParallelInitArgs,
    PixelSizeModel,
)

# Re-export OverwriteMode from models for backwards compatibility
from ome_zarr_converters_tools.models import OverwriteMode

__all__ = [
    "AcquisitionOptions",
    "ChannelInfo",
    "ConvertParallelInitArgs",
    "ImageListUpdateDict",
    "OverwriteMode",
    "PixelSizeModel",
    "generic_compute_task",
    "setup_images_for_conversion",
]
