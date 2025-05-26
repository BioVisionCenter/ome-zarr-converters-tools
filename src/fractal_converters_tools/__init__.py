"""Tooling to build ome-zarr HCS plate converters."""

from importlib.metadata import PackageNotFoundError, version

from fractal_converters_tools._microplate_utils import wellid_to_row_column
from fractal_converters_tools._omezarr_plate_writers import initiate_ome_zarr_plates
from fractal_converters_tools._task_common_models import (
    AdvancedComputeOptions,
    ConvertParallelInitArgs,
)
from fractal_converters_tools._task_compute_tools import generic_compute_task
from fractal_converters_tools._task_init_tools import build_parallelization_list
from fractal_converters_tools._tile import OriginDict, Point, Tile, Vector
from fractal_converters_tools._tiled_image import (
    PathBuilder,
    PlatePathBuilder,
    SimplePathBuilder,
    TiledImage,
)

try:
    __version__ = version("fractal-converters-tools")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

__all__ = [
    "AdvancedComputeOptions",
    "ConvertParallelInitArgs",
    "OriginDict",
    "PathBuilder",
    "PlatePathBuilder",
    "Point",
    "SimplePathBuilder",
    "Tile",
    "TiledImage",
    "Vector",
    "build_parallelization_list",
    "generic_compute_task",
    "initiate_ome_zarr_plates",
    "wellid_to_row_column",
]
