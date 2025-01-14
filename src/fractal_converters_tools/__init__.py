"""Tooling to build ome-zarr HCS plate converters."""

from importlib.metadata import PackageNotFoundError, version

from fractal_converters_tools.tiled_image import TiledImage
from fractal_converters_tools.ome_plate_meta import initiate_ome_zarr_plate

try:
    __version__ = version("fractal-converters-tools")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

__all__ = ["TiledImage", "initiate_ome_zarr_plate"]
