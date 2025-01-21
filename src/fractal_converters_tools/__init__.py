"""Tooling to build ome-zarr HCS plate converters."""

from importlib.metadata import PackageNotFoundError, version

from fractal_converters_tools.omezarr_plate_writers import initiate_ome_zarr_plates
from fractal_converters_tools.tile import Point, Tile, Vector
from fractal_converters_tools.tiled_image import TiledImage

try:
    __version__ = version("fractal-converters-tools")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

__all__ = ["Point", "Tile", "TiledImage", "Vector", "initiate_ome_zarr_plates"]
