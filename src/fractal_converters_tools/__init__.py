"""Tooling to build ome-zarr HCS plate converters."""

from importlib.metadata import PackageNotFoundError, version

from fractal_converters_tools.image import ImageInWell
from fractal_converters_tools.ome_meta_utils import initiate_ome_zarr_plate

try:
    __version__ = version("fractal-converters-tools")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Lorenzo Cerrone"
__email__ = "lorenzo.cerrone@uzh.ch"

__all__ = ["ImageInWell", "initiate_ome_zarr_plate"]
