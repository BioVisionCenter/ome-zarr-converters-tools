"""Models and types definitions for the ome_zarr_converters_tools."""

from zarr.storage import FsspecStore, LocalStore

from ome_zarr_converters_tools.models._acquisition import (
    AcquisitionDetails,
    AlignmentCorrections,
    BackendType,
    ContextModel,
    ConverterOptions,
    DefaultNgffVersion,
    NgffVersions,
    OmeZarrOptions,
    OverwriteMode,
    TilingMode,
)
from ome_zarr_converters_tools.models._collection import (
    CollectionInterfaceType,
    ImageInPlate,
    SingleImage,
)
from ome_zarr_converters_tools.models._fractal import ConvertParallelInitArgs
from ome_zarr_converters_tools.models._loader import (
    DefaultImageLoader,
    ImageLoaderInterfaceType,
)
from ome_zarr_converters_tools.models._tile import Tile
from ome_zarr_converters_tools.models._tile_region import (
    TiledImage,
    TileSlice,
)

ConverterStorageType = LocalStore | FsspecStore

__all__ = [
    "AcquisitionDetails",
    "AlignmentCorrections",
    "BackendType",
    "CollectionInterfaceType",
    "ContextModel",
    "ConvertParallelInitArgs",
    "ConverterOptions",
    "ConverterOptions",
    "ConverterStorageType",
    "DefaultImageLoader",
    "DefaultNgffVersion",
    "ImageInPlate",
    "ImageInPlate",
    "ImageLoaderInterfaceType",
    "NgffVersions",
    "OmeZarrOptions",
    "OmeZarrOptions",
    "OverwriteMode",
    "SingleImage",
    "SingleImage",
    "Tile",
    "TileSlice",
    "TiledImage",
    "TilingMode",
]
