"""Models and types definitions for the ome_zarr_converters_tools."""

from zarr.storage import FsspecStore, LocalStore

from ome_zarr_converters_tools.core._tile_region import (
    TiledImage,
    TileSlice,
)
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
    CollectionInterface,
    CollectionInterfaceType,
    ImageInPlate,
    SingleImage,
)
from ome_zarr_converters_tools.models._loader import (
    DefaultImageLoader,
    ImageLoaderInterfaceType,
)
from ome_zarr_converters_tools.models._tile import Tile

ConverterStorageType = LocalStore | FsspecStore

__all__ = [
    "AcquisitionDetails",
    "AlignmentCorrections",
    "BackendType",
    "CollectionInterface",
    "CollectionInterfaceType",
    "ContextModel",
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
