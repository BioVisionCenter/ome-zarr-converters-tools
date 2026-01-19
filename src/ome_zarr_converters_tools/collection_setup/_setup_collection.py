"""Collection setup functions for OME-Zarr converters tools."""

from typing import NamedTuple, Protocol

from ngio import DefaultNgffVersion, NgffVersions

from ome_zarr_converters_tools.collection_setup._plate_setup import setup_plates
from ome_zarr_converters_tools.collection_setup._store_utils import ConverterStorageType
from ome_zarr_converters_tools.models._acquisition import OVERWRITE_MODES
from ome_zarr_converters_tools.models._tile_region import TiledImage


class CollectionConfig(NamedTuple):
    """Configuration for collection setup step."""

    collection_type: str
    store: ConverterStorageType
    ngff_version: NgffVersions
    overwrite_mode: OVERWRITE_MODES


class SetupCollectionFunction(Protocol):
    """Protocol for collection setup handler functions.

    The function is responsible for setting up the collection structure
    in the zarr store, and creating any necessary metadata.
    """

    __name__: str

    def __call__(
        self,
        store: ConverterStorageType,
        tiled_images: list[TiledImage],
        ngff_version: NgffVersions = DefaultNgffVersion,
        overwrite_mode: OVERWRITE_MODES = "no_overwrite",
    ) -> None:
        """Set up the collection in the Zarr store."""
        ...


_collection_setup_registry: dict[str, SetupCollectionFunction] = {
    "ImageInPlate": setup_plates,
}


def add_collection_handler(
    function: SetupCollectionFunction,
    collection_type: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new collection setup handler.

    The collection setup handler is responsible for setting up the
    collection structure and metadata in the Zarr group.

    Args:
        collection_type: Name of the collection setup handler. By convention,
            the name of the CollectionInterfaceType, e.g., 'SingleImage'
            or 'ImageInPlate'.
        function: Function that performs the collection setup step.
        overwrite: Whether to overwrite an existing collection setup step
            with the same name.
    """
    if collection_type is None:
        collection_type = function.__name__
    if not overwrite and collection_type in _collection_setup_registry:
        raise ValueError(
            f"Collection setup handler '{collection_type}' is already registered."
        )
    _collection_setup_registry[collection_type] = function


def setup_collection(
    tiled_images: list[TiledImage],
    setup_collection_step: CollectionConfig,
) -> None:
    """Set up the collection in the Zarr group using the specified handler.

    Args:
        tiled_images: List of TiledImage to set up the collection for.
        setup_collection_step: Configuration for the collection setup step.

    Returns:
        The list of TiledImage after applying the collection setup handler.
    """
    collection_type = setup_collection_step.collection_type
    setup_function = _collection_setup_registry.get(collection_type)
    if setup_function is None:
        raise ValueError(
            f"Collection setup handler '{setup_collection_step.collection_type}' "
            "is not registered."
        )
    return setup_function(
        tiled_images=tiled_images,
        store=setup_collection_step.store,
        ngff_version=setup_collection_step.ngff_version,
        overwrite_mode=setup_collection_step.overwrite_mode,
    )
