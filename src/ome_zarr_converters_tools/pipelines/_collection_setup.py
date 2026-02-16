"""Collection setup functions for OME-Zarr converters tools."""

from typing import Protocol

import polars as pl
import zarr
from ngio import DefaultNgffVersion, NgffVersions
from ngio.hcs import create_empty_plate, open_ome_zarr_plate
from ngio.hcs._plate import ImageInWellPath
from ngio.tables import ConditionTable

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import (
    ImageInPlate,
    OverwriteMode,
)
from ome_zarr_converters_tools.models._url_utils import join_url_paths


def _setup_condition_table(
    tiled_images: list[TiledImage],
) -> pl.DataFrame | None:
    """Set up the condition table."""
    condition_table = {
        "row": [],
        "column": [],
        "acquisition": [],
        "path_in_well": [],
    }
    for tile in tiled_images:
        row = tile.collection.row
        col = tile.collection.column
        acq = tile.collection.acquisition
        _num_rows_dict = {}
        for attr_name, attr_value in tile.attributes.items():
            if attr_name not in condition_table:
                condition_table[attr_name] = []
            condition_table[attr_name].extend(attr_value)
            _num_rows_dict[attr_name] = len(attr_value)

        if len(set(_num_rows_dict.values())) > 1:
            raise ValueError(
                "All attributes must have the same number of values. "
                f"Got attributes {tile.attributes}."
            )
        if len(_num_rows_dict) == 0:
            # No additional attributes, no need to create a condition table entry
            continue
        _num_rows = next(iter(_num_rows_dict.values()))
        assert isinstance(tile.collection, ImageInPlate)
        row = tile.collection.row
        col = tile.collection.column
        acq = tile.collection.acquisition
        path_in_well = tile.collection.path_in_well()
        condition_table["row"].extend([row] * _num_rows)
        condition_table["column"].extend([col] * _num_rows)
        condition_table["acquisition"].extend([acq] * _num_rows)
        condition_table["path_in_well"].extend([path_in_well] * _num_rows)

    if set(condition_table.keys()) == {"row", "column", "acquisition", "path_in_well"}:
        # No additional attributes, no need to create a condition table
        return None
    return pl.DataFrame(condition_table)


def setup_plates(
    zarr_dir: str,
    tiled_images: list[TiledImage],
    ngff_version: NgffVersions = DefaultNgffVersion,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
) -> None:
    """Set up an ImageInPlate collection in the Zarr group."""
    assert isinstance(tiled_images[0].collection, ImageInPlate)
    zarr_format = 2 if ngff_version == "0.4" else 3
    if overwrite_mode == OverwriteMode.NO_OVERWRITE:
        mode = "w-"
    elif overwrite_mode == OverwriteMode.OVERWRITE:
        mode = "w"
    else:  # extend
        mode = "a"

    images_grouped_by_plate: dict[str, list[TiledImage]] = {}
    for tiled_image in tiled_images:
        plate_path = tiled_image.collection.plate_path()
        if plate_path not in images_grouped_by_plate:
            images_grouped_by_plate[plate_path] = []
        images_grouped_by_plate[plate_path].append(tiled_image)

    for plate_path, tile_images in images_grouped_by_plate.items():
        plante_url = join_url_paths(zarr_dir, plate_path)
        group = zarr.open_group(store=plante_url, mode=mode, zarr_format=zarr_format)
        try:
            # This can only succeed in "extend" mode if the group already exists
            plate = open_ome_zarr_plate(group, cache=True)
        except Exception:
            plate = create_empty_plate(
                store=group,
                name=plate_path,
                ngff_version=ngff_version,
                overwrite=True,
                cache=True,
            )
        existing_image = plate.images_paths()
        for image in tile_images:
            image_collection = image.collection
            if not isinstance(image_collection, ImageInPlate):
                raise ValueError(
                    f"Expected ImageInPlate collection, got {type(image_collection)}"
                )
            image_in_well = ImageInWellPath(
                row=image_collection.row,
                column=image_collection.column,
                path=image_collection.path_in_well(),
                acquisition_id=image_collection.acquisition,
                acquisition_name=str(image_collection.acquisition),
            )
            image_path = image_collection.image_in_well_path()
            if image_path in existing_image:
                # Image already exists in the plate, skip adding
                # This can only happen in 'extend' mode
                # other modes would have overwritten or raised an error
                continue
            plate.add_image(
                row=image_in_well.row,
                column=image_in_well.column,
                image_path=image_in_well.path,
                acquisition_id=image_in_well.acquisition_id,
                acquisition_name=image_in_well.acquisition_name,
            )
            condition_table = _setup_condition_table(tiled_images)
            if condition_table is not None:
                condition_table = ConditionTable(table_data=condition_table)
                plate.add_table(
                    "condition_table", condition_table, backend="csv", overwrite=True
                )


class SetupCollectionFunction(Protocol):
    """Protocol for collection setup handler functions.

    The function is responsible for setting up the collection structure
    in the zarr store, and creating any necessary metadata.
    """

    __name__: str

    def __call__(
        self,
        zarr_dir: str,
        tiled_images: list[TiledImage],
        ngff_version: NgffVersions = DefaultNgffVersion,
        overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
    ) -> None:
        """Set up the collection in the Zarr store."""
        ...


_collection_setup_registry: dict[str, SetupCollectionFunction] = {
    "ImageInPlate": setup_plates,
}


def add_collection_handler(
    *,
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


def setup_ome_zarr_collection(
    *,
    tiled_images: list[TiledImage],
    collection_type: str,
    zarr_dir: str,
    ngff_version: NgffVersions = DefaultNgffVersion,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
) -> None:
    """Set up the collection in the Zarr group using the specified handler.

    Args:
        tiled_images: List of TiledImage to set up the collection for.
        collection_type: Type of collection setup handler to use.
        zarr_dir: The base directory for the zarr data.
        ngff_version: NGFF version to use for the collection setup.
        overwrite_mode: Overwrite mode to use for the collection setup.

    Returns:
        The list of TiledImage after applying the collection setup handler.
    """
    collection_type = collection_type
    setup_function = _collection_setup_registry.get(collection_type)
    if setup_function is None:
        raise ValueError(
            f"Collection setup handler '{collection_type}' is not registered."
        )
    return setup_function(
        tiled_images=tiled_images,
        zarr_dir=zarr_dir,
        ngff_version=ngff_version,
        overwrite_mode=overwrite_mode,
    )
