"""Collection setup functions for OME-Zarr converters tools."""

from typing import Protocol

import polars as pl
from ngio import DefaultNgffVersion, ImageInWellPath, NgffVersions
from ngio.hcs import create_empty_plate, open_ome_zarr_plate
from ngio.tables import ConditionTable

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import (
    ImageInPlate,
    OverwriteMode,
    SingleImage,
)
from ome_zarr_converters_tools.models._url_utils import (
    filesystem_for_url,
    join_url_paths,
)
from ome_zarr_converters_tools.pipelines._registry import Registry

_RESERVED_CONDITION_COLUMNS = ("row", "column", "acquisition", "path_in_well")


def _setup_condition_table(
    tiled_images: list[TiledImage],
) -> pl.DataFrame | None:
    """Set up the condition table."""
    with_attributes = [tile for tile in tiled_images if tile.attributes]
    if not with_attributes:
        # No additional attributes, no need to create a condition table
        return None

    key_sets = {tuple(sorted(tile.attributes.keys())) for tile in with_attributes}
    if len(key_sets) > 1:
        keys_by_image = {
            tile.path: sorted(tile.attributes.keys()) for tile in with_attributes
        }
        raise ValueError(
            "All images with attributes must define the same attribute keys to "
            f"build a condition table; got {keys_by_image}. Add the missing "
            "attribute columns to the tiles table (None values are allowed) or "
            "remove the extra ones."
        )
    reserved = set(next(iter(key_sets))) & set(_RESERVED_CONDITION_COLUMNS)
    if reserved:
        raise ValueError(
            f"Attribute keys {sorted(reserved)} collide with the condition "
            f"table's reserved columns {_RESERVED_CONDITION_COLUMNS}. Rename "
            "these columns in the tiles table."
        )

    condition_table: dict[str, list] = {col: [] for col in _RESERVED_CONDITION_COLUMNS}
    for tile in with_attributes:
        num_rows_by_key = {key: len(values) for key, values in tile.attributes.items()}
        if len(set(num_rows_by_key.values())) > 1:
            raise ValueError(
                f"All attributes of image '{tile.path}' must have the same "
                f"number of values; got {num_rows_by_key}. Pad the shorter "
                "attributes with None values."
            )
        for attr_name, attr_value in tile.attributes.items():
            condition_table.setdefault(attr_name, []).extend(attr_value)

        num_rows = next(iter(num_rows_by_key.values()))
        assert isinstance(tile.collection, ImageInPlate)
        condition_table["row"].extend([tile.collection.row] * num_rows)
        condition_table["column"].extend([tile.collection.column] * num_rows)
        condition_table["acquisition"].extend([tile.collection.acquisition] * num_rows)
        condition_table["path_in_well"].extend(
            [tile.collection.path_in_well()] * num_rows
        )
    return pl.DataFrame(condition_table)


def setup_plates(
    zarr_dir: str,
    tiled_images: list[TiledImage],
    ngff_version: NgffVersions = DefaultNgffVersion,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
) -> None:
    """Set up an ImageInPlate collection in the Zarr group."""
    if not isinstance(tiled_images[0].collection, ImageInPlate):
        raise TypeError(
            "setup_plates requires ImageInPlate collections, got "
            f"{type(tiled_images[0].collection).__name__}."
        )
    images_grouped_by_plate: dict[str, list[TiledImage]] = {}
    for tiled_image in tiled_images:
        plate_path = tiled_image.collection.plate_path()
        if plate_path not in images_grouped_by_plate:
            images_grouped_by_plate[plate_path] = []
        images_grouped_by_plate[plate_path].append(tiled_image)

    for plate_path, tile_images in images_grouped_by_plate.items():
        plate_url = join_url_paths(zarr_dir, plate_path)
        if overwrite_mode == OverwriteMode.NO_OVERWRITE:
            fs = filesystem_for_url(plate_url)
            if fs.exists(plate_url):
                raise FileExistsError(
                    f"A Plate already exists at {plate_url} "
                    f"(overwrite_mode={OverwriteMode.NO_OVERWRITE.value}). "
                    f"Set overwrite_mode={OverwriteMode.OVERWRITE.value} to "
                    f"replace it, or "
                    f"overwrite_mode={OverwriteMode.EXTEND.value} to add to it."
                )
        if overwrite_mode == OverwriteMode.OVERWRITE:
            plate = create_empty_plate(
                store=plate_url,
                name=plate_path,
                ngff_version=ngff_version,
                overwrite=True,
                cache=True,
            )
        elif overwrite_mode == OverwriteMode.EXTEND:
            # Distinguish "does not exist yet" from "exists but fails to open":
            # only the former should create a fresh plate. A corrupt/unreadable
            # existing store must raise from open, never be silently overwritten.
            if filesystem_for_url(plate_url).exists(plate_url):
                plate = open_ome_zarr_plate(plate_url, cache=True)
            else:
                plate = create_empty_plate(
                    store=plate_url,
                    name=plate_path,
                    ngff_version=ngff_version,
                    overwrite=False,
                    cache=True,
                )
        else:
            # NO_OVERWRITE — pre-flight check above already raised if plate existed
            plate = create_empty_plate(
                store=plate_url,
                name=plate_path,
                ngff_version=ngff_version,
                overwrite=False,
                cache=True,
            )
        # `max_workers="auto"` opts in to ngio 1.2's default now (the 1.1
        # default, serial reads, emits an `NgioFutureWarning`).
        existing_image = plate.images_paths(max_workers="auto")
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

        # Build the condition table once per plate, from this plate's images only.
        condition_table = _setup_condition_table(tile_images)
        if condition_table is not None:
            condition_table = ConditionTable(table_data=condition_table)
            plate.add_table(
                "condition_table", condition_table, backend="csv", overwrite=True
            )


def setup_singleimage(
    zarr_dir: str,
    tiled_images: list[TiledImage],
    ngff_version: NgffVersions = DefaultNgffVersion,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
) -> None:
    """Set up a SingleImage collection (overwrite-mode enforcement only).

    SingleImage outputs do not need an upfront zarr skeleton — the zarr
    group is created during the compute task. This handler only enforces
    the OverwriteMode contract.
    """
    for tiled_image in tiled_images:
        collection = tiled_image.collection
        if not isinstance(collection, SingleImage):
            raise ValueError(f"Expected SingleImage collection, got {type(collection)}")
        zarr_url = join_url_paths(zarr_dir, collection.path())
        if overwrite_mode == OverwriteMode.NO_OVERWRITE:
            fs = filesystem_for_url(zarr_url)
            if fs.exists(zarr_url):
                raise FileExistsError(
                    f"A zarr already exists at {zarr_url} "
                    f"(overwrite_mode={OverwriteMode.NO_OVERWRITE.value}). "
                    f"Set overwrite_mode={OverwriteMode.OVERWRITE.value} to "
                    f"replace it."
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


_collection_setup_registry: Registry[SetupCollectionFunction] = Registry(
    "Collection setup handler",
    "add_collection_handler",
    {
        "ImageInPlate": setup_plates,
        "SingleImage": setup_singleimage,
    },
)


def add_collection_handler(
    *,
    function: SetupCollectionFunction,
    collection_type: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new collection setup handler.

    The collection setup handler is responsible for setting up the
    collection structure and metadata in the Zarr group.

    Note:
        Registrations are process-global: under `MultiprocessingRunner`,
        worker processes re-import the consumer's modules, so custom handlers
        must be registered at import time of the module that defines them to
        be visible in workers.

    Args:
        collection_type: Name of the collection setup handler. By convention,
            the name of the CollectionInterfaceType, e.g., 'SingleImage'
            or 'ImageInPlate'. Defaults to `function.__name__`.
        function: Function that performs the collection setup step.
        overwrite: Whether to overwrite an existing collection setup step
            with the same name.
    """
    _collection_setup_registry.add(
        function=function, name=collection_type, overwrite=overwrite
    )


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
    """
    setup_function = _collection_setup_registry.get(collection_type)
    return setup_function(
        tiled_images=tiled_images,
        zarr_dir=zarr_dir,
        ngff_version=ngff_version,
        overwrite_mode=overwrite_mode,
    )
