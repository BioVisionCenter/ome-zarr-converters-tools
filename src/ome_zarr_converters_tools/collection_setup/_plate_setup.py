"""Collection setup functions for OME-Zarr converters tools."""

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
    }
    for tile in tiled_images:
        row = tile.collection.row
        col = tile.collection.column
        acq = tile.collection.acquisition
        _num_rows_dict = {}
        for attr_name, attr_value in tile.attributes.items():
            condition_table[attr_name] = attr_value
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
        row = tile.collection.row
        col = tile.collection.column
        acq = tile.collection.acquisition
        condition_table["row"].extend([row] * _num_rows)
        condition_table["column"].extend([col] * _num_rows)
        condition_table["acquisition"].extend([acq] * _num_rows)

    print(condition_table)
    if set(condition_table.keys()) == {"row", "column", "acquisition"}:
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
