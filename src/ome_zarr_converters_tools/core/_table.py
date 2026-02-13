"""Functions to build TiledImage models from Tile models."""

from typing import Any

import pandas as pd

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    DefaultImageLoader,
    ImageInPlate,
    SingleImage,
)


def _build_default_image_loader(
    *,
    data: dict[str, Any],
) -> tuple[DefaultImageLoader, dict[str, Any]]:
    """Build an image loader for a tile row dictionary."""
    image_loader_data = {}
    out_data = {}
    for key, value in data.items():
        if key in DefaultImageLoader.model_fields.keys():
            image_loader_data[key] = value
        else:
            out_data[key] = value
    image_loader = DefaultImageLoader(**image_loader_data)
    return image_loader, out_data


def _build_plate_collection(
    *, data: dict[str, Any], plate_name: str, acquisition: int
) -> tuple[ImageInPlate, dict[str, Any]]:
    """Build an ImageInPlate collection for a tile row dictionary."""
    collection_data = {}
    out_data = {}
    for key, value in data.items():
        if key in ImageInPlate.model_fields.keys():
            collection_data[key] = value
        else:
            out_data[key] = value
    collection = ImageInPlate(
        **collection_data, plate_name=plate_name, acquisition=acquisition
    )
    return collection, out_data


def _build_single_image_collection(
    *, data: dict[str, Any]
) -> tuple[SingleImage, dict[str, Any]]:
    """Build a SingleImage collection for a tile row dictionary."""
    collection_data = {}
    out_data = {}
    for key, value in data.items():
        if key in SingleImage.model_fields.keys():
            collection_data[key] = value
        else:
            out_data[key] = value
    collection = SingleImage(**collection_data)
    return collection, out_data


def build_tile_data_and_attributes_from_row(
    *, data: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a tile row dictionary into tile data and attributes."""
    tile_data = {}
    attributes_data = {}
    for key, value in data.items():
        if key in Tile.model_fields.keys():
            tile_data[key] = value
        else:
            # Extra fields will be added as attributes later
            # Since we support condition tables with
            # multiple values per attribute, we need to store them as lists
            attributes_data[key] = [value]
    return tile_data, attributes_data


def hcs_images_from_dataframe(
    *,
    tiles_table: pd.DataFrame,
    acquisition_details: AcquisitionDetails,
    plate_name: str | None = None,
    acquisition_id: int = 0,
) -> list[Tile]:
    """Build a list of TiledImages belonging to an HCS acquisition.

    Args:
        tiles_table: DataFrame containing the tiles table.
        acquisition_details: AcquisitionDetails model for the acquisition.
        plate_name: Optional name of the plate.
        acquisition_id: Acquisition index.
    """
    plate_name = plate_name or "Plate"
    tiles = []
    for _, row in tiles_table.iterrows():
        row_dict = row.to_dict()
        image_loader, row_dict = _build_default_image_loader(data=row_dict)
        collection, row_dict = _build_plate_collection(
            data=row_dict,
            plate_name=plate_name,
            acquisition=acquisition_id,
        )
        tile_data, attributes_data = build_tile_data_and_attributes_from_row(
            data=row_dict
        )
        tile = Tile(
            **tile_data,
            image_loader=image_loader,
            collection=collection,
            acquisition_details=acquisition_details,
            attributes=attributes_data,
        )
        tiles.append(tile)
    return tiles


def single_images_from_dataframe(
    *,
    tiles_table: pd.DataFrame,
    acquisition_details: AcquisitionDetails,
) -> list[Tile]:
    """Build a list of TiledImages belonging to an HCS acquisition.

    Args:
        tiles_table: DataFrame containing the tiles table.
        acquisition_details: AcquisitionDetails model for the acquisition.
    """
    tiles = []
    for _, row in tiles_table.iterrows():
        row_dict = row.to_dict()
        image_loader, row_dict = _build_default_image_loader(data=row_dict)
        collection, row_dict = _build_single_image_collection(
            data=row_dict,
        )
        tile_data, attributes_data = build_tile_data_and_attributes_from_row(
            data=row_dict
        )
        tile = Tile(
            **tile_data,
            image_loader=image_loader,
            collection=collection,
            acquisition_details=acquisition_details,
            attributes=attributes_data,
        )
        tiles.append(tile)
    return tiles
