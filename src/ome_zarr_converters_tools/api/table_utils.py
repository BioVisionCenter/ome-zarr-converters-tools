"""Functions to build TiledImage models from Tile models."""

from pathlib import Path
from typing import Any

import pandas as pd
import toml

from ome_zarr_converters_tools.api.tiles_preprocessing_pipeline import (
    tiles_preprocessing_pipeline,
)
from ome_zarr_converters_tools.filters import FilterStep
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ConverterOptions,
    DefaultImageLoader,
    HCSContextModel,
    ImageInPlate,
    Tile,
    TiledImageWithContext,
)
from ome_zarr_converters_tools.validators import ValidatorStep


def _build_default_image_loader(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Build an image loader for a tile row dictionary."""
    image_loader_data = {}
    out_data = {}
    for key, value in data.items():
        if key in DefaultImageLoader.model_fields.keys():
            image_loader_data[key] = value
        else:
            out_data[key] = value
    image_loader = DefaultImageLoader(**image_loader_data)
    out_data["image_loader"] = image_loader
    return out_data


def _build_plate_collection(
    data: dict[str, Any], plate_name: str, acquisition: int
) -> dict[str, Any]:
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
    out_data["collection"] = collection
    return out_data


def _open_hcs_dir(
    acquisition_path: Path,
    table_name: str = "tiles.csv",
    acquisition_details_name: str = "acquisition_details.toml",
) -> tuple[pd.DataFrame, AcquisitionDetails]:
    """Open the HCS directory and read the tiles table.

    Args:
        acquisition_path: Path to the acquisition directory.
        table_name: Name of the table file.
        acquisition_details_name: Name of the acquisition details file.

    Returns:
        A tuple of the tiles DataFrame and AcquisitionDetails model.

    """
    table_path = acquisition_path / table_name
    df = pd.read_csv(table_path)

    with open(acquisition_path / acquisition_details_name) as f:
        acquisition_details_dict = toml.load(f)
        acquisition_details = AcquisitionDetails.model_validate(
            acquisition_details_dict
        )

    return df, acquisition_details


def hcs_images_from_dataframe(
    tiles_table: pd.DataFrame,
    context: HCSContextModel,
    filters: list[FilterStep] | None = None,
    validators: list[ValidatorStep] | None = None,
) -> list[TiledImageWithContext]:
    """Build a list of TiledImages belonging to an HCS acquisition.

    Args:
        tiles_table: DataFrame containing the tiles table.
        context: Full context model for the conversion.
        filters: Optional list of filter steps to apply to the tiles.
        validators: Optional list of validator steps to apply to the tiles.
        resource: Optional resource to pass to image loaders.
    """
    tiles = []
    for _, row in tiles_table.iterrows():
        row_dict = row.to_dict()
        row_dict = _build_default_image_loader(row_dict)
        row_dict = _build_plate_collection(
            row_dict,
            plate_name=context.plate_name,
            acquisition=context.acquisition_id,
        )

        tile = Tile[ImageInPlate, DefaultImageLoader].model_validate(
            row_dict,
            context=context,
        )
        tiles.append(tile)
    tiled_images = tiles_preprocessing_pipeline(
        tiles=tiles,
        context=context,
        validators=validators,
        filters=filters,
    )
    return tiled_images


def hcs_images_from_csv(
    acquisition_path: Path,
    plate_name: str,
    acquisition: int,
    converter_options: ConverterOptions,
    table_name: str = "tiles.csv",
    acquisition_details_name: str = "acquisition_details.toml",
    filters: list[FilterStep] | None = None,
    validators: list[ValidatorStep] | None = None,
) -> list[TiledImageWithContext]:
    """Build tiles for HCS data from a table.

    Args:
        acquisition_path: Path to the acquisition directory.
        plate_name: Name of the plate.
        acquisition: Acquisition index.
        converter_options: Converter options.
        table_name: Name of the table file.
        acquisition_details_name: Name of the acquisition details file.
        filters: Optional list of filter steps to apply to the tiles.
        validators: Optional list of validator steps to apply to the tiles.

    Returns:
        A list of TiledImage models created from the tiles and the context model.
    """
    df, acquisition_details = _open_hcs_dir(
        acquisition_path=acquisition_path,
        table_name=table_name,
        acquisition_details_name=acquisition_details_name,
    )
    context = HCSContextModel(
        plate_name=plate_name,
        acquisition_id=acquisition,
        acquisition_details=acquisition_details,
        converter_options=converter_options,
    )
    images = hcs_images_from_dataframe(
        tiles_table=df,
        context=context,
        filters=filters,
        validators=validators,
    )
    return images
