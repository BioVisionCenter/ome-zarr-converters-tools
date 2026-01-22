"""Tile Preprocessing Pipeline API."""

from typing import Any

from ome_zarr_converters_tools.filters import FilterStep, apply_filter_pipeline
from ome_zarr_converters_tools.models import (
    ConverterOptions,
    Tile,
    TiledImage,
)
from ome_zarr_converters_tools.utils._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.validators import ValidatorStep, apply_validator_pipeline


def tiles_preprocessing_pipeline(
    tiles: list[Tile],
    *,
    converter_options: ConverterOptions,
    filters: list[FilterStep] | None = None,
    validators: list[ValidatorStep] | None = None,
    resource: Any | None = None,
) -> list[TiledImage]:
    """Process tiles through the preprocessing pipeline to create TiledImages.

    This function applies optional filters to the input tiles and then
    constructs TiledImage models from the processed tiles.

    Args:
        tiles: List of Tile models to process.
        converter_options: ConverterOptions model for the conversion.
        filters: Optional list of filter steps to apply to the tiles.
        validators: Optional list of validator steps to apply to the tiles.
        resource: Optional resource to assist in processing.

    Returns:
        A list of TiledImage models created from the processed tiles.
    """
    if filters is not None:
        tiles = apply_filter_pipeline(tiles, filters_config=filters)
    tiled_images = tiled_image_from_tiles(
        tiles=tiles,
        converter_options=converter_options,
        resource=resource,
    )
    if validators is not None:
        tiled_images = apply_validator_pipeline(
            tiled_images, validators_config=validators
        )
    return tiled_images
