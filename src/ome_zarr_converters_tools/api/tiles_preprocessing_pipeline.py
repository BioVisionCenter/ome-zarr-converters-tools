"""Tile Preprocessing Pipeline API."""

from typing import Any

from ome_zarr_converters_tools.filters import FilterStep, apply_filter_pipeline
from ome_zarr_converters_tools.models import (
    ContextModel,
    Tile,
    TiledImageWithContext,
)
from ome_zarr_converters_tools.utils import tiled_image_from_tiles
from ome_zarr_converters_tools.validators import ValidatorStep, apply_validator_pipeline


def tiles_preprocessing_pipeline(
    tiles: list[Tile],
    *,
    context: ContextModel,
    filters: list[FilterStep] | None = None,
    validators: list[ValidatorStep] | None = None,
    resource: Any | None = None,
) -> list[TiledImageWithContext]:
    """Process tiles through the preprocessing pipeline to create TiledImages.

    This function applies optional filters to the input tiles and then
    constructs TiledImage models from the processed tiles.

    Args:
        tiles: List of Tile models to process.
        context: Full context model for the conversion.
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
        context=context,
        resource=resource,
    )
    if validators is not None:
        tiled_images = apply_validator_pipeline(
            tiled_images, validators_config=validators
        )
    tiled_images = [
        TiledImageWithContext(tiled_image=ti, context=context) for ti in tiled_images
    ]
    return tiled_images
