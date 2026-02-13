"""Tile Preprocessing Pipeline API."""

from collections.abc import Sequence
from typing import Any

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.core._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.models import ConverterOptions
from ome_zarr_converters_tools.pipelines._filters import (
    FilterModel,
    apply_filter_pipeline,
)
from ome_zarr_converters_tools.pipelines._validators import (
    ValidatorStep,
    apply_validator_pipeline,
)


def tiles_aggregation_pipeline(
    tiles: list[Tile],
    *,
    converter_options: ConverterOptions,
    filters: Sequence[FilterModel] | None = None,
    validators: Sequence[ValidatorStep] | None = None,
    resource: Any | None = None,
) -> list[TiledImage]:
    """Process tiles and aggregates them into TiledImages.

    This function applies optional filters to the input tiles and then
    constructs TiledImage models from the processed tiles.

    Args:
        tiles: List of Tile models to process.
        converter_options: ConverterOptions model for the conversion.
        filters: Optional sequence of filter steps to apply to the tiles.
        validators: Optional sequence of validator steps to apply to the tiles.
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
