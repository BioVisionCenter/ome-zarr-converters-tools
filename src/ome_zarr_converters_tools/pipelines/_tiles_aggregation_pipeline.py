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
    ValidatorModel,
    apply_validator_pipeline,
)


def tiles_aggregation_pipeline(
    tiles: list[Tile],
    *,
    converter_options: ConverterOptions,
    filters: Sequence[FilterModel] | None = None,
    validators: Sequence[ValidatorModel] | None = None,
    resource: Any | None = None,
) -> list[TiledImage]:
    """Filter, validate and aggregate tiles into `TiledImage` objects.

    Applies the optional filters to the tiles, groups what remains into
    `TiledImage` objects according to `converter_options.grouping`, then runs
    the optional validators on each image.

    Args:
        tiles: List of Tile models to process.
        converter_options: ConverterOptions model for the conversion.
        filters: Optional sequence of filter steps to apply to the tiles.
        validators: Optional sequence of validator steps to apply to the tiles.
        resource: Passed unchanged to every loader call (`load_data`,
            `find_data_type`, `preflight`); for `DefaultImageLoader` this is
            the base directory of the raw files.

    Returns:
        A list of TiledImage models created from the processed tiles.
    """
    if filters is not None:
        tiles = apply_filter_pipeline(tiles, filters_config=filters)
    tiled_images = tiled_image_from_tiles(
        tiles=tiles,
        split_per_fov=converter_options.grouping.split_per_fov,
        resource=resource,
    )
    if validators is not None:
        tiled_images = apply_validator_pipeline(
            tiled_images, validators_config=validators, resource=resource
        )
    return tiled_images
