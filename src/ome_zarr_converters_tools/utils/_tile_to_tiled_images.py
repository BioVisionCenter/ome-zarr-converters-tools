"""Functions to build TiledImage models from Tile models."""

from typing import Any

from ome_zarr_converters_tools.models._acquisition import (
    ContextModel,
)
from ome_zarr_converters_tools.models._tile import Tile
from ome_zarr_converters_tools.models._tile_region import (
    TiledImage,
)


def tiled_image_from_tiles(
    tiles: list[Tile],
    context: ContextModel,
    resource: Any | None = None,
) -> list[TiledImage]:
    """Create a TiledImage from a dictionary.

    Args:
        tiles: List of Tile models to build the TiledImage from.
        context: Full context model for the conversion.
        resource: Optional resource to assist in processing.

    Returns:
        A list of TiledImage models created from the tiles.

    """
    split_tiles = context.converter_options.tiling_mode == "no_tiling"
    tiled_images = {}

    if len(tiles) == 0:
        raise ValueError("No tiles provided to build TiledImage.")
    if context.acquisition_details.data_type is not None:
        data_type = context.acquisition_details.data_type
    else:
        data_type = tiles[0].image_loader.find_data_type(resource)

    for tile in tiles:
        suffix = "" if not split_tiles else f"_{tile.fov_name}"
        tile.collection.suffix = suffix
        path = tile.collection.path()
        attributes = tile.model_extra or {}
        if path not in tiled_images:
            tiled_images[path] = TiledImage(
                path=path,
                regions=[],
                data_type=data_type,
                channel_names=tile.channel_names,
                wavelengths=tile.wavelengths,
                pixelsize=tile.pixelsize,
                z_spacing=tile.z_spacing,
                t_spacing=tile.t_spacing,
                axes=tile.axes,
                collection=tile.collection,
                attributes=attributes,
            )
        tiled_images[path].add_tile(tile)

    return list(tiled_images.values())
