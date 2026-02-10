"""Functions to build TiledImage models from Tile models."""

from typing import Any

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import ConverterOptions, TilingMode


def tiled_image_from_tiles(
    *,
    tiles: list[Tile],
    converter_options: ConverterOptions,
    resource: Any | None = None,
) -> list[TiledImage]:
    """Create a TiledImage from a dictionary.

    Args:
        tiles: List of Tile models to build the TiledImage from.
        converter_options: ConverterOptions model for the conversion.
        resource: Optional resource to assist in processing.

    Returns:
        A list of TiledImage models created from the tiles.

    """
    split_tiles = converter_options.tiling_mode == TilingMode.NO_TILING
    tiled_images = {}

    if len(tiles) == 0:
        raise ValueError("No tiles provided to build TiledImage.")
    data_type = tiles[0].find_data_type(resource=resource)
    for tile in tiles:
        suffix = "" if not split_tiles else f"_{tile.fov_name}"
        tile.collection._suffix = suffix
        path = tile.collection.path()
        if path not in tiled_images:
            acquisition_details = tile.acquisition_details
            tiled_images[path] = TiledImage(
                path=path,
                regions=[],
                data_type=data_type,
                channels=acquisition_details.channels,
                pixelsize=acquisition_details.pixelsize,
                z_spacing=acquisition_details.z_spacing,
                t_spacing=acquisition_details.t_spacing,
                axes=acquisition_details.axes,
                collection=tile.collection,
                attributes=tile.attributes,
            )
        tiled_images[path].add_tile(tile)

    return list(tiled_images.values())
