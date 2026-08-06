"""Functions to build TiledImage models from Tile models."""

from typing import Any

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.core._tile_region import TiledImage


def tiled_image_from_tiles(
    *,
    tiles: list[Tile],
    split_per_fov: bool = False,
    resource: Any | None = None,
) -> list[TiledImage]:
    """Build a list of TiledImages from a list of Tiles.

    Args:
        tiles: List of Tile models to build the TiledImages from.
        split_per_fov: If True, each field of view becomes its own TiledImage;
            if False, all fields of view are aggregated into one mosaic image.
        resource: Optional resource to assist in processing.

    Returns:
        A list of TiledImage models created from the tiles.

    """
    tiled_images = {}

    if len(tiles) == 0:
        raise ValueError(
            "No tiles provided to build TiledImage. If filters are configured, "
            "they may have removed every tile; check that their patterns and "
            "Include/Exclude modes match the parsed acquisition."
        )
    data_type = tiles[0].find_data_type(resource=resource)
    for tile in tiles:
        if split_per_fov:
            suffix = f"_{tile.fov_name}"
            add_translation = True
        else:
            suffix = ""
            add_translation = False
        tile.collection.set_suffix(suffix)
        path = tile.collection.path()
        if path not in tiled_images:
            acquisition_details = tile.acquisition_details
            tiled_images[path] = TiledImage(
                path=path,
                regions=[],
                data_type=data_type,
                channels=acquisition_details.channels,
                xy_pixel_size=acquisition_details.xy_pixel_size,
                z_spacing=acquisition_details.z_spacing,
                t_spacing=acquisition_details.t_spacing,
                axes=acquisition_details.axes,
                collection=tile.collection,
                attributes=tile.attributes,
            )
        tiled_images[path].add_tile(tile, add_translation=add_translation)

    return list(tiled_images.values())
