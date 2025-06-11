"""Utility functions for the tiles module."""

import copy
from typing import Literal

import numpy as np

from fractal_converters_tools._grid_utils import (
    GridSetup,
    _find_grid_size,
    check_if_regular_grid,
)
from fractal_converters_tools._tile import Point, Tile, TileSpace, Vector


def check_tiles_coplanar(tiles: list[Tile]) -> None:
    """Check if all the Tiles are coplanar on the XY plane."""
    if len(tiles) == 0:
        return None

    if all(tiles[0].is_coplanar(tile) for tile in tiles):
        return None

    raise ValueError(
        "Tiles are not coplanar. "
        "All tiles in a tiled image must have the same Z, C, "
        "and T coordinates."
    )


def _min_point(tiles: list[Tile]) -> Point:
    """Find the minimum point of a list of tiles."""
    min_x = min([tile.top_l.x for tile in tiles])
    min_y = min([tile.top_l.y for tile in tiles])
    return Point(min_x, min_y, z=0, c=0, t=0)


def sort_tiles_by_distance(tiles: list[Tile]) -> list[Tile]:
    """Sort a list of tiles by distance from the origin."""
    min_point = _min_point(tiles)
    return sorted(tiles, key=lambda x: (x.top_l - min_point).lengthXY())


def remove_tiles_offset_xy(tiles: list[Tile]) -> list[Tile]:
    """Remove the offset from a list of tiles in the XY dimensions."""
    min_point = _min_point(tiles)
    offset_vector = Vector(-min_point.x, -min_point.y, z=0, c=0, t=0)
    return [tile.move_by(vec=offset_vector) for tile in tiles]


def remove_tiles_offset_zt(tiles: list[Tile]) -> list[Tile]:
    """Remove the offset from a list of tiles in the Z and T dimensions."""
    return [
        tile.move_by(vec=Vector(x=0, y=0, z=-tile.top_l.z, t=-tile.top_l.t))
        for tile in tiles
    ]


def tiles_to_pixel_space(tiles: list[Tile]) -> list[Tile]:
    """Convert a list of tiles from real space to pixel space."""
    return [tile.to_pixel_space() for tile in tiles]


def tiles_to_real_space(tiles: list[Tile]) -> list[Tile]:
    """Convert a list of tiles from pixel space to real space."""
    return [tile.to_real_space() for tile in tiles]


def swap_xy_tiles(tiles: list[Tile]):
    """Swap x and y of the tiles."""
    swapped_tiles = []
    for tile in tiles:
        top_l = Point(
            tile.top_l.y, tile.top_l.x, z=tile.top_l.z, c=tile.top_l.c, t=tile.top_l.t
        )
        diag = Vector(
            tile.diag.y, tile.diag.x, z=tile.diag.z, c=tile.diag.c, t=tile.diag.t
        )
        swapped_tiles.append(tile.derive_from_diag(top_l, diag))
    return swapped_tiles


def invert_x_tiles(tiles: list[Tile]) -> list[Tile]:
    """Invert the x coordinate of the tiles."""
    inverted_tiles = []
    for tile in tiles:
        top_l = Point(
            -tile.top_l.x, tile.top_l.y, z=tile.top_l.z, c=tile.top_l.c, t=tile.top_l.t
        )
        inverted_tiles.append(tile.derive_from_diag(top_l, tile.diag))
    return inverted_tiles


def invert_y_tiles(tiles: list[Tile]) -> list[Tile]:
    """Invert the y coordinate of the tiles."""
    inverted_tiles = []
    for tile in tiles:
        top_l = Point(
            tile.top_l.x, -tile.top_l.y, z=tile.top_l.z, c=tile.top_l.c, t=tile.top_l.t
        )
        inverted_tiles.append(tile.derive_from_diag(top_l, tile.diag))
    return inverted_tiles


def reset_tiles_origin(tiles: list[Tile]) -> list[Tile]:
    """Reset the tiles to their original position."""
    return [tile.reset_origin() for tile in tiles]


def _remove_tile_XY_overalap(
    ref_tile: Tile,
    query_tile: Tile,
    eps: float = 1e-6,
    speed: float = 0.1,
) -> Tile:
    """Move the query_tile to remove the overlap with the ref_tile."""
    if speed <= 0 or speed > 1:
        raise ValueError("Speed must be in the range (0, 1]")

    lengths = []
    vectors = []
    for corner in ref_tile.cornersXY():
        vec = corner - query_tile.top_l
        moved_bbox = query_tile.move_by(vec)
        iou = moved_bbox.iouXY(ref_tile)
        if iou < eps:
            lengths.append(vec.lengthXY())
            vectors.append(vec)

    min_idx = np.argmin(lengths)
    best_vector: Vector = vectors[min_idx]
    best_vector = best_vector * speed

    best_moved_bbox = query_tile.move_by(best_vector)
    return best_moved_bbox


def resolve_random_tiles_overlap(tiles: list[Tile], eps: float = 1e-6) -> list[Tile]:
    """Remove the overlap from any list of tiles."""
    tiles = copy.deepcopy(tiles)
    n_overlap = np.inf
    while n_overlap > 0:
        tiles = sort_tiles_by_distance(tiles)
        n_overlap = 0
        for i in range(len(tiles)):
            tile = tiles[i]
            for j in range(i + 1, len(tiles)):
                query_tile = tiles[j]
                if tile.is_overlappingXY(query_tile, eps=eps):
                    bbox_no = _remove_tile_XY_overalap(tile, query_tile, speed=1)
                    tiles[j] = bbox_no
                    n_overlap += 1
                    break
    return tiles


def resolve_grid_tiles_overlap(tiles: list[Tile], grid_setup: GridSetup) -> list[Tile]:
    """Remove overlap from a list of tiles that follow a regular grid."""
    tiles = sort_tiles_by_distance(tiles)

    z, c, t = tiles[0].top_l.z, tiles[0].top_l.c, tiles[0].top_l.t

    output_tiles = []
    # The grid tolerance is set to 1% of the grid length
    grid_tolerance = min(grid_setup.length_x, grid_setup.length_y) / 100
    for i in range(grid_setup.num_x):
        for j in range(grid_setup.num_y):
            # X-Y position in the input grid
            x_in = i * grid_setup.offset_x
            y_in = j * grid_setup.offset_y

            # X-Y position in the output grid
            x_out = i * grid_setup.length_x
            y_out = j * grid_setup.length_y

            # Find if a bounding box is close to the (x_in, y_in) position
            point = Point(x_in, y_in, z=z, c=c, t=t)
            distances = [(point - bbox.top_l).lengthXY() for bbox in tiles]
            min_dist = np.min(distances)
            closest_bbox = tiles[np.argmin(distances)]

            if min_dist < grid_tolerance:
                # Move the bounding box to the (x_out, y_out) position
                top_l = Point(x_out, y_out, z=z, c=c, t=t)
                new_tile = closest_bbox.derive_from_diag(top_l, diag=closest_bbox.diag)
                output_tiles.append(new_tile)

    if len(output_tiles) != len(tiles):
        raise ValueError("Something went wrong with the grid tiling resolution.")
    return output_tiles


def _resolve_auto_mode(tiles: list[Tile]) -> tuple[list[Tile], str]:
    """Resolve the overlap of a list of tiles."""
    error_message_or_none, grid_setup = check_if_regular_grid(tiles)
    if error_message_or_none is None:
        return resolve_grid_tiles_overlap(tiles, grid_setup), "grid"
    return resolve_random_tiles_overlap(tiles), "free"


def _resolve_grid_mode(tiles: list[Tile]) -> list[Tile]:
    """Resolve the overlap of a list of tiles."""
    error_message_or_none, grid_setup = check_if_regular_grid(tiles)
    if error_message_or_none is not None:
        raise ValueError(
            "The input tiles are not on a regular grid "
            f"because: {error_message_or_none}. Please set 'mode=free'."
        )
    return resolve_grid_tiles_overlap(tiles, grid_setup)


def _resolve_free_mode(tiles: list[Tile]) -> list[Tile]:
    """Resolve the overlap of a list of tiles."""
    return resolve_random_tiles_overlap(tiles)


def resolve_tiles_overlap(
    tiles: list[Tile],
    mode: Literal["auto", "grid", "free", "none"] = "auto",
) -> tuple[list[Tile], str]:
    """Remove the overlap from any list of tiles."""
    if mode not in ["auto", "grid", "free", "none"]:
        raise ValueError("Mode must be 'auto', 'grid', 'free', or 'none'")

    match mode:
        case "auto":
            return _resolve_auto_mode(tiles)
        case "grid":
            return _resolve_grid_mode(tiles), "grid"
        case "free":
            return _resolve_free_mode(tiles), "free"
        case "none":
            return tiles, "none"


def remove_pixel_gaps(tiles: list[Tile], max_gap: int = 1) -> list[Tile]:
    """Remove evenutal pixel gaps from a grid of tiles."""
    assert len(tiles) > 0, "The input list of tiles is empty"
    assert tiles[0].space == TileSpace.PIXEL, "Tiles must be in pixel space"

    offset_x = tiles[0].diag.x
    offset_y = tiles[0].diag.y
    num_x, num_y = _find_grid_size(tiles, offset_x, offset_y)

    z, c, t = tiles[0].top_l.z, tiles[0].top_l.c, tiles[0].top_l.t
    # The max_gap is set to the diagonal of the tiles
    # So if there is a gap of 1 pixel on the x and y axis
    # the max_gap will be sqrt(2) ~= 1.41 + eps
    max_gap = np.sqrt(max_gap**2 + max_gap**2) + 1e-6

    out_tiles = []
    for i in range(num_x):
        for j in range(num_y):
            x_in = i * offset_x
            y_in = j * offset_y

            point = Point(x_in, y_in, z=z, c=c, t=t)
            distances = [(point - bbox.top_l).lengthXY() for bbox in tiles]
            min_dist = np.min(distances)
            closest_bbox = tiles[np.argmin(distances)]

            if min_dist <= max_gap:
                top_l = Point(x_in, y_in, z=z, c=c, t=t)
                new_tile = closest_bbox.derive_from_diag(top_l, diag=closest_bbox.diag)
                out_tiles.append(new_tile)
    return out_tiles


def standard_stitching_pipe(
    tiles: list[Tile],
    mode: Literal["auto", "grid", "free", "none"] = "auto",
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
) -> list[Tile]:
    """Standard stitching pipe for a list of tiles."""
    # The standard stitching pipe will is implemented for
    # coplanar tiles only.
    check_tiles_coplanar(tiles)
    tiles = copy.deepcopy(tiles)
    if swap_xy:
        tiles = swap_xy_tiles(tiles)
    if invert_x:
        tiles = invert_x_tiles(tiles)
    if invert_y:
        tiles = invert_y_tiles(tiles)

    if any([swap_xy, invert_x, invert_y]):
        tiles = reset_tiles_origin(tiles)

    tiles = sort_tiles_by_distance(tiles)
    tiles = remove_tiles_offset_xy(tiles)
    tiles = remove_tiles_offset_zt(tiles)
    tiles, _mode = resolve_tiles_overlap(tiles, mode=mode)
    tiles = tiles_to_pixel_space(tiles)
    if _mode == "grid":
        tiles = remove_pixel_gaps(tiles)
    return tiles
