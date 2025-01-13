"""Utilities to validate a regular grid of tiles."""

from dataclasses import dataclass

import numpy as np

from fractal_converters_tools.tiles import Tile


def _first_if_allclose(values: list[float]) -> tuple[bool, float]:
    """Return the first value if all values are close."""
    if len(values) == 0:
        raise ValueError("Empty list of values")

    if np.allclose(values, values[0]):
        return True, values[0]
    return False, 0.0


def _find_grid_size(tiles: list[Tile], offset_x, offset_y) -> tuple[int, int]:
    """Find the grid size of a list of tiles."""
    x = [tile.top_l.x for tile in tiles]
    y = [tile.top_l.y for tile in tiles]

    num_x = int(max(x) // offset_x) + 1
    num_y = int(max(y) // offset_y) + 1
    return num_x, num_y


@dataclass
class GridSetup:
    """Grid setup for a list of tiles.

    Attributes:
        size_x (float): Size of each tile in the x direction.
        size_y (float): Size of each tile in the y direction.
        offset_x (float): Offset of each tile in the x direction.
        offset_y (float): Offset of each tile in the y direction.
        num_x (int): Number of tiles in the x direction.
        num_y (int): Number of tiles in the y direction.

    """

    length_x: float = 0.0
    length_y: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    num_x: int = 0
    num_y: int = 0


def check_if_regular_grid(tiles: list[Tile]) -> tuple[str | None, GridSetup]:
    """Find the grid size of a list of tiles."""
    tiles_length_x = [bbox.bot_r.x - bbox.top_l.x for bbox in tiles]
    check, length_x = _first_if_allclose(tiles_length_x)
    if not check:
        all_lengths = np.unique(tiles_length_x)
        return f"Not all lengths are the same: {all_lengths}", GridSetup()

    tiles_length_y = [bbox.bot_r.y - bbox.top_l.y for bbox in tiles]
    check, length_y = _first_if_allclose(tiles_length_y)
    if not check:
        all_lengths = np.unique(tiles_length_y)
        return f"Not all lengths are the same: {all_lengths}", GridSetup()

    offsets_x, offsets_y, num_neighbors, diag_ious = [], [], [], []
    for i, tile in enumerate(tiles):
        neighbors = 0
        for j in range(len(tiles)):
            if i == j:
                continue
            query_tile = tiles[j]
            if tile.is_overlappingXY(query_tile):
                neighbors += 1
                if np.allclose(tile.top_l.x, query_tile.top_l.x):
                    offsets_y.append(abs(tile.top_l.y - query_tile.top_l.y))
                elif np.allclose(tile.top_l.y, query_tile.top_l.y):
                    offsets_x.append(abs(tile.top_l.x - query_tile.top_l.x))
                else:
                    diag_ious.append(tile.iouXY(query_tile))

        if neighbors == 0:
            # a tile with no neighbors means that this grid:
            # 1. has no overlap (no need to move the bboxes)
            # 2. the list of bboxes does not represent a grid
            return "A tile has no neighbors", GridSetup()

        num_neighbors.append(neighbors)

    check, offset_x = _first_if_allclose(offsets_x)
    if not check:
        # all x_offset are not the same
        unique_offsets = np.unique(offsets_x)
        return f"Not all x offsets are the same: {unique_offsets}", GridSetup()

    check, offset_y = _first_if_allclose(offsets_y)
    if not check:
        # all y_offset are not the same
        unique_offsets = np.unique(offsets_y)
        return f"Not all y offsets are the same: {unique_offsets}", GridSetup()

    check, diag_ious = _first_if_allclose(diag_ious)
    if not check:
        # all diagonal overlaps are not the same
        unique_diag_ious = np.unique(diag_ious)
        return (
            f"Not all diagonal overlaps are the same: {unique_diag_ious}",
            GridSetup(),
        )

    # The number of neighbors should be 3, 5 or 8
    # but let's not check this for now
    # because we can have a grid with missing bboxes
    # if set(np.unique(num_neighbors)) != {3,  5, 8}:
    #    return False, 0., 0.

    num_x, num_y = _find_grid_size(tiles, offset_x, offset_y)
    return None, GridSetup(
        length_x=length_x,
        length_y=length_y,
        offset_x=offset_x,
        offset_y=offset_y,
        num_x=num_x,
        num_y=num_y,
    )
