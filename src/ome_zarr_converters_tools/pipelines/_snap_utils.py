"""Utilities to validate a regular grid of tiles."""

from itertools import product
from typing import NamedTuple

import numpy as np

from ome_zarr_converters_tools.core import TileSlice


class NotAGridError(Exception):
    """Exception raised when tiles do not form a regular grid."""

    pass


class NotTilableError(Exception):
    """Exception raised when tiles cannot be tiled."""

    pass


class BBox(NamedTuple):
    x: float
    y: float
    x_len: float
    y_len: float


class GripPoint(NamedTuple):
    x: float
    y: float


class GridSetup(NamedTuple):
    length_x: float
    length_y: float
    offset_x: float
    offset_y: float


def tiles_to_boxes(tiles: list[TileSlice]) -> list[BBox]:
    """Convert a list of TileSlice to a list of Box."""
    if len(tiles) == 0:
        raise ValueError("Tile list is empty, something went wrong.")

    boxes = []
    for tile in tiles:
        roi = tile.roi
        if roi.space == "world":
            raise ValueError("Tiling is only supported for tiles in pixel coordinates.")
        slice_x = tile.roi.get("x")
        assert slice_x is not None
        start_x = slice_x.start
        length_x = slice_x.length
        assert start_x is not None and length_x is not None

        slice_y = tile.roi.get("y")
        assert slice_y is not None
        start_y = slice_y.start
        length_y = slice_y.length
        assert start_y is not None and length_y is not None
        box = BBox(x=start_x, y=start_y, x_len=length_x, y_len=length_y)
        boxes.append(box)

    if len(boxes) <= 1:
        return boxes
    # Consistency check: all boxes should have the same size
    first_box = boxes[0]
    len_x = [box.x_len for box in boxes[1:]]
    len_y = [box.y_len for box in boxes[1:]]
    if not np.allclose(len_x, first_box.x_len):
        raise NotTilableError(
            "Tiling is not possible when tiles have different x length."
        )
    if not np.allclose(len_y, first_box.y_len):
        raise NotTilableError(
            "Tiling is not possible when tiles have different y length."
        )
    return boxes


def check_if_regular_grid(tiles: list[TileSlice], eps: float = 1e-6) -> GridSetup:
    """Find the grid size of a list of tiles."""
    bboxes = tiles_to_boxes(tiles)
    if len(tiles) == 1:
        # Trivial case of a single tile
        return GridSetup(
            length_x=bboxes[0].x_len,
            length_y=bboxes[0].y_len,
            offset_x=1.0,
            offset_y=1.0,
        )
    # Test 1: Check if all offsets are the same
    pos_top_l_x = [box.x for box in bboxes]
    pos_top_l_x = np.sort(pos_top_l_x)
    offsets_x = np.diff(pos_top_l_x)
    offsets_x = offsets_x[offsets_x > eps].tolist()

    if len(offsets_x) == 0:
        offset_x = 1.0
    elif np.allclose(offsets_x, offsets_x[0]):
        offset_x = offsets_x[0]
    else:
        # Not all offsets are the same
        unique_offsets = np.unique(offsets_x)
        raise NotAGridError(
            "Cannot tile to a regular grid: not all x offsets are "
            f"the same: {unique_offsets}"
        )
    pos_top_l_y = [box.y for box in bboxes]
    pos_top_l_y = np.sort(pos_top_l_y)
    offsets_y = np.diff(pos_top_l_y)
    offsets_y = offsets_y[offsets_y > eps].tolist()
    if len(offsets_y) == 0:
        offset_y = 1.0
    elif np.allclose(offsets_y, offsets_y[0]):
        offset_y = offsets_y[0]
    else:
        # Not all offsets are the same
        unique_offsets = np.unique(offsets_y)
        raise NotAGridError(
            "Cannot tile to a regular grid: not all y offsets are "
            f"the same: {unique_offsets}"
        )
    # Test 2: Check the edge case where the grid is slanted
    if len(bboxes) > 2:
        for bbox in bboxes[1:]:
            slant_x = bbox.x - bboxes[0].x
            slant_y = bbox.y - bboxes[0].y
            if slant_x < eps or slant_y < eps:
                # All good the grid is not slanted
                break
        else:
            raise NotAGridError("Cannot tile to a regular grid: the grid is slanted.")
    return GridSetup(
        length_x=bboxes[0].x_len,
        length_y=bboxes[0].y_len,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def calculate_snap_to_grid_offset(
    tiles: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    """Remove overlap from a list of tiles by snapping them to a regular grid."""
    grid_setup = check_if_regular_grid(list(tiles.values()))
    offsets = {}
    for name, tile in tiles.items():
        # Find the x grid position
        slice_x = tile.roi.get("x")
        assert slice_x is not None
        x = slice_x.start
        assert x is not None
        x_grid = (x / grid_setup.offset_x) * grid_setup.length_x
        # Find the y grid position
        slice_y = tile.roi.get("y")
        assert slice_y is not None
        y = slice_y.start
        assert y is not None
        x_grid = (x / grid_setup.offset_x) * grid_setup.length_x
        y_grid = (y / grid_setup.offset_y) * grid_setup.length_y
        # Store the new start positions
        offsets[name] = {"x": x_grid - x, "y": y_grid - y}
    return offsets


def _build_perfect_grid_points(
    length_x: float, length_y: float, num_x: int, num_y: int
) -> list[GripPoint]:
    """Build a grid of points given the grid size and number of points."""
    grid_points = []
    for i, j in product(range(num_x), range(num_y)):
        point = GripPoint(x=i * length_x, y=j * length_y)
        grid_points.append(point)
    return grid_points


def calculate_snap_to_corner_offset(
    tiles: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    """Remove overlap from a list of tiles by snapping them to a regular grid."""
    boxes = tiles_to_boxes(list(tiles.values()))
    len_x, len_y = boxes[0].x_len, boxes[0].y_len  # Length consistency already checked
    num_x, num_y = len(tiles), len(tiles)  # Upper buound to the number of tiles
    perfect_grid = _build_perfect_grid_points(len_x, len_y, num_x, num_y)
    offsets = {}
    for name, box in zip(tiles.keys(), boxes, strict=True):
        min_distance = float("inf")
        min_id = -1
        for i, point in enumerate(perfect_grid):
            distance = np.sqrt((box.x - point.x) ** 2 + (box.y - point.y) ** 2)
            if distance < min_distance:
                min_distance = distance
                min_id = i
                offsets[name] = {"x": point.x - box.x, "y": point.y - box.y}
        # remove the used point from the perfect grid
        if min_id == -1:
            raise ValueError("Could not find a matching point in the perfect grid.")
        perfect_grid.pop(min_id)
    return offsets
