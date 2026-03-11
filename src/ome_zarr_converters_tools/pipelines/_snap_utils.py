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
        if slice_x is None:
            raise ValueError("Tile ROI is missing the 'x' axis slice.")
        start_x = slice_x.start
        length_x = slice_x.length
        if start_x is None or length_x is None:
            raise ValueError("Tile ROI 'x' axis slice is missing start or length.")

        slice_y = tile.roi.get("y")
        if slice_y is None:
            raise ValueError("Tile ROI is missing the 'y' axis slice.")
        start_y = slice_y.start
        length_y = slice_y.length
        if start_y is None or length_y is None:
            raise ValueError("Tile ROI 'y' axis slice is missing start or length.")
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
    sorted_x = np.sort([box.x for box in bboxes])
    offsets_x = np.diff(sorted_x)
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
    sorted_y = np.sort([box.y for box in bboxes])
    offsets_y = np.diff(sorted_y)
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
    # Test 2: Verify that tile positions form the full Cartesian product of unique
    # x and y coordinates (catches slanted, incomplete, or duplicated grids).
    actual_positions = {(round(b.x, 6), round(b.y, 6)) for b in bboxes}
    unique_x = sorted({round(b.x, 6) for b in bboxes})
    unique_y = sorted({round(b.y, 6) for b in bboxes})
    for xi in unique_x:
        for yj in unique_y:
            if (xi, yj) not in actual_positions:
                raise NotAGridError(
                    f"Cannot tile to a regular grid: missing tile at position "
                    f"({xi}, {yj}). The grid may be incomplete or slanted."
                )
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
    if len(tiles) == 1:
        name = next(iter(tiles))
        return {name: {"x": 0.0, "y": 0.0}}
    grid_setup = check_if_regular_grid(list(tiles.values()))

    def _get_start(tile: TileSlice, axis: str) -> float:
        s = tile.roi.get(axis)
        if s is None:
            raise ValueError(f"Tile ROI is missing the '{axis}' axis slice.")
        if s.start is None:
            raise ValueError(f"Tile ROI '{axis}' axis slice is missing start.")
        return float(s.start)

    starts_x = [_get_start(t, "x") for t in tiles.values()]
    starts_y = [_get_start(t, "y") for t in tiles.values()]
    min_x = min(starts_x)
    min_y = min(starts_y)

    offsets = {}
    xy_pairs = zip(starts_x, starts_y, strict=True)
    for name, (x, y) in zip(tiles.keys(), xy_pairs, strict=True):
        x_grid = min_x + ((x - min_x) / grid_setup.offset_x) * grid_setup.length_x
        y_grid = min_y + ((y - min_y) / grid_setup.offset_y) * grid_setup.length_y
        offsets[name] = {"x": x_grid - x, "y": y_grid - y}
    return offsets


def _build_perfect_grid_points(
    length_x: float,
    length_y: float,
    num_x: int,
    num_y: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> list[GripPoint]:
    """Build a grid of points given the grid size, number of points and origin."""
    grid_points = []
    for i, j in product(range(num_x), range(num_y)):
        point = GripPoint(x=origin_x + i * length_x, y=origin_y + j * length_y)
        grid_points.append(point)
    return grid_points


def calculate_snap_to_corner_offset(
    tiles: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    """Remove overlap from a list of tiles by snapping them to a regular grid."""
    boxes = tiles_to_boxes(list(tiles.values()))
    len_x, len_y = boxes[0].x_len, boxes[0].y_len  # Length consistency already checked
    num_x, num_y = len(tiles), len(tiles)  # Upper bound to the number of tiles
    origin_x = min(b.x for b in boxes)
    origin_y = min(b.y for b in boxes)
    perfect_grid = _build_perfect_grid_points(
        len_x, len_y, num_x, num_y, origin_x, origin_y
    )
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
