"""Utilities to validate a regular grid of tiles."""

from itertools import product
from typing import Literal, NamedTuple

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
    num_x: int
    num_y: int


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


def _find_offset(
    bboxes: list[BBox], ax: Literal["x", "y"], tolerance: float = 0.0
) -> float:
    """Find the offset to snap the tiles to a regular grid."""
    index = 0 if ax == "x" else 1
    sorted_pos = np.sort([box[index] for box in bboxes])
    offsets = np.diff(sorted_pos)
    offsets = offsets[offsets > 1e-6].tolist()

    if len(offsets) == 0:
        return 1.0
    median_offset = float(np.median(offsets))
    if np.allclose(offsets, median_offset, atol=tolerance, rtol=0):
        return median_offset

    unique_offsets = np.unique(offsets)
    raise NotAGridError(
        f"Cannot tile to a regular grid: not all {ax} offsets are "
        f"the same (tolerance={tolerance}): {unique_offsets}"
    )


def _find_grid_size(
    bboxes: list[BBox], offset_x: float, offset_y: float
) -> tuple[int, int]:
    """Find the grid size (number of tiles in x and y)."""
    min_x = min(box.x for box in bboxes)
    min_y = min(box.y for box in bboxes)
    max_x = max(box.x for box in bboxes)
    max_y = max(box.y for box in bboxes)
    num_x = round((max_x - min_x) / offset_x) + 1
    num_y = round((max_y - min_y) / offset_y) + 1
    return num_x, num_y


def check_if_regular_grid(tiles: list[TileSlice], tolerance: float = 0.0) -> GridSetup:
    """Find the grid size of a list of tiles.

    Args:
        tiles: List of TileSlice objects to check.
        tolerance: Maximum allowed deviation from the mean step size, in the same
            units as the tile coordinates (e.g. pixels). Use this when stage
            positioning introduces small jitter so that inter-tile spacing is not
            perfectly uniform. Default 0.0 requires near-exact uniformity.
    """
    bboxes = tiles_to_boxes(tiles)
    if len(tiles) == 1:
        # Trivial case of a single tile
        return GridSetup(
            length_x=bboxes[0].x_len,
            length_y=bboxes[0].y_len,
            offset_x=1.0,
            offset_y=1.0,
            num_x=1,
            num_y=1,
        )
    offset_x = _find_offset(bboxes, "x", tolerance)
    offset_y = _find_offset(bboxes, "y", tolerance)
    num_x, num_y = _find_grid_size(bboxes, offset_x, offset_y)
    return GridSetup(
        length_x=bboxes[0].x_len,
        length_y=bboxes[0].y_len,
        offset_x=offset_x,
        offset_y=offset_y,
        num_x=num_x,
        num_y=num_y,
    )


def _get_start(tile: TileSlice, axis: str) -> float:
    s = tile.roi.get(axis)
    if s is None:
        raise ValueError(f"Tile ROI is missing the '{axis}' axis slice.")
    if s.start is None:
        raise ValueError(f"Tile ROI '{axis}' axis slice is missing start.")
    return float(s.start)


def _match_to_perfect_grid(
    x_grid: float, y_grid: float, perfect_grid_positions: list[GripPoint]
) -> tuple[float, float]:
    """Find the closest perfect grid position to the given grid position."""
    min_distance = float("inf")
    closest_position = (0.0, 0.0)
    for point in perfect_grid_positions:
        distance = np.sqrt((x_grid - point.x) ** 2 + (y_grid - point.y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_position = (point.x, point.y)
    return closest_position


def calculate_snap_to_grid_offset(
    tiles: dict[str, TileSlice],
    tolerance: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Remove overlap from a list of tiles by snapping them to a regular grid."""
    if len(tiles) == 1:
        name = next(iter(tiles))
        return {name: {"x": 0.0, "y": 0.0}}
    grid_setup = check_if_regular_grid(list(tiles.values()), tolerance=tolerance)

    starts_x = [_get_start(t, "x") for t in tiles.values()]
    starts_y = [_get_start(t, "y") for t in tiles.values()]
    min_x = min(starts_x)
    min_y = min(starts_y)

    offsets = {}
    xy_pairs = zip(starts_x, starts_y, strict=True)
    perfect_grid_positions = _build_perfect_grid_points(
        grid_setup.length_x,
        grid_setup.length_y,
        grid_setup.num_x,
        grid_setup.num_y,
        origin_x=min_x,
        origin_y=min_y,
    )
    for name, (x, y) in zip(tiles.keys(), xy_pairs, strict=True):
        x_grid = min_x + ((x - min_x) / grid_setup.offset_x) * grid_setup.length_x
        y_grid = min_y + ((y - min_y) / grid_setup.offset_y) * grid_setup.length_y
        x_grid, y_grid = _match_to_perfect_grid(x_grid, y_grid, perfect_grid_positions)
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
