"""Utilities to validate a regular grid of tiles."""

import math
from itertools import product
from typing import Literal, NamedTuple

import numpy as np

from ome_zarr_converters_tools.core import TileSlice
from ome_zarr_converters_tools.models._errors import (
    ConvertersToolsInternalError,
    ConvertersToolsUserError,
)


class NotAGridError(ConvertersToolsUserError):
    """Error raised when the tiles do not form a regular grid.

    Under `AutoTiling` it is caught and the tiles are snapped to the nearest
    grid corners instead; under `SnapToGridTiling` it reaches the user.
    """


class NotTilableError(ConvertersToolsUserError):
    """Error raised when tiles of different sizes cannot be snapped to a grid.

    No tiling strategy falls back from it: use `InplaceTiling`, or make the
    parser emit tiles of one size.
    """


class BBox(NamedTuple):
    x: float
    y: float
    x_len: float
    y_len: float


class GridPoint(NamedTuple):
    x: float
    y: float


class GridSetup(NamedTuple):
    length_x: float
    length_y: float
    offset_x: float
    offset_y: float
    num_x: int
    num_y: int


def _pixel_extent(tile: TileSlice, axis: str) -> tuple[float, float]:
    """Return `(start, length)` of `axis` for a tile in pixel coordinates."""
    axis_slice = tile.roi.get(axis)
    if axis_slice is None:
        raise ConvertersToolsInternalError(
            f"Tile ROI '{tile.roi.name}' is missing the '{axis}' axis slice."
        )
    if axis_slice.start is None or axis_slice.length is None:
        raise ConvertersToolsInternalError(
            f"Tile ROI '{tile.roi.name}' '{axis}' axis slice is missing start or "
            f"length (start={axis_slice.start}, length={axis_slice.length})."
        )
    return float(axis_slice.start), float(axis_slice.length)


def _describe_size_mismatch(
    tiles: list[TileSlice], boxes: list[BBox], axis: str, reference: float
) -> str:
    """List up to five tiles whose `axis` length differs from `reference`."""
    index = 2 if axis == "x" else 3
    differing = [
        f"'{tile.roi.name}'={box[index]}"
        for tile, box in zip(tiles, boxes, strict=True)
        if not math.isclose(box[index], reference)
    ]
    shown = ", ".join(differing[:5])
    more = f" and {len(differing) - 5} more" if len(differing) > 5 else ""
    return f"{shown}{more}"


def tiles_to_boxes(tiles: list[TileSlice]) -> list[BBox]:
    """Convert a list of TileSlice to a list of Box."""
    if len(tiles) == 0:
        raise ConvertersToolsInternalError(
            "Tiling received an empty tile list; the registration pipeline must "
            "not run a tiling step on an image without regions."
        )

    boxes = []
    for tile in tiles:
        if tile.roi.space == "world":
            raise ConvertersToolsInternalError(
                f"Tile ROI '{tile.roi.name}' is still in world coordinates; tiling "
                "requires pixel coordinates, which the registration pipeline "
                "converts to before any tiling step."
            )
        start_x, length_x = _pixel_extent(tile, "x")
        start_y, length_y = _pixel_extent(tile, "y")
        boxes.append(BBox(x=start_x, y=start_y, x_len=length_x, y_len=length_y))

    if len(boxes) <= 1:
        return boxes
    # Consistency check: all boxes should have the same size
    first_box = boxes[0]
    for axis, reference in (("x", first_box.x_len), ("y", first_box.y_len)):
        lengths = [box.x_len if axis == "x" else box.y_len for box in boxes[1:]]
        if not np.allclose(lengths, reference):
            differing = _describe_size_mismatch(tiles, boxes, axis, reference)
            raise NotTilableError(
                f"Tiles cannot be snapped to a grid because their {axis} lengths "
                f"differ: tile '{tiles[0].roi.name}' has {axis} length "
                f"{reference} but {differing}. Snapping needs tiles of one size "
                "and no tiling strategy falls back from this; use "
                "`InplaceTiling`, or make the parser emit equally sized tiles."
            )
    return boxes


def _find_offset(
    bboxes: list[BBox], ax: Literal["x", "y"], tolerance: float = 0.0
) -> float:
    """Find the offset to snap the tiles to a regular grid."""
    index = 0 if ax == "x" else 1
    sorted_pos = np.sort([box[index] for box in bboxes])
    offsets = np.diff(sorted_pos)
    offsets = offsets[offsets > tolerance].tolist()

    if len(offsets) == 0:
        return 1.0
    median_offset = float(np.median(offsets))
    if np.allclose(offsets, median_offset, atol=tolerance, rtol=0):
        return median_offset

    unique_offsets = np.unique(offsets)
    raise NotAGridError(
        f"Cannot snap the tiles to a regular grid: the {ax} steps between "
        f"neighbouring tiles are not all the same (tolerance={tolerance}); found "
        f"steps {unique_offsets}. Raise the tiling `tolerance` to cover the stage "
        "jitter, or use `SnapToCornersTiling` (or `AutoTiling`, which falls back "
        "to it) to snap each tile to its nearest grid corner instead."
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
        raise ConvertersToolsInternalError(
            f"Tile ROI '{tile.roi.name}' is missing the '{axis}' axis slice."
        )
    if s.start is None:
        raise ConvertersToolsInternalError(
            f"Tile ROI '{tile.roi.name}' '{axis}' axis slice is missing start."
        )
    return float(s.start)


def _match_to_perfect_grid(
    x_grid: float, y_grid: float, perfect_grid_positions: list[GridPoint]
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
) -> list[GridPoint]:
    """Build a grid of points given the grid size, number of points and origin."""
    grid_points = []
    for i, j in product(range(num_x), range(num_y)):
        point = GridPoint(x=origin_x + i * length_x, y=origin_y + j * length_y)
        grid_points.append(point)
    return grid_points


def calculate_snap_to_corner_offset(
    tiles: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    """Remove overlap from a list of tiles by snapping them to a regular grid."""
    boxes = tiles_to_boxes(list(tiles.values()))
    len_x, len_y = boxes[0].x_len, boxes[0].y_len  # Length consistency already checked
    origin_x = min(b.x for b in boxes)
    origin_y = min(b.y for b in boxes)
    # Size the candidate grid from the tiles' bounding box (plus one cell of
    # slack per axis) rather than num_tiles x num_tiles points: a tile's
    # nearest free corner is always within a cell of its own position.
    num_x = math.ceil((max(b.x for b in boxes) - origin_x) / len_x) + 2
    num_y = math.ceil((max(b.y for b in boxes) - origin_y) / len_y) + 2
    # Each tile consumes one grid point, so the grid must hold at least one
    # point per tile even when tiles overlap heavily.
    while num_x * num_y < len(boxes):
        num_x += 1
        num_y += 1
    grid = np.array(
        [
            (origin_x + i * len_x, origin_y + j * len_y)
            for i, j in product(range(num_x), range(num_y))
        ]
    )
    available = np.ones(len(grid), dtype=bool)
    offsets = {}
    for name, box in zip(tiles.keys(), boxes, strict=True):
        distances = (grid[:, 0] - box.x) ** 2 + (grid[:, 1] - box.y) ** 2
        distances[~available] = np.inf
        idx = int(np.argmin(distances))
        available[idx] = False
        offsets[name] = {
            "x": float(grid[idx, 0] - box.x),
            "y": float(grid[idx, 1] - box.y),
        }
    return offsets
