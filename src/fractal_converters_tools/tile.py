"""This module contains the classes to handle an abstact 5D (t, c, z, y, x) tile."""

from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from dask.array.core import Array


def _find_prec(a: float | int) -> int:
    """Find the precision of a float."""
    if isinstance(a, int):
        return 0
    return len(str(a).split(".")[1])


def _round_ops(a: float | int, b: float | int, op: Callable) -> float | int:
    """Round and apply an operation to two numbers."""
    prec_a, prec_b = _find_prec(a), _find_prec(b)
    prec = max(prec_a, prec_b)
    return round(op(a, b), prec)


def _round_add(a: float | int, b: float | int) -> float | int:
    """Round and add two numbers."""
    return _round_ops(a, b, lambda x, y: x + y)


def _round_sub(a: float | int, b: float | int) -> float | int:
    """Round and subtract two numbers."""
    return _round_ops(a, b, lambda x, y: x - y)


@dataclass
class Vector:
    """Basic 5D vector class."""

    x: int | float
    y: int | float
    z: int | float = 0.0
    c: int = 0
    t: int = 0

    def __add__(self, other: "Vector") -> "Vector":
        """Add two vectors."""
        return Vector(
            _round_add(self.x, other.x),
            _round_add(self.y, other.y),
            _round_add(self.z, other.z),
            self.c + other.c,
            self.t + other.t,
        )

    def __sub__(self, other: "Vector") -> "Vector":
        """Subtract two vectors."""
        return Vector(
            _round_sub(self.x, other.x),
            _round_sub(self.y, other.y),
            _round_sub(self.z, other.z),
            self.c - other.c,
            self.t - other.t,
        )

    def __mul__(self, scalar: float) -> "Vector":
        """Multiply a vector by a scalar."""
        return Vector(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            int(self.c * scalar),
            int(self.t * scalar),
        )

    def normalize(self) -> "Vector":
        """Normalize the vector."""
        length = self.length()
        return Vector(
            self.x / length,
            self.y / length,
            self.z / length,
            int(self.c / length),
            int(self.t / length),
        )

    def length(self) -> float:
        """Compute the length of the vector."""
        return (self.x**2 + self.y**2 + self.z**2 + self.c**2 + self.t**2) ** 0.5

    def to_pixel_space(self, xy_scale: float = 1.0, z_scale: float = 1.0) -> "Vector":
        """Convert the vector to pixel space."""
        x = int(self.x * xy_scale)
        y = int(self.y * xy_scale)
        z = int(self.z * z_scale)
        return Vector(x, y, z=z, c=self.c, t=self.t)

    def to_real_space(self, xy_scale: float = 1.0, z_scale: float = 1.0) -> "Vector":
        """Convert the vector to real space."""
        x = self.x / xy_scale
        y = self.y / xy_scale
        z = self.z / z_scale
        return Vector(x, y, z=z, c=self.c, t=self.t)


@dataclass
class Point:
    """Basic 5D point class."""

    x: int | float
    y: int | float
    z: int | float = 0.0
    c: int = 0
    t: int = 0

    def __add__(self, other: Vector) -> "Point":
        """Add a vector to a point."""
        return Point(
            _round_add(self.x, other.x),
            _round_add(self.y, other.y),
            _round_add(self.z, other.z),
            self.c + other.c,
            self.t + other.t,
        )

    def __sub__(self, other: "Point") -> "Vector":
        """Subtract two points."""
        return Vector(
            _round_sub(self.x, other.x),
            _round_sub(self.y, other.y),
            _round_sub(self.z, other.z),
            self.c - other.c,
            self.t - other.t,
        )

    def to_pixel_space(self, xy_scale: float = 1.0, z_scale: float = 1.0) -> "Point":
        """Convert the point to pixel space."""
        x = int(self.x * xy_scale)
        y = int(self.y * xy_scale)
        z = int(self.z * z_scale)
        return Point(x, y, z=z, c=self.c, t=self.t)

    def to_real_space(self, xy_scale: float = 1.0, z_scale: float = 1.0) -> "Point":
        """Convert the point to real space."""
        x = self.x / xy_scale
        y = self.y / xy_scale
        z = self.z / z_scale
        return Point(x, y, z=z, c=self.c, t=self.t)


class TileLoader(Protocol):
    """Tile loader interface."""

    def __call__(self) -> np.ndarray | Array:
        """Load the tile data into a numpy array."""
        ...


class TileSpace(Enum):
    """Tile space enumeration."""

    PIXEL = "pixel"
    REAL = "real"


OriginDict = namedtuple(
    "OriginDict",
    [
        "x_micrometer_original",
        "y_micrometer_original",
        "z_micrometer_original",
        "t_original",
    ],
    defaults=[0.0, 0.0, 0.0, 0.0],
)


class Tile:
    """5D tile class.

    The tile is defined by two points: the top-left and bottom-right corners.

    The origin attribute is used to keep track of the original tile position when
    moving
    """

    def __init__(
        self,
        top_l: Point,
        diag: Vector,
        origin: OriginDict | None = None,
        space: TileSpace = TileSpace.REAL,
        data_loader: TileLoader | None = None,
        channel_names: list[str] | None = None,
        xy_scale: float = 1.0,
        z_scale: float = 1.0,
    ):
        """Initialize the tile with the top-left corner and the diagonal vector."""
        self._top_l = top_l
        self._diag = diag

        if origin is None:
            self._origin = OriginDict(
                x_micrometer_original=top_l.x, y_micrometer_original=top_l.y
            )
        else:
            self._origin = origin

        self._data_loader = data_loader
        self._space = space
        self._channel_names = channel_names
        self._xy_scale = xy_scale
        self._z_scale = z_scale

    def __repr__(self) -> str:
        """String representation of the tile."""
        x, y, z, c, t = (
            self.top_l.x,
            self.top_l.y,
            self.top_l.z,
            self.top_l.c,
            self.top_l.t,
        )
        s_x, s_y, s_z, s_c, s_t = (
            self.diag.x,
            self.diag.y,
            self.diag.z,
            self.diag.c,
            self.diag.t,
        )
        o_x, o_y = (
            self.origin.x_micrometer_original,
            self.origin.y_micrometer_original,
        )
        return (
            f"Tile(xyzct=({x}, {y}, {z}, {c}, {t}), "
            f"diag=({s_x}, {s_y}, {s_z}, {s_c}, {s_t})), "
            f"origin=({o_x}, {o_y}), "
            f"space={self.space})"
        )

    @property
    def top_l(self) -> Point:
        """Return the top-left corner of the tile."""
        return self._top_l

    @property
    def diag(self) -> Vector:
        """Return the diagonal vector of the tile."""
        return self._diag

    @property
    def bot_r(self) -> Point:
        """Return the bottom-right corner of the tile."""
        return self._top_l + self._diag

    @property
    def origin(self) -> OriginDict:
        """Return the origin reference of the tile."""
        return self._origin

    @property
    def space(self) -> TileSpace:
        """Return the space of the tile."""
        return self._space

    @property
    def channel_names(self) -> list[str] | None:
        """Return the channel names of the tile."""
        return self._channel_names

    @property
    def xy_scale(self) -> float:
        """Return the XY scale of the tile."""
        return self._xy_scale

    @property
    def z_scale(self) -> float:
        """Return the Z scale of the tile."""
        return self._z_scale

    @classmethod
    def from_points(
        cls,
        top_l: Point,
        bot_r: Point,
        origin: OriginDict | None = None,
        space: TileSpace = TileSpace.REAL,
        data_loader: TileLoader | None = None,
        channel_names: list[str] | None = None,
        xy_scale: float = 1.0,
        z_scale: float = 1.0,
    ):
        """Create a tile from two points (top-left and bottom-right corners)."""
        diag = bot_r - top_l
        return cls(
            top_l=top_l,
            diag=diag,
            origin=origin,
            space=space,
            data_loader=data_loader,
            channel_names=channel_names,
            xy_scale=xy_scale,
            z_scale=z_scale,
        )

    def derive_from_diag(self, top_l: Point, diag: Vector) -> "Tile":
        """Create a new tile keeping the origin."""
        return Tile(
            top_l,
            diag,
            origin=self._origin,
            data_loader=self._data_loader,
            space=self._space,
            channel_names=self._channel_names,
            xy_scale=self._xy_scale,
            z_scale=self._z_scale,
        )

    def derive_from_points(self, top_l: Point, bot_r: Point) -> "Tile":
        """Create a new tile keeping the origin."""
        diag = bot_r - top_l
        return Tile(
            top_l,
            diag,
            origin=self._origin,
            data_loader=self._data_loader,
            space=self._space,
            channel_names=self._channel_names,
            xy_scale=self._xy_scale,
            z_scale=self._z_scale,
        )

    def move_by(self, vec: Vector) -> "Tile":
        """Move the tile by a vector keeping the origin reference."""
        return self.derive_from_diag(self.top_l + vec, self.diag)

    def move_to(self, point: Point) -> "Tile":
        """Move the tile to a new point."""
        return self.derive_from_points(point, point + self.diag)

    def reset_origin(self) -> "Tile":
        """Reset the origin reference of the tile to the current position."""
        new_origin = OriginDict(
            x_micrometer_original=self.top_l.x, y_micrometer_original=self.top_l.y
        )
        return Tile(
            self.top_l,
            self.diag,
            origin=new_origin,
            data_loader=self._data_loader,
            space=self._space,
            channel_names=self._channel_names,
            xy_scale=self._xy_scale,
            z_scale=self._z_scale,
        )

    def to_pixel_space(self) -> "Tile":
        """Convert the tile to pixel space."""
        if self.space == TileSpace.PIXEL:
            raise ValueError("Tile is already in pixel space")
        top_l = self.top_l.to_pixel_space(
            xy_scale=1 / self.xy_scale, z_scale=1 / self.z_scale
        )
        diag = self.diag.to_pixel_space(
            xy_scale=1 / self.xy_scale, z_scale=1 / self.z_scale
        )
        return Tile(
            top_l,
            diag,
            origin=self._origin,
            data_loader=self._data_loader,
            space=TileSpace.PIXEL,
            channel_names=self._channel_names,
            xy_scale=self._xy_scale,
            z_scale=self._z_scale,
        )

    def to_real_space(self) -> "Tile":
        """Convert the tile to real space."""
        if self.space == TileSpace.REAL:
            raise ValueError("Tile is already in real space")
        top_l = self.top_l.to_real_space(
            xy_scale=1 / self.xy_scale, z_scale=1 / self.z_scale
        )
        diag = self.diag.to_real_space(
            xy_scale=1 / self.xy_scale, z_scale=1 / self.z_scale
        )
        return Tile(
            top_l,
            diag,
            origin=self._origin,
            data_loader=self._data_loader,
            space=TileSpace.REAL,
            channel_names=self._channel_names,
            xy_scale=self._xy_scale,
            z_scale=self._z_scale,
        )

    def is_coplanar(self, bbox: "Tile", z_tol: float = 1e-6) -> bool:
        """Check if two tiles are coplanar on the XY plane."""
        if abs(self.top_l.z - bbox.top_l.z) > z_tol:
            return False

        if self.top_l.c != bbox.top_l.c:
            return False

        if self.top_l.t != bbox.top_l.t:
            return False

        return True

    def cornersXY(self) -> list[Point]:
        """Return the 4 corners of the tiles box in the top-XY plane."""
        corners = [
            (self.top_l.x, self.top_l.y),
            (self.top_l.x, self.bot_r.y),
            (self.bot_r.x, self.bot_r.y),
            (self.bot_r.x, self.top_l.y),
        ]

        return [
            Point(x, y, self.top_l.z, self.top_l.c, self.top_l.t) for x, y in corners
        ]

    def areaXY(self) -> float:
        """BBBox area in the XY plane."""
        return self.diag.x * self.diag.y

    def intersection_area_XY(self, other: "Tile") -> float:
        """Compute the intersection of two tiles boxes on the XY plane."""
        # Check if the bounding boxes are coplanar
        if not self.is_coplanar(other):
            raise ValueError("Bounding boxes are not coplanar")

        # Compute the intersection
        min_x = max(self.top_l.x, other.top_l.x)
        min_y = max(self.top_l.y, other.top_l.y)

        max_x = min(self.bot_r.x, other.bot_r.x)
        max_y = min(self.bot_r.y, other.bot_r.y)

        if min_x > max_x or min_y > max_y:
            return 0

        return (max_x - min_x) * (max_y - min_y)

    def iouXY(self, other: "Tile") -> float:
        """Compute the intersection over union of tiles in the XY plane."""
        if not self._is_overlappingXY(other):
            return 0

        area_inter = self.intersection_area_XY(other)
        if area_inter <= 0:
            return 0

        vol1 = self.areaXY()
        vol2 = other.areaXY()
        union = vol1 + vol2 - area_inter
        if union <= 0:
            raise ValueError("Union is less than 0")

        return area_inter / union

    def _is_overlappingXY(self, bbox: "Tile") -> bool:
        if self.top_l.x > bbox.bot_r.x or self.bot_r.x < bbox.top_l.x:
            return False
        if self.top_l.y > bbox.bot_r.y or self.bot_r.y < bbox.top_l.y:
            return False
        return True

    def is_overlappingXY(self, bbox: "Tile", eps: float = 1e-6) -> bool:
        """Check if two bounding boxes are overlapping."""
        return self._is_overlappingXY(bbox) and self.iouXY(bbox) > eps

    def load(self) -> np.ndarray | Array:
        """Load the tile data."""
        if self._data_loader is None:
            raise ValueError("No data loader provided.")
        return self._data_loader()

    def shape(self) -> dict[str, int]:
        """Return the shape of the tile."""
        if self.space == TileSpace.REAL:
            raise ValueError("Tile is in real space, cannot return shape")

        return {
            "t": int(self.diag.t),
            "c": int(self.diag.c),
            "z": int(self.diag.z),
            "y": int(self.diag.y),
            "x": int(self.diag.x),
        }

    def slices(self) -> dict[str, slice]:
        """Return the slices of the tile."""
        if self.space == TileSpace.REAL:
            raise ValueError("Tile is in real space, cannot return slices")

        return {
            "t": slice(self.top_l.t, self.bot_r.t),
            "c": slice(self.top_l.c, self.bot_r.c),
            "z": slice(self.top_l.z, self.bot_r.z),
            "y": slice(self.top_l.y, self.bot_r.y),
            "x": slice(self.top_l.x, self.bot_r.x),
        }
