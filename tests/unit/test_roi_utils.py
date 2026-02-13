"""Unit tests for ROI utility functions."""

import math

from ngio import PixelSize, Roi, RoiSlice

from ome_zarr_converters_tools.core._roi_utils import (
    bulk_roi_union,
    move_roi_by,
    move_to,
    roi_corners,
    roi_to_point_distance,
    roi_to_roi_distance,
    shape_from_rois,
    zero_roi_from_roi,
)


def _make_roi(
    x_start: float, y_start: float, x_len: float, y_len: float, name: str = "test"
) -> Roi:
    """Helper to build a 2D world-space ROI."""
    return Roi(
        name=name,
        slices=[
            RoiSlice(axis_name="x", start=x_start, length=x_len),
            RoiSlice(axis_name="y", start=y_start, length=y_len),
        ],
        space="world",
    )


class TestMoveRoiBy:
    def test_basic_shift(self) -> None:
        roi = _make_roi(10.0, 20.0, 100.0, 200.0)
        moved = move_roi_by(roi, {"x": 5.0, "y": -10.0})
        x_slice = moved.get("x")
        assert x_slice is not None
        assert x_slice.start == 15.0
        assert x_slice.length == 100.0
        y_slice = moved.get("y")
        assert y_slice is not None
        assert y_slice.start == 10.0
        assert y_slice.length == 200.0

    def test_zero_shift(self) -> None:
        roi = _make_roi(10.0, 20.0, 100.0, 200.0)
        moved = move_roi_by(roi, {"x": 0.0, "y": 0.0})
        x_slice = moved.get("x")
        assert x_slice is not None
        assert x_slice.start == 10.0
        y_slice = moved.get("y")
        assert y_slice is not None
        assert y_slice.start == 20.0

    def test_missing_axis_uses_zero(self) -> None:
        roi = _make_roi(10.0, 20.0, 100.0, 200.0)
        moved = move_roi_by(roi, {"x": 5.0})
        x_slice = moved.get("x")
        assert x_slice is not None
        assert x_slice.start == 15.0
        y_slice = moved.get("y")
        assert y_slice is not None
        assert y_slice.start == 20.0  # unchanged


class TestMoveTo:
    def test_move_to_new_origin(self) -> None:
        roi = _make_roi(10.0, 20.0, 100.0, 200.0)
        moved = move_to(roi, {"x": 0.0, "y": 0.0})
        x_slice = moved.get("x")
        assert x_slice is not None
        assert x_slice.start == 0.0
        assert x_slice.length == 100.0
        y_slice = moved.get("y")
        assert y_slice is not None
        assert y_slice.start == 0.0

    def test_partial_move(self) -> None:
        roi = _make_roi(10.0, 20.0, 100.0, 200.0)
        moved = move_to(roi, {"x": 50.0})
        x_slice = moved.get("x")
        assert x_slice is not None
        assert x_slice.start == 50.0
        y_slice = moved.get("y")
        assert y_slice is not None
        assert y_slice.start == 20.0  # unchanged


class TestDistances:
    def test_roi_to_roi_distance(self) -> None:
        roi1 = _make_roi(0.0, 0.0, 10.0, 10.0)
        roi2 = _make_roi(3.0, 4.0, 10.0, 10.0)
        dist = roi_to_roi_distance(roi1, roi2)
        assert math.isclose(dist, 5.0, rel_tol=1e-9)

    def test_roi_to_roi_distance_single_axis(self) -> None:
        roi1 = _make_roi(0.0, 0.0, 10.0, 10.0)
        roi2 = _make_roi(7.0, 100.0, 10.0, 10.0)
        dist = roi_to_roi_distance(roi1, roi2, axes=["x"])
        assert math.isclose(dist, 7.0, rel_tol=1e-9)

    def test_roi_to_point_distance(self) -> None:
        roi = _make_roi(3.0, 4.0, 10.0, 10.0)
        dist = roi_to_point_distance(roi, {"x": 0.0, "y": 0.0})
        assert math.isclose(dist, 5.0, rel_tol=1e-9)


class TestRoiCorners:
    def test_corners(self) -> None:
        roi = _make_roi(10.0, 20.0, 100.0, 200.0)
        c = roi_corners(roi)
        assert len(c) == 4
        assert c[0] == {"x": 10.0, "y": 20.0}
        assert c[3] == {"x": 110.0, "y": 220.0}


class TestZeroRoi:
    def test_zero_roi_from_roi(self) -> None:
        roi = _make_roi(50.0, 100.0, 30.0, 40.0)
        zeroed = zero_roi_from_roi(roi)
        x_slice = zeroed.get("x")
        assert x_slice is not None
        assert x_slice.start == 0.0
        assert x_slice.length == 30.0
        y_slice = zeroed.get("y")
        assert y_slice is not None
        assert y_slice.start == 0.0
        assert y_slice.length == 40.0


class TestBulkRoiUnion:
    def test_union_of_non_overlapping(self) -> None:
        rois = [
            _make_roi(0.0, 0.0, 10.0, 10.0, name="a"),
            _make_roi(10.0, 0.0, 10.0, 10.0, name="b"),
            _make_roi(0.0, 10.0, 10.0, 10.0, name="c"),
        ]
        union = bulk_roi_union(rois)
        x_slice = union.get("x")
        assert x_slice is not None
        assert x_slice.start == 0.0
        assert x_slice.end == 20.0
        y_slice = union.get("y")
        assert y_slice is not None
        assert y_slice.start == 0.0
        assert y_slice.end == 20.0

    def test_union_single_roi(self) -> None:
        roi = _make_roi(5.0, 5.0, 10.0, 10.0, name="single")
        union = bulk_roi_union([roi])
        x_slice = union.get("x")
        assert x_slice is not None
        assert x_slice.start == 5.0
        assert x_slice.length == 10.0


class TestShapeFromRois:
    def test_shape_from_rois(self) -> None:
        rois = [
            _make_roi(0.0, 0.0, 100.0, 200.0, name="a"),
            _make_roi(100.0, 0.0, 100.0, 200.0, name="b"),
        ]
        pixel_size = PixelSize(x=1.0, y=1.0, z=1.0)
        shape = shape_from_rois(rois, axes=["x", "y"], pixel_size=pixel_size)
        assert shape == (200, 200)

    def test_shape_with_pixel_size(self) -> None:
        rois = [
            _make_roi(0.0, 0.0, 1.0, 2.0, name="a"),
        ]
        pixel_size = PixelSize(x=0.5, y=0.5, z=1.0)
        shape = shape_from_rois(rois, axes=["x", "y"], pixel_size=pixel_size)
        assert shape == (2, 4)
