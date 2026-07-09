"""Unit tests for registration module (alignment, tiling, snap utils)."""

import numpy as np
import pytest
from ngio import Roi, RoiSlice

from ome_zarr_converters_tools.core._dummy_tiles import (
    DummyLoader,
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile_region import TiledImage, TileSlice
from ome_zarr_converters_tools.core._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    AutoTiling,
    ChannelInfo,
    ConverterOptions,
    InplaceTiling,
    NoTiling,
    SingleImage,
    SnapToGridTiling,
    StagePositionCorrections,
)
from ome_zarr_converters_tools.pipelines._alignment import (
    _align_t_regions,
    _align_xy_regions,
    _align_z_regions,
    apply_align_to_pixel_grid,
    apply_fov_alignment_corrections,
    apply_remove_offsets,
)
from ome_zarr_converters_tools.pipelines._snap_utils import (
    BBox,
    NotAGridError,
    calculate_snap_to_corner_offset,
    calculate_snap_to_grid_offset,
    check_if_regular_grid,
    tiles_to_boxes,
)
from ome_zarr_converters_tools.pipelines._tiling import (
    _find_tiling,
    apply_mosaic_tiling,
)


def _make_pixel_tile_slice(
    x_start: float, y_start: float, x_len: float, y_len: float, name: str = "FOV"
) -> TileSlice:
    """Helper: TileSlice with pixel-space ROI."""
    roi = Roi(
        name=name,
        slices=[
            RoiSlice(axis_name="x", start=x_start, length=x_len),
            RoiSlice(axis_name="y", start=y_start, length=y_len),
        ],
        space="pixel",
    )
    loader = DummyLoader(shape=TileShape(x=int(x_len), y=int(y_len)), text=name)
    return TileSlice(roi=roi, image_loader=loader)


def _make_world_tile_slice(
    x_start: float, y_start: float, x_len: float, y_len: float, name: str = "FOV"
) -> TileSlice:
    """Helper: TileSlice with world-space ROI."""
    roi = Roi(
        name=name,
        slices=[
            RoiSlice(axis_name="x", start=x_start, length=x_len),
            RoiSlice(axis_name="y", start=y_start, length=y_len),
        ],
        space="world",
    )
    loader = DummyLoader(shape=TileShape(x=int(x_len), y=int(y_len)), text=name)
    return TileSlice(roi=roi, image_loader=loader)


def _make_tiled_image(regions: list[TileSlice], pixelsize: float = 1.0) -> TiledImage:
    """Helper: build a TiledImage from TileSlices."""
    collection = SingleImage(image_path="test_image")
    return TiledImage(
        regions=regions,
        path="test_image",
        data_type="uint8",
        axes=["x", "y"],
        collection=collection,
        pixelsize=pixelsize,
    )


# --- Alignment tests ---


class TestAlignment:
    def test_align_xy_regions(self) -> None:
        regions = [
            _make_world_tile_slice(10.0, 20.0, 100.0, 100.0, "FOV_0"),
            _make_world_tile_slice(15.0, 25.0, 100.0, 100.0, "FOV_0"),
            _make_world_tile_slice(12.0, 22.0, 100.0, 100.0, "FOV_0"),
        ]
        aligned = _align_xy_regions(regions)
        for region in aligned:
            x_slice = region.roi.get("x")
            assert x_slice is not None
            assert x_slice.start == 10.0
            y_slice = region.roi.get("y")
            assert y_slice is not None
            assert y_slice.start == 20.0

    def test_align_z_raises(self) -> None:
        regions = [_make_world_tile_slice(0.0, 0.0, 10.0, 10.0)]
        with pytest.raises(NotImplementedError, match="not implemented"):
            _align_z_regions(regions)

    def test_align_t_raises(self) -> None:
        regions = [_make_world_tile_slice(0.0, 0.0, 10.0, 10.0)]
        with pytest.raises(NotImplementedError, match="not implemented"):
            _align_t_regions(regions)

    def test_apply_fov_alignment_corrections(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        coll = SingleImage(image_path="test_image")
        tiles = [
            build_dummy_tile(
                fov_name="FOV_0",
                start=StartPosition(x=10, y=20),
                shape=TileShape(x=64, y=64, z=1, c=1, t=1),
                collection=coll,
                acquisition_details=acq,
            ),
            build_dummy_tile(
                fov_name="FOV_0",
                start=StartPosition(x=15, y=25),
                shape=TileShape(x=64, y=64, z=1, c=1, t=1),
                collection=coll,
                acquisition_details=acq,
            ),
        ]
        images = tiled_image_from_tiles(
            tiles=tiles, converter_options=ConverterOptions()
        )
        corrections = StagePositionCorrections(align_xy=True)
        result = apply_fov_alignment_corrections(images[0], corrections)
        x_slice = result.regions[0].roi.get("x")
        assert x_slice is not None
        first_x = x_slice.start
        for region in result.regions:
            x_slice = region.roi.get("x")
            assert x_slice is not None
            assert x_slice.start == first_x

    def test_apply_align_to_pixel_grid_floor(self) -> None:
        regions = [_make_world_tile_slice(10.7, 20.3, 100.0, 100.0, "FOV")]
        img = _make_tiled_image(regions, pixelsize=1.0)
        result = apply_align_to_pixel_grid(img, mode="floor")
        roi = result.regions[0].roi
        x_slice = roi.get("x")
        assert x_slice is not None
        y_slice = roi.get("y")
        assert y_slice is not None
        assert x_slice.start == 10.0
        assert y_slice.start == 20.0

    def test_apply_align_to_pixel_grid_ceil(self) -> None:
        regions = [_make_world_tile_slice(10.1, 20.1, 100.0, 100.0, "FOV")]
        img = _make_tiled_image(regions, pixelsize=1.0)
        result = apply_align_to_pixel_grid(img, mode="ceil")
        roi = result.regions[0].roi
        x_slice = roi.get("x")
        assert x_slice is not None
        y_slice = roi.get("y")
        assert y_slice is not None
        assert x_slice.start == 11.0
        assert y_slice.start == 21.0

    def test_apply_remove_offsets(self) -> None:
        regions = [
            _make_world_tile_slice(100.0, 200.0, 64.0, 64.0, "FOV_0"),
            _make_world_tile_slice(164.0, 200.0, 64.0, 64.0, "FOV_1"),
        ]
        img = _make_tiled_image(regions)
        result = apply_remove_offsets(img)
        x_slice = result.regions[0].roi.get("x")
        assert x_slice is not None
        y_slice = result.regions[0].roi.get("y")
        assert y_slice is not None
        assert x_slice.start == 0.0
        assert y_slice.start == 0.0
        x_slice = result.regions[1].roi.get("x")
        assert x_slice is not None
        y_slice = result.regions[1].roi.get("y")
        assert y_slice is not None
        assert x_slice.start == 64.0
        assert y_slice.start == 0.0


# --- Snap utils tests ---


class TestSnapUtils:
    def test_tiles_to_boxes(self) -> None:
        tiles = [
            _make_pixel_tile_slice(0.0, 0.0, 256.0, 256.0, "A"),
            _make_pixel_tile_slice(256.0, 0.0, 256.0, 256.0, "B"),
        ]
        boxes = tiles_to_boxes(tiles)
        assert len(boxes) == 2
        assert boxes[0] == BBox(x=0.0, y=0.0, x_len=256.0, y_len=256.0)
        assert boxes[1] == BBox(x=256.0, y=0.0, x_len=256.0, y_len=256.0)

    def test_tiles_to_boxes_empty_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            tiles_to_boxes([])

    def test_check_if_regular_grid(self) -> None:
        tiles = [
            _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            _make_pixel_tile_slice(100.0, 0.0, 100.0, 100.0, "B"),
            _make_pixel_tile_slice(0.0, 100.0, 100.0, 100.0, "C"),
            _make_pixel_tile_slice(100.0, 100.0, 100.0, 100.0, "D"),
        ]
        grid = check_if_regular_grid(tiles)
        assert grid.length_x == 100.0
        assert grid.length_y == 100.0
        assert np.isclose(grid.offset_x, 100.0)
        assert np.isclose(grid.offset_y, 100.0)

    def test_check_if_regular_grid_single_tile(self) -> None:
        tiles = [_make_pixel_tile_slice(0.0, 0.0, 200.0, 200.0, "A")]
        grid = check_if_regular_grid(tiles)
        assert grid.length_x == 200.0
        assert grid.length_y == 200.0

    def test_check_if_irregular_grid_raises(self) -> None:
        tiles = [
            _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            _make_pixel_tile_slice(100.0, 0.0, 100.0, 100.0, "B"),
            _make_pixel_tile_slice(0.0, 100.0, 100.0, 100.0, "C"),
            _make_pixel_tile_slice(150.0, 100.0, 100.0, 100.0, "D"),
        ]
        with pytest.raises(NotAGridError):
            check_if_regular_grid(tiles)

    def test_check_if_regular_grid_row_only(self) -> None:
        # 3-tile single row — all share y=0, so Cartesian product is trivially satisfied
        tiles = [
            _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            _make_pixel_tile_slice(95.0, 0.0, 100.0, 100.0, "B"),
            _make_pixel_tile_slice(190.0, 0.0, 100.0, 100.0, "C"),
        ]
        grid = check_if_regular_grid(tiles)  # should not raise
        assert grid.length_x == 100.0

    def test_snap_to_grid_y_offset_correct(self) -> None:
        # Regression for the duplicate x_grid bug: y-offsets must be non-zero
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(95.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(0.0, 95.0, 100.0, 100.0, "C"),
            "D": _make_pixel_tile_slice(95.0, 95.0, 100.0, 100.0, "D"),
        }
        offsets = calculate_snap_to_grid_offset(tiles)
        assert np.isclose(offsets["A"]["x"], 0.0)
        assert np.isclose(offsets["A"]["y"], 0.0)
        assert np.isclose(offsets["B"]["x"], 5.0)
        assert np.isclose(offsets["B"]["y"], 0.0)
        assert np.isclose(offsets["C"]["x"], 0.0)
        assert np.isclose(offsets["C"]["y"], 5.0)
        assert np.isclose(offsets["D"]["x"], 5.0)
        assert np.isclose(offsets["D"]["y"], 5.0)

    def test_snap_to_grid_single_tile_zero_offset(self) -> None:
        # Regression: single tile at non-zero origin must return zero offset
        tiles = {"A": _make_pixel_tile_slice(500.0, 300.0, 100.0, 100.0, "A")}
        offsets = calculate_snap_to_grid_offset(tiles)
        assert offsets["A"] == {"x": 0.0, "y": 0.0}

    def test_snap_to_corner_non_zero_origin(self) -> None:
        # Regression: corner snapping must work without prior remove_offsets
        tiles = {
            "A": _make_pixel_tile_slice(200.0, 300.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(295.0, 300.0, 100.0, 100.0, "B"),
        }
        offsets = calculate_snap_to_corner_offset(tiles)
        assert np.isclose(offsets["A"]["x"], 0.0)
        assert np.isclose(offsets["A"]["y"], 0.0)
        assert np.isclose(offsets["B"]["x"], 5.0)  # 295 → 300
        assert np.isclose(offsets["B"]["y"], 0.0)

    def test_tiles_to_boxes_raises_value_error_missing_y(self) -> None:
        # Regression: missing y-axis should raise ValueError, not AssertionError.
        # Use x+z slices (valid Roi: ≥2 slices) so that get("y") returns None.
        roi = Roi(
            name="bad",
            slices=[
                RoiSlice(axis_name="x", start=0.0, length=100.0),
                RoiSlice(axis_name="z", start=0.0, length=1.0),
            ],
            space="pixel",
        )
        from ome_zarr_converters_tools.core._dummy_tiles import DummyLoader, TileShape

        loader = DummyLoader(shape=TileShape(x=100, y=100), text="bad")
        bad_tile = TileSlice(roi=roi, image_loader=loader)
        with pytest.raises(ValueError, match="missing the 'y' axis slice"):
            tiles_to_boxes([bad_tile])

    def test_snap_to_grid_with_tolerance_accepts_varying_step(self) -> None:
        # 3-tile row where column spacing alternates: 95 px then 96 px.
        # The two steps differ by 1 px (mean = 95.5); tolerance=2 must accept.
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(95.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(191.0, 0.0, 100.0, 100.0, "C"),
        }
        offsets = calculate_snap_to_grid_offset(tiles, tolerance=2.0)
        assert np.isclose(offsets["A"]["x"], 0.0)
        assert np.isclose(offsets["A"]["y"], 0.0)
        # C is exactly 2 mean-steps from A (191 / 95.5 == 2.0), so it snaps to 200
        assert np.isclose(offsets["C"]["x"], 9.0)  # 191 -> 200
        assert np.isclose(offsets["C"]["y"], 0.0)

    def test_snap_to_grid_without_tolerance_rejects_varying_step(self) -> None:
        # Same non-uniform spacing must raise with the default tolerance=0.
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(95.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(191.0, 0.0, 100.0, 100.0, "C"),
        }
        with pytest.raises(NotAGridError):
            calculate_snap_to_grid_offset(tiles)

    def test_check_if_regular_grid_tolerance_parameter(self) -> None:
        # 3-tile row where inter-tile spacing varies by 1 px (95, 96).
        # tolerance=0 rejects; tolerance=2 accepts.
        tiles = [
            _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            _make_pixel_tile_slice(95.0, 0.0, 100.0, 100.0, "B"),
            _make_pixel_tile_slice(191.0, 0.0, 100.0, 100.0, "C"),
        ]
        with pytest.raises(NotAGridError):
            check_if_regular_grid(tiles, tolerance=0.0)
        grid = check_if_regular_grid(tiles, tolerance=2.0)
        assert grid.length_x == 100.0
        assert np.isclose(grid.offset_x, 95.5)  # mean of 95 and 96

    def test_check_if_regular_grid_with_column_jitter(self) -> None:
        # Regression for hardcoded 1e-6 threshold: same-column tiles have +-1 px jitter.
        # Sorted x diffs are [2.0, 98.0, 2.0]; the old filter kept all three making
        # the median ~2, which caused allclose to reject the valid ~98 step.
        tiles = [
            _make_pixel_tile_slice(1.0, 0.0, 100.0, 100.0, "A"),
            _make_pixel_tile_slice(-1.0, 100.0, 100.0, 100.0, "B"),
            _make_pixel_tile_slice(99.0, 0.0, 100.0, 100.0, "C"),
            _make_pixel_tile_slice(101.0, 100.0, 100.0, 100.0, "D"),
        ]
        grid = check_if_regular_grid(tiles, tolerance=5.0)
        assert grid.num_x == 2
        assert grid.num_y == 2
        with pytest.raises(NotAGridError):
            check_if_regular_grid(tiles, tolerance=0.0)

    def test_check_if_regular_grid_2x3_jitter(self) -> None:
        # Regression: 2-col x 3-row grid matching the debug.py scenario.
        jitter = [[-1.5, 0.5, 1.0], [1.5, -0.5, -1.0]]
        tiles = [
            _make_pixel_tile_slice(
                x * 100 + jitter[x][y],
                y * 100 + jitter[x][y],
                100.0,
                100.0,
            )
            for x in range(2)
            for y in range(3)
        ]
        grid = check_if_regular_grid(tiles, tolerance=5.0)
        assert grid.num_x == 2
        assert grid.num_y == 3
        with pytest.raises(NotAGridError):
            check_if_regular_grid(tiles, tolerance=0.0)

    def test_snap_to_grid_with_column_jitter(self) -> None:
        # Regression: snapping a jittered 2x2 grid must produce a perfect grid.
        tiles = {
            "A": _make_pixel_tile_slice(1.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(-1.0, 100.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(99.0, 0.0, 100.0, 100.0, "C"),
            "D": _make_pixel_tile_slice(101.0, 100.0, 100.0, 100.0, "D"),
        }
        offsets = calculate_snap_to_grid_offset(tiles, tolerance=5.0)
        snapped_x = sorted(
            {
                1.0 + offsets["A"]["x"],
                -1.0 + offsets["B"]["x"],
                99.0 + offsets["C"]["x"],
                101.0 + offsets["D"]["x"],
            }
        )
        snapped_y = sorted(
            {
                0.0 + offsets["A"]["y"],
                100.0 + offsets["B"]["y"],
                0.0 + offsets["C"]["y"],
                100.0 + offsets["D"]["y"],
            }
        )
        assert len(snapped_x) == 2
        assert len(snapped_y) == 2
        assert np.isclose(snapped_x[1] - snapped_x[0], 100.0)
        assert np.isclose(snapped_y[1] - snapped_y[0], 100.0)

    def test_check_if_regular_grid_jitter_exceeds_tolerance_raises(self) -> None:
        # Jitter spread of 6 px (from -3 to +3) exceeds tolerance=2 → must raise.
        tiles = [
            _make_pixel_tile_slice(3.0, 0.0, 100.0, 100.0, "A"),
            _make_pixel_tile_slice(-3.0, 100.0, 100.0, 100.0, "B"),
            _make_pixel_tile_slice(99.0, 0.0, 100.0, 100.0, "C"),
            _make_pixel_tile_slice(101.0, 100.0, 100.0, 100.0, "D"),
        ]
        with pytest.raises(NotAGridError):
            check_if_regular_grid(tiles, tolerance=2.0)
        grid = check_if_regular_grid(tiles, tolerance=10.0)
        assert grid.num_x == 2


# --- Tiling tests ---


class TestTiling:
    def test_no_tiling_returns_zero_offsets(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(10.0, 20.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(200.0, 300.0, 100.0, 100.0, "B"),
        }
        offsets = _find_tiling(tiles, NoTiling())
        for offset in offsets.values():
            assert offset == {"x": 0.0, "y": 0.0}

    def test_snap_to_grid_regular(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(95.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(0.0, 95.0, 100.0, 100.0, "C"),
            "D": _make_pixel_tile_slice(95.0, 95.0, 100.0, 100.0, "D"),
        }
        offsets = _find_tiling(tiles, SnapToGridTiling())
        assert np.isclose(offsets["A"]["x"], 0.0)
        assert np.isclose(offsets["A"]["y"], 0.0)

    def test_snap_to_grid_not_a_grid_error(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(100.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(0.0, 100.0, 100.0, 100.0, "C"),
            "D": _make_pixel_tile_slice(150.0, 100.0, 100.0, 100.0, "D"),
        }
        with pytest.raises(NotAGridError):
            _find_tiling(tiles, SnapToGridTiling())

    def test_snap_to_grid_off_origin_2x1(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(500.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(595.0, 0.0, 100.0, 100.0, "B"),
        }
        offsets = _find_tiling(tiles, SnapToGridTiling())
        assert np.isclose(offsets["A"]["x"], 0.0)
        assert np.isclose(offsets["B"]["x"], 5.0)  # 595 → 600
        assert np.isclose(offsets["A"]["y"], 0.0)
        assert np.isclose(offsets["B"]["y"], 0.0)

    def test_inplace_returns_zero_offsets(self) -> None:
        tiles = {"A": _make_pixel_tile_slice(50.0, 50.0, 100.0, 100.0, "A")}
        offsets = _find_tiling(tiles, InplaceTiling())
        assert offsets["A"] == {"x": 0.0, "y": 0.0}

    def test_auto_tiling_falls_back_to_corners(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(100.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(0.0, 100.0, 100.0, 100.0, "C"),
            "D": _make_pixel_tile_slice(150.0, 100.0, 100.0, 100.0, "D"),
        }
        offsets = _find_tiling(tiles, AutoTiling())
        assert len(offsets) == 4

    def test_apply_mosaic_tiling(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        coll = SingleImage(image_path="test_image")
        tiles = [
            build_dummy_tile(
                fov_name=f"FOV_{i}",
                start=StartPosition(x=x, y=y),
                shape=TileShape(x=100, y=100, z=1, c=1, t=1),
                collection=coll,
                acquisition_details=acq,
            )
            for i, (x, y) in enumerate([(0, 0), (100, 0), (0, 100), (100, 100)])
        ]
        images = tiled_image_from_tiles(
            tiles=tiles, converter_options=ConverterOptions()
        )
        result = apply_mosaic_tiling(images[0], InplaceTiling())
        assert len(result.regions) == 4
