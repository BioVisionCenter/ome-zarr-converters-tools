"""Unit tests for registration module (alignment, tiling, snap utils)."""

import warnings

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
    AlignmentCorrections,
    ChannelInfo,
    ConverterOptions,
    SingleImage,
    TilingMode,
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

    def test_align_z_warns(self) -> None:
        regions = [_make_world_tile_slice(0.0, 0.0, 10.0, 10.0)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _align_z_regions(regions)
            assert len(w) == 1
            assert "not implemented" in str(w[0].message).lower()
        assert result == regions

    def test_align_t_warns(self) -> None:
        regions = [_make_world_tile_slice(0.0, 0.0, 10.0, 10.0)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _align_t_regions(regions)
            assert len(w) == 1
            assert "not implemented" in str(w[0].message).lower()
        assert result == regions

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
        corrections = AlignmentCorrections(align_xy=True)
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


# --- Tiling tests ---


class TestTiling:
    def test_no_tiling_returns_zero_offsets(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(10.0, 20.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(200.0, 300.0, 100.0, 100.0, "B"),
        }
        offsets = _find_tiling(tiles, TilingMode.NO_TILING)
        for offset in offsets.values():
            assert offset == {"x": 0.0, "y": 0.0}

    def test_snap_to_grid_regular(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(95.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(0.0, 95.0, 100.0, 100.0, "C"),
            "D": _make_pixel_tile_slice(95.0, 95.0, 100.0, 100.0, "D"),
        }
        offsets = _find_tiling(tiles, TilingMode.SNAP_TO_GRID)
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
            _find_tiling(tiles, TilingMode.SNAP_TO_GRID)

    def test_auto_tiling_falls_back_to_corners(self) -> None:
        tiles = {
            "A": _make_pixel_tile_slice(0.0, 0.0, 100.0, 100.0, "A"),
            "B": _make_pixel_tile_slice(100.0, 0.0, 100.0, 100.0, "B"),
            "C": _make_pixel_tile_slice(0.0, 100.0, 100.0, 100.0, "C"),
            "D": _make_pixel_tile_slice(150.0, 100.0, 100.0, 100.0, "D"),
        }
        offsets = _find_tiling(tiles, TilingMode.AUTO)
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
        result = apply_mosaic_tiling(images[0], TilingMode.INPLACE)
        assert len(result.regions) == 4
