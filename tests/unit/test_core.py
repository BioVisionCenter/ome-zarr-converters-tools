"""Unit tests for core module (Tile, TiledImage, TileSlice, tiled_image_from_tiles)."""

from typing import Any

import numpy as np
import pytest

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.core._tile_region import (
    TiledImage,
    TileFOVGroup,
    TileSlice,
)
from ome_zarr_converters_tools.core._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    ConverterOptions,
    SingleImage,
    StageOrientation,
)


class TestTile:
    def test_tile_creation(self, single_tile: Tile[Any, Any]) -> None:
        assert single_tile.fov_name == "FOV_0"
        assert single_tile.start_x == 0
        assert single_tile.start_y == 0
        assert single_tile.length_x == 256
        assert single_tile.length_y == 256

    def test_tile_to_roi(self, single_tile: Tile[Any, Any]) -> None:
        roi = single_tile.to_roi()
        assert roi.name == "FOV_0"
        x_slice = roi.get("x")
        y_slice = roi.get("y")
        assert x_slice is not None
        assert y_slice is not None
        assert x_slice.start == 0.0
        assert x_slice.length == 256.0
        assert y_slice.start == 0.0
        assert y_slice.length == 256.0

    def test_tile_to_roi_with_flip(self, default_collection: SingleImage) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
            stage_orientation=StageOrientation(flip_x=True, flip_y=False),
        )
        tile = build_dummy_tile(
            fov_name="FOV_flip",
            start=StartPosition(x=100, y=50),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=default_collection,
            acquisition_details=acq,
        )
        roi = tile.to_roi()
        # flip_x negates the x start
        x_slice = roi.get("x")
        assert x_slice is not None
        assert x_slice.start == -100.0
        y_slice = roi.get("y")
        assert y_slice is not None
        assert y_slice.start == 50.0

    def test_tile_to_roi_with_swap_xy(self, default_collection: SingleImage) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
            stage_orientation=StageOrientation(swap_xy=True),
        )
        tile = build_dummy_tile(
            fov_name="FOV_swap",
            start=StartPosition(x=100, y=200),
            shape=TileShape(x=64, y=128, z=1, c=1, t=1),
            collection=default_collection,
            acquisition_details=acq,
        )
        roi = tile.to_roi()
        # swap_xy transposes X and Y: the output x axis is built from the tile's
        # y position/length and vice versa.
        x_slice = roi.get("x")
        assert x_slice is not None
        assert x_slice.start == 200.0
        assert x_slice.length == 128.0
        y_slice = roi.get("y")
        assert y_slice is not None
        assert y_slice.start == 100.0
        assert y_slice.length == 64.0

    def test_tile_find_data_type(self, single_tile: Tile[Any, Any]) -> None:
        dtype = single_tile.find_data_type()
        assert dtype == "uint8"

    @pytest.mark.parametrize(
        "start_x,xy_pixel_size",
        [
            (-645.814, 0.5979760809567617),
            (745.34, 1.294),
        ],
    )
    def test_tile_to_roi_non_aligned_coordinates(
        self, default_collection: SingleImage, start_x: float, xy_pixel_size: float
    ) -> None:
        """Regression: coordinates that don't land exactly on the pixel grid.

        Previously raised ValueError due to strict round-trip tolerance.
        """
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=xy_pixel_size,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        tile = build_dummy_tile(
            fov_name="FOV_0",
            start=StartPosition(x=start_x, y=0),
            shape=TileShape(x=256, y=256, z=1, c=1, t=1),
            collection=default_collection,
            acquisition_details=acq,
        )
        roi = tile.to_roi()
        x_slice = roi.get("x")
        assert x_slice is not None
        assert x_slice.start is not None


class TestTileSlice:
    def test_from_tile(self, single_tile: Tile[Any, Any]) -> None:
        tile_slice: TileSlice = TileSlice.from_tile(single_tile)
        assert tile_slice.roi is not None
        assert tile_slice.roi.name == "FOV_0"

    def test_load_data(self, single_tile: Tile[Any, Any]) -> None:
        tile_slice: TileSlice = TileSlice.from_tile(single_tile)
        axes = single_tile.acquisition_details.axes
        data = tile_slice.load_data(axes=axes)
        # Expected shape: (t, c, z, y, x) = (1, 2, 1, 256, 256)
        assert data.shape == (1, 2, 1, 256, 256)
        assert data.dtype == np.uint8


class TestTiledImage:
    def test_add_tile(self, tiled_image_from_grid: TiledImage) -> None:
        assert len(tiled_image_from_grid.regions) == 4

    def test_add_tile_mismatched_channels(
        self, tiled_image_from_grid: TiledImage, default_collection: SingleImage
    ) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DIFFERENT")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        tile = build_dummy_tile(
            fov_name="FOV_bad",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=default_collection,
            acquisition_details=acq,
        )
        with pytest.raises(ValueError, match="channels"):
            tiled_image_from_grid.add_tile(tile)

    def test_group_by_fov(self, tiled_image_from_grid: TiledImage) -> None:
        groups = tiled_image_from_grid.group_by_fov()
        # Each FOV is unique, so 4 groups
        assert len(groups) == 4
        for group in groups:
            assert isinstance(group, TileFOVGroup)
            assert len(group.regions) == 1

    def test_shape(self, tiled_image_from_grid: TiledImage) -> None:
        shape = tiled_image_from_grid.shape()
        # 2x2 grid of 256x256 tiles, 2 channels, 1 z, 1 t
        # axes order: t, c, z, y, x
        assert shape == (1, 2, 1, 512, 512)

    def test_load_data(self, tiled_image_from_grid: TiledImage) -> None:
        data = tiled_image_from_grid.load_data()
        assert data.shape == (1, 2, 1, 512, 512)
        assert data.dtype == np.uint8
        # DummyLoader fills with non-zero data
        assert data.sum() > 0

    def test_load_data_non_zero_origin(
        self,
        default_acquisition_details: AcquisitionDetails,
        default_collection: SingleImage,
        default_converter_options: ConverterOptions,
    ) -> None:
        # Regions that do not start at pixel 0 must be zeroed to the union origin
        # before slicing into the union-sized buffer; otherwise their data is
        # dropped (or a broadcast error is raised). load_data must be
        # translation-invariant: an offset grid yields identical pixels to the
        # same grid at the origin.
        def _build_grid(offset: int) -> TiledImage:
            positions = [
                ("FOV_0", StartPosition(x=offset, y=offset)),
                ("FOV_1", StartPosition(x=offset + 256, y=offset)),
                ("FOV_2", StartPosition(x=offset, y=offset + 256)),
                ("FOV_3", StartPosition(x=offset + 256, y=offset + 256)),
            ]
            tiles = [
                build_dummy_tile(
                    fov_name=fov_name,
                    start=start,
                    shape=TileShape(x=256, y=256, z=1, c=2, t=1),
                    collection=default_collection,
                    acquisition_details=default_acquisition_details,
                )
                for fov_name, start in positions
            ]
            images = tiled_image_from_tiles(
                tiles=tiles, converter_options=default_converter_options
            )
            assert len(images) == 1
            return images[0]

        at_origin = _build_grid(0).load_data()
        offset = _build_grid(100).load_data()
        assert offset.shape == at_origin.shape == (1, 2, 1, 512, 512)
        np.testing.assert_array_equal(offset, at_origin)


class TestTiledImageFromTiles:
    def test_single_collection(
        self,
        grid_2x2_tiles: list[Tile[Any, Any]],
        default_converter_options: ConverterOptions,
    ) -> None:
        images = tiled_image_from_tiles(
            tiles=grid_2x2_tiles,
            converter_options=default_converter_options,
        )
        # All tiles share the same collection path -> 1 TiledImage
        assert len(images) == 1
        assert len(images[0].regions) == 4

    def test_multiple_collections(
        self, default_acquisition_details: AcquisitionDetails
    ) -> None:
        coll_a = SingleImage(image_path="image_A")
        coll_b = SingleImage(image_path="image_B")
        tiles = [
            build_dummy_tile(
                fov_name="FOV_0",
                start=StartPosition(x=0, y=0),
                shape=TileShape(x=64, y=64, z=1, c=2, t=1),
                collection=coll_a,
                acquisition_details=default_acquisition_details,
            ),
            build_dummy_tile(
                fov_name="FOV_1",
                start=StartPosition(x=0, y=0),
                shape=TileShape(x=64, y=64, z=1, c=2, t=1),
                collection=coll_b,
                acquisition_details=default_acquisition_details,
            ),
        ]
        images = tiled_image_from_tiles(
            tiles=tiles,
            converter_options=ConverterOptions(),
        )
        assert len(images) == 2

    def test_empty_tiles_error(self) -> None:
        with pytest.raises(ValueError, match="No tiles"):
            tiled_image_from_tiles(
                tiles=[],
                converter_options=ConverterOptions(),
            )
