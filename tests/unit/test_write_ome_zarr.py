"""Unit tests for pipelines._write_ome_zarr helper functions."""

import polars as pl
import pytest
from ngio import PixelSize, Roi, RoiSlice

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
    ChannelInfo,
    ConverterOptions,
    FixedSizeChunking,
    FovBasedChunking,
    OmeZarrOptions,
    SingleImage,
)
from ome_zarr_converters_tools.pipelines._write_ome_zarr import (
    _attribute_to_condition_table,
    _compute_chunk_size,
    _region_to_pixel_coordinates,
    build_channels_meta,
)


def _make_tiled_image_with_channels(
    channels: list[ChannelInfo] | None = None,
    pixelsize: float = 1.0,
) -> TiledImage:
    """Build a simple TiledImage for testing."""
    acq = AcquisitionDetails(
        channels=channels,
        pixelsize=pixelsize,
        z_spacing=1.0,
        t_spacing=1.0,
    )
    coll = SingleImage(image_path="test")
    tiles = [
        build_dummy_tile(
            fov_name="FOV_0",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=256, y=256, z=1, c=1, t=1),
            collection=coll,
            acquisition_details=acq,
        ),
    ]
    images = tiled_image_from_tiles(tiles=tiles, converter_options=ConverterOptions())
    return images[0]


class TestComputeChunkSize:
    def test_default_fov_based_chunking(self) -> None:
        img = _make_tiled_image_with_channels(
            channels=[ChannelInfo(channel_label="DAPI")]
        )
        options = OmeZarrOptions(chunks=FovBasedChunking())
        chunks = _compute_chunk_size(img, options)
        # axes: c, z, y, x -> chunks should be (c_chunk, z_chunk, fov_y, fov_x)
        assert len(chunks) == len(img.axes)
        # XY chunks should match FOV shape with scaling=1
        fov_shape = img.group_by_fov()[0].shape()
        x_idx = img.axes.index("x")
        y_idx = img.axes.index("y")
        assert chunks[x_idx] == fov_shape[x_idx]
        assert chunks[y_idx] == fov_shape[y_idx]

    def test_fixed_size_chunking(self) -> None:
        img = _make_tiled_image_with_channels(
            channels=[ChannelInfo(channel_label="DAPI")]
        )
        options = OmeZarrOptions(chunks=FixedSizeChunking(xy_chunk=128))
        chunks = _compute_chunk_size(img, options)
        x_idx = img.axes.index("x")
        y_idx = img.axes.index("y")
        assert chunks[x_idx] == 128
        assert chunks[y_idx] == 128

    def test_c_and_z_chunks(self) -> None:
        img = _make_tiled_image_with_channels(
            channels=[ChannelInfo(channel_label="DAPI")]
        )
        options = OmeZarrOptions(chunks=FovBasedChunking(z_chunk=5, c_chunk=2))
        chunks = _compute_chunk_size(img, options)
        c_idx = img.axes.index("c")
        z_idx = img.axes.index("z")
        assert chunks[c_idx] == 2
        assert chunks[z_idx] == 5

    def test_with_time_axis(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
            axes=["t", "c", "z", "y", "x"],
        )
        coll = SingleImage(image_path="test")
        tiles = [
            build_dummy_tile(
                fov_name="FOV_0",
                start=StartPosition(x=0, y=0),
                shape=TileShape(x=64, y=64, z=1, c=1, t=1),
                collection=coll,
                acquisition_details=acq,
            ),
        ]
        images = tiled_image_from_tiles(
            tiles=tiles, converter_options=ConverterOptions()
        )
        options = OmeZarrOptions(chunks=FovBasedChunking(t_chunk=3))
        chunks = _compute_chunk_size(images[0], options)
        t_idx = images[0].axes.index("t")
        assert chunks[t_idx] == 3


class TestRegionToPixelCoordinates:
    def test_world_to_pixel_conversion(self) -> None:
        roi = Roi(
            name="FOV",
            slices=[
                RoiSlice(axis_name="x", start=10.0, length=100.0),
                RoiSlice(axis_name="y", start=20.0, length=200.0),
            ],
            space="world",
        )
        loader = DummyLoader(shape=TileShape(x=100, y=200), text="FOV")
        regions = [TileSlice(roi=roi, image_loader=loader)]
        pixel_size = PixelSize(x=0.5, y=0.5, z=1.0, t=1.0)
        result = _region_to_pixel_coordinates(regions, pixel_size)
        assert len(result) == 1
        x_slice = result[0].roi.get("x")
        y_slice = result[0].roi.get("y")
        assert x_slice is not None
        assert y_slice is not None
        # 10.0 / 0.5 = 20, 100.0 / 0.5 = 200
        assert x_slice.start == 20
        assert x_slice.length == 200
        # 20.0 / 0.5 = 40, 200.0 / 0.5 = 400
        assert y_slice.start == 40
        assert y_slice.length == 400

    def test_rounds_to_nearest_integer(self) -> None:
        roi = Roi(
            name="FOV",
            slices=[
                RoiSlice(axis_name="x", start=0.65, length=1.95),
                RoiSlice(axis_name="y", start=0.3, length=3.7),
            ],
            space="world",
        )
        loader = DummyLoader(shape=TileShape(x=2, y=4), text="FOV")
        regions = [TileSlice(roi=roi, image_loader=loader)]
        pixel_size = PixelSize(x=1.0, y=1.0, z=1.0, t=1.0)
        result = _region_to_pixel_coordinates(regions, pixel_size)
        x_slice = result[0].roi.get("x")
        assert x_slice is not None
        assert x_slice.start == round(0.65)
        assert x_slice.length == round(1.95)

    def test_multiple_regions(self) -> None:
        regions = []
        for i in range(3):
            roi = Roi(
                name=f"FOV_{i}",
                slices=[
                    RoiSlice(axis_name="x", start=float(i * 10), length=10.0),
                    RoiSlice(axis_name="y", start=0.0, length=10.0),
                ],
                space="world",
            )
            loader = DummyLoader(shape=TileShape(x=10, y=10), text=f"FOV_{i}")
            regions.append(TileSlice(roi=roi, image_loader=loader))
        pixel_size = PixelSize(x=1.0, y=1.0, z=1.0, t=1.0)
        result = _region_to_pixel_coordinates(regions, pixel_size)
        assert len(result) == 3


class TestAttributeToConditionTable:
    def test_basic_attributes(self) -> None:
        attrs = {"drug": ["DMSO", "Compound_A"], "dose": [0.0, 1.0]}
        table = _attribute_to_condition_table(attrs)
        assert table is not None
        df = table.table_data
        # table_data may be LazyFrame or DataFrame depending on ngio version
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 2)
        assert "drug" in df.columns
        assert "dose" in df.columns

    def test_empty_attributes(self) -> None:
        result = _attribute_to_condition_table({})
        assert result is None

    def test_mismatched_lengths_raises(self) -> None:
        attrs = {"drug": ["DMSO"], "dose": [0.0, 1.0]}
        with pytest.raises(ValueError, match="same number of values"):
            _attribute_to_condition_table(attrs)

    def test_single_attribute(self) -> None:
        attrs = {"tissue": ["brain"]}
        table = _attribute_to_condition_table(attrs)  # type: ignore
        assert table is not None
        df = table.table_data
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        assert df.shape == (1, 1)


class TestBuildChannelsMeta:
    def test_no_channels(self) -> None:
        img = _make_tiled_image_with_channels(channels=None)
        result = build_channels_meta(img)
        assert result is None

    def test_single_channel(self) -> None:
        img = _make_tiled_image_with_channels(
            channels=[ChannelInfo(channel_label="DAPI", wavelength_id="405")]
        )
        result = build_channels_meta(img)
        assert result is not None
        assert len(result) == 1
        assert result[0].label == "DAPI"
        assert result[0].wavelength_id == "405"

    def test_multiple_channels(self) -> None:
        img = _make_tiled_image_with_channels(
            channels=[
                ChannelInfo(channel_label="DAPI", wavelength_id="405"),
                ChannelInfo(channel_label="GFP", wavelength_id="488"),
                ChannelInfo(channel_label="TRITC", wavelength_id="561"),
            ]
        )
        result = build_channels_meta(img)
        assert result is not None
        assert len(result) == 3
        assert [c.label for c in result] == ["DAPI", "GFP", "TRITC"]

    def test_channel_color_hex(self) -> None:
        img = _make_tiled_image_with_channels(
            channels=[ChannelInfo(channel_label="DAPI")]
        )
        result = build_channels_meta(img)
        assert result is not None
        # Default color is blue -> 0000FF (with or without # prefix)
        assert "0000FF" in result[0].channel_visualisation.color  # type: ignore
