"""Unit tests for fractal._models."""

import pytest

from ome_zarr_converters_tools.fractal._models import (
    AcquisitionOptions,
    ConvertParallelInitArgs,
    PixelSizeModel,
    converters_tools_models,
)
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    ConverterOptions,
    OverwriteMode,
)
from ome_zarr_converters_tools.models._acquisition import DataTypeEnum


class TestConvertParallelInitArgs:
    def test_basic_construction(self) -> None:
        args = ConvertParallelInitArgs(
            tiled_image_json_dump_url="/tmp/test.json",
            converter_options=ConverterOptions(),
        )
        assert args.tiled_image_json_dump_url == "/tmp/test.json"
        assert args.overwrite_mode == OverwriteMode.NO_OVERWRITE

    def test_custom_overwrite_mode(self) -> None:
        args = ConvertParallelInitArgs(
            tiled_image_json_dump_url="/tmp/test.json",
            converter_options=ConverterOptions(),
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        assert args.overwrite_mode == OverwriteMode.OVERWRITE


class TestPixelSizeModel:
    def test_construction(self) -> None:
        model = PixelSizeModel(pixelsize=0.65, z_spacing=5.0, t_spacing=1.0)
        assert model.pixelsize == 0.65
        assert model.z_spacing == 5.0
        assert model.t_spacing == 1.0


class TestAcquisitionOptions:
    def test_defaults(self) -> None:
        opts = AcquisitionOptions()
        assert opts.channels is None
        assert opts.pixel_info is None
        assert opts.axes is None
        assert opts.data_type is None
        assert opts.condition_table_path is None
        assert opts.filters == []

    def test_to_axes_list_none(self) -> None:
        opts = AcquisitionOptions()
        assert opts.to_axes_list() is None

    def test_to_axes_list_valid(self) -> None:
        opts = AcquisitionOptions(axes="czyx")
        result = opts.to_axes_list()
        assert result == ["c", "z", "y", "x"]

    def test_to_axes_list_with_time(self) -> None:
        opts = AcquisitionOptions(axes="tczyx")
        result = opts.to_axes_list()
        assert result == ["t", "c", "z", "y", "x"]

    def test_to_axes_list_invalid_axis_raises(self) -> None:
        opts = AcquisitionOptions(axes="abcd")
        with pytest.raises(ValueError, match="Invalid axis"):
            opts.to_axes_list()

    def test_update_acquisition_details_no_changes(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        opts = AcquisitionOptions()
        updated = opts.update_acquisition_details(acq)
        assert updated.channels == acq.channels
        assert updated.pixelsize == acq.pixelsize

    def test_update_acquisition_details_channels(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        new_channels = [
            ChannelInfo(channel_label="GFP"),
            ChannelInfo(channel_label="RFP"),
        ]
        opts = AcquisitionOptions(channels=new_channels)
        updated = opts.update_acquisition_details(acq)
        channels = updated.channels
        assert channels is not None
        assert len(channels) == 2
        assert channels[0].channel_label == "GFP"

    def test_update_acquisition_details_pixel_info(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        opts = AcquisitionOptions(
            pixel_info=PixelSizeModel(pixelsize=0.65, z_spacing=5.0, t_spacing=2.0)
        )
        updated = opts.update_acquisition_details(acq)
        assert updated.pixelsize == 0.65
        assert updated.z_spacing == 5.0
        assert updated.t_spacing == 2.0

    def test_update_acquisition_details_axes(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        opts = AcquisitionOptions(axes="tczyx")
        updated = opts.update_acquisition_details(acq)
        assert updated.axes == ["t", "c", "z", "y", "x"]

    def test_update_acquisition_details_data_type(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        opts = AcquisitionOptions(data_type=DataTypeEnum.UINT16)
        updated = opts.update_acquisition_details(acq)
        assert updated.data_type == DataTypeEnum.UINT16

    def test_update_acquisition_details_condition_table(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        opts = AcquisitionOptions(condition_table_path="/tmp/conditions.csv")
        updated = opts.update_acquisition_details(acq)
        assert updated.condition_table_path == "/tmp/conditions.csv"

    def test_update_does_not_mutate_original(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        opts = AcquisitionOptions(
            pixel_info=PixelSizeModel(pixelsize=0.5, z_spacing=2.0, t_spacing=3.0)
        )
        updated = opts.update_acquisition_details(acq)
        assert acq.pixelsize == 1.0  # Original unchanged
        assert updated.pixelsize == 0.5


class TestConvertersToolsModels:
    def test_returns_list_of_tuples(self) -> None:
        result = converters_tools_models()
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 3 for t in result)

    def test_default_base(self) -> None:
        result = converters_tools_models()
        for base, _, _ in result:
            assert base == "ome_zarr_converters_tools"

    def test_custom_base(self) -> None:
        result = converters_tools_models(base="my_package")
        for base, _, _ in result:
            assert base == "my_package"

    def test_contains_acquisition_options(self) -> None:
        result = converters_tools_models()
        names = [name for _, _, name in result]
        assert "AcquisitionOptions" in names

    def test_contains_converter_options(self) -> None:
        result = converters_tools_models()
        names = [name for _, _, name in result]
        assert "ConverterOptions" in names
