"""Unit tests for pydantic models."""

import pytest
from pydantic import ValidationError

from ome_zarr_converters_tools import AcquisitionDetails, ChannelInfo
from ome_zarr_converters_tools.models import (
    ConverterOptions,
    ImageInPlate,
    SingleImage,
    StageOrientation,
    StagePositionCorrections,
)
from ome_zarr_converters_tools.models._collection import validate_zarr_name
from ome_zarr_converters_tools.models._converter_options import (
    AutoTiling,
    FovBasedChunking,
    WriterMode,
)


class TestAcquisitionDetails:
    """Tests for the AcquisitionDetails model."""

    def test_acquisition_details_creation(
        self, sample_acquisition_details: AcquisitionDetails
    ) -> None:
        """Test basic acquisition details creation."""
        assert sample_acquisition_details is not None
        assert sample_acquisition_details.xy_pixel_size == 0.65
        assert sample_acquisition_details.channels == [
            ChannelInfo(channel_label="Channel 1"),
            ChannelInfo(channel_label="Channel 2"),
        ]

    def test_acquisition_details_validation(self) -> None:
        """Test acquisition details validation."""
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                xy_pixel_size=-1.0,
                z_spacing=1.0,
                t_spacing=1.0,
            )
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                xy_pixel_size=1.0,
                z_spacing=0.0,
                t_spacing=1.0,
            )
        # Extra fields should be forbidden
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                xy_pixel_size=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                unknown_field="value",  # type: ignore
            )

    def test_acquisition_details_axes_order(self) -> None:
        """Test axes order validation."""
        # Valid order: subset of t, c, z, y, x in canonical order
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
            axes=["c", "z", "y", "x"],
        )
        assert acq.axes == ["c", "z", "y", "x"]

        # Invalid order: x before y
        with pytest.raises(ValueError, match="canonical order"):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                xy_pixel_size=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                axes=["x", "y"],
            )

        # Invalid: duplicate axis
        with pytest.raises(ValueError, match="canonical order"):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                xy_pixel_size=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                axes=["y", "y", "x"],
            )

        # Too few axes
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                xy_pixel_size=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                axes=["x"],
            )


class TestChannelInfo:
    """Tests for ChannelInfo color assignment logic."""

    def test_wavelength_uv_gives_magenta(self) -> None:
        ch = ChannelInfo(channel_label="UV", wavelength_id="350")
        assert ch.color == "#FF00FF"

    def test_wavelength_violet_gives_blue(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="405")
        assert ch.color == "#0000FF"

    def test_wavelength_cyan_band(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="488")
        assert ch.color == "#00FFFF"

    def test_wavelength_green_band(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="510")
        assert ch.color == "#00FF00"

    def test_wavelength_lime_band(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="540")
        assert ch.color == "#00FF80"

    def test_wavelength_yellow_band(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="575")
        assert ch.color == "#FFFF00"

    def test_wavelength_orange_band(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="605")
        assert ch.color == "#FF8000"

    def test_wavelength_red_band(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="640")
        assert ch.color == "#FF0000"

    def test_wavelength_far_ir_gives_magenta(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="800")
        assert ch.color == "#FF00FF"

    def test_wavelength_nm_suffix_parsed(self) -> None:
        ch = ChannelInfo(channel_label="x", wavelength_id="561nm")
        assert ch.color == "#FFFF00"

    def test_wavelength_out_of_range_low_falls_back_to_label(self) -> None:
        ch = ChannelInfo(channel_label="DAPI", wavelength_id="1")
        assert ch.color == "#0000FF"

    def test_wavelength_out_of_range_high_falls_back_to_label(self) -> None:
        ch = ChannelInfo(channel_label="DAPI", wavelength_id="5000")
        assert ch.color == "#0000FF"

    def test_wavelength_non_numeric_falls_back_to_label(self) -> None:
        ch = ChannelInfo(channel_label="DAPI", wavelength_id="GFP")
        assert ch.color == "#0000FF"

    def test_wavelength_none_falls_back_to_label(self) -> None:
        ch = ChannelInfo(channel_label="DAPI", wavelength_id=None)
        assert ch.color == "#0000FF"

    def test_label_dapi_gives_blue(self) -> None:
        ch = ChannelInfo(channel_label="DAPI")
        assert ch.color == "#0000FF"

    def test_label_gfp_gives_green(self) -> None:
        ch = ChannelInfo(channel_label="GFP")
        assert ch.color == "#00FF00"

    def test_label_rfp_gives_red(self) -> None:
        ch = ChannelInfo(channel_label="RFP")
        assert ch.color == "#FF0000"

    def test_label_mcherry_gives_red(self) -> None:
        ch = ChannelInfo(channel_label="mCherry")
        assert ch.color == "#FF0000"

    def test_label_cfp_gives_cyan(self) -> None:
        ch = ChannelInfo(channel_label="CFP")
        assert ch.color == "#00FFFF"

    def test_label_case_insensitive(self) -> None:
        assert (
            ChannelInfo(channel_label="dapi").color
            == ChannelInfo(channel_label="DAPI").color
        )

    def test_explicit_color_not_overwritten(self) -> None:
        ch = ChannelInfo(channel_label="DAPI", color="#FF0000")
        assert ch.color == "#FF0000"

    def test_same_label_produces_same_color(self) -> None:
        assert ChannelInfo(channel_label="Channel 1") == ChannelInfo(
            channel_label="Channel 1"
        )

    def test_color_is_always_set_after_validation(self) -> None:
        ch = ChannelInfo(channel_label="unknown_fluorophore_xyz")
        assert ch.color is not None
        assert ch.color.startswith("#")


class TestConverterOptions:
    """Tests for the ConverterOptions model."""

    def test_converter_options_creation(
        self, sample_converter_options: ConverterOptions
    ) -> None:
        """Test basic converter options creation."""
        assert sample_converter_options is not None

    def test_converter_options_defaults(self) -> None:
        """Test default values."""
        opts = ConverterOptions()
        assert isinstance(opts.tiling_strategy, AutoTiling)
        assert opts.writer_mode == WriterMode.BY_FOV
        assert opts.stage_position_corrections.remove_xy_offset == "Global"
        assert opts.stage_position_corrections.remove_z_offset == "Global"
        assert opts.stage_position_corrections.remove_t_offset == "Global"
        assert opts.stage_position_corrections.remove_xy_jitter is True
        assert opts.stage_position_corrections.reindex_channels is True
        assert opts.omezarr_options.num_levels == 5
        assert isinstance(opts.omezarr_options.chunks, FovBasedChunking)
        assert (
            opts.runtime_settings.temp_json_options.temp_url == "{zarr_dir}/_tmp_json"
        )


class TestStageOrientation:
    """Tests for the StageOrientation model."""

    def test_stage_corrections_creation(
        self, sample_stage_corrections: StageOrientation
    ) -> None:
        """Test basic stage corrections creation."""
        assert sample_stage_corrections is not None


class TestStagePositionCorrections:
    """Tests for the StagePositionCorrections model."""

    def test_alignment_corrections_creation(
        self, sample_alignment_corrections: StagePositionCorrections
    ) -> None:
        """Test basic alignment corrections creation."""
        assert sample_alignment_corrections is not None


class TestCollectionModels:
    """Tests for collection models (SingleImage, ImageInPlate)."""

    def test_single_image_creation(self, sample_single_image: SingleImage) -> None:
        """Test SingleImage creation."""
        assert sample_single_image is not None
        assert sample_single_image.image_path == "test_image"

    def test_image_in_plate_creation(self, sample_image_in_plate: ImageInPlate) -> None:
        """Test ImageInPlate creation."""
        assert sample_image_in_plate is not None
        assert sample_image_in_plate.plate_name == "test_plate"
        assert sample_image_in_plate.row == "A"
        assert sample_image_in_plate.column == 1

    def test_image_in_plate_well_property(
        self, sample_image_in_plate: ImageInPlate
    ) -> None:
        """Test well property combines row and column."""
        assert sample_image_in_plate.well == "A01"

    def test_collection_path_generation(self) -> None:
        """Test path generation for collections."""
        single = SingleImage(image_path="my_image")
        assert single.path() == "my_image.zarr"

        plate = ImageInPlate(plate_name="MyPlate", row="B", column=3, acquisition=0)
        assert plate.plate_path() == "MyPlate.zarr"
        assert plate.well_path() == "B/03"
        assert plate.path_in_well() == "0"
        assert plate.path() == "MyPlate.zarr/B/03/0"

    def test_row_integer_index_converts_to_letter(self) -> None:
        """Integer rows map 1->A ... 26->Z; 26 (Z) must not be rejected."""
        assert ImageInPlate(plate_name="P", row=1, column=1, acquisition=0).row == "A"
        assert ImageInPlate(plate_name="P", row=26, column=1, acquisition=0).row == "Z"

    def test_row_integer_index_out_of_range_raises(self) -> None:
        """Integer rows outside 1..26 are rejected."""
        for bad_row in (0, 27):
            with pytest.raises(ValidationError, match="out of range"):
                ImageInPlate(plate_name="P", row=bad_row, column=1, acquisition=0)

    def test_validate_zarr_name_valid(self) -> None:
        """Test valid Zarr names are accepted."""
        for string in [
            "hello",
            "hello.world",
            "my-file_name.txt",
            "_single_underscore",
            "a.",
            ".a",
            "123",
            "hello world",
        ]:
            validate_zarr_name(string)  # Should not raise

    def test_validate_zarr_name_invalid(self) -> None:
        """Test invalid Zarr names are rejected."""
        for string in [
            "path/to/file",  # contains /
            ".",  # only periods
            "..",  # only periods
            "...",  # only periods
            "path#",  # contains invalid character #
            "file$name",  # contains invalid character $
            "file%name",  # contains invalid character %
            "file&name",  # contains invalid character &
            "file(name)",  # contains invalid character ()
            "file\U0001f60aname",  # contains emoji
            "__dunder",  # starts with __
            "__",  # starts with __
            "",  # empty string
            "caf\u00e9",  # non-ASCII
            " hello world",  # Leading space
            "hello world ",  # Trailing space
        ]:
            with pytest.raises(ValueError):
                validate_zarr_name(string)
