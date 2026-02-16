"""Unit tests for pydantic models."""

import pytest

from ome_zarr_converters_tools import AcquisitionDetails, ChannelInfo
from ome_zarr_converters_tools.models import (
    AlignmentCorrections,
    ConverterOptions,
    ImageInPlate,
    SingleImage,
    StageCorrections,
)
from ome_zarr_converters_tools.models._collection import validate_zarr_name
from ome_zarr_converters_tools.models._converter_options import (
    FovBasedChunking,
    TilingMode,
    WriterMode,
)


class TestAcquisitionDetails:
    """Tests for the AcquisitionDetails model."""

    def test_acquisition_details_creation(
        self, sample_acquisition_details: AcquisitionDetails
    ) -> None:
        """Test basic acquisition details creation."""
        assert sample_acquisition_details is not None
        assert sample_acquisition_details.pixelsize == 0.65
        assert sample_acquisition_details.channels == [
            ChannelInfo(channel_label="Channel 1"),
            ChannelInfo(channel_label="Channel 2"),
        ]

    def test_acquisition_details_validation(self) -> None:
        """Test acquisition details validation."""
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                pixelsize=-1.0,
                z_spacing=1.0,
                t_spacing=1.0,
            )
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                pixelsize=1.0,
                z_spacing=0.0,
                t_spacing=1.0,
            )
        # Extra fields should be forbidden
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                pixelsize=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                unknown_field="value",  # type: ignore
            )

    def test_acquisition_details_axes_order(self) -> None:
        """Test axes order validation."""
        # Valid order: subset of t, c, z, y, x in canonical order
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            pixelsize=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
            axes=["c", "z", "y", "x"],
        )
        assert acq.axes == ["c", "z", "y", "x"]

        # Invalid order: x before y
        with pytest.raises(ValueError, match="canonical order"):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                pixelsize=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                axes=["x", "y"],
            )

        # Invalid: duplicate axis
        with pytest.raises(ValueError, match="canonical order"):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                pixelsize=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                axes=["y", "y", "x"],
            )

        # Too few axes
        with pytest.raises(ValueError):
            AcquisitionDetails(
                channels=[ChannelInfo(channel_label="DAPI")],
                pixelsize=1.0,
                z_spacing=1.0,
                t_spacing=1.0,
                axes=["x"],
            )


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
        assert opts.tiling_mode == TilingMode.AUTO
        assert opts.writer_mode == WriterMode.BY_FOV
        assert opts.alignment_correction.align_xy is False
        assert opts.alignment_correction.align_z is False
        assert opts.alignment_correction.align_t is False
        assert opts.omezarr_options.num_levels == 5
        assert isinstance(opts.omezarr_options.chunks, FovBasedChunking)
        assert opts.temp_json_options.temp_url == "{zarr_dir}/_tmp_json"


class TestStageCorrections:
    """Tests for the StageCorrections model."""

    def test_stage_corrections_creation(
        self, sample_stage_corrections: StageCorrections
    ) -> None:
        """Test basic stage corrections creation."""
        assert sample_stage_corrections is not None


class TestAlignmentCorrections:
    """Tests for the AlignmentCorrections model."""

    def test_alignment_corrections_creation(
        self, sample_alignment_corrections: AlignmentCorrections
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
