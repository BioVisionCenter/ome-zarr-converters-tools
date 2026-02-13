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

    @pytest.mark.skip(reason="Not implemented yet")
    def test_acquisition_details_validation(self) -> None:
        """Test acquisition details validation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_acquisition_details_axes_order(self) -> None:
        """Test axes order validation."""


class TestConverterOptions:
    """Tests for the ConverterOptions model."""

    def test_converter_options_creation(
        self, sample_converter_options: ConverterOptions
    ) -> None:
        """Test basic converter options creation."""
        assert sample_converter_options is not None

    @pytest.mark.skip(reason="Not implemented yet")
    def test_converter_options_defaults(self) -> None:
        """Test default values."""


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

    @pytest.mark.skip(reason="Not implemented yet")
    def test_collection_path_generation(self) -> None:
        """Test path generation for collections."""

    def test_validate_zarr_name(self) -> None:
        """Test path sanitization."""
        # Should match
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
