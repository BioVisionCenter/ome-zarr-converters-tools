"""Unit test fixtures for model factories and class instances."""

import numpy as np
import pytest

from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    AlignmentCorrections,
    ChannelInfo,
    ConverterOptions,
    ImageInPlate,
    SingleImage,
    StageCorrections,
)


class DummyLoader:
    """Mock image loader for testing without real image data."""

    def __init__(
        self,
        shape: tuple[int, ...] = (1, 1, 1, 100, 100),
        dtype: str = "uint16",
    ):
        self.shape = shape
        self.dtype = dtype

    def load_data(self, resource: object = None) -> np.ndarray:
        """Return a zero-filled array of the configured shape."""
        return np.zeros(self.shape, dtype=self.dtype)


@pytest.fixture
def dummy_loader() -> DummyLoader:
    """Provide a mock image loader."""
    return DummyLoader()


@pytest.fixture
def sample_acquisition_details() -> AcquisitionDetails:
    """Provide sample acquisition details for testing."""
    return AcquisitionDetails(
        channels=[
            ChannelInfo(channel_label="Channel 1"),
            ChannelInfo(channel_label="Channel 2"),
        ],
        pixelsize=0.65,
        z_spacing=1.0,
        t_spacing=1.0,
    )


@pytest.fixture
def sample_converter_options() -> ConverterOptions:
    """Provide sample converter options for testing."""
    return ConverterOptions()


@pytest.fixture
def sample_stage_corrections() -> StageCorrections:
    """Provide sample stage corrections for testing."""
    return StageCorrections()


@pytest.fixture
def sample_alignment_corrections() -> AlignmentCorrections:
    """Provide sample alignment corrections for testing."""
    return AlignmentCorrections()


@pytest.fixture
def sample_single_image() -> SingleImage:
    """Provide a sample SingleImage collection for testing."""
    return SingleImage(image_path="/test/path/test_image")


@pytest.fixture
def sample_image_in_plate() -> ImageInPlate:
    """Provide a sample ImageInPlate collection for testing."""
    return ImageInPlate(
        plate_name="test_plate",
        row="A",
        column=1,
        acquisition=0,
    )
