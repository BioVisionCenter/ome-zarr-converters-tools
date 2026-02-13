"""Unit test fixtures using DummyLoader and build_dummy_tile."""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.core._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    AlignmentCorrections,
    ChannelInfo,
    ConverterOptions,
    ImageInPlate,
    SingleImage,
    StageCorrections,
)


@pytest.fixture
def default_acquisition_details() -> AcquisitionDetails:
    """AcquisitionDetails with 2 channels and pixelsize=1.0."""
    return AcquisitionDetails(
        channels=[
            ChannelInfo(channel_label="DAPI"),
            ChannelInfo(channel_label="GFP"),
        ],
        pixelsize=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )


@pytest.fixture
def default_collection() -> SingleImage:
    """SingleImage collection for testing."""
    return SingleImage(image_path="image_01")


@pytest.fixture
def plate_collection() -> ImageInPlate:
    """ImageInPlate collection for testing."""
    return ImageInPlate(
        plate_name="test_plate",
        row="A",
        column=1,
        acquisition=0,
    )


@pytest.fixture
def default_converter_options() -> ConverterOptions:
    """Default ConverterOptions."""
    return ConverterOptions()


@pytest.fixture
def default_alignment_corrections() -> AlignmentCorrections:
    """Default AlignmentCorrections."""
    return AlignmentCorrections()


@pytest.fixture
def default_stage_corrections() -> StageCorrections:
    """Default StageCorrections."""
    return StageCorrections()


@pytest.fixture
def single_tile(
    default_acquisition_details: AcquisitionDetails,
    default_collection: SingleImage,
) -> Tile[Any, Any]:
    """A single tile at the origin."""
    return build_dummy_tile(
        fov_name="FOV_0",
        start=StartPosition(x=0, y=0),
        shape=TileShape(x=256, y=256, z=1, c=2, t=1),
        collection=default_collection,
        acquisition_details=default_acquisition_details,
    )


@pytest.fixture
def grid_2x2_tiles(
    default_acquisition_details: AcquisitionDetails,
    default_collection: SingleImage,
) -> list[Tile[Any, Any]]:
    """Four tiles arranged in a 2x2 grid (no overlap)."""
    positions = [
        ("FOV_0", StartPosition(x=0, y=0)),
        ("FOV_1", StartPosition(x=256, y=0)),
        ("FOV_2", StartPosition(x=0, y=256)),
        ("FOV_3", StartPosition(x=256, y=256)),
    ]
    tiles = []
    for fov_name, start in positions:
        tile = build_dummy_tile(
            fov_name=fov_name,
            start=start,
            shape=TileShape(x=256, y=256, z=1, c=2, t=1),
            collection=default_collection,
            acquisition_details=default_acquisition_details,
        )
        tiles.append(tile)
    return tiles


@pytest.fixture
def tiled_image_from_grid(
    grid_2x2_tiles: list[Tile[Any, Any]],
    default_converter_options: ConverterOptions,
) -> TiledImage:
    """A TiledImage built from the 2x2 grid tiles."""
    images = tiled_image_from_tiles(
        tiles=grid_2x2_tiles,
        converter_options=default_converter_options,
    )
    assert len(images) == 1
    return images[0]


# --- Legacy fixtures for test_models.py compatibility ---


class DummyLoader:
    """Mock image loader for testing without real image data."""

    def __init__(
        self,
        shape: tuple[int, ...] = (1, 1, 1, 100, 100),
        dtype: str = "uint16",
    ) -> None:
        self.shape = shape
        self.dtype = dtype

    def load_data(self, resource: object = None) -> NDArray[Any]:
        return np.zeros(self.shape, dtype=self.dtype)


@pytest.fixture
def dummy_loader() -> DummyLoader:
    return DummyLoader()


@pytest.fixture
def sample_acquisition_details() -> AcquisitionDetails:
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
    return ConverterOptions()


@pytest.fixture
def sample_stage_corrections() -> StageCorrections:
    return StageCorrections()


@pytest.fixture
def sample_alignment_corrections() -> AlignmentCorrections:
    return AlignmentCorrections()


@pytest.fixture
def sample_single_image() -> SingleImage:
    return SingleImage(image_path="test_image")


@pytest.fixture
def sample_image_in_plate() -> ImageInPlate:
    return ImageInPlate(
        plate_name="test_plate",
        row="A",
        column=1,
        acquisition=0,
    )
