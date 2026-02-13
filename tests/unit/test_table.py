"""Unit tests for core._table"""

from pathlib import Path

import pandas as pd
import pytest

from ome_zarr_converters_tools.core._table import (
    hcs_images_from_dataframe,
    single_images_from_dataframe,
)
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    DefaultImageLoader,
    ImageInPlate,
    SingleImage,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
HCS_EXAMPLE_DIR = EXAMPLES_DIR / "hcs_plate"
SINGLE_EXAMPLE_DIR = EXAMPLES_DIR / "single_acquisitions"


@pytest.fixture
def hcs_acquisition_details() -> AcquisitionDetails:
    return AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        pixelsize=0.65,
        z_spacing=5.0,
        t_spacing=1.0,
        start_x_coo="world",
        start_y_coo="world",
        start_z_coo="pixel",
        start_t_coo="pixel",
        axes=["t", "c", "z", "y", "x"],
    )


@pytest.fixture
def single_acquisition_details() -> AcquisitionDetails:
    return AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        pixelsize=0.65,
        z_spacing=5.0,
        t_spacing=1.0,
        start_x_coo="world",
        start_y_coo="world",
        start_z_coo="pixel",
        start_t_coo="pixel",
        axes=["t", "c", "z", "y", "x"],
    )


class TestHcsImagesFromDataframe:
    def test_load_from_example_csv(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
            plate_name="TestPlate",
            acquisition_id=0,
        )
        assert len(tiles) == len(df)

    def test_tile_collection_is_image_in_plate(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
            plate_name="MyPlate",
            acquisition_id=1,
        )
        for tile in tiles:
            assert isinstance(tile.collection, ImageInPlate)
            assert tile.collection.plate_name == "MyPlate"
            assert tile.collection.acquisition == 1

    def test_tile_loader_has_file_path(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
        )
        for tile in tiles:
            assert isinstance(tile.image_loader, DefaultImageLoader)
            assert tile.image_loader.file_path != ""

    def test_tile_positions_and_sizes(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
        )
        first = tiles[0]
        assert first.start_x == 10.0
        assert first.start_y == 10.0
        assert first.length_x == 2560
        assert first.length_y == 2160
        assert first.fov_name == "FOV_1"

    def test_well_row_and_column(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
        )
        for tile in tiles:
            assert tile.collection.row == "A"
            assert tile.collection.column == 1

    def test_extra_columns_become_attributes(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
        )
        for tile in tiles:
            assert "drug" in tile.attributes
            assert tile.attributes["drug"] == ["DMSO"]

    def test_default_plate_name(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
        )
        assert tiles[0].collection.plate_name == "Plate"

    def test_multiple_fovs(self, hcs_acquisition_details: AcquisitionDetails) -> None:
        df = pd.read_csv(HCS_EXAMPLE_DIR / "tiles.csv")
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
        )
        fov_names = {tile.fov_name for tile in tiles}
        assert len(fov_names) == 3  # FOV_1, FOV_2, FOV_3

    def test_integer_row_conversion(
        self, hcs_acquisition_details: AcquisitionDetails
    ) -> None:
        """Test that integer row indices are converted to letters."""
        df = pd.DataFrame(
            {
                "file_path": ["img.tif"],
                "row": [2],
                "column": [3],
                "fov_name": ["FOV_1"],
                "start_x": [0.0],
                "start_y": [0.0],
                "length_x": [100],
                "length_y": [100],
            }
        )
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
        )
        assert tiles[0].collection.row == "B"
        assert tiles[0].collection.column == 3

    def test_minimal_columns(self, hcs_acquisition_details: AcquisitionDetails) -> None:
        """Test with only required columns."""
        df = pd.DataFrame(
            {
                "file_path": ["a.tif", "b.tif"],
                "row": ["A", "B"],
                "column": [1, 2],
                "fov_name": ["FOV_1", "FOV_2"],
                "start_x": [0.0, 100.0],
                "start_y": [0.0, 200.0],
                "length_x": [512, 512],
                "length_y": [512, 512],
            }
        )
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=hcs_acquisition_details,
            plate_name="MinPlate",
        )
        assert len(tiles) == 2
        assert tiles[0].collection.row == "A"
        assert tiles[1].collection.row == "B"


class TestSingleImagesFromDataframe:
    def test_load_from_example_csv(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        assert len(tiles) == len(df)

    def test_tile_collection_is_single_image(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        for tile in tiles:
            assert isinstance(tile.collection, SingleImage)

    def test_image_path_set_correctly(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        for tile in tiles:
            assert tile.collection.image_path == "cardiomyocyte_scan"

    def test_tile_loader_has_file_path(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        for tile in tiles:
            assert isinstance(tile.image_loader, DefaultImageLoader)
            assert (
                "20200812-CardiomyocyteDifferentiation14" in tile.image_loader.file_path
            )

    def test_tile_positions(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        # FOV_1 tiles at (10, 10)
        assert tiles[0].start_x == 10.0
        assert tiles[0].start_y == 10.0
        # FOV_2 tiles at (1000, 1000)
        assert tiles[2].start_x == 1000.0
        assert tiles[2].start_y == 1000.0

    def test_multiple_fovs(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        fov_names = {tile.fov_name for tile in tiles}
        assert fov_names == {"FOV_1", "FOV_2"}

    def test_z_offsets(self, single_acquisition_details: AcquisitionDetails) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        # First tile: z=0
        assert tiles[0].start_z == 0.0
        # Second tile: z=1
        assert tiles[1].start_z == 1.0

    def test_collection_path(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        df = pd.read_csv(SINGLE_EXAMPLE_DIR / "tiles.csv")
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        assert tiles[0].collection.path() == "cardiomyocyte_scan.zarr"

    def test_minimal_single_image(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        """Test with only required columns."""
        df = pd.DataFrame(
            {
                "file_path": ["scan_01.tif"],
                "image_path": ["my_image"],
                "fov_name": ["FOV_1"],
                "start_x": [0.0],
                "start_y": [0.0],
                "length_x": [256],
                "length_y": [256],
            }
        )
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        assert len(tiles) == 1
        assert tiles[0].collection.image_path == "my_image"
        assert tiles[0].image_loader.file_path == "scan_01.tif"

    def test_extra_columns_become_attributes(
        self, single_acquisition_details: AcquisitionDetails
    ) -> None:
        """Test that extra columns not matching any model field become attributes."""
        df = pd.DataFrame(
            {
                "file_path": ["scan.tif"],
                "image_path": ["img"],
                "fov_name": ["FOV_1"],
                "start_x": [0.0],
                "start_y": [0.0],
                "length_x": [100],
                "length_y": [100],
                "tissue_type": ["brain"],
                "sample_id": [42],
            }
        )
        tiles = single_images_from_dataframe(
            tiles_table=df,
            acquisition_details=single_acquisition_details,
        )
        assert tiles[0].attributes["tissue_type"] == ["brain"]
        assert tiles[0].attributes["sample_id"] == [42]
