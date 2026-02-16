"""Integration tests for conversion pipelines."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._table import (
    hcs_images_from_dataframe,
    single_images_from_dataframe,
)
from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    AlignmentCorrections,
    ChannelInfo,
    ConverterOptions,
    ImageInPlate,
    OverwriteMode,
    SingleImage,
    TilingMode,
    WriterMode,
)
from ome_zarr_converters_tools.pipelines import (
    tiled_image_creation_pipeline,
    tiles_aggregation_pipeline,
)
from ome_zarr_converters_tools.pipelines._filters import RegexIncludeFilter
from ome_zarr_converters_tools.pipelines._registration_pipeline import (
    apply_registration_pipeline,
    build_default_registration_pipeline,
)

# ---------------------------------------------------------------------------
# Project root / example paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HCS_EXAMPLE_DIR = _PROJECT_ROOT / "examples" / "hcs_plate"
_SINGLE_EXAMPLE_DIR = _PROJECT_ROOT / "examples" / "single_acquisitions"
_HCS_DATA_DIR = _HCS_EXAMPLE_DIR / "data"


def _example_acq_details() -> AcquisitionDetails:
    """AcquisitionDetails matching the example TOML files."""
    return AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI", wavelength_id="405")],
        pixelsize=0.65,
        z_spacing=5.0,
        t_spacing=1.0,
        axes=["t", "c", "z", "y", "x"],
        start_x_coo="world",
        start_y_coo="world",
        start_z_coo="pixel",
        start_t_coo="pixel",
    )


# ---------------------------------------------------------------------------
# Existing dummy-tile helpers (kept for backward compat)
# ---------------------------------------------------------------------------


def _acq(n_channels: int = 1) -> AcquisitionDetails:
    return AcquisitionDetails(
        channels=[ChannelInfo(channel_label=f"CH{i}") for i in range(n_channels)],
        pixelsize=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )


def _make_tiles(collection: SingleImage, n_channels: int = 1) -> list[Tile[Any, Any]]:
    """Build a 2x2 grid of tiles."""
    acq = _acq(n_channels)
    positions = [(0, 0), (64, 0), (0, 64), (64, 64)]
    return [
        build_dummy_tile(
            fov_name=f"FOV_{i}",
            start=StartPosition(x=x, y=y),
            shape=TileShape(x=64, y=64, z=1, c=n_channels, t=1),
            collection=collection,
            acquisition_details=acq,
        )
        for i, (x, y) in enumerate(positions)
    ]


# ===================================================================
# Existing tests (synthetic / DummyLoader)
# ===================================================================


class TestTilesAggregationPipeline:
    def test_basic_aggregation(self) -> None:
        coll = SingleImage(image_path="test_agg")
        tiles = _make_tiles(coll)
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(tiles=tiles, converter_options=opts)
        assert len(images) == 1
        assert len(images[0].regions) == 4

    def test_aggregation_with_filter(self) -> None:
        coll = SingleImage(image_path="img_keep")
        tiles_keep = _make_tiles(coll)[:2]
        coll2 = SingleImage(image_path="img_drop")
        tiles_drop = _make_tiles(coll2)[2:]
        all_tiles = tiles_keep + tiles_drop
        opts = ConverterOptions()
        f = RegexIncludeFilter(regex=".*keep.*")
        images = tiles_aggregation_pipeline(
            tiles=all_tiles, converter_options=opts, filters=[f]
        )
        assert len(images) == 1
        assert "keep" in images[0].path


class TestTiledImageCreationPipeline:
    def test_write_single_image(self, tmp_path: Path) -> None:
        coll = SingleImage(image_path="test_write")
        tiles = _make_tiles(coll)
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(tiles=tiles, converter_options=opts)
        tiled_image = images[0]

        pipeline = build_default_registration_pipeline(
            AlignmentCorrections(), TilingMode.INPLACE
        )
        zarr_url = str(tmp_path / "output.zarr")
        omezarr = tiled_image_creation_pipeline(
            zarr_url=zarr_url,
            tiled_image=tiled_image,
            registration_pipeline=pipeline,
            converter_options=opts,
            writer_mode=WriterMode.BY_FOV_DASK,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        assert omezarr is not None
        # Verify the written data is readable
        img = omezarr.get_image()
        data = img.get_array()
        assert data.shape[-2:] == (128, 128)  # 2x2 grid of 64x64
        assert np.any(data > 0)


# ===================================================================
# Real-data integration tests (examples/ PNGs)
# ===================================================================


class TestHCSPlateEndToEnd:
    """End-to-end tests using the HCS plate example with real PNG images."""

    def test_build_tiles_from_csv(self) -> None:
        df = pd.read_csv(_HCS_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = hcs_images_from_dataframe(
            tiles_table=df,
            acquisition_details=acq,
            plate_name="TestPlate",
            acquisition_id=0,
        )

        assert len(tiles) == 6
        fov_names = {t.fov_name for t in tiles}
        assert fov_names == {"FOV_1", "FOV_2", "FOV_3"}

        # All tiles belong to well A/1
        for tile in tiles:
            assert isinstance(tile.collection, ImageInPlate)
            assert tile.collection.row == "A"
            assert tile.collection.column == 1
            assert tile.collection.plate_name == "TestPlate"

        # Drug attribute present on every tile
        for tile in tiles:
            assert "drug" in tile.attributes
            assert tile.attributes["drug"] == ["DMSO"]

    def test_aggregation_produces_single_well(self) -> None:
        df = pd.read_csv(_HCS_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = hcs_images_from_dataframe(
            tiles_table=df, acquisition_details=acq, plate_name="TestPlate"
        )
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(
            tiles=tiles, converter_options=opts, resource=str(_HCS_DATA_DIR)
        )

        # All tiles share the same well → 1 TiledImage
        assert len(images) == 1
        tiled_image = images[0]
        assert len(tiled_image.regions) == 6

        # 3 FOVs with 2 Z-slices each
        fov_groups = tiled_image.group_by_fov()
        assert len(fov_groups) == 3

    def test_registration_aligns_positions(self) -> None:
        df = pd.read_csv(_HCS_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = hcs_images_from_dataframe(
            tiles_table=df, acquisition_details=acq, plate_name="TestPlate"
        )
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(
            tiles=tiles, converter_options=opts, resource=str(_HCS_DATA_DIR)
        )
        tiled_image = images[0]

        pipeline = build_default_registration_pipeline(
            AlignmentCorrections(), TilingMode.AUTO
        )
        registered = apply_registration_pipeline(tiled_image, pipeline)

        # After registration, all start positions should be pixel-aligned
        # (i.e. no fractional values like 10.1 remaining)
        for region in registered.regions:
            for s in region.roi.slices:
                if s.start is not None:
                    assert float(s.start) == int(
                        s.start
                    ), f"start={s.start} is not pixel-aligned"

    def test_full_pipeline_writes_omezarr(self, tmp_path: Path) -> None:
        df = pd.read_csv(_HCS_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = hcs_images_from_dataframe(
            tiles_table=df, acquisition_details=acq, plate_name="TestPlate"
        )
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(
            tiles=tiles, converter_options=opts, resource=str(_HCS_DATA_DIR)
        )
        tiled_image = images[0]

        pipeline = build_default_registration_pipeline(
            AlignmentCorrections(), TilingMode.AUTO
        )
        zarr_url = str(tmp_path / "output.zarr")
        omezarr = tiled_image_creation_pipeline(
            zarr_url=zarr_url,
            tiled_image=tiled_image,
            registration_pipeline=pipeline,
            converter_options=opts,
            writer_mode=WriterMode.BY_FOV,
            overwrite_mode=OverwriteMode.OVERWRITE,
            resource=str(_HCS_DATA_DIR),
        )

        # Basic existence checks
        assert omezarr is not None
        assert Path(zarr_url).exists()

        # Image is readable with correct properties
        img = omezarr.get_image()
        data = img.get_array()
        assert data.ndim == 5  # t, c, z, y, x
        assert data.shape[2] == 2  # 2 Z-slices
        assert data.shape[1] == 1  # 1 channel (DAPI)
        assert np.any(data > 0)  # Real image data was loaded

        # Channel metadata
        assert img.num_channels == 1
        assert "DAPI" in img.channel_labels

        # ROI tables were written
        table_names = omezarr.list_tables()
        assert "FOV_ROI_table" in table_names
        assert "well_ROI_table" in table_names

    @pytest.mark.parametrize(
        "writer_mode",
        [
            WriterMode.BY_TILE,
            WriterMode.BY_FOV,
            WriterMode.BY_FOV_DASK,
            WriterMode.IN_MEMORY,
        ],
    )
    def test_writer_modes_produce_same_shape(
        self, tmp_path: Path, writer_mode: WriterMode
    ) -> None:
        df = pd.read_csv(_HCS_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = hcs_images_from_dataframe(
            tiles_table=df, acquisition_details=acq, plate_name="TestPlate"
        )
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(
            tiles=tiles, converter_options=opts, resource=str(_HCS_DATA_DIR)
        )
        tiled_image = images[0]

        pipeline = build_default_registration_pipeline(
            AlignmentCorrections(), TilingMode.AUTO
        )
        zarr_url = str(tmp_path / f"output_{writer_mode.value}.zarr")
        omezarr = tiled_image_creation_pipeline(
            zarr_url=zarr_url,
            tiled_image=tiled_image,
            registration_pipeline=pipeline,
            converter_options=opts,
            writer_mode=writer_mode,
            overwrite_mode=OverwriteMode.OVERWRITE,
            resource=str(_HCS_DATA_DIR),
        )

        img = omezarr.get_image()
        data = img.get_array()
        # All writer modes should produce same dimensions
        assert data.ndim == 5
        assert data.shape[1] == 1  # 1 channel
        assert data.shape[2] == 2  # 2 Z-slices
        assert np.any(data > 0)


class TestSingleImageEndToEnd:
    """End-to-end tests using the single acquisitions example."""

    def test_build_tiles_from_csv(self) -> None:
        df = pd.read_csv(_SINGLE_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = single_images_from_dataframe(tiles_table=df, acquisition_details=acq)

        assert len(tiles) == 4
        for tile in tiles:
            assert isinstance(tile.collection, SingleImage)
            assert tile.collection.image_path == "cardiomyocyte_scan"

        fov_names = {t.fov_name for t in tiles}
        assert fov_names == {"FOV_1", "FOV_2"}

    def test_full_pipeline_writes_omezarr(self, tmp_path: Path) -> None:
        df = pd.read_csv(_SINGLE_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = single_images_from_dataframe(tiles_table=df, acquisition_details=acq)
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(
            tiles=tiles, converter_options=opts, resource=str(_HCS_DATA_DIR)
        )
        assert len(images) == 1
        tiled_image = images[0]

        pipeline = build_default_registration_pipeline(
            AlignmentCorrections(), TilingMode.AUTO
        )
        zarr_url = str(tmp_path / "single_output.zarr")
        omezarr = tiled_image_creation_pipeline(
            zarr_url=zarr_url,
            tiled_image=tiled_image,
            registration_pipeline=pipeline,
            converter_options=opts,
            writer_mode=WriterMode.BY_FOV,
            overwrite_mode=OverwriteMode.OVERWRITE,
            resource=str(_HCS_DATA_DIR),
        )

        assert omezarr is not None
        img = omezarr.get_image()
        data = img.get_array()
        assert data.ndim == 5
        assert data.shape[2] == 2  # 2 Z-slices
        assert data.shape[1] == 1  # 1 channel
        assert np.any(data > 0)

        # ROI tables
        table_names = omezarr.list_tables()
        assert "FOV_ROI_table" in table_names
        assert "well_ROI_table" in table_names


class TestHCSPlateWithAttributes:
    """Test that CSV attributes (e.g. drug column) flow through to OME-Zarr."""

    def test_condition_table_written(self, tmp_path: Path) -> None:
        df = pd.read_csv(_HCS_EXAMPLE_DIR / "tiles.csv")
        acq = _example_acq_details()
        tiles = hcs_images_from_dataframe(
            tiles_table=df, acquisition_details=acq, plate_name="TestPlate"
        )
        opts = ConverterOptions()
        images = tiles_aggregation_pipeline(
            tiles=tiles, converter_options=opts, resource=str(_HCS_DATA_DIR)
        )
        tiled_image = images[0]

        # Verify attributes were collected
        assert "drug" in tiled_image.attributes

        pipeline = build_default_registration_pipeline(
            AlignmentCorrections(), TilingMode.AUTO
        )
        zarr_url = str(tmp_path / "attrs_output.zarr")
        omezarr = tiled_image_creation_pipeline(
            zarr_url=zarr_url,
            tiled_image=tiled_image,
            registration_pipeline=pipeline,
            converter_options=opts,
            writer_mode=WriterMode.BY_FOV,
            overwrite_mode=OverwriteMode.OVERWRITE,
            resource=str(_HCS_DATA_DIR),
        )

        table_names = omezarr.list_tables()
        assert "condition_table" in table_names
