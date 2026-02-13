"""Integration tests for conversion pipelines."""

from pathlib import Path
from typing import Any

import numpy as np

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    AlignmentCorrections,
    ChannelInfo,
    ConverterOptions,
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
    build_default_registration_pipeline,
)


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
