"""Unit tests for pipelines._to_zarr writing functions."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from ngio import Roi

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import WriterMode
from ome_zarr_converters_tools.pipelines._to_zarr import (
    dask_parallel_fov_writing,
    dask_parallel_tile_writing,
    in_memory_writing,
    sequential_fov_writing,
    sequential_tile_writing,
    write_to_zarr,
)


@pytest.fixture
def mock_image() -> MagicMock:
    """Mock ngio.Image with a set_roi method that records calls."""
    image = MagicMock()
    image.set_roi = MagicMock()
    return image


class TestSequentialTileWriting:
    def test_writes_each_region(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        sequential_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        assert mock_image.set_roi.call_count == len(tiled_image_from_grid.regions)

    def test_passes_correct_roi_and_data(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        sequential_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        for call_args in mock_image.set_roi.call_args_list:
            assert "roi" in call_args.kwargs
            assert "patch" in call_args.kwargs
            assert isinstance(call_args.kwargs["roi"], Roi)
            assert isinstance(call_args.kwargs["patch"], np.ndarray)

    def test_with_resource(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        sequential_tile_writing(tiled_image_from_grid, mock_image, resource="/data")
        assert mock_image.set_roi.call_count == len(tiled_image_from_grid.regions)


class TestDaskParallelTileWriting:
    def test_writes_single_call(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        # Dask parallel loads the full image and writes once
        assert mock_image.set_roi.call_count == 1

    def test_passes_roi_and_dask_array(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        call_args = mock_image.set_roi.call_args
        assert isinstance(call_args.kwargs["roi"], Roi)


class TestSequentialFovWriting:
    def test_writes_per_fov(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        sequential_fov_writing(tiled_image_from_grid, mock_image, resource=None)
        num_fovs = len(tiled_image_from_grid.group_by_fov())
        assert mock_image.set_roi.call_count == num_fovs

    def test_passes_correct_roi_and_data(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        sequential_fov_writing(tiled_image_from_grid, mock_image, resource=None)
        for call_args in mock_image.set_roi.call_args_list:
            assert isinstance(call_args.kwargs["roi"], Roi)
            assert isinstance(call_args.kwargs["patch"], np.ndarray)


class TestDaskParallelFovWriting:
    def test_writes_per_fov(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        dask_parallel_fov_writing(tiled_image_from_grid, mock_image, resource=None)
        num_fovs = len(tiled_image_from_grid.group_by_fov())
        assert mock_image.set_roi.call_count == num_fovs


class TestInMemoryWriting:
    def test_writes_single_call(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        in_memory_writing(tiled_image_from_grid, mock_image, resource=None)
        assert mock_image.set_roi.call_count == 1

    def test_data_shape_matches_tiled_image(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        in_memory_writing(tiled_image_from_grid, mock_image, resource=None)
        call_args = mock_image.set_roi.call_args
        patch = call_args.kwargs["patch"]
        assert patch.shape == tiled_image_from_grid.shape()

    def test_data_is_non_zero(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        in_memory_writing(tiled_image_from_grid, mock_image, resource=None)
        call_args = mock_image.set_roi.call_args
        patch = call_args.kwargs["patch"]
        assert patch.sum() > 0


class TestWriteToZarr:
    def test_by_tile_mode(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        write_to_zarr(
            image=mock_image,
            tiled_image=tiled_image_from_grid,
            resource=None,
            writer_mode=WriterMode.BY_TILE,
        )
        assert mock_image.set_roi.call_count == len(tiled_image_from_grid.regions)

    def test_by_tile_dask_mode(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        write_to_zarr(
            image=mock_image,
            tiled_image=tiled_image_from_grid,
            resource=None,
            writer_mode=WriterMode.BY_TILE_DASK,
        )
        assert mock_image.set_roi.call_count == 1

    def test_by_fov_mode(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        write_to_zarr(
            image=mock_image,
            tiled_image=tiled_image_from_grid,
            resource=None,
            writer_mode=WriterMode.BY_FOV,
        )
        num_fovs = len(tiled_image_from_grid.group_by_fov())
        assert mock_image.set_roi.call_count == num_fovs

    def test_by_fov_dask_mode(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        write_to_zarr(
            image=mock_image,
            tiled_image=tiled_image_from_grid,
            resource=None,
            writer_mode=WriterMode.BY_FOV_DASK,
        )
        num_fovs = len(tiled_image_from_grid.group_by_fov())
        assert mock_image.set_roi.call_count == num_fovs

    def test_in_memory_mode(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        write_to_zarr(
            image=mock_image,
            tiled_image=tiled_image_from_grid,
            resource=None,
            writer_mode=WriterMode.IN_MEMORY,
        )
        assert mock_image.set_roi.call_count == 1

    def test_unknown_mode_raises(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown writer mode"):
            write_to_zarr(
                image=mock_image,
                tiled_image=tiled_image_from_grid,
                resource=None,
                writer_mode="invalid_mode",  # type: ignore[arg-type]
            )
