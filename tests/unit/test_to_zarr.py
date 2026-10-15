"""Unit tests for pipelines._to_zarr writing functions."""

from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import pytest
from ngio import Roi

from ome_zarr_converters_tools.core._dask_lazy_loader import aligned_chunks
from ome_zarr_converters_tools.core._roi_utils import move_roi_by
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import ConvertersToolsInternalError, WriterMode
from ome_zarr_converters_tools.pipelines._to_zarr import (
    dask_parallel_fov_writing,
    dask_parallel_tile_writing,
    dask_write_chunks,
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


def _write_unit_for(tiled_image: TiledImage, xy: int) -> tuple[int, ...]:
    return tuple(xy if ax in ("y", "x") else 1 for ax in tiled_image.axes)


def _attach_write_unit(mock_image: MagicMock, tiled_image: TiledImage, xy: int) -> None:
    """Give the mock image a write unit of `xy` pixels on y and x."""
    mock_image.write_granularity = _write_unit_for(tiled_image, xy)


def _assert_cut_on_unit(
    chunks: tuple[tuple[int, ...], ...],
    start: tuple[int, ...],
    shape: tuple[int, ...],
    unit: tuple[int, ...],
) -> None:
    """Every interior block edge of `chunks`, offset by `start`, is on the grid."""
    for s, n, u, sizes in zip(start, shape, unit, chunks, strict=True):
        assert isinstance(sizes, tuple)
        assert sum(sizes) == n
        edges = [s + sum(sizes[:i]) for i in range(1, len(sizes))]
        assert all(edge % u == 0 for edge in edges)


class TestDaskWriteChunks:
    def test_image_without_write_unit_is_a_single_block(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        chunks = dask_write_chunks(
            tiled_image_from_grid.roi(),
            tiled_image_from_grid.shape(),
            tiled_image_from_grid.axes,
            tiled_image_from_grid.pixel_size,
            mock_image,
        )
        assert chunks is None

    def test_blocks_are_cut_on_the_write_unit_in_array_coordinates(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_write_unit(mock_image, tiled_image_from_grid, 16)
        unit = mock_image.write_granularity
        axes = tiled_image_from_grid.axes
        shape = tiled_image_from_grid.shape()
        pixel_size = tiled_image_from_grid.pixel_size
        # The region starts off the grid: at pixel 10 on x and 5 on y.
        roi = move_roi_by(
            tiled_image_from_grid.roi(), {"x": 10 * pixel_size.x, "y": 5 * pixel_size.y}
        )
        chunks = dask_write_chunks(roi, shape, axes, pixel_size, mock_image)
        assert chunks is not None
        start = tuple(10 if ax == "x" else 5 if ax == "y" else 0 for ax in axes)
        _assert_cut_on_unit(chunks, start, shape, unit)
        assert chunks == aligned_chunks(start, shape, unit)
        # Off-grid start: the first x block is shorter than the unit.
        assert chunks[axes.index("x")][0] == 6


class TestDaskParallelTileWriting:
    def test_dask_patch_is_cut_on_the_write_unit(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_write_unit(mock_image, tiled_image_from_grid, 48)
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        patch = mock_image.set_roi.call_args.kwargs["patch"]
        assert isinstance(patch, da.Array)
        shape = tiled_image_from_grid.shape()
        unit = mock_image.write_granularity
        assert patch.chunks == aligned_chunks((0,) * len(shape), shape, unit)
        assert patch.numblocks != (1,) * len(shape)
        np.testing.assert_array_equal(
            patch.compute(), tiled_image_from_grid.load_data()
        )

    def test_image_without_write_unit_gets_a_single_block(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        patch = mock_image.set_roi.call_args.kwargs["patch"]
        assert isinstance(patch, da.Array)
        assert patch.numblocks == (1,) * patch.ndim

    def test_writes_single_call(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_write_unit(mock_image, tiled_image_from_grid, 48)
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        # Dask parallel loads the full image and writes once
        assert mock_image.set_roi.call_count == 1

    def test_passes_roi_and_dask_array(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_write_unit(mock_image, tiled_image_from_grid, 48)
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        call_args = mock_image.set_roi.call_args
        assert isinstance(call_args.kwargs["roi"], Roi)
        assert isinstance(call_args.kwargs["patch"], da.Array)


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
    def test_fov_patches_are_aligned_to_write_unit(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_write_unit(mock_image, tiled_image_from_grid, 48)
        dask_parallel_fov_writing(tiled_image_from_grid, mock_image, resource=None)
        groups = tiled_image_from_grid.group_by_fov()
        pixel_size = tiled_image_from_grid.pixel_size
        for group, call_args in zip(
            groups, mock_image.set_roi.call_args_list, strict=True
        ):
            patch = call_args.kwargs["patch"]
            assert isinstance(patch, da.Array)
            pixel_roi = group.roi().to_pixel(pixel_size=pixel_size)
            for ax, sizes in zip(group.axes, patch.chunks, strict=True):
                if ax not in ("y", "x"):
                    continue
                roi_slice = pixel_roi.get(ax)
                assert roi_slice is not None
                start = round(roi_slice.start)
                edges = [start + sum(sizes[:i]) for i in range(1, len(sizes))]
                assert all(edge % 48 == 0 for edge in edges)
            np.testing.assert_array_equal(patch.compute(), group.load_data())

    def test_writes_per_fov(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_write_unit(mock_image, tiled_image_from_grid, 48)
        dask_parallel_fov_writing(tiled_image_from_grid, mock_image, resource=None)
        num_fovs = len(tiled_image_from_grid.group_by_fov())
        assert mock_image.set_roi.call_count == num_fovs
        for call_args in mock_image.set_roi.call_args_list:
            assert isinstance(call_args.kwargs["patch"], da.Array)


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
        _attach_write_unit(mock_image, tiled_image_from_grid, 48)
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
        with pytest.raises(ConvertersToolsInternalError, match="Unknown writer mode"):
            write_to_zarr(
                image=mock_image,
                tiled_image=tiled_image_from_grid,
                resource=None,
                writer_mode="invalid_mode",  # type: ignore[arg-type]
            )
