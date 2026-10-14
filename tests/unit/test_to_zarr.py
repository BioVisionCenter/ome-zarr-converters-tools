"""Unit tests for pipelines._to_zarr writing functions."""

from unittest.mock import MagicMock

import dask
import dask.array as da
import numpy as np
import pytest
import zarr
from ngio import Roi

import ome_zarr_converters_tools.pipelines._to_zarr as to_zarr_module
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import WriterMode
from ome_zarr_converters_tools.pipelines._to_zarr import (
    dask_parallel_fov_writing,
    dask_parallel_tile_writing,
    dask_write_chunks,
    dask_write_grid,
    grid_is_unit_aligned,
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


def _attach_array(mock_image: MagicMock, tiled_image: TiledImage, xy: int) -> None:
    """Give the mock image a real in-memory zarr array with `xy` chunks."""
    mock_image.zarr_array = zarr.create_array(
        store={},
        shape=tiled_image.shape(),
        chunks=_write_unit_for(tiled_image, xy),
        dtype=tiled_image.data_type,
    )


def _full_region(shape: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(0, n) for n in shape)


class TestDaskWriteGrid:
    def _array(
        self, shape: tuple[int, ...], xy: int, dtype: str = "uint16"
    ) -> "zarr.Array":
        chunks = tuple(xy if i >= len(shape) - 2 else 1 for i in range(len(shape)))
        return zarr.create_array(store={}, shape=shape, chunks=chunks, dtype=dtype)

    def test_blocks_are_bounded_multiples_of_the_write_unit(self) -> None:
        z = self._array((1, 1, 1, 8192, 8192), 1024)  # 2 MiB write units
        grid = dask_write_grid(z, _full_region(z.shape))
        block = tuple(sizes[0] for sizes in grid)
        assert all(b % c == 0 for b, c in zip(block, z.chunks, strict=True))
        assert np.prod(block) * 2 <= 8 * 2**20  # ngio's default write-block cap
        assert np.prod(block) * 2 > 2 * 2**20  # but larger than one chunk
        assert grid_is_unit_aligned(grid, _full_region(z.shape), z.chunks)

    def test_grid_patch_is_not_rechunked_by_to_zarr(self) -> None:
        z = self._array((1, 1, 1, 8192, 8192), 1024)
        grid = dask_write_grid(z, _full_region(z.shape))
        patch = da.zeros(z.shape, chunks=grid, dtype=z.dtype)
        # ngio applies its budget as `array.chunk-size` around `da.to_zarr`.
        with dask.config.set({"array.chunk-size": 8 * 2**20}):
            delayed = da.to_zarr(patch, z, compute=False)
        keys = {k[0] if isinstance(k, tuple) else k for k in delayed.dask}
        assert not any(str(k).startswith("rechunk") for k in keys)

    def test_region_grid_is_cut_from_the_array_grid(self) -> None:
        z = self._array((1, 1, 1, 4096, 4096), 1024)
        region = (
            slice(0, 1),
            slice(0, 1),
            slice(0, 1),
            slice(512, 2560),
            slice(0, 4096),
        )
        grid = dask_write_grid(z, region)
        assert sum(grid[3]) == 2048
        assert grid_is_unit_aligned(grid, region, z.chunks)

    @pytest.mark.parametrize(
        "shape, xy, dtype",
        [
            # Write units of 8 MiB (ngio's block budget) and axes 1.5 units
            # long with two channels: dask balances each axis into two blocks
            # instead of cutting it on the unit.
            ((1, 2, 1, 3072, 3072), 2048, "uint16"),
            ((1, 2, 1, 1536, 1536), 1024, "float64"),
        ],
    )
    def test_detects_a_grid_that_splits_a_write_unit(
        self, shape: tuple[int, ...], xy: int, dtype: str
    ) -> None:
        z = self._array(shape, xy, dtype)
        region = _full_region(shape)
        grid = dask_write_grid(z, region)
        assert not grid_is_unit_aligned(grid, region, z.chunks)

    def test_alignment_is_judged_in_array_coordinates(self) -> None:
        unit = (4,)
        # Region [2, 10): one interior edge at 4 is on the grid ...
        assert grid_is_unit_aligned(((2, 6),), (slice(2, 10),), unit)
        # ... one at 5 is not.
        assert not grid_is_unit_aligned(((3, 5),), (slice(2, 10),), unit)
        # Region edges inside a unit are fine: a single writer touches them.
        assert grid_is_unit_aligned(((7,),), (slice(2, 9),), unit)


class TestDaskWriteChunks:
    def test_mock_image_without_zarr_array_is_unsafe(
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

    def test_returns_the_dask_write_grid(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_array(mock_image, tiled_image_from_grid, 16)
        shape = tiled_image_from_grid.shape()
        chunks = dask_write_chunks(
            tiled_image_from_grid.roi(),
            shape,
            tiled_image_from_grid.axes,
            tiled_image_from_grid.pixel_size,
            mock_image,
        )
        assert chunks == dask_write_grid(mock_image.zarr_array, _full_region(shape))
        for n, sizes in zip(shape, chunks, strict=True):
            assert isinstance(sizes, tuple)
            assert sum(sizes) == n

    def test_unsafe_grid_returns_none(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_array(mock_image, tiled_image_from_grid, 16)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(to_zarr_module, "grid_is_unit_aligned", lambda *a: False)
            chunks = dask_write_chunks(
                tiled_image_from_grid.roi(),
                tiled_image_from_grid.shape(),
                tiled_image_from_grid.axes,
                tiled_image_from_grid.pixel_size,
                mock_image,
            )
        assert chunks is None


class TestDaskParallelTileWriting:
    def test_dask_patch_is_built_on_the_write_grid(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_array(mock_image, tiled_image_from_grid, 48)
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        patch = mock_image.set_roi.call_args.kwargs["patch"]
        assert isinstance(patch, da.Array)
        shape = tiled_image_from_grid.shape()
        assert patch.chunks == dask_write_grid(
            mock_image.zarr_array, _full_region(shape)
        )
        np.testing.assert_array_equal(
            patch.compute(), tiled_image_from_grid.load_data()
        )

    def test_writes_single_call(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_array(mock_image, tiled_image_from_grid, 48)
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        # Dask parallel loads the full image and writes once
        assert mock_image.set_roi.call_count == 1

    def test_passes_roi_and_dask_array(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
    ) -> None:
        _attach_array(mock_image, tiled_image_from_grid, 48)
        dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        call_args = mock_image.set_roi.call_args
        assert isinstance(call_args.kwargs["roi"], Roi)
        assert isinstance(call_args.kwargs["patch"], da.Array)

    def test_unsafe_grid_falls_back_to_sequential_tiles(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _attach_array(mock_image, tiled_image_from_grid, 48)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(to_zarr_module, "grid_is_unit_aligned", lambda *a: False)
            dask_parallel_tile_writing(tiled_image_from_grid, mock_image, resource=None)
        assert mock_image.set_roi.call_count == len(tiled_image_from_grid.regions)
        for call_args in mock_image.set_roi.call_args_list:
            assert isinstance(call_args.kwargs["patch"], np.ndarray)
        assert "Writing it sequentially" in caplog.text


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
        _attach_array(mock_image, tiled_image_from_grid, 48)
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
        _attach_array(mock_image, tiled_image_from_grid, 48)
        dask_parallel_fov_writing(tiled_image_from_grid, mock_image, resource=None)
        num_fovs = len(tiled_image_from_grid.group_by_fov())
        assert mock_image.set_roi.call_count == num_fovs
        for call_args in mock_image.set_roi.call_args_list:
            assert isinstance(call_args.kwargs["patch"], da.Array)

    def test_unsafe_grid_writes_the_fov_as_numpy(
        self,
        tiled_image_from_grid: TiledImage,
        mock_image: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _attach_array(mock_image, tiled_image_from_grid, 48)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(to_zarr_module, "grid_is_unit_aligned", lambda *a: False)
            dask_parallel_fov_writing(tiled_image_from_grid, mock_image, resource=None)
        groups = tiled_image_from_grid.group_by_fov()
        assert mock_image.set_roi.call_count == len(groups)
        for group, call_args in zip(
            groups, mock_image.set_roi.call_args_list, strict=True
        ):
            patch = call_args.kwargs["patch"]
            assert isinstance(patch, np.ndarray)
            np.testing.assert_array_equal(patch, group.load_data())
        assert "Writing it sequentially" in caplog.text


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
        _attach_array(mock_image, tiled_image_from_grid, 48)
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
