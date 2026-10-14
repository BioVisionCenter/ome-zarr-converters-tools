import time
from collections.abc import Sequence
from logging import getLogger
from typing import Any

import dask
import numpy as np
import zarr
from dask.array.core import normalize_chunks
from dask.array.slicing import new_blockdim, normalize_index
from dask.utils import parse_bytes
from ngio import Image, PixelSize, Roi, get_config

from ome_zarr_converters_tools.core._dask_lazy_loader import ChunksLike
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import WriterMode

logger = getLogger(__name__)

Region = tuple[slice, ...]


def dask_write_grid(
    zarr_array: zarr.Array, region: Region
) -> tuple[tuple[int, ...], ...]:
    """Block grid `da.to_zarr` rechunks a patch for `region` of `zarr_array` onto.

    Dask sizes the blocks with `normalize_chunks("auto", previous_chunks=unit)`
    under `array.chunk-size`, where `unit` is the array's write unit (shards or
    chunks), then cuts that whole-array grid to the region. ngio caps the
    budget with `DaskConfig.write_block_max_bytes` (8 MiB by default) and never
    lets it undercut one write unit; the same computation is reproduced here.
    """
    unit = zarr_array.shards or zarr_array.chunks
    resident = [min(u, s) for u, s in zip(unit, zarr_array.shape, strict=True)]
    unit_bytes = int(np.prod(resident)) * zarr_array.dtype.itemsize
    budget = parse_bytes(dask.config.get("array.chunk-size"))
    cap = get_config().dask.write_block_max_bytes
    if cap is not None:
        budget = min(budget, cap)
    budget = max(budget, unit_bytes)
    with dask.config.set({"array.chunk-size": budget}):
        grid = normalize_chunks(
            "auto",
            shape=zarr_array.shape,
            dtype=zarr_array.dtype,
            previous_chunks=unit,
        )
    index = normalize_index(region, zarr_array.shape)
    return tuple(
        tuple(int(b) for b in new_blockdim(size, sizes, idx))
        for size, sizes, idx in zip(zarr_array.shape, grid, index, strict=True)
    )


def grid_is_unit_aligned(
    grid: tuple[tuple[int, ...], ...], region: Region, unit: tuple[int, ...]
) -> bool:
    """Whether every interior block edge of `grid` lands on a write-unit edge.

    `da.to_zarr` stores the blocks without a lock, so a unit shared by two
    blocks is read-modify-written by both and one update is lost. The region
    edges may sit inside a unit (that unit has a single writer); only the
    edges between blocks matter, taken in array coordinates.
    """
    for sizes, index, step in zip(grid, region, unit, strict=True):
        edge = index.start or 0
        for size in sizes[:-1]:
            edge += size
            if edge % step:
                return False
    return True


def dask_write_chunks(
    roi: Roi,
    shape: tuple[int, ...],
    axes: Sequence[str],
    pixel_size: PixelSize,
    image: Image,
) -> ChunksLike | None:
    """Dask block sizes for a safe, rechunk-free write of `roi` into `image`.

    ngio writes a dask patch through `da.to_zarr`, which rechunks it onto the
    grid of `dask_write_grid` and stores the blocks in parallel without a
    lock. Building the patch on that grid up front makes the rechunk a no-op,
    so the data streams a few blocks at a time instead of being materialized
    first. Dask's grid is not always cut on the write unit, though: when the
    unit is at least as large as ngio's block budget (8 MiB by default) and
    the array is not a multiple of it, dask balances the axes instead, and
    two blocks sharing a chunk silently lose data.

    Returns `None` when that is the case, or when `image` has no zarr array
    to read the grid from; the caller then writes the region sequentially.
    """
    zarr_array = getattr(image, "zarr_array", None)
    if not isinstance(zarr_array, zarr.Array) or len(zarr_array.shape) != len(axes):
        return None
    slicing = roi.to_slicing_dict(pixel_size=pixel_size)
    region = []
    for axis, length in zip(axes, shape, strict=True):
        axis_slice = slicing.get(axis)
        start = 0 if axis_slice is None else axis_slice.start or 0
        start = round(start)
        region.append(slice(start, start + length))
    region_t = tuple(region)
    grid = dask_write_grid(zarr_array, region_t)
    unit = zarr_array.shards or zarr_array.chunks
    if not grid_is_unit_aligned(grid, region_t, unit):
        return None
    return grid


_UNSAFE_DASK_WRITE = (
    "Dask would split a storage chunk of this region between two parallel "
    "writers, which can lose data (ngio 1.1 `store_dask` with chunks or shards "
    "of 8 MiB or more and an array that is not a multiple of them). Writing it "
    "sequentially instead."
)


def sequential_tile_writing(
    tiled_image: TiledImage, image: Image, resource: Any
) -> None:
    """Write tiles sequentially to the OME-Zarr image.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    regions = tiled_image.regions
    num_regions = len(regions)
    logger.info(f"Starting sequential tile writing - Number of tiles: {num_regions}.")
    timer = time.time()
    for idx, region in enumerate(regions):
        region_data = region.load_data(axes=tiled_image.axes, resource=resource)
        image.set_roi(roi=region.roi, patch=region_data)
        if idx == 0:
            elapsed = time.time() - timer
            estimated_total = elapsed * num_regions
            logger.info(
                f"Estimated total time for tile writing: {estimated_total:.2f} seconds."
            )


def dask_parallel_tile_writing(
    tiled_image: TiledImage, image: Image, resource: Any
) -> None:
    """Write tiles in memory to the OME-Zarr image using Dask.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    roi = tiled_image.roi()
    chunks = dask_write_chunks(
        roi, tiled_image.shape(), tiled_image.axes, tiled_image.pixel_size, image
    )
    if chunks is None:
        logger.warning(_UNSAFE_DASK_WRITE)
        sequential_tile_writing(tiled_image=tiled_image, image=image, resource=resource)
        return
    logger.info("Starting Dask in-memory writing.")
    timer = time.time()
    full_image = tiled_image.load_data_dask(resource=resource, chunks=chunks)
    image.set_roi(roi=roi, patch=full_image)
    elapsed = time.time() - timer
    logger.info(f"Elapsed time for Dask in-memory writing: {elapsed:.2f} seconds.")


def sequential_fov_writing(
    tiled_image: TiledImage, image: Image, resource: Any
) -> None:
    """Write tiles sequentially to the OME-Zarr image.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    groups = tiled_image.group_by_fov()
    num_groups = len(groups)
    logger.info(f"Starting sequential FOV writing - Number of FOVs: {num_groups}.")
    timer = time.time()
    for idx, group in enumerate(groups):
        roi = group.roi()
        group_data = group.load_data(resource=resource)
        image.set_roi(roi=roi, patch=group_data)
        if idx == 0:
            elapsed = time.time() - timer
            estimated_total = elapsed * num_groups
            logger.info(
                f"Estimated total time for FOV writing: {estimated_total:.2f} seconds."
            )


def dask_parallel_fov_writing(
    tiled_image: TiledImage, image: Image, resource: Any
) -> None:
    """Write tiles in parallel to the OME-Zarr image using Dask.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    groups = tiled_image.group_by_fov()
    num_groups = len(groups)
    logger.info(f"Starting Dask parallel FOV writing - Number of FOVs: {num_groups}.")
    timer = time.time()
    for idx, group in enumerate(groups):
        roi = group.roi()
        chunks = dask_write_chunks(
            roi, group.shape(), group.axes, group.pixel_size, image
        )
        if chunks is None:
            logger.warning(_UNSAFE_DASK_WRITE)
            image.set_roi(roi=roi, patch=group.load_data(resource=resource))
        else:
            group_data = group.load_data_dask(resource=resource, chunks=chunks)
            image.set_roi(roi=roi, patch=group_data)
        if idx == 0:
            elapsed = time.time() - timer
            estimated_total = elapsed * num_groups
            logger.info(
                "Estimated total time for Dask parallel "
                f"FOV writing: {estimated_total:.2f} seconds."
            )


def in_memory_writing(tiled_image: TiledImage, image: Image, resource: Any) -> None:
    """Write tiles in memory to the OME-Zarr image.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    logger.info("Starting in-memory writing.")
    timer = time.time()
    full_image = tiled_image.load_data(resource=resource)
    roi = tiled_image.roi()
    image.set_roi(roi=roi, patch=full_image)
    elapsed = time.time() - timer
    logger.info(f"Elapsed time for in-memory writing: {elapsed:.2f} seconds.")


def write_to_zarr(
    *,
    image: Image,
    tiled_image: TiledImage,
    resource: Any | None,
    writer_mode: WriterMode,
) -> None:
    if writer_mode == WriterMode.BY_TILE:
        sequential_tile_writing(tiled_image=tiled_image, image=image, resource=resource)
    elif writer_mode == WriterMode.BY_TILE_DASK:
        dask_parallel_tile_writing(
            tiled_image=tiled_image, image=image, resource=resource
        )
    elif writer_mode == WriterMode.BY_FOV:
        sequential_fov_writing(tiled_image=tiled_image, image=image, resource=resource)
    elif writer_mode == WriterMode.BY_FOV_DASK:
        dask_parallel_fov_writing(
            tiled_image=tiled_image, image=image, resource=resource
        )
    elif writer_mode == WriterMode.IN_MEMORY:
        in_memory_writing(tiled_image=tiled_image, image=image, resource=resource)
    else:
        raise ValueError(f"Unknown writer mode: {writer_mode}")
