import time
from logging import getLogger
from typing import Any

from ngio import Image

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import WriterMode

logger = getLogger(__name__)


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
    logger.info("Starting Dask in-memory writing.")
    timer = time.time()
    full_image = tiled_image.load_data_dask(resource=resource)
    roi = tiled_image.roi()
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
        group_data = group.load_data_dask(resource=resource)
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
