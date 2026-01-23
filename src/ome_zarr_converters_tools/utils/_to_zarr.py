from typing import Any

from ngio import Image

from ome_zarr_converters_tools.models import TiledImage
from ome_zarr_converters_tools.models._acquisition import WriterMode


def sequential_tile_writing(
    tiled_image: TiledImage, image: Image, resource: Any
) -> None:
    """Write tiles sequentially to the OME-Zarr image.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    for region in tiled_image.regions:
        region_data = region.load_data(axes=tiled_image.axes, resource=resource)
        image.set_roi(roi=region.roi, patch=region_data)


def dask_map_block_tile_writing(
    tiled_image: TiledImage, image: Image, resource: Any
) -> None:
    """Write tiles in parallel to the OME-Zarr image.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    raise NotImplementedError("Parallel tile writing is not implemented yet.")


def sequential_fov_writing(
    tiled_image: TiledImage, image: Image, resource: Any
) -> None:
    """Write tiles sequentially to the OME-Zarr image.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    for group in tiled_image.group_by_fov():
        roi = group.roi()
        group_data = group.load_data(resource=resource)
        image.set_roi(roi=roi, patch=group_data)


def in_memory_writing(tiled_image: TiledImage, image: Image, resource: Any) -> None:
    """Write tiles in memory to the OME-Zarr image.

    For each region in the TiledImage, load the data and write it to the
    corresponding ROI in the OME-Zarr image.
    """
    full_image = tiled_image.load_data(resource=resource)
    roi = tiled_image.roi()
    image.set_roi(roi=roi, patch=full_image)


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
        dask_map_block_tile_writing(
            tiled_image=tiled_image, image=image, resource=resource
        )
    elif writer_mode == WriterMode.BY_FOV:
        sequential_fov_writing(tiled_image=tiled_image, image=image, resource=resource)
    elif writer_mode == WriterMode.IN_MEMORY:
        in_memory_writing(tiled_image=tiled_image, image=image, resource=resource)
    else:
        raise ValueError(f"Unknown writer mode: {writer_mode}")
