import math
import warnings
from typing import Literal

from ngio import RoiSlice

from ome_zarr_converters_tools.core import (
    TiledImage,
    TileSlice,
)
from ome_zarr_converters_tools.core._roi_utils import move_roi_by, move_to
from ome_zarr_converters_tools.models import AlignmentCorrections


def _align_xy_regions(regions: list[TileSlice]) -> list[TileSlice]:
    ref_region = regions[0]
    ref_start = {}
    for ax in ["x", "y"]:
        slice_ = ref_region.roi.get(ax)
        assert slice_ is not None
        ref_start[ax] = slice_.start or 0.0
    update_regions = [ref_region.model_copy()]
    for region in regions[1:]:
        region.roi = move_to(region.roi, ref_start)
        update_regions.append(region)
    return update_regions


def _align_z_regions(regions: list[TileSlice]) -> list[TileSlice]:
    warnings.warn(
        "Z alignment is not implemented yet. Returning regions unchanged.",
        UserWarning,
        stacklevel=2,
    )
    return regions


def _align_t_regions(regions: list[TileSlice]) -> list[TileSlice]:
    warnings.warn(
        "T alignment is not implemented yet. Returning regions unchanged.",
        UserWarning,
        stacklevel=2,
    )
    return regions


def _align_regions(
    regions: list[TileSlice],
    alignment_corrections: AlignmentCorrections,
) -> list[TileSlice]:
    if alignment_corrections.align_xy:
        regions = _align_xy_regions(regions)
    if alignment_corrections.align_z:
        regions = _align_z_regions(regions)
    if alignment_corrections.align_t:
        regions = _align_t_regions(regions)
    return regions


def apply_fov_alignment_corrections(
    tiled_image: TiledImage, alignment_corrections: AlignmentCorrections
) -> TiledImage:
    """Align all the regions in a TiledImage to be consistent.

    The function:
        -groups regions by their field of view (FOV)
        -applies alignment corrections to each group
        -updates the TiledImage with the aligned regions

    Args:
        tiled_image: TiledImage model to align.
        alignment_corrections: AlignmentCorrections model specifying which
            corrections to apply.

    """
    fov_tiles = tiled_image.group_by_fov()
    aligned_regions = []
    for fov_tile in fov_tiles:
        aligned = _align_regions(fov_tile.regions, alignment_corrections)
        aligned_regions.extend(aligned)
    tiled_image.regions = aligned_regions
    return tiled_image


def apply_align_to_pixel_grid(
    tiled_image: TiledImage, mode: Literal["round", "floor", "ceil"] = "floor"
) -> TiledImage:
    """Align the start position to the pixel grid.

    For each tile, adjust the start position to the nearest pixel grid position.
    """
    if mode == "round":
        op = round
    elif mode == "floor":
        op = math.floor
    elif mode == "ceil":
        op = math.ceil
    else:
        raise ValueError(f"Mode '{mode}' is not recognized.")

    pixel_size = tiled_image.pixel_size
    for region in tiled_image.regions:
        adjusted_slices = []
        roi = region.roi.to_pixel(pixel_size=pixel_size)
        for roi_slice in roi.slices:
            start = roi_slice.start
            length = roi_slice.length
            assert start is not None and length is not None
            new_start = op(start)
            new_length = op(length)
            adjusted_slices.append(
                RoiSlice(
                    axis_name=roi_slice.axis_name,
                    start=new_start,
                    length=new_length,
                )
            )
        updated_roi = roi.model_copy(update={"slices": adjusted_slices})
        region.roi = updated_roi
    return tiled_image


def apply_remove_offsets(tiled_image: TiledImage) -> TiledImage:
    """Remove any offsets from the tile positions.

    This will find the minimum position in each dimension and
        subtract it from all tiles.
    """
    tile_slices = tiled_image.regions

    # Find the minimum start position for each axis
    min_starts = {}
    for tile in tile_slices:
        for roi_slice in tile.roi.slices:
            axis = roi_slice.axis_name
            start = roi_slice.start or 0.0
            if axis not in min_starts:
                min_starts[axis] = start
            else:
                min_starts[axis] = min(min_starts[axis], start)

    # Compute the vector shifts to move the minimum to zero
    offset_shifts = {axis: -min_start for axis, min_start in min_starts.items()}

    # Apply the shifts to each tile's ROI
    for tile in tile_slices:
        tile.roi = move_roi_by(tile.roi, offset_shifts)
    tiled_image = tiled_image.model_copy(update={"regions": tile_slices})
    return tiled_image
