from ome_zarr_converters_tools.core._roi_utils import move_roi_by
from ome_zarr_converters_tools.core._tile_region import TiledImage, TileSlice
from ome_zarr_converters_tools.models import TilingMode
from ome_zarr_converters_tools.pipelines._snap_utils import (
    NotAGridError,
    calculate_snap_to_corner_offset,
    calculate_snap_to_grid_offset,
)


def _no_tiling(
    regions: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    return {key: {"x": 0.0, "y": 0.0} for key in regions.keys()}


def _snap_to_corners_tiling(
    regions: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    return calculate_snap_to_corner_offset(regions)


def _snap_to_grid_tiling(
    regions: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    return calculate_snap_to_grid_offset(regions)


def _auto_tiling(
    regions: dict[str, TileSlice],
) -> dict[str, dict[str, float]]:
    try:
        return _snap_to_grid_tiling(regions)
    except NotAGridError:
        return _snap_to_corners_tiling(regions)


def _find_tiling(
    regions: dict[str, TileSlice],
    tiling_mode: TilingMode,
) -> dict[str, dict[str, float]]:
    if tiling_mode == TilingMode.INPLACE or tiling_mode == TilingMode.NO_TILING:
        return _no_tiling(regions)
    if tiling_mode in [TilingMode.INPLACE, TilingMode.NO_TILING]:
        return _no_tiling(regions)
    if tiling_mode == TilingMode.AUTO:
        return _auto_tiling(regions)
    if tiling_mode == TilingMode.SNAP_TO_CORNERS:
        return _snap_to_corners_tiling(regions)
    if tiling_mode == TilingMode.SNAP_TO_GRID:
        return _snap_to_grid_tiling(regions)
    raise ValueError(f"Tiling mode '{tiling_mode}' is not recognized.")


def _tile_regions(
    regions: list[TileSlice], vector: dict[str, float]
) -> list[TileSlice]:
    for region in regions:
        region.roi = move_roi_by(region.roi, vector)
    return regions


def apply_mosaic_tiling(
    tiled_image: TiledImage,
    tiling_mode: TilingMode,
) -> TiledImage:
    """Tile all the TiledImages to the reference region of the first TiledImage.

    This function modifies the TiledImages in place.

    Args:
        tiled_image: TiledImage model to tile.
        tiling_mode: Tiling mode to use.

    """
    fov_tiles = tiled_image.group_by_fov()

    reference_regions = {}
    for fov_tile in fov_tiles:
        reference_regions[fov_tile.fov_name] = fov_tile.ref_slice()

    tiling_instructions = _find_tiling(reference_regions, tiling_mode)
    aligned_regions = []
    for fov_tile in fov_tiles:
        instruction = tiling_instructions[fov_tile.fov_name]
        tiled = _tile_regions(fov_tile.regions, instruction)
        aligned_regions.extend(tiled)
    tiled_image.regions = aligned_regions
    return tiled_image
