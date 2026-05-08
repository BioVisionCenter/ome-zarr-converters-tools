from ome_zarr_converters_tools.core._roi_utils import move_roi_by
from ome_zarr_converters_tools.core._tile_region import TiledImage, TileSlice
from ome_zarr_converters_tools.models import (
    AutoTiling,
    InplaceTiling,
    NoTiling,
    SnapToCornersTiling,
    SnapToGridTiling,
    TilingStrategy,
)
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
    tolerance: float = 0.0,
) -> dict[str, dict[str, float]]:
    return calculate_snap_to_grid_offset(regions, tolerance)


def _auto_tiling(
    regions: dict[str, TileSlice],
    tolerance: float = 0.0,
) -> dict[str, dict[str, float]]:
    try:
        return _snap_to_grid_tiling(regions, tolerance)
    except NotAGridError:
        return _snap_to_corners_tiling(regions)


def _find_tiling(
    regions: dict[str, TileSlice],
    strategy: TilingStrategy,
) -> dict[str, dict[str, float]]:
    if isinstance(strategy, (InplaceTiling, NoTiling)):
        return _no_tiling(regions)
    if isinstance(strategy, AutoTiling):
        return _auto_tiling(regions, strategy.tolerance)
    if isinstance(strategy, SnapToCornersTiling):
        return _snap_to_corners_tiling(regions)
    if isinstance(strategy, SnapToGridTiling):
        return _snap_to_grid_tiling(regions, strategy.tolerance)
    raise ValueError(f"Tiling strategy '{strategy}' is not recognized.")


def _tile_regions(
    regions: list[TileSlice], vector: dict[str, float]
) -> list[TileSlice]:
    for region in regions:
        region.roi = move_roi_by(region.roi, vector)
    return regions


def apply_mosaic_tiling(
    tiled_image: TiledImage,
    tiling_strategy: TilingStrategy,
) -> TiledImage:
    """Tile all the TiledImages to the reference region of the first TiledImage.

    This function modifies the TiledImages in place.

    Args:
        tiled_image: TiledImage model to tile.
        tiling_strategy: Tiling strategy to use.
    """
    fov_tiles = tiled_image.group_by_fov()

    reference_regions = {}
    for fov_tile in fov_tiles:
        reference_regions[fov_tile.fov_name] = fov_tile.ref_slice()

    tiling_instructions = _find_tiling(reference_regions, tiling_strategy)
    aligned_regions = []
    for fov_tile in fov_tiles:
        instruction = tiling_instructions[fov_tile.fov_name]
        tiled = _tile_regions(fov_tile.regions, instruction)
        aligned_regions.extend(tiled)
    tiled_image.regions = aligned_regions
    return tiled_image
