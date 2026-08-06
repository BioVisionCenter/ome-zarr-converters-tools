import math
from typing import Literal

from ngio import RoiSlice

from ome_zarr_converters_tools.core import (
    TiledImage,
    TileSlice,
)
from ome_zarr_converters_tools.core._roi_utils import move_roi_by, move_to
from ome_zarr_converters_tools.models import StagePositionCorrections


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


def apply_xy_jitter_correction(
    tiled_image: TiledImage, corrections: StagePositionCorrections
) -> TiledImage:
    """Remove intra-FOV XY stage-position jitter.

    Groups regions by FOV and snaps each group's sub-tiles to a shared XY origin
    (the group's reference region). No-op when `remove_xy_jitter` is False.
    """
    if not corrections.remove_xy_jitter:
        return tiled_image
    aligned_regions = []
    for fov_tile in tiled_image.group_by_fov():
        aligned_regions.extend(_align_xy_regions(fov_tile.regions))
    tiled_image.regions = aligned_regions
    return tiled_image


def _axis_min(regions: list[TileSlice], axis: str) -> float | None:
    """Minimum start of `axis` across `regions`, or None if the axis is absent."""
    starts = [
        s.start
        for region in regions
        if (s := region.roi.get(axis)) is not None and s.start is not None
    ]
    return min(starts) if starts else None


def _shift_axes_to_zero(regions: list[TileSlice], axes: list[str]) -> None:
    """Translate `regions` in place so each axis's minimum start becomes 0."""
    shifts = {}
    for axis in axes:
        axis_min = _axis_min(regions, axis)
        if axis_min is not None:
            shifts[axis] = -axis_min
    for region in regions:
        region.roi = move_roi_by(region.roi, shifts)


def _validate_non_negative(regions: list[TileSlice], axes: list[str]) -> None:
    """Raise if any region has a negative start on one of `axes` (Keep mode)."""
    field = {"x": "xy", "y": "xy", "z": "z", "t": "t"}
    for region in regions:
        for axis in axes:
            slice_ = region.roi.get(axis)
            if slice_ is not None and slice_.start is not None and slice_.start < 0:
                raise ValueError(
                    f"remove_{field[axis]}_offset='Keep' requires non-negative "
                    f"{axis} positions, got {slice_.start}."
                )


def apply_offset_removal(
    tiled_image: TiledImage, corrections: StagePositionCorrections
) -> TiledImage:
    """Remove per-axis stage offsets according to the correction modes.

    - `Global`: translate so the axis's global minimum start is 0.
    - `Per-FOV` (z only): translate each FOV group's z independently to 0.
    - `Keep`: keep absolute positions (raise on negatives; positives left-pad).
    """
    if corrections.remove_xy_offset == "Global":
        _shift_axes_to_zero(tiled_image.regions, ["x", "y"])
    else:
        _validate_non_negative(tiled_image.regions, ["x", "y"])

    if corrections.remove_z_offset == "Global":
        _shift_axes_to_zero(tiled_image.regions, ["z"])
    elif corrections.remove_z_offset == "Per-FOV":
        new_regions = []
        for fov_tile in tiled_image.group_by_fov():
            _shift_axes_to_zero(fov_tile.regions, ["z"])
            new_regions.extend(fov_tile.regions)
        tiled_image.regions = new_regions
    else:
        _validate_non_negative(tiled_image.regions, ["z"])

    if corrections.remove_t_offset == "Global":
        _shift_axes_to_zero(tiled_image.regions, ["t"])
    else:
        _validate_non_negative(tiled_image.regions, ["t"])

    return tiled_image


def _channel_range(region: TileSlice) -> tuple[int, int]:
    """Return a region's channel span as `(start, length)` in integer indices."""
    slice_ = region.roi.get("c")
    if slice_ is None or slice_.start is None:
        raise ValueError("Tile ROI is missing the 'c' axis slice.")
    length = 1 if slice_.length is None else round(slice_.length)
    return round(slice_.start), length


def apply_reindex_channels(
    tiled_image: TiledImage, corrections: StagePositionCorrections
) -> TiledImage:
    """Compact channel indices to consecutive integers (drop filtered channels).

    When `reindex_channels` is True, the distinct channel indices present across
    the tiles are mapped to `0, 1, 2, …` and the channel metadata is compacted to
    match. Both happen even when the present indices are already dense, since the
    metadata can still declare channels the image does not acquire.

    When False, both the indices and the metadata are left alone, and
    `TiledImage.output_shape` then sizes the `c` axis from `channels`, so every
    declared channel gets a plane and the ones no tile covers are written empty.

    Note:
        Tiles spanning several channels (`length_c > 1`) are remapped by their
        start index alone, and their length is left untouched. That is exact, not
        an approximation: `present` holds every index any tile covers, so a tile's
        span is a run of consecutive integers all of which are in `present`, and
        consecutive integers always get consecutive ranks. Compaction can
        therefore never split a tile's span or change which spans overlap.
    """
    if not corrections.reindex_channels:
        return tiled_image

    if "c" not in tiled_image.axes:
        return tiled_image  # no channel axis, nothing to reindex or compact

    ranges = [_channel_range(region) for region in tiled_image.regions]
    present = sorted(
        {c for start, length in ranges for c in range(start, start + length)}
    )
    if not present:
        return tiled_image  # no tiles, nothing to reindex or compact

    if tiled_image.channels is not None:
        if present[-1] >= len(tiled_image.channels):
            # Unreachable: Tile/TiledImage validators enforce this at construction.
            raise ValueError(
                f"Internal error: TiledImage '{tiled_image.path}' references "
                f"channel index {present[-1]} but channels has "
                f"{len(tiled_image.channels)} entries; this should have been "
                "rejected at tile building. Please report this as a bug."
            )
        tiled_image.channels = [tiled_image.channels[c] for c in present]

    if present == list(range(len(present))):
        # Indices are already dense, and the metadata above now matches them.
        # This check must stay *below* the compaction: a dense index set says
        # nothing about how many channels the metadata declares.
        return tiled_image

    remap = {old: new for new, old in enumerate(present)}
    for region, (start, _) in zip(tiled_image.regions, ranges, strict=True):
        region.roi = move_to(region.roi, {"c": float(remap[start])})
    return tiled_image


def apply_align_to_pixel_grid(
    tiled_image: TiledImage, mode: Literal["round", "floor", "ceil"] = "round"
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
            # Snap the interval endpoints, not start and length independently:
            # op(start) + op(length) can differ from op(start + length) by one
            # pixel, creating gaps/overlaps at tile boundaries.
            new_start = op(start)
            new_length = op(start + length) - new_start
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
