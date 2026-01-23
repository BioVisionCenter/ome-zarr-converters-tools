import math
from collections.abc import Sequence

import numpy as np
from ngio import PixelSize, Roi, RoiSlice


def move_roi_by(
    roi: Roi,
    vector: dict[str, float],
) -> Roi:
    """Move a ROI by specified deltas in each axis.

    Args:
        roi: The ROI to move.
        vector: A dictionary mapping axis names to deltas.

    Returns:
        The moved ROI.
    """
    new_slices = []
    for roi_slice in roi.slices:
        delta = vector.get(roi_slice.axis_name, 0.0)
        assert roi_slice.start is not None
        new_start = roi_slice.start + delta
        new_slice = RoiSlice(
            axis_name=roi_slice.axis_name,
            start=new_start,
            length=roi_slice.length,
        )
        new_slices.append(new_slice)
    return roi.model_copy(update={"slices": new_slices})


def move_to(
    roi: Roi,
    new_starts: dict[str, float],
) -> Roi:
    """Move a ROI to specified origins in each axis.

    Args:
        roi: The ROI to move.
        new_starts: A dictionary mapping axis names to new origins.

    Returns:
        The moved ROI.
    """
    new_slices = []
    for roi_slice in roi.slices:
        new_start = new_starts.get(roi_slice.axis_name, roi_slice.start)
        assert new_start is not None
        new_slice = RoiSlice(
            axis_name=roi_slice.axis_name,
            start=new_start,
            length=roi_slice.length,
        )
        new_slices.append(new_slice)
    return roi.model_copy(update={"slices": new_slices})


def roi_to_roi_distance(
    roi1: Roi,
    roi2: Roi,
    axes: Sequence[str] | None = None,
) -> float:
    """Calculate the distance between two ROIs along a specified axis.

    Args:
        roi1: The first ROI.
        roi2: The second ROI.
        axes: List of axis names to calculate the distance for. If None, use all axes.

    Returns:
        The distance between the two ROIs along the specified axis.
    """
    sum_squared = 0.0
    axes_to_consider = axes if axes is not None else [s.axis_name for s in roi1.slices]
    for axis in axes_to_consider:
        slice1 = roi1.get(axis_name=axis)
        slice2 = roi2.get(axis_name=axis)
        assert slice1 is not None and slice2 is not None
        assert slice1.start is not None and slice2.start is not None
        diff = slice2.start - slice1.start
        sum_squared += diff * diff
    return sum_squared**0.5


def roi_to_point_distance(
    roi: Roi,
    point: dict[str, float],
) -> float:
    """Calculate the distance between a ROI and a point along specified axes.

    Args:
        roi: The ROI.
        point: A dictionary mapping axis names to point coordinates.
    """
    sum_squared = 0.0
    for axis, coord in point.items():
        slice_ = roi.get(axis_name=axis)
        assert slice_ is not None
        assert slice_.start is not None
        diff = coord - slice_.start
        sum_squared += diff * diff
    return sum_squared**0.5


def roi_corners(
    roi: Roi,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Get the corner coordinates of a ROI.

    Args:
        roi: The ROI to get the corners for.

    Returns:
        A dictionary mapping axis names to corner coordinates.
    """
    x_slice = roi.get(axis_name="x")
    y_slice = roi.get(axis_name="y")
    assert x_slice is not None and y_slice is not None
    x_start, x_end = x_slice.start, x_slice.end
    y_start, y_end = y_slice.start, y_slice.end
    assert x_start is not None and x_end is not None
    assert y_start is not None and y_end is not None
    corners = (
        {"x": x_start, "y": y_start},
        {"x": x_start, "y": y_end},
        {"x": x_end, "y": y_start},
        {"x": x_end, "y": y_end},
    )
    return corners


def zero_roi_from_roi(
    roi: Roi,
) -> Roi:
    """Create a zero-origin ROI with the same shape as the input ROI.

    Args:
        roi: The ROI to create a zero-origin ROI from.

    Returns:
        The zero-origin ROI.
    """
    new_slices = []
    for roi_slice in roi.slices:
        new_slice = RoiSlice(
            axis_name=roi_slice.axis_name,
            start=0.0,
            length=roi_slice.length,
        )
        new_slices.append(new_slice)
    return roi.model_copy(update={"slices": new_slices})


def bulk_roi_union(
    rois: Sequence[Roi],
) -> Roi:
    """Calculate the union of multiple ROIs.

    To avoit to build the union of all ROIs which can be computationally expensive,
    this function find the min and max ROIs along each axis and builds the union from
    them.

    Args:
        rois: List of ROIs to union.

    Returns:
        The union ROI.
    """
    min_max_rois = {}
    ref_roi = rois[0]
    for ax in ref_roi.slices:
        min_, max_ = np.inf, -np.inf
        min_roi, max_roi = ref_roi, ref_roi
        for roi in rois:
            current_slice = roi.get(axis_name=ax.axis_name)
            assert current_slice is not None
            assert current_slice.start is not None
            if current_slice.start < min_:
                min_ = current_slice.start
                min_roi = roi
            end = current_slice.end
            assert end is not None
            if end > max_:
                max_ = end
                max_roi = roi
        min_max_rois[f"{min_roi.get_name()}_{ax.axis_name}_min"] = min_roi
        min_max_rois[f"{max_roi.get_name()}_{ax.axis_name}_max"] = max_roi

    min_max_rois_values = list(min_max_rois.values())
    union_roi = min_max_rois_values[0]
    for roi in min_max_rois_values[1:]:
        union_roi = union_roi.union(roi)
    return union_roi


def shape_from_rois(
    rois: Sequence[Roi], axes: Sequence[str], pixel_size: PixelSize
) -> tuple[int, ...]:
    """Get the shape from a list of ROIs."""
    axes_shape = {}
    roi_union = bulk_roi_union(rois)
    roi_union = roi_union.to_pixel(pixel_size)
    for roi_slice in roi_union.slices:
        length = roi_slice.length
        assert length is not None
        axes_shape[roi_slice.axis_name] = math.ceil(length)  # TODO remove ceil?
    return tuple(axes_shape[ax] for ax in axes)
