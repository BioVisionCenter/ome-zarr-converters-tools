from collections.abc import Sequence
from logging import getLogger
from typing import Any

import polars as pl
import zarr
from ngio import (
    OmeZarrContainer,
    PixelSize,
    RoiSlice,
    create_empty_ome_zarr,
    open_ome_zarr_container,
)
from ngio.ome_zarr_meta import Channel, ChannelVisualisation
from ngio.tables import ConditionTable, RoiTable

from ome_zarr_converters_tools.core._tile_region import (
    AttributeType,
    TiledImage,
    TileSlice,
)
from ome_zarr_converters_tools.models import (
    ConverterOptions,
    NoSharding,
    OmeZarrOptions,
    OverwriteMode,
    WriterMode,
)
from ome_zarr_converters_tools.models._errors import (
    ConvertersToolsInternalError,
    ConvertersToolsUserError,
)
from ome_zarr_converters_tools.models._url_utils import filesystem_for_url
from ome_zarr_converters_tools.pipelines._to_zarr import write_to_zarr

logger = getLogger(__name__)


def _compute_chunk_size(
    tiled_image: TiledImage, ome_zarr_options: OmeZarrOptions
) -> tuple[int, ...]:
    """Compute the chunk size for the tiled image."""
    axes = tiled_image.axes
    chunks_strategy = ome_zarr_options.chunks
    fov_regions = tiled_image.group_by_fov()[0]
    fov_shape = fov_regions.shape()
    chunks = []
    for ax, fov_sh in zip(axes, fov_shape, strict=True):
        if ax == "x" or ax == "y":
            chunks.append(chunks_strategy.get_xy_chunk(fov_sh))
        elif ax == "z":
            chunks.append(chunks_strategy.z_chunk)
        elif ax == "c":
            chunks.append(chunks_strategy.c_chunk)
        elif ax == "t":
            chunks.append(chunks_strategy.t_chunk)
        else:
            raise ConvertersToolsInternalError(
                f"Axis '{ax}' of image '{tiled_image.path}' has no chunking rule "
                f"({axes=}); axes must be among t, c, z, y, x."
            )
    return tuple(chunks)


def _compute_shard_size(
    chunks: tuple[int, ...], axes: Sequence[str], ome_zarr_options: OmeZarrOptions
) -> tuple[int, ...] | None:
    """Shard shape as a per-axis multiple of `chunks`, or `None` when unsharded.

    A whole multiple by construction, which is what zarr requires; ngio clips
    shards that exceed the array shape.
    """
    strategy = ome_zarr_options.shards
    if isinstance(strategy, NoSharding):
        return None
    return tuple(
        chunk * strategy.get_factor(ax) for ax, chunk in zip(axes, chunks, strict=True)
    )


def _region_to_pixel_coordinates(
    regions: list[TileSlice],
    pixel_size: PixelSize,
) -> list[TileSlice]:
    """Convert TileSlice ROIs to rounded (integer) pixel coordinates.

    This function modifies the TileSlices in place.

    Args:
        regions: List of TileSlice models to convert.
        pixel_size: PixelSize model to use for conversion.
    """
    for region in regions:
        roi = region.roi.to_pixel(pixel_size=pixel_size)
        rounded_slices = []
        for ax_slice in roi.slices:
            start = ax_slice.start
            length = ax_slice.length
            assert start is not None and length is not None
            # Round the interval endpoints, not start and length independently:
            # round(start) + round(length) can differ from round(start + length)
            # by one pixel, creating gaps/overlaps at tile boundaries.
            rounded_start = round(start)
            rounded_slice = RoiSlice(
                axis_name=ax_slice.axis_name,
                start=rounded_start,
                length=round(start + length) - rounded_start,
            )
            rounded_slices.append(rounded_slice)
        region.roi = roi.model_copy(update={"slices": rounded_slices})
    return regions


def _attribute_to_condition_table(
    attributes: dict[str, AttributeType],
    image_path: str | None = None,
) -> ConditionTable | None:
    """Convert attributes to a condition table.

    Args:
        attributes: Dictionary of attribute names to lists of attribute values.
        image_path: Path of the image the attributes belong to, named in errors.

    Returns:
        A `ConditionTable`, or `None` if no attributes are provided.
    """
    condition_table = {}
    num_rows_dict = {}
    for attr_name, attr_values in attributes.items():
        condition_table[attr_name] = attr_values
        num_rows_dict[attr_name] = len(attr_values)

    if len(set(num_rows_dict.values())) > 1:
        of_image = f" of image '{image_path}'" if image_path is not None else ""
        raise ConvertersToolsUserError(
            f"All attributes{of_image} must have the same number of values; got "
            f"{num_rows_dict}. Pad the shorter attributes with None values."
        )
    if len(num_rows_dict) == 0:
        # No attributes, no need to create a condition table
        return None
    return ConditionTable(table_data=pl.DataFrame(condition_table))


def build_channels_meta(tiled_image: TiledImage) -> list[Channel] | None:
    """Build channel metadata from a TiledImage.

    Args:
        tiled_image: TiledImage model to extract channel metadata from.

    Returns:
        List of Channel metadata or None if no channel names are provided.
    """
    if tiled_image.channels is None:
        return None
    channels = []
    for channel in tiled_image.channels:
        channel = Channel(
            label=channel.channel_label,
            wavelength_id=channel.wavelength_id,
            channel_visualisation=ChannelVisualisation(color=channel.color),
        )
        channels.append(channel)
    return channels


def _validate_channels_axis(
    tiled_image: TiledImage, channels_meta: list[Channel] | None
) -> None:
    """Raise if channel metadata is declared for an image that has no `c` axis.

    The `c` length itself needs no check here: `TiledImage.output_shape` derives
    it from `channels`, so the array and the metadata cannot disagree.
    """
    if channels_meta is None or "c" in tiled_image.axes:
        return
    labels = ", ".join(channel.label for channel in channels_meta)
    raise ConvertersToolsUserError(
        f"TiledImage '{tiled_image.path}' declares {len(channels_meta)} "
        f"channel(s) ({labels}) but its axes {tiled_image.axes} have no 'c' "
        "axis, so the image has nowhere to put them. Either add 'c' to "
        "AcquisitionDetails.axes, or set channels=None."
    )


def write_tiled_image_as_zarr(
    *,
    zarr_url: str,
    tiled_image: TiledImage,
    converter_options: ConverterOptions,
    writer_mode: WriterMode,
    overwrite_mode: OverwriteMode,
    resource: Any | None = None,
) -> OmeZarrContainer:
    """Write a TiledImage as a Zarr file.

    Args:
        zarr_url: URL to write the Zarr file to.
        tiled_image: TiledImage model to write.
        converter_options: Options for the OME-Zarr conversion.
        writer_mode: Mode for writing the data.
        overwrite_mode: Mode to handle existing data.
        resource: Optional resource to pass to the image loaders.

    Returns:
        The written OME-Zarr container.
    """
    if overwrite_mode == OverwriteMode.NO_OVERWRITE:
        mode = "w-"
    elif overwrite_mode == OverwriteMode.OVERWRITE:
        mode = "w"
    else:  # extend
        mode = "a"
    zarr_format = 2 if converter_options.omezarr_options.ngff_version == "0.4" else 3
    # Whether a store already exists must be checked before `zarr.open_group`,
    # since mode "w" would truncate it and mode "a" would create it.
    store_pre_exists = filesystem_for_url(zarr_url).exists(zarr_url)
    base_group = zarr.open_group(store=zarr_url, mode=mode, zarr_format=zarr_format)
    omezarr_options = converter_options.omezarr_options
    if overwrite_mode == OverwriteMode.EXTEND and store_pre_exists:
        # Reuse the existing container. A failure to open here (corrupt/partial
        # store, version mismatch) is a real error that must surface, never be
        # silently overwritten with a fresh, empty container.
        return open_ome_zarr_container(base_group, cache=True)

    tiled_image.regions = _region_to_pixel_coordinates(
        tiled_image.regions,
        tiled_image.pixel_size,
    )
    channels_meta = build_channels_meta(tiled_image)
    _validate_channels_axis(tiled_image, channels_meta)
    chunks = _compute_chunk_size(tiled_image, omezarr_options)
    ome_zarr = create_empty_ome_zarr(
        store=base_group,
        axes_names=tiled_image.axes,
        shape=tiled_image.output_shape(),
        chunks=chunks,
        shards=_compute_shard_size(chunks, tiled_image.axes, omezarr_options),
        pixelsize=tiled_image.xy_pixel_size,
        z_spacing=tiled_image.z_spacing,
        time_spacing=tiled_image.t_spacing,
        levels=omezarr_options.levels.to_ngio_levels(),
        channels_meta=channels_meta,
        translation=tiled_image.translation,
        overwrite=True,
        ngff_version=omezarr_options.ngff_version,
        dtype=tiled_image.data_type,
    )
    image = ome_zarr.get_image()
    write_to_zarr(
        image=image,
        tiled_image=tiled_image,
        resource=resource,
        writer_mode=writer_mode,
    )
    # `mode="auto"` opts in to ngio 1.2's default now: it takes the faster
    # in-memory pyramid path only where it is provably identical to the dask one.
    image.consolidate(mode="auto")
    ome_zarr.set_channel_windows_with_percentiles()
    logger.info("OME-Zarr image creation and data writing complete.")

    fov_tiles = tiled_image.group_by_fov()
    if len(fov_tiles) > 1:
        rois = []
        for fov_tile in fov_tiles:
            roi_union = fov_tile.roi().to_world(pixel_size=tiled_image.pixel_size)
            rois.append(roi_union)

        roi_table = RoiTable(rois=rois)
        ome_zarr.add_table(
            "FOV_ROI_table", roi_table, backend=omezarr_options.table_backend
        )

    well_roi = ome_zarr.build_image_roi_table()
    ome_zarr.add_table(
        "well_ROI_table", well_roi, backend=omezarr_options.table_backend
    )
    condition_table = _attribute_to_condition_table(
        tiled_image.attributes, image_path=tiled_image.path
    )
    if condition_table is not None:
        ome_zarr.add_table("condition_table", condition_table, backend="csv")
    logger.info("Finished writing OME-Zarr Tables and metadata.")
    return ome_zarr
