"""Functions to write TiledImage models from Tile models."""

import logging
from typing import Any

from ngio import OmeZarrContainer

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import (
    ConverterOptions,
    OverwriteMode,
    WriterMode,
)
from ome_zarr_converters_tools.pipelines._registration_pipeline import (
    RegistrationStep,
    apply_registration_pipeline,
)
from ome_zarr_converters_tools.pipelines._write_ome_zarr import (
    write_tiled_image_as_zarr,
)

logger = logging.getLogger(__name__)


def tiled_image_creation_pipeline(
    *,
    zarr_url: str,
    tiled_image: TiledImage,
    registration_pipeline: list[RegistrationStep],
    converter_options: ConverterOptions,
    writer_mode: WriterMode,
    overwrite_mode: OverwriteMode,
    resource: Any | None = None,
) -> OmeZarrContainer:
    """Register and write one `TiledImage` as an OME-Zarr image.

    Applies `registration_pipeline` (see `build_default_registration_pipeline`)
    and then `write_tiled_image_as_zarr`, inside
    `converter_options.runtime_settings.apply()`.

    Args:
        zarr_url: Full URL of the image to write, i.e. the output directory
            joined with `tiled_image.path`.
        tiled_image: The image to register and write.
        registration_pipeline: Registration steps applied in order before
            writing.
        converter_options: Supplies `omezarr_options` and `runtime_settings`.
            Its `writer_mode` is not read here; pass `writer_mode` explicitly.
        writer_mode: How the data is loaded and written.
        overwrite_mode: What to do when the output already exists.
        resource: Passed unchanged to every image-loader call.

    Returns:
        The written image.

    Raises:
        ConvertersToolsOutputExistsError: The output exists and `overwrite_mode`
            is `NO_OVERWRITE`.
        ConvertersToolsUserError: The tiles cannot be laid out or written, e.g.
            negative positions under `Keep`, unequal tile sizes under a snapping
            strategy, or channels declared without a `c` axis.
    """
    with converter_options.runtime_settings.apply():
        logger.info("Applying registration pipeline to TiledImage.")
        tiled_image = apply_registration_pipeline(tiled_image, registration_pipeline)
        logger.info("Starting to write TiledImage as OME-Zarr.")
        omezarr = write_tiled_image_as_zarr(
            zarr_url=zarr_url,
            tiled_image=tiled_image,
            converter_options=converter_options,
            writer_mode=writer_mode,
            overwrite_mode=overwrite_mode,
            resource=resource,
        )
        return omezarr
