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
    """Write a TiledImage from a dictionary."""
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
