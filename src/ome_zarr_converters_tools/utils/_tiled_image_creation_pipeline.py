"""Functions to write TiledImage models from Tile models."""

from typing import Any

from ngio import OmeZarrContainer

from ome_zarr_converters_tools.models import (
    ConverterOptions,
    OverwriteMode,
    TiledImage,
)
from ome_zarr_converters_tools.registration import (
    RegistrationStep,
    apply_registration_pipeline,
)
from ome_zarr_converters_tools.utils._write_ome_zarr import write_tiled_image_as_zarr


def tiled_image_creation_pipeline(
    *,
    zarr_url: str,
    tiled_image: TiledImage,
    registration_pipeline: list[RegistrationStep],
    converter_options: ConverterOptions,
    overwrite_mode: OverwriteMode,
    resource: Any | None = None,
) -> OmeZarrContainer:
    """Write a TiledImage from a dictionary."""
    tiled_image = apply_registration_pipeline(tiled_image, registration_pipeline)
    omezarr = write_tiled_image_as_zarr(
        zarr_url=zarr_url,
        tiled_image=tiled_image,
        converter_options=converter_options,
        overwrite_mode=overwrite_mode,
        resource=resource,
    )
    return omezarr
