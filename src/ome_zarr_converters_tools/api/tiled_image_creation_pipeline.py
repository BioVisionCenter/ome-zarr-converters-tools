"""Functions to write TiledImage models from Tile models."""

import logging
import time
from typing import Any

from ngio.utils._zarr_utils import NgioSupportedStore
from zarr.storage import LocalStore

from ome_zarr_converters_tools.models import (
    CollectionInterfaceType,
    ConverterOptions,
    ConvertParallelInitArgs,
    ImageLoaderInterfaceType,
    OverwriteMode,
    TiledImage,
)
from ome_zarr_converters_tools.registration import (
    RegistrationStep,
    apply_registration_pipeline,
)
from ome_zarr_converters_tools.registration._registration_pipeline import (
    build_default_registration_pipeline,
)
from ome_zarr_converters_tools.utils._json_utils import (
    remove_json,
    tiled_image_from_json,
)
from ome_zarr_converters_tools.utils._write_ome_zarr import write_tiled_image_as_zarr

logger = logging.getLogger(__name__)


def tiled_image_creation_pipeline(
    *,
    base_store: NgioSupportedStore,
    tiled_image: TiledImage,
    registration_pipeline: list[RegistrationStep],
    converter_options: ConverterOptions,
    overwrite_mode: OverwriteMode,
    resource: Any | None = None,
) -> dict[str, Any]:
    """Write a TiledImage from a dictionary."""
    tiled_image = apply_registration_pipeline(tiled_image, registration_pipeline)
    updates = write_tiled_image_as_zarr(
        base_store=base_store,
        tiled_image=tiled_image,
        converter_options=converter_options,
        overwrite_mode=overwrite_mode,
        resource=resource,
    )
    return updates


def generic_compute_task(
    *,
    # Fractal parameters
    init_args: ConvertParallelInitArgs,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
    resource: Any = None,
):
    """Initialize the task to convert a LIF plate to OME-Zarr.

    Args:
        init_args (ConvertParallelInitArgs): Arguments from the initialization task.
        collection_type (Any): The collection type to use when loading the TiledImage.
        image_loader_type (Any): The image loader type to use when loading the
            TiledImage.
        resource (Any): The resource to associate with the context model.
    """
    # Build List of TiledImage models
    if init_args.store_type == "local":
        store = LocalStore(init_args.store_url)
    else:
        raise NotImplementedError("Only local store is currently supported.")

    for t in range(3):  # Retry up to 3 times
        try:
            tiled_image_loaded = tiled_image_from_json(
                json_file_name=init_args.json_file_name,
                store=store,
                collection_type=collection_type,
                image_loader_type=image_loader_type,
            )
            break  # Exit loop if successful
        except FileNotFoundError:
            logger.error(
                f"JSON file does not exist: {init_args.json_file_name}, retrying..."
            )
            sleep_time = 2 ** (t + 1)
            time.sleep(sleep_time)
    else:
        raise FileNotFoundError(
            f"JSON file does not exist after 3 retries: {init_args.json_file_name}"
        )
    registration_pipeline = build_default_registration_pipeline(
        alignment_corrections=init_args.converter_options.alignment_correction,
        tiling_mode=init_args.converter_options.tiling_mode,
    )

    tiled_image_creation_pipeline(
        base_store=store,
        tiled_image=tiled_image_loaded,
        registration_pipeline=registration_pipeline,
        converter_options=init_args.converter_options,
        overwrite_mode=init_args.overwrite_mode,
        resource=resource,
    )
    remove_json(init_args.json_file_name, store)
    return {}
