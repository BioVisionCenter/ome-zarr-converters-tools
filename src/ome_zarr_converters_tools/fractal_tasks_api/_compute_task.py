"""Functions to write TiledImage models from Tile models."""

import logging
import time
from typing import Any, TypedDict

from ngio import OmeZarrContainer

from ome_zarr_converters_tools.models import (
    CollectionInterface,
    CollectionInterfaceType,
    ConvertParallelInitArgs,
    ImageInPlate,
    ImageLoaderInterfaceType,
)
from ome_zarr_converters_tools.registration import (
    build_default_registration_pipeline,
)
from ome_zarr_converters_tools.utils import (
    remove_json,
    tiled_image_creation_pipeline,
    tiled_image_from_json,
)

logger = logging.getLogger(__name__)


class UpdateDict(TypedDict):
    zarr_url: str
    types: dict[str, Any]
    attributes: dict[str, Any]


class ImageListUpdateDict(TypedDict):
    image_list_updates: list[UpdateDict]


def _build_image_list_update(
    zarr_url: str,
    ome_zarr: OmeZarrContainer,
    collection: CollectionInterface,
    attributes: dict[str, Any],
) -> ImageListUpdateDict:
    _types = {"is_3D": ome_zarr.is_3d}
    if ome_zarr.is_time_series:
        _types["is_time_series"] = True

    if isinstance(collection, ImageInPlate):
        attributes["plate"] = collection.plate_name
        attributes["well"] = collection.well
        attributes["acquisition"] = collection.acquisition

    _update_dict = UpdateDict(
        zarr_url=zarr_url,
        types=_types,
        attributes=attributes,
    )
    return ImageListUpdateDict(image_list_updates=[_update_dict])


def generic_compute_task(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: ConvertParallelInitArgs,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
    resource: Any = None,
) -> ImageListUpdateDict:
    """Initialize the task to convert a LIF plate to OME-Zarr.

    Args:
        zarr_url (str): URL to the OME-Zarr file.
        init_args (ConvertParallelInitArgs): Arguments from the initialization task.
        collection_type (type[CollectionInterfaceType]): The collection type to use
            when loading the TiledImage.
        image_loader_type (type[ImageLoaderInterfaceType]): The image loader type to
            use when loading the TiledImage.
        resource (Any): The resource to associate with the context model.
    """
    for t in range(3):  # Retry up to 3 times
        try:
            tiled_image_loaded = tiled_image_from_json(
                tiled_image_json_dump_url=init_args.tiled_image_json_dump_url,
                collection_type=collection_type,
                image_loader_type=image_loader_type,
            )
            break  # Exit loop if successful
        except FileNotFoundError:
            logger.error(
                f"JSON file does not exist: "
                f"{init_args.tiled_image_json_dump_url}, retrying..."
            )
            sleep_time = 2 ** (t + 1)
            time.sleep(sleep_time)
    else:
        raise FileNotFoundError(
            f"JSON file does not exist after 3 retries: "
            f"{init_args.tiled_image_json_dump_url}"
        )
    registration_pipeline = build_default_registration_pipeline(
        alignment_corrections=init_args.converter_options.alignment_correction,
        tiling_mode=init_args.converter_options.tiling_mode,
    )

    ome_zarr = tiled_image_creation_pipeline(
        zarr_url=zarr_url,
        tiled_image=tiled_image_loaded,
        registration_pipeline=registration_pipeline,
        converter_options=init_args.converter_options,
        overwrite_mode=init_args.overwrite_mode,
        resource=resource,
    )
    remove_json(init_args.tiled_image_json_dump_url)
    return _build_image_list_update(
        zarr_url=zarr_url,
        ome_zarr=ome_zarr,
        collection=tiled_image_loaded.collection,
        attributes=tiled_image_loaded.attributes,
    )
