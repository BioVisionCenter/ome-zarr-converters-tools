"""Functions to write TiledImage models from Tile models."""

import logging
import time
from typing import Any, TypedDict

from ngio import OmeZarrContainer

from ome_zarr_converters_tools.core import AttributeType
from ome_zarr_converters_tools.fractal._json_utils import (
    remove_json,
    tiled_image_from_json,
    tiled_image_from_json_str,
)
from ome_zarr_converters_tools.fractal._models import (
    ConvertParallelInitArgs,
)
from ome_zarr_converters_tools.models import (
    CollectionInterface,
    CollectionInterfaceType,
    ImageInPlate,
    ImageLoaderInterfaceType,
)
from ome_zarr_converters_tools.pipelines import (
    build_default_registration_pipeline,
    tiled_image_creation_pipeline,
)

logger = logging.getLogger(__name__)


class UpdateDict(TypedDict):
    zarr_url: str
    types: dict[str, Any]
    attributes: dict[str, str | int | float | bool | None]


class ImageListUpdateDict(TypedDict):
    image_list_updates: list[UpdateDict]


def _format_attribute_value(
    value: AttributeType,
) -> str | int | float | bool | None:
    """Format an attribute value for inclusion in the update dictionary."""
    if len(value) == 1:
        return value[0]
    return " & ".join(str(v) for v in value)


def _build_image_list_update(
    zarr_url: str,
    ome_zarr: OmeZarrContainer,
    collection: CollectionInterface,
    attributes: dict[str, AttributeType],
) -> ImageListUpdateDict:
    _types = {"is_3D": ome_zarr.is_3d}
    if ome_zarr.is_time_series:
        _types["is_time_series"] = True

    _attributes = {k: _format_attribute_value(v) for k, v in attributes.items()}
    if isinstance(collection, ImageInPlate):
        _attributes["plate"] = collection.plate_path()
        _attributes["well"] = collection.well
        _attributes["acquisition"] = collection.acquisition

    _update_dict = UpdateDict(
        zarr_url=zarr_url,
        types=_types,
        attributes=_attributes,
    )
    return ImageListUpdateDict(image_list_updates=[_update_dict])


def generic_compute_task(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: ConvertParallelInitArgs | dict,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
    resource: Any = None,
) -> ImageListUpdateDict:
    """Initialize the task to convert a LIF plate to OME-Zarr.

    Args:
        zarr_url (str): URL to the OME-Zarr file.
        init_args (ConvertParallelInitArgs | dict): Arguments from the initialization
            task. Accepts either a ``ConvertParallelInitArgs`` instance or a plain dict
            (as produced by Fractal's orchestration layer).
        collection_type (type[CollectionInterfaceType]): The collection type to use
            when loading the TiledImage.
        image_loader_type (type[ImageLoaderInterfaceType]): The image loader type to
            use when loading the TiledImage.
        resource (Any): The resource to associate with the context model.
    """
    logger.info(f"Starting conversion for Zarr URL: {zarr_url}")
    parsed_args = ConvertParallelInitArgs.model_validate(init_args)

    if parsed_args.tiled_image_json_str is not None:
        tiled_image_loaded = tiled_image_from_json_str(
            json_str=parsed_args.tiled_image_json_str,
            collection_type=collection_type,
            image_loader_type=image_loader_type,
        )
    else:
        for t in range(3):  # Retry up to 3 times
            try:
                tiled_image_loaded = tiled_image_from_json(
                    tiled_image_json_dump_url=parsed_args.tiled_image_json_dump_url,  # ty:ignore
                    collection_type=collection_type,
                    image_loader_type=image_loader_type,
                )
                logger.info(
                    "Successfully loaded JSON file: "
                    f"{parsed_args.tiled_image_json_dump_url}"
                )
                break  # Exit loop if successful
            except FileNotFoundError:
                logger.error(
                    f"JSON file does not exist: "
                    f"{parsed_args.tiled_image_json_dump_url}, retrying..."
                )
                sleep_time = 2 ** (t + 1)
                time.sleep(sleep_time)
        else:
            raise FileNotFoundError(
                f"JSON file does not exist after 3 retries: "
                f"{parsed_args.tiled_image_json_dump_url}"
            )

    registration_pipeline = build_default_registration_pipeline(
        alignment_corrections=parsed_args.converter_options.stage_position_corrections,
        tiling_strategy=parsed_args.converter_options.tiling_strategy,
    )
    ome_zarr = tiled_image_creation_pipeline(
        zarr_url=zarr_url,
        tiled_image=tiled_image_loaded,
        registration_pipeline=registration_pipeline,
        converter_options=parsed_args.converter_options,
        writer_mode=parsed_args.converter_options.writer_mode,
        overwrite_mode=parsed_args.overwrite_mode,
        resource=resource,
    )
    if parsed_args.tiled_image_json_dump_url is not None:
        remove_json(parsed_args.tiled_image_json_dump_url)
    logger.info("Conversion complete")
    return _build_image_list_update(
        zarr_url=zarr_url,
        ome_zarr=ome_zarr,
        collection=tiled_image_loaded.collection,
        attributes=tiled_image_loaded.attributes,
    )
