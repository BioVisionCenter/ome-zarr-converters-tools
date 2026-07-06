"""Utils for serializing and deserializing tiled images to/from pickle files."""

import logging
import os
import time
from uuid import uuid4

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import (
    CollectionInterfaceType,
    ImageLoaderInterfaceType,
)
from ome_zarr_converters_tools.models._url_utils import (
    filesystem_for_url,
    join_url_paths,
    parent_url,
)

logger = logging.getLogger(__name__)


def dump_json_str(temp_json_url: str, json_str: str) -> str:
    """Write a pre-serialized JSON string to a unique file in temp_json_url."""
    fs = filesystem_for_url(temp_json_url, error_msg_prefix="Dumping JSON")
    fs.makedirs(temp_json_url, exist_ok=True)
    unique_json_filename = f"{uuid4()}.json"
    tile_json_name = join_url_paths(temp_json_url, unique_json_filename)
    with fs.open(tile_json_name, "w") as f:
        f.write(json_str)
    logger.debug(f"JSON file created: {tile_json_name}")
    return tile_json_name


def dump_to_json(temp_json_url: str, tiled_image: TiledImage) -> str:
    """Create a JSON file for the tiled image."""
    return dump_json_str(
        temp_json_url=temp_json_url, json_str=tiled_image.model_dump_json()
    )


def tiled_image_from_json(
    tiled_image_json_dump_url: str,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
) -> TiledImage:
    """Load the json TiledImage object.

    Since TiledImage is a generic model, we need to specify the concrete types
    when loading it from json otherwise pydantic cannot infer them.

    Args:
        tiled_image_json_dump_url: The URL to the json file.
        collection_type: The concrete collection type of the `TiledImage`.
        image_loader_type: The concrete image loader type of the `TiledImage`.

    Returns:
        The loaded `TiledImage` object.
    """
    num_retries = int(os.getenv("CONVERTERS_TOOLS_NUM_RETRIES", 5))

    if num_retries < 1:
        raise ValueError("NUM_RETRIES must be greater than 0")

    for t in range(num_retries):
        try:
            fs = filesystem_for_url(
                tiled_image_json_dump_url, error_msg_prefix="Loading JSON"
            )
            with fs.open(tiled_image_json_dump_url, "r") as f:
                # Concretely specify the types to load the generic TiledImage
                tiled_image = TiledImage[
                    collection_type, image_loader_type  # ty:ignore[invalid-type-form]
                ].model_validate_json(f.read())

            return tiled_image

        except FileNotFoundError:
            logger.error(
                f"JSON file does not exist: {tiled_image_json_dump_url}, retrying..."
            )
            sleep_time = 2 ** (t + 1)
            time.sleep(sleep_time)

    raise FileNotFoundError(
        f"JSON file does not exist after {num_retries} "
        f"retries: {tiled_image_json_dump_url}"
    )


def tiled_image_from_json_str(
    json_str: str,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
) -> TiledImage:
    """Deserialize a TiledImage from a JSON string (no filesystem I/O).

    Args:
        json_str: The JSON string to deserialize.
        collection_type: The concrete collection type of the TiledImage.
        image_loader_type: The concrete image loader type of the TiledImage.

    Returns:
        The loaded `TiledImage` object.
    """
    return TiledImage[
        collection_type, image_loader_type  # ty:ignore[invalid-type-form]
    ].model_validate_json(json_str)


def remove_json(
    tiled_image_json_dump_url: str,
):
    """Clean up the JSON file and the directory if it is empty.

    Args:
        tiled_image_json_dump_url: The URL to the json file.
    """
    fs = filesystem_for_url(
        tiled_image_json_dump_url, error_msg_prefix="Cleaning up JSON"
    )

    try:
        fs.rm(tiled_image_json_dump_url)
        parent_dir = parent_url(tiled_image_json_dump_url)
        try:
            # no-op if non-empty; avoids a listdir on potentially large directories
            # for distributed filesystems like CephFS where listdir can be expensive
            # also avoids a race condition where another process has already removed
            # the file and directory
            fs.rmdir(parent_dir)
        except OSError:
            pass
    except Exception as e:
        logger.error(
            f"An error occurred while cleaning up the JSON file: {e}. "
            f"You can safely remove the store: {tiled_image_json_dump_url}"
        )


def cleanup_if_exists(temp_json_url: str):
    """Clean up the temporary JSON directory if it exists.

    If cleaning up is not possible, log an error message, but do not raise.

    Args:
        temp_json_url: The URL to the temporary JSON directory.
    """
    fs = filesystem_for_url(temp_json_url, error_msg_prefix="Cleanup")

    if not fs.exists(temp_json_url):
        return
    if not fs.isdir(temp_json_url):
        raise ValueError(
            f"Expected a directory for a cleanup, but got a file: {temp_json_url}"
        )
    try:
        # Limit to depth 1: temp JSON store is flat (files only, no subdirs)
        fs.rm(temp_json_url, recursive=True, maxdepth=1)
    except Exception as e:
        logger.error(
            f"An error occurred while cleaning up the temporary JSON directory: {e}. "
            f"You can safely remove the store: {temp_json_url}"
        )
