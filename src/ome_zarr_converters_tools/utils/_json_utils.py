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
    UrlType,
    find_url_type,
    local_url_to_path,
)

logger = logging.getLogger(__name__)


def _dump_to_json_local_fs(temp_json_url: str, json_data: str) -> str:
    """Create a pickle file for the tiled image."""
    json_store = local_url_to_path(temp_json_url)
    json_store.mkdir(parents=True, exist_ok=True)
    unique_json_filename = f"{uuid4()}.json"
    json_path = json_store / unique_json_filename
    with open(json_path, "w") as f:
        f.write(json_data)
    tile_json_name = str(json_path)
    logger.debug(f"JSON file created: {tile_json_name}")
    return tile_json_name


def dump_to_json(temp_json_url: str, tiled_image: TiledImage) -> str:
    """Create a pickle file for the tiled image."""
    json_data = tiled_image.model_dump_json()
    url_type = find_url_type(temp_json_url)
    if url_type == UrlType.LOCAL:
        tile_json_name = _dump_to_json_local_fs(
            temp_json_url=temp_json_url, json_data=json_data
        )
        return tile_json_name
    elif url_type == UrlType.S3:
        raise NotImplementedError("Dumping JSON to S3 is not implemented yet.")
    raise NotImplementedError(
        f"Dumping JSON to URL type {url_type} is not implemented yet."
    )


def _tiled_image_from_json_local_fs(
    tiled_image_json_dump_url: str,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
) -> TiledImage[CollectionInterfaceType, ImageLoaderInterfaceType]:
    """Load the JSON file from the local filesystem."""
    json_path = local_url_to_path(tiled_image_json_dump_url)
    with open(json_path) as f:
        # Concretely specify the types to load the generic TiledImage
        tiled_image = TiledImage[
            collection_type, image_loader_type
        ].model_validate_json(f.read())
    return tiled_image


def tiled_image_from_json(
    tiled_image_json_dump_url: str,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
) -> TiledImage:
    """Load the json TiledImage object.

    Since TiledImage is a generic model, we need to specify the concrete types
    when loading it from json otherwise pydantic cannot infer them.

    Args:
        tiled_image_json_dump_url (str): The URL to the json file.
        collection_type (type[CollectionInterfaceType]): The concrete collection type
            of the TiledImage.
        image_loader_type (type[ImageLoaderInterfaceType]): The concrete image loader
            type of the TiledImage.

    Returns:
        TiledImage: The loaded TiledImage object.
    """
    num_retries = int(os.getenv("CONVERTERS_TOOLS_NUM_RETRIES", 5))

    if num_retries < 1:
        raise ValueError("NUM_RETRIES must be greater than 0")

    for t in range(num_retries):
        try:
            url_type = find_url_type(tiled_image_json_dump_url)
            if url_type == UrlType.LOCAL:
                tiled_image = _tiled_image_from_json_local_fs(
                    tiled_image_json_dump_url,
                    collection_type,
                    image_loader_type,
                )
                return tiled_image
            elif url_type == UrlType.S3:
                raise NotImplementedError(
                    "Loading JSON from S3 is not implemented yet."
                )
            raise NotImplementedError(
                f"Loading JSON from URL type {url_type} is not implemented yet."
            )
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


def _remove_json_local_fs(
    tiled_image_json_dump_url: str,
):
    """Clean up the JSON file and the directory if it is empty.

    Args:
        tiled_image_json_dump_url (str): The URL to the json file.
    """
    try:
        json_path = local_url_to_path(tiled_image_json_dump_url)
        json_path.unlink()
        if not list(json_path.parent.iterdir()):
            # Remove the parent directory if it is empty
            json_path.parent.rmdir()
    except Exception as e:
        logger.error(
            f"An error occurred while cleaning up the JSON file: {e}. "
            f"You can safely remove the store: {tiled_image_json_dump_url}"
        )


def remove_json(
    tiled_image_json_dump_url: str,
):
    """Clean up the JSON file and the directory if it is empty.

    Args:
        tiled_image_json_dump_url (str): The URL to the json file.
    """
    url_type = find_url_type(tiled_image_json_dump_url)
    if url_type == UrlType.LOCAL:
        _remove_json_local_fs(tiled_image_json_dump_url)
        return
    elif url_type == UrlType.S3:
        logger.error("Removing JSON from S3 is not implemented yet.")
        return
    logger.error(f"Cleanup for URL type {url_type} is not implemented yet.")


def _cleanup_if_exists_local_fs(temp_json_url: str):
    """Clean up the temporary JSON directory if it exists.

    If cleaning up is not possible, log an error message, but do not raise.

    Args:
        temp_json_url (str): The URL to the temporary JSON directory.
    """
    json_path = local_url_to_path(temp_json_url)
    try:
        if json_path.exists():
            for json_file in json_path.iterdir():
                json_file.unlink()
            json_path.rmdir()
    except Exception as e:
        logger.error(
            f"An error occurred while cleaning up the JSON store: {e}. "
            f"You can safely remove the store: {json_path}"
        )


def cleanup_if_exists(temp_json_url: str):
    """Clean up the temporary JSON directory if it exists.

    If cleaning up is not possible, log an error message, but do not raise.

    Args:
        temp_json_url (str): The URL to the temporary JSON directory.
    """
    url_type = find_url_type(temp_json_url)
    if url_type == UrlType.LOCAL:
        _cleanup_if_exists_local_fs(temp_json_url)
        return
    elif url_type == UrlType.S3:
        logger.error("Cleaning up JSON from S3 is not implemented yet.")
        return
    logger.error(f"Cleanup for URL type {url_type} is not implemented yet.")
