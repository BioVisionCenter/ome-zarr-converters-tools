"""Utils for serializing and deserializing tiled images to/from pickle files."""

import logging
import os
import time
from uuid import uuid4

from zarr.storage import FsspecStore, LocalStore

from ome_zarr_converters_tools.collection_setup import (
    ConverterStorageType,
    concat_storage,
)
from ome_zarr_converters_tools.models import (
    CollectionInterfaceType,
    ImageLoaderInterfaceType,
    TiledImage,
)

logger = logging.getLogger(__name__)


def _create_tmp_json_store(
    store: ConverterStorageType, dir_path: str
) -> ConverterStorageType:
    """Create a directory in the given store."""
    json_store = concat_storage(store, dir_path)
    if isinstance(json_store, LocalStore):
        json_store.root.mkdir(parents=True, exist_ok=True)
        return json_store
    elif isinstance(json_store, FsspecStore):
        raise NotImplementedError(
            "Directory creation for FsspecStore is not implemented yet."
        )
    raise ValueError(f"Unsupported store type {type(store)}")


def _dump_json_to_store(json_store: ConverterStorageType, json_data: str) -> str:
    """Dump the tiled image as a JSON file in the given store."""
    unique_json_filename = f"{uuid4()}.json"
    if isinstance(json_store, LocalStore):
        json_path = json_store.root / unique_json_filename
        with open(json_path, "w") as f:
            f.write(json_data)
        logger.info(f"JSON file created: {json_path}")
        return unique_json_filename
    elif isinstance(json_store, FsspecStore):
        raise NotImplementedError(
            "JSON dumping for FsspecStore is not implemented yet."
        )
    raise ValueError(f"Unsupported store type {type(json_store)}")


def dump_to_json(
    store: ConverterStorageType, tiled_image: TiledImage, tmp_path: str = "_tmp_json"
) -> str:
    """Create a pickle file for the tiled image."""
    json_store = _create_tmp_json_store(store, tmp_path)
    data_json = tiled_image.model_dump_json()
    tile_json_name = _dump_json_to_store(json_store=json_store, json_data=data_json)
    logger.info(f"JSON file created: {tile_json_name}")
    return tile_json_name


def tiled_image_from_json(
    json_file_name: str,
    store: ConverterStorageType,
    collection_type: type[CollectionInterfaceType],
    image_loader_type: type[ImageLoaderInterfaceType],
    tmp_path: str = "_tmp_json",
) -> TiledImage:
    """Load the json TiledImage object.

    Since TiledImage is a generic model, we need to specify the concrete types
    when loading it from json otherwise pydantic cannot infer them.

    Args:
        json_file_name (str): Name of the json file.
        store (ConverterStorageType): The Zarr store where the json file is located.
        collection_type (type[CollectionInterfaceType]): The concrete collection type
            of the TiledImage.
        image_loader_type (type[ImageLoaderInterfaceType]): The concrete image loader
            type of the TiledImage.
        tmp_path (str): The temporary directory where the json file is stored.

    Returns:
        TiledImage: The loaded TiledImage object.
    """
    num_retries = int(os.getenv("CONVERTERS_TOOLS_NUM_RETRIES", 5))

    if num_retries < 1:
        raise ValueError("NUM_RETRIES must be greater than 0")

    json_store = concat_storage(store, tmp_path)
    if not isinstance(json_store, LocalStore):
        raise NotImplementedError(
            "JSON loading for non-LocalStore is not implemented yet."
        )

    json_path = json_store.root / json_file_name
    for t in range(num_retries):
        try:
            with open(json_path) as f:
                # Concretely specify the types to load the generic TiledImage
                tiled_image = TiledImage[
                    collection_type, image_loader_type
                ].model_validate_json(f.read())
                if not isinstance(tiled_image, TiledImage):
                    raise ValueError(
                        f"JSON object is not a TiledImage: {type(tiled_image)}"
                    )
            return tiled_image
        except FileNotFoundError:
            logger.error(f"JSON file does not exist: {json_path}")
            logger.info("Retrying to load the JSON file...")
            sleep_time = 2 ** (t + 1)
            time.sleep(sleep_time)

    raise FileNotFoundError(
        f"JSON file does not exist after {num_retries} retries: {json_path}"
    )


def remove_json(
    json_file_name: str, store: ConverterStorageType, tmp_path: str = "_tmp_json"
):
    """Clean up the JSON file and the directory if it is empty.

    Args:
        json_file_name (str): Name of the json file.
        store (ConverterStorageType): The Zarr store where the json file is located.
        tmp_path (str): The temporary directory where the json file is stored.
    """
    json_store = concat_storage(store, tmp_path)
    try:
        if not isinstance(json_store, LocalStore):
            raise NotImplementedError(
                "JSON removal for non-LocalStore is not implemented yet."
            )
        json_path = json_store.root / json_file_name
        json_path.unlink()
        if not list(json_path.parent.iterdir()):
            # Remove the parent directory if it is empty
            json_path.parent.rmdir()
    except Exception as e:
        # This path is not tested
        # But if multiple processes are trying to clean up the same file
        # it might raise an exception.
        logger.error(
            f"An error occurred while cleaning up the JSON file: {e}. "
            f"You can safely remove the store: {json_store}"
        )


def cleanup_if_exists(store: ConverterStorageType, tmp_path: str = "_tmp_json"):
    """Clean up the temporary JSON directory if it exists.

    If cleaning up is not possible, log an error message, but do not raise.

    Args:
        store (ConverterStorageType): The Zarr store where the json files are located.
        tmp_path (str): The temporary directory where the json files are stored.
    """
    json_store = concat_storage(store, tmp_path)
    try:
        if not isinstance(json_store, LocalStore):
            raise NotImplementedError(
                "JSON removal for non-LocalStore is not implemented yet."
            )
        if json_store.root.exists():
            for json_file in json_store.root.iterdir():
                json_file.unlink()
            json_store.root.rmdir()
    except Exception as e:
        logger.error(
            f"An error occurred while cleaning up the JSON store: {e}. "
            f"You can safely remove the store: {json_store}"
        )
