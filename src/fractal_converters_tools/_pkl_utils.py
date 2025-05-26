"""Utils for serializing and deserializing tiled images to/from pickle files."""

import logging
import os
import pickle
import time
from pathlib import Path
from uuid import uuid4

from fractal_converters_tools._tiled_image import TiledImage

logger = logging.getLogger(__name__)


def create_pkl(pickle_dir: Path, tiled_image: TiledImage) -> Path:
    """Create a pickle file for the tiled image."""
    pickle_dir.mkdir(parents=True, exist_ok=True)
    tile_pickle_path = pickle_dir / f"{uuid4()}.pkl"
    with open(tile_pickle_path, "wb") as f:
        pickle.dump(tiled_image, f)

    logger.info(f"Pickled file created: {tile_pickle_path}")
    return tile_pickle_path


def load_tiled_image(pickle_path: Path) -> TiledImage:
    """Load the pickled TiledImage object.

    Args:
        pickle_path (Path): Path to the pickled file.

    Returns:
        TiledImage: The loaded TiledImage object.
    """
    num_retries = int(os.getenv("CONVERTERS_TOOLS_NUM_RETRIES", 5))

    if num_retries < 1:
        raise ValueError("NUM_RETRIES must be greater than 0")

    for t in range(num_retries):
        try:
            with open(pickle_path, "rb") as f:
                tiled_image = pickle.load(f)
                if not isinstance(tiled_image, TiledImage):
                    raise ValueError(
                        f"Pickled object is not a TiledImage: {type(tiled_image)}"
                    )
            return tiled_image
        except FileNotFoundError:
            logger.error(f"Pickled file does not exist: {pickle_path}")
            logger.info("Retrying to load the pickled file...")
            sleep_time = 2 ** (t + 1)
            time.sleep(sleep_time)

    raise FileNotFoundError(
        f"Pickled file does not exist after {num_retries} retries: {pickle_path}"
    )


def remove_pkl(pickle_path: Path):
    """Clean up the pickled file and the directory if it is empty.

    Args:
        pickle_path (Path): Path to the pickled file.
    """
    try:
        pickle_path.unlink()
        if not list(pickle_path.parent.iterdir()):
            # Remove the parent directory if it is empty
            pickle_path.parent.rmdir()
    except Exception as e:
        # This path is not tested
        # But if multiple processes are trying to clean up the same file
        # it might raise an exception.
        logger.error(
            f"An error occurred while cleaning up the pickled file: {e}. "
            f"You can safely remove the directory: {pickle_path.parent}"
        )


def remove_pkl_dir(pickle_dir: Path):
    """Remove the directory containing the pickled files."""
    try:
        if pickle_dir.exists():
            for pkl_file in pickle_dir.iterdir():
                pkl_file.unlink()
            pickle_dir.rmdir()
    except Exception as e:
        logger.error(
            f"An error occurred while removing the pickled directory: {e} "
            f"You can safely remove the directory: {pickle_dir}"
        )
