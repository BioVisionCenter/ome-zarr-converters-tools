"""A generic task to convert a LIF plate to OME-Zarr."""

import logging
import os
import pickle
import time
from functools import partial
from pathlib import Path

from fractal_converters_tools.omezarr_image_writers import write_tiled_image
from fractal_converters_tools.stitching import standard_stitching_pipe
from fractal_converters_tools.task_common_models import ConvertParallelInitArgs
from fractal_converters_tools.tiled_image import PlatePathBuilder, TiledImage

logger = logging.getLogger(__name__)

NUM_RETRIES = int(os.getenv("COVERTERS_TOOLS_NUM_RETRIES", 5))


def _check_if_pickled_file_exists(pickle_path: Path) -> None:
    """Check if the pickled file exists.

    Args:
        pickle_path (Path): Path to the pickled file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    for t in range(NUM_RETRIES):
        try:
            if pickle_path.exists():
                return
            else:
                raise FileNotFoundError(f"Pickled file {pickle_path} does not exist.")
        except FileNotFoundError as e:
            logger.error(f"An error occurred while loading the pickled file: {e}")
            logger.info("Retrying to load the pickled file...")
            sleep_time = 2 ** (t + 1)  # Exponential backoff
            time.sleep(sleep_time)


def _load_tiled_image(pickle_path: Path) -> TiledImage:
    """Load the pickled TiledImage object.

    Args:
        pickle_path (Path): Path to the pickled file.

    Returns:
        TiledImage: The loaded TiledImage object.
    """
    _check_if_pickled_file_exists(pickle_path)
    with open(pickle_path, "rb") as f:
        tiled_image = pickle.load(f)
        if not isinstance(tiled_image, TiledImage):
            raise ValueError(f"Pickled object is not a TiledImage: {type(tiled_image)}")
    return tiled_image


def _clean_up_pickled_file(pickle_path: Path):
    """Clean up the pickled file and the directory if it is empty.

    Args:
        pickle_path (Path): Path to the pickled file.
    """
    _check_if_pickled_file_exists(pickle_path)
    try:
        pickle_path.unlink()
        if not list(pickle_path.parent.iterdir()):
            pickle_path.parent.rmdir()
    except Exception as e:
        # This path is not tested
        # But if multiple processes are trying to clean up the same file
        # it might raise an exception.
        logger.error(f"An error occurred while cleaning up the pickled file: {e}")


def generic_compute_task(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: ConvertParallelInitArgs,
):
    """Initialize the task to convert a LIF plate to OME-Zarr.

    Args:
        zarr_url (str): URL to the OME-Zarr file.
        init_args (ConvertScanrInitArgs): Arguments for the initialization task.
    """
    pickle_path = Path(init_args.tiled_image_pickled_path)
    tiled_image = _load_tiled_image(pickle_path)

    try:
        stitching_pipe = partial(
            standard_stitching_pipe,
            mode=init_args.advanced_compute_options.tiling_mode,
            swap_xy=init_args.advanced_compute_options.swap_xy,
            invert_x=init_args.advanced_compute_options.invert_x,
            invert_y=init_args.advanced_compute_options.invert_y,
        )

        new_zarr_url, is_3d, is_time_series = write_tiled_image(
            zarr_dir=zarr_url,
            tiled_image=tiled_image,
            stiching_pipe=stitching_pipe,
            num_levels=init_args.advanced_compute_options.num_levels,
            max_xy_chunk=init_args.advanced_compute_options.max_xy_chunk,
            z_chunk=init_args.advanced_compute_options.z_chunk,
            c_chunk=init_args.advanced_compute_options.c_chunk,
            t_chunk=init_args.advanced_compute_options.t_chunk,
            overwrite=init_args.overwrite,
        )
    except Exception as e:
        logger.error(f"An error occurred while processing {tiled_image}.")
        _clean_up_pickled_file(pickle_path)
        raise e

    p_types = {"is_3D": is_3d}

    if isinstance(tiled_image.path_builder, PlatePathBuilder):
        attributes = {
            "well": f"{tiled_image.path_builder.row}{tiled_image.path_builder.column}",
            "plate": tiled_image.path_builder.plate_path,
        }
    else:
        attributes = {}

    if tiled_image.attributes is not None:
        attributes = {
            **attributes,
            **tiled_image.attributes,
        }

    _clean_up_pickled_file(pickle_path)

    return {
        "image_list_updates": [
            {"zarr_url": new_zarr_url, "types": p_types, "attributes": attributes}
        ]
    }
