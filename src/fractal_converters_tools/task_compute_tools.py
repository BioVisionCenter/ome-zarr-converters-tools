"""A generic task to convert a LIF plate to OME-Zarr."""

import logging
import pickle
from functools import partial
from pathlib import Path

from fractal_converters_tools.omezarr_image_writers import write_tiled_image
from fractal_converters_tools.stitching import standard_stitching_pipe
from fractal_converters_tools.task_common_models import ConvertParallelInitArgs
from fractal_converters_tools.tiled_image import PlatePathBuilder

logger = logging.getLogger(__name__)


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
    if not pickle_path.exists():
        logger.error(f"Pickled file {pickle_path} does not exist.")
        raise FileNotFoundError(f"Pickled file {pickle_path} does not exist.")

    with open(pickle_path, "rb") as f:
        tiled_image = pickle.load(f)

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
        raise e

    p_types = {"is_3D": is_3d}

    if isinstance(tiled_image.path_builder, PlatePathBuilder):
        attributes = {
            "well": f"{tiled_image.path_builder.row}{tiled_image.path_builder.column}",
            "plate": tiled_image.path_builder.plate_path,
        }
    else:
        attributes = {}

    # Clean up the pickled file and the directory if it is empty

    try:
        Path(init_args.tiled_image_pickled_path).unlink()
        if not list(Path(init_args.tiled_image_pickled_path).parent.iterdir()):
            Path(init_args.tiled_image_pickled_path).parent.rmdir()
    except Exception as e:
        logger.error(f"An error occurred while cleaning up the pickled file: {e}")

    return {
        "image_list_updates": [
            {"zarr_url": new_zarr_url, "types": p_types, "attributes": attributes}
        ]
    }
