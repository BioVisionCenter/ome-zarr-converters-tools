"""A generic task to convert a LIF plate to OME-Zarr."""

import logging
from functools import partial
from pathlib import Path

from fractal_converters_tools._omezarr_image_writers import write_tiled_image
from fractal_converters_tools._pkl_utils import load_tiled_image, remove_pkl
from fractal_converters_tools._stitching import standard_stitching_pipe
from fractal_converters_tools._task_common_models import ConvertParallelInitArgs
from fractal_converters_tools._tiled_image import PlatePathBuilder

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
    tiled_image = load_tiled_image(pickle_path)

    try:
        stitching_pipe = partial(
            standard_stitching_pipe,
            mode=init_args.advanced_compute_options.tiling_mode,
            swap_xy=init_args.advanced_compute_options.swap_xy,
            invert_x=init_args.advanced_compute_options.invert_x,
            invert_y=init_args.advanced_compute_options.invert_y,
        )

        im_list_types = write_tiled_image(
            zarr_url=zarr_url,
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
        remove_pkl(pickle_path)
        logger.error(f"An error occurred while processing {tiled_image}.")
        logger.exception(e)
        raise e

    if isinstance(tiled_image.path_builder, PlatePathBuilder):
        plate_attributes = {
            "well": f"{tiled_image.path_builder.row}{tiled_image.path_builder.column}",
            "plate": tiled_image.path_builder.plate_path,
            "acquisition": str(tiled_image.path_builder.acquisition_id),
        }
        tiled_image.update_attributes(plate_attributes)

    remove_pkl(pickle_path)

    return {
        "image_list_updates": [
            {
                "zarr_url": zarr_url,
                "types": im_list_types,
                "attributes": tiled_image.attributes,
            }
        ]
    }
