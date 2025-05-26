"""Tools to initialize a conversion tasks."""

from pathlib import Path

from fractal_converters_tools._pkl_utils import create_pkl, remove_pkl_dir
from fractal_converters_tools._task_common_models import (
    AdvancedComputeOptions,
    ConvertParallelInitArgs,
)
from fractal_converters_tools._tiled_image import TiledImage


def build_parallelization_list(
    zarr_dir: str | Path,
    tiled_images: list[TiledImage],
    overwrite: bool,
    advanced_compute_options: AdvancedComputeOptions,
    tmp_dir_name: str = "_tmp_converter_dir",
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        zarr_dir (str): The path to the zarr directory.
        tiled_images (list[TiledImage]): A list of tiled images objects to convert.
        overwrite (bool): Overwrite the existing zarr directory.
        advanced_compute_options (AdvancedComputeOptions): The advanced compute options.
        tmp_dir_name (str): The name of the temporary directory to store the
            pickled tiled images.
    """
    parallelization_list = []
    if isinstance(zarr_dir, str):
        zarr_dir = Path(zarr_dir)

    pickle_dir = zarr_dir / tmp_dir_name

    if pickle_dir.exists():
        # Reinitialize the directory
        remove_pkl_dir(pickle_dir)

    for tile in tiled_images:
        tile_pickle_path = create_pkl(pickle_dir=pickle_dir, tiled_image=tile)
        zarr_url = str(zarr_dir / tile.path)
        parallelization_list.append(
            {
                "zarr_url": zarr_url,
                "init_args": ConvertParallelInitArgs(
                    tiled_image_pickled_path=str(tile_pickle_path),
                    overwrite=overwrite,
                    advanced_compute_options=advanced_compute_options,
                ).model_dump(),
            }
        )
    return parallelization_list
