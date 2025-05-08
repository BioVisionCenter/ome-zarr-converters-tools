"""Tools to initialize a conversion tasks."""

import pickle
from pathlib import Path
from uuid import uuid4

from fractal_converters_tools.task_common_models import (
    AdvancedComputeOptions,
    ConvertParallelInitArgs,
)
from fractal_converters_tools.tiled_image import PlatePathBuilder, TiledImage


def _add_acquisition_attributes(tiled_images: list[TiledImage]) -> list[TiledImage]:
    """Add acquisition attributes to the tiled images.

    Args:
        tiled_images (list[TiledImage]): A list of tiled images objects to convert.
    """
    unique_acquisition_ids = set()
    for tile in tiled_images:
        path_builder = tile.path_builder
        if isinstance(path_builder, PlatePathBuilder):
            unique_acquisition_ids.add(path_builder.acquisition_id)

    if len(unique_acquisition_ids) <= 1:
        return tiled_images

    new_tiled_images = []
    for tile in tiled_images:
        path_builder = tile.path_builder
        if isinstance(path_builder, PlatePathBuilder):
            if tile._attributes is None:
                tile._attributes = {}
            tile._attributes["acquisition"] = str(path_builder.acquisition_id)
        new_tiled_images.append(tile)
    return new_tiled_images


def build_parallelization_list(
    zarr_dir: str | Path,
    tiled_images: list[TiledImage],
    overwrite: bool,
    advanced_compute_options: AdvancedComputeOptions,
    tmp_dir_name: str | None = None,
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        zarr_dir (str): The path to the zarr directory.
        tiled_images (list[TiledImage]): A list of tiled images objects to convert.
        overwrite (bool): Overwrite the existing zarr directory.
        advanced_compute_options (AdvancedComputeOptions): The advanced compute options.
        tmp_dir_name (str, optional): The name of the temporary directory to store the
            pickled tiled images.
    """
    parallelization_list = []

    tmp_dir_name = tmp_dir_name if tmp_dir_name else "_tmp_coverter_dir"
    pickle_dir = Path(zarr_dir) / tmp_dir_name
    pickle_dir.mkdir(parents=True, exist_ok=True)

    tiled_images = _add_acquisition_attributes(tiled_images)

    for tile in tiled_images:
        tile_pickle_path = pickle_dir / f"{uuid4()}.pkl"
        with open(tile_pickle_path, "wb") as f:
            pickle.dump(tile, f)
        parallelization_list.append(
            {
                "zarr_url": str(zarr_dir),
                "init_args": ConvertParallelInitArgs(
                    tiled_image_pickled_path=str(tile_pickle_path),
                    overwrite=overwrite,
                    advanced_compute_options=advanced_compute_options,
                ).model_dump(),
            }
        )
    return parallelization_list
