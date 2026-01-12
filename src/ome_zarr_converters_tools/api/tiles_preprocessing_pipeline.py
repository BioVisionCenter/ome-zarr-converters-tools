"""Functions to build TiledImage models from Tile models."""

from ome_zarr_converters_tools.collection_setup import (
    SetupCollectionStep,
    setup_collection,
)
from ome_zarr_converters_tools.filters import FilterStep, apply_filter_pipeline
from ome_zarr_converters_tools.models import (
    BaseTile,
    ContextModel,
    TiledImage,
)
from ome_zarr_converters_tools.utils import tiled_image_from_tiles
from ome_zarr_converters_tools.validators import ValidatorStep, apply_validator_pipeline


def tiles_preprocessing_pipeline(
    tiles: list[BaseTile],
    context: ContextModel,
    filters: list[FilterStep] | None = None,
    validators: list[ValidatorStep] | None = None,
    setup_collection_step: SetupCollectionStep | None = None,
) -> list[TiledImage]:
    """Process tiles through the preprocessing pipeline to create TiledImages.

    This function applies optional filters to the input tiles and then
    constructs TiledImage models from the processed tiles.

    Args:
        tiles: List of Tile models to process.
        context: Full context model for the conversion.
        filters: Optional list of filter steps to apply to the tiles.
        validators: Optional list of validator steps to apply to the tiles.
        setup_collection_step: Optional configuration for the collection setup step.

    Returns:
        A list of TiledImage models created from the processed tiles.
    """
    if filters is not None:
        tiles = apply_filter_pipeline(tiles, filters_config=filters)
    tiled_images = tiled_image_from_tiles(
        tiles=tiles,
        context=context,
    )
    if validators is not None:
        tiled_images = apply_validator_pipeline(
            tiled_images, validators_config=validators
        )
    if setup_collection_step is not None:
        setup_collection(
            tiled_images=tiled_images,
            setup_collection_step=setup_collection_step,
        )
    return tiled_images


def remove_pkl_dir(pickle_dir):
    pass


def create_pkl(pickle_dir, tiled_image):
    pass


def build_parallelization_list(
    tiled_images: list[TiledImage],
    tmp_dir_name: str = "_tmp_converter_dir",
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        tiled_images (list[TiledImage]): A list of tiled images objects to convert.
        tmp_dir_name (str): The name of the temporary directory to store the
            pickled tiled images.
    """
    parallelization_list = []
    for image in tiled_images:
        zarr_url = str(image.path)
    """
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
    """
    return parallelization_list
