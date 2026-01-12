"""Functions to build TiledImage models from Tile models."""

from zarr.storage import LocalStore

from ome_zarr_converters_tools.collection_setup import (
    SetupCollectionStep,
    setup_collection,
)
from ome_zarr_converters_tools.collection_setup._store_utils import ConverterStorageType
from ome_zarr_converters_tools.filters import FilterStep, apply_filter_pipeline
from ome_zarr_converters_tools.models import (
    BaseTile,
    ContextModel,
    ConvertParallelInitArgs,
    TiledImage,
)
from ome_zarr_converters_tools.utils import tiled_image_from_tiles
from ome_zarr_converters_tools.utils._json_utils import cleanup_if_exists, dump_to_json
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


def build_parallelization_list(
    store: ConverterStorageType,
    tiled_images: list[TiledImage],
    context: ContextModel,
    tmp_path: str = "_tmp_json",
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        store (ConverterStorageType): The base store for the zarr data.
        tiled_images (list[TiledImage]): A list of tiled images objects to convert.
        context (ContextModel): Full context model for the conversion.
        tmp_path (str): The name of the temporary directory to store the
            pickled tiled images.
    """
    if isinstance(store, LocalStore):
        zarr_base = str(store.root)
    else:
        raise NotImplementedError(
            "Parallelization list building is only implemented for LocalStore."
        )
    cleanup_if_exists(store, tmp_path=tmp_path)
    parallelization_list = []
    for image in tiled_images:
        json_name = dump_to_json(store, image, tmp_path=tmp_path)
        # This is not used directly but kept for api consistency
        zarr_url = f"{zarr_base}/{image.path}"
        parallelization_list.append(
            {
                "zarr_url": zarr_url,
                "init_args": ConvertParallelInitArgs(
                    store_type="local",  # fsspec not yet supported
                    store_url=store.root.as_uri(),
                    json_file_name=json_name,
                    converter_options=context.converter_options,
                    acquisition_details=context.acquisition_details,
                    overwrite_mode=context.overwrite_mode,
                ).model_dump(),
            }
        )
    return parallelization_list
