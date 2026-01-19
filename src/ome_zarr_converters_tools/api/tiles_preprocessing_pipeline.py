"""Functions to build TiledImage models from Tile models."""

from typing import Any

from zarr.storage import LocalStore

from ome_zarr_converters_tools.collection_setup import (
    ConverterStorageType,
    setup_ome_zarr_collection,
)
from ome_zarr_converters_tools.filters import FilterStep, apply_filter_pipeline
from ome_zarr_converters_tools.models import (
    ContextModel,
    ConvertParallelInitArgs,
    OverwriteMode,
    Tile,
    TiledImageWithContext,
)
from ome_zarr_converters_tools.utils import tiled_image_from_tiles
from ome_zarr_converters_tools.utils._json_utils import cleanup_if_exists, dump_to_json
from ome_zarr_converters_tools.validators import ValidatorStep, apply_validator_pipeline


def tiles_preprocessing_pipeline(
    tiles: list[Tile],
    *,
    context: ContextModel,
    filters: list[FilterStep] | None = None,
    validators: list[ValidatorStep] | None = None,
    resource: Any | None = None,
) -> list[TiledImageWithContext]:
    """Process tiles through the preprocessing pipeline to create TiledImages.

    This function applies optional filters to the input tiles and then
    constructs TiledImage models from the processed tiles.

    Args:
        tiles: List of Tile models to process.
        context: Full context model for the conversion.
        filters: Optional list of filter steps to apply to the tiles.
        validators: Optional list of validator steps to apply to the tiles.
        resource: Optional resource to assist in processing.

    Returns:
        A list of TiledImage models created from the processed tiles.
    """
    if filters is not None:
        tiles = apply_filter_pipeline(tiles, filters_config=filters)
    tiled_images = tiled_image_from_tiles(
        tiles=tiles,
        context=context,
        resource=resource,
    )
    if validators is not None:
        tiled_images = apply_validator_pipeline(
            tiled_images, validators_config=validators
        )
    tiled_images = [
        TiledImageWithContext(tiled_image=ti, context=context) for ti in tiled_images
    ]
    return tiled_images


def build_parallelization_list(
    tiled_images: list[TiledImageWithContext],
    *,
    store: ConverterStorageType,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
    tmp_path: str = "_tmp_json",
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        store (ConverterStorageType): The base store for the zarr data.
        tiled_images (list[TiledImageWithContext]): A list of tiled images objects
            to convert.
        overwrite_mode (OVERWRITE_MODES): Mode to handle existing data.
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
    for image, context in tiled_images:
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
                    overwrite_mode=overwrite_mode,
                ).model_dump(),
            }
        )
    return parallelization_list


def setup_images_from_conversion(
    tiled_images: list[TiledImageWithContext],
    *,
    store: ConverterStorageType,
    collection_type: str,
) -> None:
    """Setup the OME-Zarr collection from converted tiled images.

    Args:
        tiled_images: List of TiledImageWithContext models that have been converted.
        store: The base store for the zarr data.
        collection_type: The type of collection to set up.
    """
    setup_ome_zarr_collection(
        tiled_images=tiled_images,
        collection_type=collection_type,
        store=store,
    )
    build_parallelization_list(
        store=store,
        tiled_images=tiled_images,
    )
