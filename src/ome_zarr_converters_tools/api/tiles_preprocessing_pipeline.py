"""Functions to build TiledImage models from Tile models."""

from zarr.storage import LocalStore

from ome_zarr_converters_tools.collection_setup._store_utils import ConverterStorageType
from ome_zarr_converters_tools.filters import FilterStep, apply_filter_pipeline
from ome_zarr_converters_tools.models import (
    ContextModel,
    ConvertParallelInitArgs,
    Tile,
    TiledImageWithContext,
)
from ome_zarr_converters_tools.utils import tiled_image_from_tiles
from ome_zarr_converters_tools.utils._json_utils import cleanup_if_exists, dump_to_json
from ome_zarr_converters_tools.validators import ValidatorStep, apply_validator_pipeline


def tiles_preprocessing_pipeline(
    tiles: list[Tile],
    context: ContextModel,
    filters: list[FilterStep] | None = None,
    validators: list[ValidatorStep] | None = None,
) -> list[TiledImageWithContext]:
    """Process tiles through the preprocessing pipeline to create TiledImages.

    This function applies optional filters to the input tiles and then
    constructs TiledImage models from the processed tiles.

    Args:
        tiles: List of Tile models to process.
        context: Full context model for the conversion.
        filters: Optional list of filter steps to apply to the tiles.
        validators: Optional list of validator steps to apply to the tiles.

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
    # Wrap each TiledImage with its context
    tiled_images = [
        TiledImageWithContext(tiled_image=ti, context=context) for ti in tiled_images
    ]
    return tiled_images


def stup_ome_zarr_collection(
    store: ConverterStorageType, tiled_image: list[TiledImageWithContext]
) -> None:
    """Set up an OME-Zarr collection in the given store.

    Args:
        store: The Zarr store where the collection will be set up.
        tiled_image: A list of TiledImageWithContext objects to set up the
            collection for.
    """
    # Currently a placeholder for potential future setup steps
    from ome_zarr_converters_tools.collection_setup import (
        SetupCollectionStep,
        setup_collection,
    )

    step = SetupCollectionStep(
        name="ImageInPlate",
        store=store,
        ngff_version=tiled_image[
            0
        ].context.converter_options.omezarr_options.ngff_version,
        overwrite_mode=tiled_image[0].context.overwrite_mode,
    )
    setup_collection(
        tiled_images=[ti.tiled_image for ti in tiled_image], setup_collection_step=step
    )


def build_parallelization_list(
    store: ConverterStorageType,
    tiled_images: list[TiledImageWithContext],
    tmp_path: str = "_tmp_json",
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        store (ConverterStorageType): The base store for the zarr data.
        tiled_images (list[TiledImageWithContext]): A list of tiled images objects
            to convert, since tiled images can come from different acquisitions,
            each tiled image keeps its own context.
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
                    overwrite_mode=context.overwrite_mode,
                ).model_dump(),
            }
        )
    return parallelization_list
