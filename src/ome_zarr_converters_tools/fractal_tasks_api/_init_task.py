"""Utilities for converters init tasks in Fractal."""

from zarr.storage import LocalStore

from ome_zarr_converters_tools.collection_setup import (
    ConverterStorageType,
    setup_ome_zarr_collection,
)
from ome_zarr_converters_tools.models import (
    ConverterOptions,
    ConvertParallelInitArgs,
    DefaultNgffVersion,
    NgffVersions,
    OverwriteMode,
    TiledImage,
)
from ome_zarr_converters_tools.utils import cleanup_if_exists, dump_to_json


def build_parallelization_list(
    tiled_images: list[TiledImage],
    *,
    store: ConverterStorageType,
    converter_options: ConverterOptions,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
    tmp_path: str = "_tmp_json",
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        store (ConverterStorageType): The base store for the zarr data.
        tiled_images (list[TiledImageWithContext]): A list of tiled images objects
            to convert.
        converter_options (ConverterOptions): The converter options to use during
            conversion.
        overwrite_mode (OverwriteMode): The overwrite mode to use when writing the data.
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
                    store_url=store.root.absolute().as_uri(),
                    json_file_name=json_name,
                    converter_options=converter_options,
                    overwrite_mode=overwrite_mode,
                ).model_dump_json(exclude_none=True),
            }
        )
    return parallelization_list


def setup_images_for_conversion(
    tiled_images: list[TiledImage],
    *,
    store: ConverterStorageType,
    collection_type: str,
    converter_options: ConverterOptions,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
    ngff_version: NgffVersions = DefaultNgffVersion,
) -> list[dict]:
    """Setup the OME-Zarr collection from converted tiled images.

    This function run all the necessary steps to setup before parallel conversion.
        - Build the OME-Zarr collection structure.
        - Build the parallelization list (used by the fractal compute task).

    Args:
        tiled_images: List of TiledImageWithContext models that have been converted.
        store: The base store for the zarr data.
        collection_type: The type of collection to set up.
        converter_options: The converter options to use during conversion.
        overwrite_mode: The overwrite mode to use when writing the data.
        ngff_version: The NGFF version to use when setting up the collection.
    """
    setup_ome_zarr_collection(
        tiled_images=tiled_images,
        collection_type=collection_type,
        store=store,
        ngff_version=ngff_version,
        overwrite_mode=overwrite_mode,
    )
    return build_parallelization_list(
        store=store,
        tiled_images=tiled_images,
        converter_options=converter_options,
        overwrite_mode=overwrite_mode,
    )
