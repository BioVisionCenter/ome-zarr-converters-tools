"""Utilities for converters init tasks in Fractal."""

from ome_zarr_converters_tools.core._tile_region import (
    TiledImage,
)
from ome_zarr_converters_tools.fractal._json_utils import (
    cleanup_if_exists,
    dump_json_str,
)
from ome_zarr_converters_tools.fractal._models import (
    ConvertParallelInitArgs,
)
from ome_zarr_converters_tools.models import (
    ConverterOptions,
    DefaultNgffVersion,
    NgffVersions,
    OverwriteMode,
)
from ome_zarr_converters_tools.models._url_utils import join_url_paths
from ome_zarr_converters_tools.pipelines._collection_setup import (
    setup_ome_zarr_collection,
)


def build_parallelization_list(
    tiled_images: list[TiledImage],
    *,
    zarr_dir: str,
    converter_options: ConverterOptions,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
) -> list[dict]:
    """Build a list of dictionaries to parallelize the conversion.

    Args:
        tiled_images: List of tiled images to convert.
        zarr_dir: The base directory for the zarr data.
        converter_options: The converter options to use during conversion.
        overwrite_mode: The overwrite mode to use when writing the data.

    Returns:
        One dict per image, each with a `zarr_url` and `init_args` entry, ready
        to be consumed by the Fractal compute task.
    """
    # Determine whether to use in-memory JSON strings or temporary JSON files
    # based on the total size of the serialized tiled images and the temp_json_options.
    temp_json_options = converter_options.runtime_settings.temp_json_options
    json_strs = [image.model_dump_json() for image in tiled_images]
    total_bytes = sum(len(s.encode()) for s in json_strs)
    in_memory = temp_json_options.use_in_memory(total_bytes)

    temp_json_url = temp_json_options.format_temp_url(zarr_dir=zarr_dir)
    if not in_memory:
        cleanup_if_exists(temp_json_url=temp_json_url)

    parallelization_list = []
    for image, json_str in zip(tiled_images, json_strs, strict=True):
        zarr_url = join_url_paths(zarr_dir, image.path)
        if in_memory:
            init_args = ConvertParallelInitArgs(
                tiled_image_json_str=json_str,
                converter_options=converter_options,
                overwrite_mode=overwrite_mode,
            )
        else:
            tiled_image_json_dump_url = dump_json_str(
                temp_json_url=temp_json_url, json_str=json_str
            )
            init_args = ConvertParallelInitArgs(
                tiled_image_json_dump_url=tiled_image_json_dump_url,
                converter_options=converter_options,
                overwrite_mode=overwrite_mode,
            )
        parallelization_list.append(
            {
                "zarr_url": zarr_url,
                "init_args": init_args.model_dump(exclude=None),
            }
        )
    return parallelization_list


def setup_images_for_conversion(
    tiled_images: list[TiledImage],
    *,
    zarr_dir: str,
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
        tiled_images: List of tiled images that have been converted.
        zarr_dir: The base directory for the zarr data.
        collection_type: The type of collection to set up.
        converter_options: The converter options to use during conversion.
        overwrite_mode: The overwrite mode to use when writing the data.
        ngff_version: The NGFF version to use when setting up the collection.
    """
    setup_ome_zarr_collection(
        tiled_images=tiled_images,
        collection_type=collection_type,
        zarr_dir=zarr_dir,
        ngff_version=ngff_version,
        overwrite_mode=overwrite_mode,
    )
    return build_parallelization_list(
        zarr_dir=zarr_dir,
        tiled_images=tiled_images,
        converter_options=converter_options,
        overwrite_mode=overwrite_mode,
    )
