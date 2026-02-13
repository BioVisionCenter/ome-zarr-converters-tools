"""Utilities for converters init tasks in Fractal."""

from ome_zarr_converters_tools.core._tile_region import (
    TiledImage,
)
from ome_zarr_converters_tools.fractal._json_utils import (
    cleanup_if_exists,
    dump_to_json,
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
        tiled_images (list[TiledImageWithContext]): A list of tiled images objects
            to convert.
        zarr_dir (str): The base directory for the zarr data.
        converter_options (ConverterOptions): The converter options to use during
            conversion.
        overwrite_mode (OverwriteMode): The overwrite mode to use when writing the data.
        tmp_path (str): The name of the temporary directory to store the
            pickled tiled images.
    """
    temp_json_url = converter_options.temp_json_options.format_temp_url(
        zarr_dir=zarr_dir
    )
    cleanup_if_exists(temp_json_url=temp_json_url)
    parallelization_list = []
    for image in tiled_images:
        tiled_image_json_dump_url = dump_to_json(
            temp_json_url=temp_json_url, tiled_image=image
        )
        # This is not used directly but kept for api consistency
        zarr_url = join_url_paths(zarr_dir, image.path)
        parallelization_list.append(
            {
                "zarr_url": zarr_url,
                "init_args": ConvertParallelInitArgs(
                    tiled_image_json_dump_url=tiled_image_json_dump_url,
                    converter_options=converter_options,
                    overwrite_mode=overwrite_mode,
                ).model_dump(exclude=None),
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
        tiled_images: List of TiledImageWithContext models that have been converted.
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
