"""Pipeline modules for OME-Zarr converters tools."""

from ome_zarr_converters_tools.pipelines._collection_setup import (
    add_collection_handler,
    setup_ome_zarr_collection,
)
from ome_zarr_converters_tools.pipelines._filters import (
    FilterModel,
    ImplementedFilters,
    add_filter,
    apply_filter_pipeline,
)
from ome_zarr_converters_tools.pipelines._registration_pipeline import (
    RegistrationStep,
    add_registration_func,
    apply_registration_pipeline,
    build_default_registration_pipeline,
)
from ome_zarr_converters_tools.pipelines._tiled_image_creation_pipeline import (
    tiled_image_creation_pipeline,
)
from ome_zarr_converters_tools.pipelines._tiles_aggregation_pipeline import (
    tiles_aggregation_pipeline,
)
from ome_zarr_converters_tools.pipelines._validators import (
    ValidatorStep,
    add_validator,
    apply_validator_pipeline,
)
from ome_zarr_converters_tools.pipelines._write_ome_zarr import (
    write_tiled_image_as_zarr,
)

__all__ = [
    "FilterModel",
    "ImplementedFilters",
    "RegistrationStep",
    "ValidatorStep",
    "add_collection_handler",
    "add_filter",
    "add_registration_func",
    "add_validator",
    "apply_filter_pipeline",
    "apply_registration_pipeline",
    "apply_validator_pipeline",
    "build_default_registration_pipeline",
    "setup_ome_zarr_collection",
    "tiled_image_creation_pipeline",
    "tiles_aggregation_pipeline",
    "write_tiled_image_as_zarr",
]
