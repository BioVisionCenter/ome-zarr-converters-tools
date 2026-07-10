"""Stand-in Fractal task exposing the full downstream model surface.

Not a real task: its only purpose is to mirror the signatures of the
downstream converter tasks (e.g. fractal-uzh-converters) so that
fractal-task-tools generates a JSON schema whose `$defs` contain every
model this package exposes to Fractal manifests. Consumed exclusively by
`test_json_schema_compat.py`.
"""

from ome_zarr_converters_tools import (
    AcquisitionOptions,
    ConverterOptions,
    ConvertParallelInitArgs,
    OverwriteMode,
)

default_converter_options = ConverterOptions()


def schema_surface_task(
    *,
    zarr_dir: str,
    acquisitions: list[AcquisitionOptions],
    converter_options: ConverterOptions = default_converter_options,
    overwrite: OverwriteMode = OverwriteMode.NO_OVERWRITE,
    init_args: ConvertParallelInitArgs,
) -> None:
    """Stand-in task covering the models exposed to downstream converters.

    Args:
        zarr_dir: Directory to store the Zarr files.
        acquisitions: List of raw acquisitions to convert to OME-Zarr.
        converter_options: Advanced converter options.
        overwrite: Overwrite mode for existing data.
        init_args: Arguments forwarded from the init task to the compute task.
    """
    raise NotImplementedError("Schema-generation stand-in, never executed.")
