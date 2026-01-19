from typing import Literal

from pydantic import BaseModel

from ome_zarr_converters_tools.models._acquisition import (
    AcquisitionDetails,
    ConverterOptions,
    OverwriteMode,
)


class ConvertParallelInitArgs(BaseModel):
    """Arguments for the compute task."""

    store_url: str
    json_file_name: str
    store_type: Literal["local", "fsspec"] = "local"
    converter_options: ConverterOptions
    acquisition_details: AcquisitionDetails
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE
