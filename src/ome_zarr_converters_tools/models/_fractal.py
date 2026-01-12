from typing import Literal

from pydantic import BaseModel

from ome_zarr_converters_tools.models import ConverterOptions
from ome_zarr_converters_tools.models._acquisition import OVERWRITE_MODES


class ConvertParallelInitArgs(BaseModel):
    """Arguments for the compute task."""

    store_url: str
    json_file_name: str
    store_type: Literal["local", "fsspec"] = "local"
    converter_options: ConverterOptions
    overwrite_mode: OVERWRITE_MODES = "no_overwrite"
