"""Models for defining regions to be converted into OME-Zarr format."""

from enum import StrEnum
from typing import Literal, NamedTuple

from ngio import DefaultNgffVersion, NgffVersions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from zarr.storage import FsspecStore, LocalStore

ConverterStorageType = LocalStore | FsspecStore


CANONICAL_AXES_TYPE = Literal["t", "c", "z", "y", "x"]
canonical_axes: list[CANONICAL_AXES_TYPE] = ["t", "c", "z", "y", "x"]
COO_SYSTEM_TYPE = Literal["world", "pixel"]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class OverwriteMode(StrEnum):
    NO_OVERWRITE = "No Overwrite"
    OVERWRITE = "Overwrite"
    EXTEND = "Extend"


class TilingMode(StrEnum):
    AUTO = "Auto"
    SNAP_TO_GRID = "Snap to Grid"
    SNAP_TO_CORNERS = "Snap to Corners"
    INPLACE = "Inplace"
    NO_TILING = "No Tiling"


class BackendType(StrEnum):
    ANNDATA = "anndata"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


class AcquisitionDetails(BaseModel):
    """Details about the acquisition.

    These attributes are known and fixed prior to conversion.
    (Either parsed from metadata or manually serialized by the user beforehand.)
    """

    # Determine the coordinate system for start and length values
    start_x_coo: COO_SYSTEM_TYPE = "world"
    start_y_coo: COO_SYSTEM_TYPE = "world"
    start_z_coo: COO_SYSTEM_TYPE = "world"
    start_t_coo: COO_SYSTEM_TYPE = "world"
    length_x_coo: COO_SYSTEM_TYPE = "pixel"
    length_y_coo: COO_SYSTEM_TYPE = "pixel"
    length_z_coo: COO_SYSTEM_TYPE = "pixel"
    length_t_coo: COO_SYSTEM_TYPE = "pixel"
    # Spacing information
    pixelsize: float = Field(default=1.0, gt=0.0)  # in micrometers
    z_spacing: float = Field(default=1.0, gt=0.0)  # in micrometers
    t_spacing: float = Field(default=1.0, gt=0.0)  # in micrometers

    # Channel information
    channel_names: list[str] | None = None
    wavelength_ids: list[str] | None = None
    colors: list[str] | None = None

    # Axes order to be used for the data (should be a subset of canonical axes)
    axes: list[CANONICAL_AXES_TYPE] = Field(
        default_factory=lambda: canonical_axes.copy(), min_length=2, max_length=5
    )

    # Data type of the image data (if known)
    data_type: str | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @field_validator("axes")
    def validate_axes(cls, v: list[CANONICAL_AXES_TYPE]) -> list[CANONICAL_AXES_TYPE]:
        """Validate that axes are in canonical order."""
        for i in range(1, len(v)):
            if canonical_axes.index(v[i]) <= canonical_axes.index(v[i - 1]):
                raise ValueError("Axes must be in canonical order: t, c, z, y, x")
        return v


class StageCorrections(BaseModel):
    flip_x: bool = False
    flip_y: bool = False
    swap_xy: bool = False
    model_config = ConfigDict(extra="forbid")


class AlignmentCorrections(BaseModel):
    align_xy: bool = False
    align_z: bool = False
    align_t: bool = False
    model_config = ConfigDict(extra="forbid")


class OmeZarrOptions(BaseModel):
    num_levels: int = Field(default=5, ge=1)
    max_xy_chunk: int = Field(default=4096, ge=1)
    z_chunk: int = Field(default=10, ge=1)
    c_chunk: int = Field(default=1, ge=1)
    t_chunk: int = Field(default=1, ge=1)
    ngff_version: NgffVersions = DefaultNgffVersion
    table_backend: BackendType = BackendType.ANNDATA
    model_config = ConfigDict(extra="forbid")


class TempJsonOptions(BaseModel):
    temp_url: str = "{zarr_dir}/_tmp_json"

    def format_temp_url(self, zarr_dir: str) -> str:
        return self.temp_url.format(zarr_dir=zarr_dir)


class ConverterOptions(BaseModel):
    tiling_mode: TilingMode = TilingMode.AUTO
    stage_correction: StageCorrections = Field(default_factory=StageCorrections)
    alignment_correction: AlignmentCorrections = Field(
        default_factory=AlignmentCorrections
    )
    omezarr_options: OmeZarrOptions = Field(default_factory=OmeZarrOptions)
    temp_json_options: TempJsonOptions = Field(default_factory=TempJsonOptions)
    model_config = ConfigDict(extra="forbid")


class ContextModel(NamedTuple):
    """Base model for context information during conversion.

    This models holds the all context information needed during the conversion
    process, including acquisition details and converter options.
    """

    acquisition_details: AcquisitionDetails
    converter_options: ConverterOptions
