"""Models for defining regions to be converted into OME-Zarr format."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

CANONICAL_AXES_TYPE = Literal["t", "c", "z", "y", "x"]
canonical_axes: list[CANONICAL_AXES_TYPE] = ["t", "c", "z", "y", "x"]
COO_SYSTEM_TYPE = Literal["world", "pixel"]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class DataTypeEnum(StrEnum):
    """Data type enumeration."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"


def default_axes_builder(is_time_series: bool) -> list[CANONICAL_AXES_TYPE]:
    """Build default axes list."""
    if is_time_series:
        return ["t", "c", "z", "y", "x"]
    else:
        return ["c", "z", "y", "x"]


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
    wavelength_ids: Sequence[str | None] | None = None
    colors: Sequence[str | None] | None = None

    # Axes order to be used for the data (should be a subset of canonical axes)
    axes: list[CANONICAL_AXES_TYPE] = Field(
        default_factory=lambda: canonical_axes.copy(), min_length=2, max_length=5
    )

    # Data type of the image data (if known)
    data_type: DataTypeEnum | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @field_validator("axes")
    def validate_axes(cls, v: list[CANONICAL_AXES_TYPE]) -> list[CANONICAL_AXES_TYPE]:
        """Validate that axes are in canonical order."""
        for i in range(1, len(v)):
            if canonical_axes.index(v[i]) <= canonical_axes.index(v[i - 1]):
                raise ValueError("Axes must be in canonical order: t, c, z, y, x")
        return v
