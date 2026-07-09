"""Models for defining regions to be converted into OME-Zarr format."""

import re
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

CANONICAL_AXES_TYPE = Literal["t", "c", "z", "y", "x"]
canonical_axes: list[CANONICAL_AXES_TYPE] = ["t", "c", "z", "y", "x"]
COO_SYSTEM_TYPE = Literal["world", "pixel"]
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class DataTypeEnum(StrEnum):
    """Data type enumeration."""

    AUTODETECT = "autodetect"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"


def default_axes_builder(is_time_series: bool) -> list[CANONICAL_AXES_TYPE]:
    """Build default axes list."""
    if is_time_series:
        return ["t", "c", "z", "y", "x"]
    else:
        return ["c", "z", "y", "x"]


_WAVELENGTH_COLOR_TABLE: list[tuple[int, str]] = [
    (380, "#FF00FF"),  # UV (<380 nm)      → Magenta
    (450, "#0000FF"),  # Violet/Blue       → Blue
    (495, "#00FFFF"),  # Cyan              → Cyan
    (520, "#00FF00"),  # Green             → Green
    (560, "#00FF80"),  # Lime/Yellow-green → Lime
    (590, "#FFFF00"),  # Yellow            → Yellow
    (620, "#FF8000"),  # Orange            → Orange
    (750, "#FF0000"),  # Red               → Red
]
_WAVELENGTH_VALID_RANGE = (200, 1500)

_LABEL_COLOR_MAP: dict[str, str] = {
    "dapi": "#0000FF",
    "hoechst": "#0000FF",
    "blue": "#0000FF",
    "cfp": "#00FFFF",
    "cyan": "#00FFFF",
    "gfp": "#00FF00",
    "egfp": "#00FF00",
    "fitc": "#00FF00",
    "green": "#00FF00",
    "yfp": "#FFFF00",
    "yellow": "#FFFF00",
    "cy3": "#FFFF00",
    "rfp": "#FF0000",
    "mcherry": "#FF0000",
    "cy5": "#FF0000",
    "red": "#FF0000",
    "cy7": "#FF00FF",
    "magenta": "#FF00FF",
    "brightfield": "#808080",
    "gray": "#808080",
}


def _color_from_wavelength_id(wavelength_id: str | None) -> str | None:
    """Assign a default color based on the wavelength ID."""
    if wavelength_id is None:
        return None
    match = re.search(r"\d+", wavelength_id)
    if match is None:
        return None
    wavelength = int(match.group())
    lo, hi = _WAVELENGTH_VALID_RANGE
    # If wavelength is out of visible range,
    # return None to allow fallback to label-based color assignment
    if not (lo <= wavelength <= hi):
        return None
    # Assign color based on wavelength ranges corresponding to visible light
    for upper_bound, color in _WAVELENGTH_COLOR_TABLE:
        if wavelength < upper_bound:
            return color
    # The final visible band (Red) is inclusive of its 750 nm upper edge;
    # strictly-longer near-IR wavelengths fall back to magenta.
    if wavelength == _WAVELENGTH_COLOR_TABLE[-1][0]:
        return _WAVELENGTH_COLOR_TABLE[-1][1]
    return "#FF00FF"  # > 750 nm: far-red/IR → Magenta


def _color_from_channel_label(channel_label: str) -> str:
    """Assign a default color based on the channel label using fuzzy matching."""
    lower = channel_label.lower()
    # Use fuzzy string matching to find the closest label in the color map
    # This allows for flexible matching of common fluorophore names and colors
    similarity = {
        key: SequenceMatcher(None, lower, key).ratio() for key in _LABEL_COLOR_MAP
    }
    best = max(similarity, key=lambda key: similarity[key])
    return _LABEL_COLOR_MAP[best]


class ChannelInfo(BaseModel):
    """Channel information."""

    channel_label: str
    """Label of the channel."""
    wavelength_id: str | None = None
    """
    The wavelength ID of the channel.
    This field can be used in some tasks as alternative to channel_label,
    e.g. for multiplexed acquisitions it can be used for applying illumination
    correction based on wavelength ID instead of channel name.
    """
    color: str | None = Field(
        default=None, pattern=r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
    )
    """
    The color associated with the channel in hex format,
    e.g. for visualization purposes.
    """

    @model_validator(mode="after")
    def validate_channel_info(self) -> "ChannelInfo":
        """Assign a default color if None is provided."""
        if self.color is not None:
            return self
        color = _color_from_wavelength_id(self.wavelength_id)
        if color is None:
            color = _color_from_channel_label(self.channel_label)
        self.color = color
        return self


class StageOrientation(BaseModel):
    """Stage orientation corrections."""

    flip_x: bool = Field(default=False, title="Flip X")
    """Whether to flip the position along the X axis."""
    flip_y: bool = Field(default=False, title="Flip Y")
    """Whether to flip the position along the Y axis."""
    swap_xy: bool = Field(default=False, title="Swap XY")
    """Whether to swap the positions along the X and Y axes."""
    model_config = ConfigDict(extra="forbid")


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
    channels: list[ChannelInfo] | None = None

    # Axes order to be used for the data (should be a subset of canonical axes)
    axes: list[CANONICAL_AXES_TYPE] = Field(
        default_factory=lambda: canonical_axes.copy(), min_length=2, max_length=5
    )

    # Data type of the image data (if known)
    data_type: DataTypeEnum = DataTypeEnum.AUTODETECT

    # Condition table path (if applicable)
    condition_table_path: str | None = None

    # Stage orientation corrections
    stage_orientation: StageOrientation = Field(default_factory=StageOrientation)

    model_config = ConfigDict(extra="forbid")

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, v: list[CANONICAL_AXES_TYPE]) -> list[CANONICAL_AXES_TYPE]:
        """Validate that axes are in canonical order."""
        for i in range(1, len(v)):
            if canonical_axes.index(v[i]) <= canonical_axes.index(v[i - 1]):
                raise ValueError("Axes must be in canonical order: t, c, z, y, x")
        return v
