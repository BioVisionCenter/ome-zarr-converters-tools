"""Models for defining regions to be converted into OME-Zarr format."""

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


class DefaultColors(StrEnum):
    """Default colors for the channels."""

    blue = "Blue (0000FF)"
    red = "Red (FF0000)"
    yellow = "Yellow (FFFF00)"
    magenta = "Magenta (FF00FF)"
    cyan = "Cyan (00FFFF)"
    gray = "Gray (808080)"
    green = "Green (00FF00)"
    orange = "Orange (FF8000)"
    purple = "Purple (8000FF)"
    teal = "Teal (008080)"
    lime = "Lime (00FF80)"
    amber = "Amber (FFBF00)"
    pink = "Pink (FF0080)"
    navy = "Navy (000080)"
    maroon = "Maroon (800000)"
    olive = "Olive (808000)"
    coral = "Coral (FF7F50)"
    violet = "Violet (8000FF)"

    def to_hex(self) -> str:
        """Convert the color to hex format."""
        _color_mapping = {
            DefaultColors.blue: "#0000FF",
            DefaultColors.red: "#FF0000",
            DefaultColors.yellow: "#FFFF00",
            DefaultColors.magenta: "#FF00FF",
            DefaultColors.cyan: "#00FFFF",
            DefaultColors.gray: "#808080",
            DefaultColors.green: "#00FF00",
            DefaultColors.orange: "#FF8000",
            DefaultColors.purple: "#8000FF",
            DefaultColors.teal: "#008080",
            DefaultColors.lime: "#00FF80",
            DefaultColors.amber: "#FFBF00",
            DefaultColors.pink: "#FF0080",
            DefaultColors.navy: "#000080",
            DefaultColors.maroon: "#800000",
            DefaultColors.olive: "#808000",
            DefaultColors.coral: "#FF7F50",
            DefaultColors.violet: "#8000FF",
        }
        return _color_mapping[self]


class ChannelInfo(BaseModel):
    """Channel information.

    Attributes:
        channel_label: Label of the channel.
        wavelength_id: The wavelength ID of the channel.
            This field can be used in some tasks as alternative to channel_label,
            e.g. for multiplexed acquisitions it can be used for applying illumination
            correction based on wavelength ID instead of channel name.
        colors: The color associated with the channel, e.g. for visualization purposes.
    """

    channel_label: str
    wavelength_id: str | None = None
    colors: DefaultColors = DefaultColors.blue


class StageCorrections(BaseModel):
    """Stage orientation corrections.

    Attributes:
        flip_x: Whether to flip the position along the X axis.
        flip_y: Whether to flip the position along the Y axis.
        swap_xy: Whether to swap the positions along the X and Y axes.
    """

    flip_x: bool = Field(default=False, title="Flip X")
    flip_y: bool = Field(default=False, title="Flip Y")
    swap_xy: bool = Field(default=False, title="Swap XY")
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
    data_type: DataTypeEnum | None = None

    # Condition table path (if applicable)
    condition_table_path: str | None = None

    # Stage orientation corrections
    stage_corrections: StageCorrections = Field(default_factory=StageCorrections)

    model_config = ConfigDict(extra="forbid")

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, v: list[CANONICAL_AXES_TYPE]) -> list[CANONICAL_AXES_TYPE]:
        """Validate that axes are in canonical order."""
        for i in range(1, len(v)):
            if canonical_axes.index(v[i]) <= canonical_axes.index(v[i - 1]):
                raise ValueError("Axes must be in canonical order: t, c, z, y, x")
        return v
