"""Models to be used with Fractal tasks API."""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from ome_zarr_converters_tools.models._acquisition import (
    CANONICAL_AXES_TYPE,
    AcquisitionDetails,
    ChannelInfo,
    DataTypeEnum,
    StageOrientation,
    canonical_axes,
)
from ome_zarr_converters_tools.models._converter_options import (
    ConverterOptions,
    OverwriteMode,
)
from ome_zarr_converters_tools.pipelines._filters import ImplementedFilters


class ConvertParallelInitArgs(BaseModel):
    """Arguments for the compute task."""

    tiled_image_json_dump_url: str | None = None
    tiled_image_json_str: str | None = None
    converter_options: ConverterOptions
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE

    @model_validator(mode="after")
    def _validate_exactly_one_source(self) -> "ConvertParallelInitArgs":
        if (self.tiled_image_json_dump_url is None) == (
            self.tiled_image_json_str is None
        ):
            raise ValueError(
                "Exactly one of tiled_image_json_dump_url or "
                "tiled_image_json_str must be set."
            )
        return self


class PixelSizeModel(BaseModel):
    """Pixel size model."""

    xy_pixel_size: float
    """
    XY pixel size in micrometers.
    """
    z_spacing: float
    """
    Z spacing in micrometers.
    """
    t_spacing: float
    """
    Time spacing in seconds.
    """


named_colors = {
    "Blue": "#0000FF",
    "Red": "#FF0000",
    "Yellow": "#FFFF00",
    "Magenta": "#FF00FF",
    "Cyan": "#00FFFF",
    "Gray": "#808080",
    "Green": "#00FF00",
    "Orange": "#FF8000",
    "Purple": "#8000FF",
    "Teal": "#008080",
    "Lime": "#00FF80",
    "Amber": "#FFBF00",
    "Pink": "#FF0080",
    "Navy": "#000080",
    "Maroon": "#800000",
    "Olive": "#808000",
    "Coral": "#FF7F50",
    "Violet": "#EE82EE",
}


class ColorMenuBase(StrEnum):
    """Default color conversion for the channels."""

    def to_hexstr(self) -> str | None:
        if self.name == "Auto":
            # Auto color assignment is handled by ChannelInfo's model
            # validator when color=None
            return None
        color = named_colors.get(self.name)
        if color is None:
            raise ValueError(f"No default color found for {self.name=}")
        return color


_color_menu = {
    "Auto": "Auto",
    **{name: f"{name} ({val})" for name, val in named_colors.items()},
}

ColorMenu = StrEnum(
    "ColorMenu",
    _color_menu,
    type=ColorMenuBase,
)


class ChannelInfoUI(BaseModel):
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
    color: ColorMenu = ColorMenu.Auto
    """The color associated with the channel, e.g. for visualization purposes."""


class AcquisitionOptions(BaseModel):
    """Acquisition options for conversion.

    These are option that can be specified per acquisition.
    by the user at conversion time.
    This is not to be confused with AcquisitionDetails,
    this model is used in fractal tasks to override/update
    details from AcquisitionDetails model.
    """

    channels: list[ChannelInfoUI] | None = None
    """List of channel information."""
    pixel_info: PixelSizeModel | None = Field(
        default=None, title="Pixel Size Information"
    )
    """Pixel size information."""
    condition_table_path: str | None = None
    """Optional path to a condition table CSV file."""
    axes: str | None = None
    """Axes to use for the image data, e.g. "czyx"."""
    data_type: DataTypeEnum = Field(default=DataTypeEnum.AUTODETECT, title="Data Type")
    """Data type of the image data."""
    stage_orientation: StageOrientation = Field(
        default_factory=StageOrientation, title="Stage Orientation"
    )
    """Stage orientation corrections."""
    filters: list[ImplementedFilters] = Field(default_factory=list)
    """List of filters to apply."""

    def to_axes_list(self) -> list[CANONICAL_AXES_TYPE] | None:
        """Convert axes string to list of axes."""
        if self.axes is None:
            return None
        _axes = []
        for ax in self.axes:
            if ax not in canonical_axes:
                raise ValueError(f"Invalid axis '{ax}' in axes string.")
            _axes.append(ax)
        return _axes  # type: ignore

    def update_acquisition_details(
        self,
        acquisition_details: AcquisitionDetails,
    ) -> AcquisitionDetails:
        """Update AcquisitionDetails model with options from this model.

        Args:
            acquisition_details: AcquisitionDetails model to update.

        Returns:
            Updated AcquisitionDetails model.

        """
        updated_details = acquisition_details.model_copy()
        if self.channels is not None:
            _updated_channels = []
            for channel in self.channels:
                _updated_channels.append(
                    ChannelInfo(
                        channel_label=channel.channel_label,
                        wavelength_id=channel.wavelength_id,
                        color=channel.color.to_hexstr(),
                    )
                )
            updated_details.channels = _updated_channels
        if self.pixel_info is not None:
            updated_details.xy_pixel_size = self.pixel_info.xy_pixel_size
            updated_details.z_spacing = self.pixel_info.z_spacing
            updated_details.t_spacing = self.pixel_info.t_spacing
        axes = self.to_axes_list()
        if axes is not None:
            updated_details.axes = axes
        if self.data_type != DataTypeEnum.AUTODETECT:
            updated_details.data_type = self.data_type
        if self.condition_table_path is not None:
            updated_details.condition_table_path = self.condition_table_path
        return updated_details


def converters_tools_models(
    base: str = "ome_zarr_converters_tools",
) -> list[tuple[str, str, str]]:
    """Get all input models for Fractal tasks API.

    Returns:
        List of input models.
    """
    return [
        (
            base,
            "fractal/_models.py",
            "AcquisitionOptions",
        ),
        (
            base,
            "pipelines/_filters.py",
            "WellFilter",
        ),
        (
            base,
            "pipelines/_filters.py",
            "RegexFilter",
        ),
        (
            base,
            "models/_converter_options.py",
            "ConverterOptions",
        ),
        (
            base,
            "models/_acquisition.py",
            "StageOrientation",
        ),
        (
            base,
            "models/_converter_options.py",
            "StagePositionCorrections",
        ),
        (
            base,
            "models/_converter_options.py",
            "OmeZarrOptions",
        ),
        (
            base,
            "models/_runtime_settings.py",
            "TempJsonOptions",
        ),
        (
            base,
            "models/_converter_options.py",
            "FovBasedChunking",
        ),
        (
            base,
            "models/_converter_options.py",
            "FixedSizeChunking",
        ),
        (
            base,
            "models/_acquisition.py",
            "ChannelInfo",
        ),
    ]
