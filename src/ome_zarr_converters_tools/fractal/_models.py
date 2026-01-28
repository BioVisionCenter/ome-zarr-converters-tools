"""Models to be used with Fractal tasks API."""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from ome_zarr_converters_tools.filters._filter_pipeline import ImplementedFilters
from ome_zarr_converters_tools.models._acquisition import (
    CANONICAL_AXES_TYPE,
    AcquisitionDetails,
    DataTypeEnum,
    canonical_axes,
)
from ome_zarr_converters_tools.models._converter_options import (
    ConverterOptions,
)
from ome_zarr_converters_tools.models._shared import (
    OverwriteMode,
)


class ConvertParallelInitArgs(BaseModel):
    """Arguments for the compute task."""

    tiled_image_json_dump_url: str
    converter_options: ConverterOptions
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE


class ChannelInfo(BaseModel):
    """Channel information.

    Attributes:
        name: Name of the channel.
        wavelength_id: Wavelength ID of the channel.
        colors: Color information of the channel.
    """

    name: str
    wavelength_id: str | None = None
    colors: str | None = None


class PixelSizeModel(BaseModel):
    """Pixel size model 2.

    Attributes:
        pixelsize: Pixel size in micrometers.
        z_spacing: Z spacing in micrometers.
        t_spacing: Time spacing in seconds.
    """

    pixelsize: float
    z_spacing: float
    t_spacing: float


class AcquisitionOptions(BaseModel):
    """Acquisition options for conversion.

    These are option that can be specified per acquisition.
    by the user at conversion time.
    This is not to be confused with AcquisitionDetails,
    this model is used in fractal tasks to override/update
    details from AcquisitionDetails model.

    Attributes:
        channels: List of channel information.
        pixel_info: Pixel size information.
        axes: Axes to use for the image data, e.g. "czyx".
        data_type: Data type of the image data.
        filters: List of filters to apply.
    """

    channels: list[ChannelInfo] | None = None
    pixel_info: PixelSizeModel | None = Field(
        default=None, title="Pixel Size Information"
    )
    axes: str | None = None
    data_type: DataTypeEnum | None = None
    filters: list[ImplementedFilters] = Field(default_factory=list)

    # Validate channels to ensure that either all wavelength_id and colors are provided
    @model_validator(mode="after")
    def check_channels(self) -> Self:
        """Check that channels have consistent wavelength_id and colors."""
        if self.channels is None:
            return self
        wavelength_ids = [ch.wavelength_id for ch in self.channels]
        colors = [ch.colors for ch in self.channels]
        if any(wavelength_ids) and not all(wavelength_ids):
            raise ValueError(
                "Either all or none of the channels must have wavelength_id."
            )
        if any(colors) and not all(colors):
            raise ValueError("Either all or none of the channels must have colors.")
        return self

    def channel_names_list(self) -> list[str] | None:
        """Convert channels to list of channel names."""
        if self.channels is None:
            return None
        return [ch.name for ch in self.channels]

    def wavelength_ids_list(self) -> list[str] | None:
        """Convert channels to list of wavelength IDs."""
        if self.channels is None:
            return None
        return [
            ch.wavelength_id for ch in self.channels if ch.wavelength_id is not None
        ]

    def colors_list(self) -> list[str] | None:
        """Convert channels to list of colors."""
        if self.channels is None:
            return None
        return [ch.colors for ch in self.channels if ch.colors is not None]

    def to_axes_list(self) -> list[CANONICAL_AXES_TYPE] | None:
        """Convert axes string to list of axes."""
        if self.axes is None:
            return None
        _axes = []
        for ax in self.axes:
            if ax not in canonical_axes:
                raise ValueError(f"Invalid axis '{ax}' in axes string.")
            _axes.append(ax)
        return _axes

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
            updated_details.channel_names = self.channel_names_list()
            updated_details.wavelength_ids = self.wavelength_ids_list()
            updated_details.colors = self.colors_list()
        if self.pixel_info is not None:
            updated_details.pixelsize = self.pixel_info.pixelsize
            updated_details.z_spacing = self.pixel_info.z_spacing
            updated_details.t_spacing = self.pixel_info.t_spacing
        axes = self.to_axes_list()
        if axes is not None:
            updated_details.axes = axes
        if self.data_type is not None:
            updated_details.data_type = self.data_type
        return updated_details


def converters_tools_models_list(
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
            "filters/_filter_pipeline.py",
            "WellFilter",
        ),
        (
            base,
            "filters/_filter_pipeline.py",
            "RegexIncludeFilter",
        ),
        (
            base,
            "filters/_filter_pipeline.py",
            "RegexExcludeFilter",
        ),
        (
            base,
            "models/_converter_options.py",
            "ConverterOptions",
        ),
        (
            base,
            "models/_converter_options.py",
            "StageCorrections",
        ),
        (
            base,
            "models/_converter_options.py",
            "AlignmentCorrections",
        ),
        (
            base,
            "models/_converter_options.py",
            "OmeZarrOptions",
        ),
        (
            base,
            "models/_converter_options.py",
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
    ]
