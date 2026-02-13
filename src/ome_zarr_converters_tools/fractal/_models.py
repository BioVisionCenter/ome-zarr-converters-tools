"""Models to be used with Fractal tasks API."""

from pydantic import BaseModel, Field

from ome_zarr_converters_tools.models._acquisition import (
    CANONICAL_AXES_TYPE,
    AcquisitionDetails,
    ChannelInfo,
    DataTypeEnum,
    StageCorrections,
    canonical_axes,
)
from ome_zarr_converters_tools.models._converter_options import (
    ConverterOptions,
    OverwriteMode,
)
from ome_zarr_converters_tools.pipelines._filters import ImplementedFilters


class ConvertParallelInitArgs(BaseModel):
    """Arguments for the compute task."""

    tiled_image_json_dump_url: str
    converter_options: ConverterOptions
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE


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
        condition_table_path: Optional path to a condition table CSV file.
        axes: Axes to use for the image data, e.g. "czyx".
        data_type: Data type of the image data.
        stage_corrections: Stage orientation corrections.
        filters: List of filters to apply.
    """

    channels: list[ChannelInfo] | None = None
    pixel_info: PixelSizeModel | None = Field(
        default=None, title="Pixel Size Information"
    )
    condition_table_path: str | None = None
    axes: str | None = None
    data_type: DataTypeEnum | None = Field(default=None, title="Data Type")
    stage_corrections: StageCorrections = Field(
        default_factory=StageCorrections, title="Stage Corrections"
    )
    filters: list[ImplementedFilters] = Field(default_factory=list)

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
            updated_details.channels = self.channels
        if self.pixel_info is not None:
            updated_details.pixelsize = self.pixel_info.pixelsize
            updated_details.z_spacing = self.pixel_info.z_spacing
            updated_details.t_spacing = self.pixel_info.t_spacing
        axes = self.to_axes_list()
        if axes is not None:
            updated_details.axes = axes
        if self.data_type is not None:
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
            "RegexIncludeFilter",
        ),
        (
            base,
            "pipelines/_filters.py",
            "RegexExcludeFilter",
        ),
        (
            base,
            "models/_converter_options.py",
            "ConverterOptions",
        ),
        (
            base,
            "models/_acquisition.py",
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
        (
            base,
            "models/_acquisition.py",
            "ChannelInfo",
        ),
    ]
