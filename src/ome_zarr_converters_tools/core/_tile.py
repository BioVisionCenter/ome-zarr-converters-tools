"""Models for defining regions to be converted into OME-Zarr format."""

from typing import Any, Generic, TypeAlias

from ngio.common._roi import Roi, RoiSlice, pixel_to_world, world_to_pixel
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ome_zarr_converters_tools.models._acquisition import (
    SPACE_TYPE,
    AcquisitionDetails,
    DataTypeEnum,
)
from ome_zarr_converters_tools.models._collection import (
    CollectionInterfaceType,
)
from ome_zarr_converters_tools.models._loader import (
    ImageLoaderInterfaceType,
)


def safe_to_world(
    *,
    start: float,
    spacing: float,
    space: SPACE_TYPE,
) -> float:
    """Convert coordinates to world space, normalizing through pixel grid."""
    if space == "world":
        pixel_coord = world_to_pixel(start, spacing)
        return pixel_to_world(pixel_coord, spacing)
    return pixel_to_world(start, spacing)


AttributeType: TypeAlias = (
    list[str | None] | list[int | None] | list[float | None] | list[bool | None]
)


class Tile(BaseModel, Generic[CollectionInterfaceType, ImageLoaderInterfaceType]):
    """A tile representing a region of an image to be converted.

    This model is a complete definition of a tile, including its position,
    size, how to load the image data, and additional metadata. This model is the
    basic entry point for defining what regions of an acquisition to convert.

    Attributes:
        fov_name: Name of the field of view (FOV) this tile belongs to.
        start_x: Starting position in the X dimension.
        start_y: Starting position in the Y dimension.
        start_z: Starting position in the Z dimension.
        start_c: Starting position in the C (channel) dimension. Channel indices
            index into `acquisition_details.channels` when channel metadata is
            provided, so `start_c + length_c` must not exceed its length.
        start_t: Starting position in the T (time) dimension.
        length_x: Length of the tile in the X dimension.
        length_y: Length of the tile in the Y dimension.
        length_z: Length of the tile in the Z dimension.
        length_c: Length of the tile in the C (channel) dimension. Values above 1
            describe a tile that carries several channels at once (e.g. a
            two-camera acquisition); the channels it covers must be *adjacent* in
            `acquisition_details.channels`, since the span is the contiguous
            range `start_c … start_c + length_c - 1`. Channels acquired together
            but not adjacent in that list must be emitted as separate tiles.
        length_t: Length of the tile in the T (time) dimension.
        collection: Collection model defining how to build the path to the image(s).
        image_loader: Image loader model defining how to load the image data.
        acquisition_details: Acquisition specific details that will be used to validate
            and convert the tile.
        attributes: Additional attributes for the these will be passed to
            the fractal image list as key-value pairs.

    """

    fov_name: str
    # Positions
    start_x: float
    start_y: float
    start_z: float = 0.0
    start_c: int = 0
    start_t: float = 0.0

    # Sizes
    length_x: float = Field(gt=0)
    length_y: float = Field(gt=0)
    length_z: float = Field(default=1.0, gt=0)
    length_c: int = Field(default=1, gt=0)
    length_t: float = Field(default=1.0, gt=0)

    # Additional attribute for the tile
    attributes: dict[str, AttributeType] = Field(default_factory=dict)
    # Collection model defining how to build the path to the image(s)
    collection: CollectionInterfaceType
    # Image loader model defining how to load the image data
    # This model will need to wrap all the necessary context
    # to load the image data for this tile
    image_loader: ImageLoaderInterfaceType
    # Acquisition specific details that will be used to validate and convert
    # the tile
    acquisition_details: AcquisitionDetails

    # Pydantic configuration
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_channel_range(self) -> "Tile":
        """Enforce that channel indices resolve into `acquisition_details.channels`."""
        if self.start_c < 0:
            raise ValueError(
                f"Tile '{self.fov_name}' has start_c={self.start_c}; "
                "channel indices must be >= 0."
            )
        channels = self.acquisition_details.channels
        if channels is not None and self.start_c + self.length_c > len(channels):
            max_index = self.start_c + self.length_c - 1
            raise ValueError(
                f"Tile '{self.fov_name}' references channel index {max_index} but "
                f"acquisition_details.channels has {len(channels)} entries. Provide "
                "one ChannelInfo per channel index (padding unused instrument slots "
                'with e.g. ChannelInfo(channel_label="unused_1")), or set '
                "channels=None to use auto-generated channel names."
            )
        return self

    def to_roi(self) -> Roi:
        """Convert the Tile to a Roi."""
        acquisition_details = self.acquisition_details
        stage_corrections = acquisition_details.stage_orientation
        spacing = {
            "x": acquisition_details.xy_pixel_size,
            "y": acquisition_details.xy_pixel_size,
            "z": acquisition_details.z_spacing,
            "t": acquisition_details.t_spacing,
        }
        origins = {}
        roi_slices = {}
        for ax in acquisition_details.axes:
            # `swap_xy` transposes the X and Y stage axes: the output x slice is
            # built from this tile's y position and vice versa. The output axis
            # label (`axis_name`) stays `ax`; only the source field is swapped.
            source_ax = ax
            if stage_corrections.swap_xy and ax == "x":
                source_ax = "y"
            elif stage_corrections.swap_xy and ax == "y":
                source_ax = "x"

            start_field = f"start_{source_ax}"
            start = getattr(self, start_field)
            start_space = getattr(acquisition_details, f"{start_field}_space", None)
            if start_space is not None:
                start = safe_to_world(
                    start=start,
                    spacing=spacing[source_ax],
                    space=start_space,
                )

            if ax == "x" and stage_corrections.flip_x:
                start = -start
            if ax == "y" and stage_corrections.flip_y:
                start = -start

            length_field = f"length_{source_ax}"
            length = getattr(self, length_field)
            length_space = getattr(acquisition_details, f"{length_field}_space", None)
            if length_space is not None:
                length = safe_to_world(
                    start=length,
                    spacing=spacing[source_ax],
                    space=length_space,
                )
            roi_slices[ax] = RoiSlice(start=start, length=length, axis_name=ax)
            if ax in ["x", "y", "z"]:
                origins[f"{ax}_micrometer_original"] = start

        return Roi(
            name=self.fov_name,
            slices=list(roi_slices.values()),
            space="world",
            **origins,
        )

    def find_data_type(self, resource: Any | None = None) -> str:
        """Find the data type of the image data."""
        if self.acquisition_details.data_type != DataTypeEnum.AUTODETECT:
            return self.acquisition_details.data_type
        return self.image_loader.find_data_type(resource)
