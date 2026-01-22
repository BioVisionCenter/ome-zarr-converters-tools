"""Models for defining regions to be converted into OME-Zarr format."""

from typing import Any, Generic

from ngio.common._roi import Roi, RoiSlice, pixel_to_world, world_to_pixel
from pydantic import BaseModel, ConfigDict, Field

from ome_zarr_converters_tools.models._acquisition import (
    CANONICAL_AXES_TYPE,
    COO_SYSTEM_TYPE,
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
    coo_system: COO_SYSTEM_TYPE,
    eps: float = 1e-6,
) -> float:
    """Convert from world to pixel and back to world to ensure alignment."""
    if coo_system == "world":
        pixel_coord = world_to_pixel(start, spacing)
        world_coord = pixel_to_world(pixel_coord, spacing)
        if abs(world_coord - start) > eps:
            raise ValueError(
                f"Coordinate {start}, with spacing {spacing}, "
                f"cannot be accurately represented in pixel coordinates."
            )
        return world_coord
    world_coord = pixel_to_world(start, spacing)
    pixel_coord = world_to_pixel(world_coord, spacing)
    if abs(pixel_coord - start) > eps:
        raise ValueError(
            f"Coordinate {start}, with spacing {spacing}, "
            f"cannot be accurately represented in world coordinates."
        )
    return world_coord


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
        start_c: Starting position in the C (channel) dimension.
        start_t: Starting position in the T (time) dimension.
        length_x: Length of the tile in the X dimension.
        length_y: Length of the tile in the Y dimension.
        length_z: Length of the tile in the Z dimension.
        length_c: Length of the tile in the C (channel) dimension.
        length_t: Length of the tile in the T (time) dimension.
        collection: Collection model defining how to build the path to the image(s).
        image_loader: Image loader model defining how to load the image data.
        start_x_coo: Coordinate system for the start_x value.
        start_y_coo: Coordinate system for the start_y value.
        start_z_coo: Coordinate system for the start_z value.
        start_t_coo: Coordinate system for the start_t value.
        length_x_coo: Coordinate system for the length_x value.
        length_y_coo: Coordinate system for the length_y value.
        length_z_coo: Coordinate system for the length_z value.
        length_t_coo: Coordinate system for the length_t value.
        pixelsize: Pixel size in micrometers.
        z_spacing: Z spacing in micrometers.
        t_spacing: T spacing in seconds.
        channel_names: List of channel names.
        wavelength_ids: List of wavelength IDs.
        colors: List of color information.
        axes: Axes order to be used for the data.
        data_type: Data type of the image data.
        flip_x: Whether to flip the tile in the X dimension.
        flip_y: Whether to flip the tile in the Y dimension.
        swap_xy: Whether to swap the X and Y dimensions.
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

    # Collection model defining how to build the path to the image(s)
    collection: CollectionInterfaceType
    # Image loader model defining how to load the image data
    # This model will need to wrap all the necessary context
    # to load the image data for this tile
    image_loader: ImageLoaderInterfaceType

    # Additional acquisition details
    # Coordinate system for start and length values
    start_x_coo: COO_SYSTEM_TYPE
    start_y_coo: COO_SYSTEM_TYPE
    start_z_coo: COO_SYSTEM_TYPE
    start_t_coo: COO_SYSTEM_TYPE
    length_x_coo: COO_SYSTEM_TYPE
    length_y_coo: COO_SYSTEM_TYPE
    length_z_coo: COO_SYSTEM_TYPE
    length_t_coo: COO_SYSTEM_TYPE

    # Spacing information
    pixelsize: float
    z_spacing: float
    t_spacing: float
    # Channel information
    channel_names: list[str] | None
    wavelength_ids: list[str] | None
    colors: list[str] | None

    # Axes order to be used for the data
    axes: list[CANONICAL_AXES_TYPE]

    # Data type of the image data
    data_type: str | None

    # Context from the converter options or from metadata
    flip_x: bool
    flip_y: bool
    swap_xy: bool

    attributes: dict[str, Any]
    model_config = ConfigDict(extra="forbid")

    def to_roi(self) -> Roi:
        """Convert the Tile to a Roi."""
        spacing = {
            "x": self.pixelsize,
            "y": self.pixelsize,
            "z": self.z_spacing,
            "t": self.t_spacing,
        }
        origins = {}
        roi_slices = {}
        for ax in self.axes:
            if ax == "x" and self.swap_xy:
                ax = "y"
            elif ax == "y" and self.swap_xy:
                ax = "x"

            start_field = f"start_{ax}"
            start = getattr(self, start_field)
            start_coo_system = getattr(self, f"{start_field}_coo", None)
            if start_coo_system is not None:
                start = safe_to_world(
                    start=start, spacing=spacing[ax], coo_system=start_coo_system
                )

            if ax == "x" and self.flip_x:
                start = -start
            if ax == "y" and self.flip_y:
                start = -start

            length_field = f"length_{ax}"
            length = getattr(self, length_field)
            length_coo_system = getattr(self, f"{length_field}_coo", None)
            if length_coo_system is not None:
                length = safe_to_world(
                    start=length, spacing=spacing[ax], coo_system=length_coo_system
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
        if self.data_type is not None:
            return self.data_type
        return self.image_loader.find_data_type(resource)
