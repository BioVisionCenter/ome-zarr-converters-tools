"""Models for defining regions to be converted into OME-Zarr format."""

from typing import Any, Generic, TypeAlias

from ngio.common._roi import Roi, RoiSlice, pixel_to_world, world_to_pixel
from pydantic import BaseModel, ConfigDict, Field

from ome_zarr_converters_tools.models._acquisition import (
    COO_SYSTEM_TYPE,
    AcquisitionDetails,
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
        start_c: Starting position in the C (channel) dimension.
        start_t: Starting position in the T (time) dimension.
        length_x: Length of the tile in the X dimension.
        length_y: Length of the tile in the Y dimension.
        length_z: Length of the tile in the Z dimension.
        length_c: Length of the tile in the C (channel) dimension.
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

    def to_roi(self) -> Roi:
        """Convert the Tile to a Roi."""
        acquisition_details = self.acquisition_details
        stage_corrections = acquisition_details.stage_corrections
        spacing = {
            "x": acquisition_details.pixelsize,
            "y": acquisition_details.pixelsize,
            "z": acquisition_details.z_spacing,
            "t": acquisition_details.t_spacing,
        }
        origins = {}
        roi_slices = {}
        for ax in acquisition_details.axes:
            if ax == "x" and stage_corrections.swap_xy:
                ax = "y"
            elif ax == "y" and stage_corrections.swap_xy:
                ax = "x"

            start_field = f"start_{ax}"
            start = getattr(self, start_field)
            start_coo_system = getattr(acquisition_details, f"{start_field}_coo", None)
            if start_coo_system is not None:
                start = safe_to_world(
                    start=start, spacing=spacing[ax], coo_system=start_coo_system
                )

            if ax == "x" and stage_corrections.flip_x:
                start = -start
            if ax == "y" and stage_corrections.flip_y:
                start = -start

            length_field = f"length_{ax}"
            length = getattr(self, length_field)
            length_coo_system = getattr(
                acquisition_details, f"{length_field}_coo", None
            )
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
        if self.acquisition_details.data_type is not None:
            return self.acquisition_details.data_type
        return self.image_loader.find_data_type(resource)
