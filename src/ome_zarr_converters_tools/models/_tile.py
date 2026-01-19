"""Models for defining regions to be converted into OME-Zarr format."""

from typing import Any, Generic, Literal

from ngio.common._roi import Roi, RoiSlice, pixel_to_world, world_to_pixel
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ome_zarr_converters_tools.models._acquisition import (
    AcquisitionDetails,
)
from ome_zarr_converters_tools.models._collection import (
    CollectionInterfaceType,
)
from ome_zarr_converters_tools.models._loader import (
    ImageLoaderInterfaceType,
)

CANONICAL_AXES_TYPE = Literal["t", "c", "z", "y", "x"]
COO_TYPE = Literal["world", "pixel"]


def safe_to_world(
    start: float,
    spacing: float,
    coo_system: COO_TYPE,
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
    # Image loader
    image_loader: ImageLoaderInterfaceType

    # Additional acquisition details
    # This if not provided, will be filled from context
    # Coordinate system for start and length values
    start_x_coo: COO_TYPE
    start_y_coo: COO_TYPE
    start_z_coo: COO_TYPE
    start_t_coo: COO_TYPE
    length_x_coo: COO_TYPE
    length_y_coo: COO_TYPE
    length_z_coo: COO_TYPE
    length_t_coo: COO_TYPE

    pixelsize: float
    z_spacing: float
    t_spacing: float
    channel_names: list[str]
    wavelengths: list[float]
    axes: list[CANONICAL_AXES_TYPE]

    # Context from the converter options
    # Stage corrections
    flip_x: bool
    flip_y: bool
    swap_xy: bool

    model_config = ConfigDict(extra="allow")

    @field_validator("channel_names", "wavelengths", mode="before")
    def split_channel_names(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [name.strip() for name in v.split("/")]
        return v

    @model_validator(mode="before")
    def fill_from_context(cls, data: dict[str, Any], info: Any) -> dict[str, Any]:
        """Fill missing fields from context acquisition details."""
        if info.context is None:
            return data

        acq_details: AcquisitionDetails = info.context.acquisition_details
        acq_fields = AcquisitionDetails.model_fields.keys()
        self_fields = Tile.model_fields.keys()
        for field in acq_fields:
            if field in self_fields and field not in data:
                data[field] = getattr(acq_details, field)

        stage_corrections = info.context.converter_options.stage_correction
        if "flip_x" not in data:
            data["flip_x"] = stage_corrections.flip_x
        if "flip_y" not in data:
            data["flip_y"] = stage_corrections.flip_y
        if "swap_xy" not in data:
            data["swap_xy"] = stage_corrections.swap_xy
        return data

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
                start = safe_to_world(start, spacing[ax], start_coo_system)

            if ax == "x" and self.flip_x:
                start = -start
            if ax == "y" and self.flip_y:
                start = -start

            length_field = f"length_{ax}"
            length = getattr(self, length_field)
            length_coo_system = getattr(self, f"{length_field}_coo", None)
            if length_coo_system is not None:
                length = safe_to_world(length, spacing[ax], length_coo_system)
            roi_slices[ax] = RoiSlice(start=start, length=length, axis_name=ax)
            if ax in ["x", "y", "z"]:
                origins[f"{ax}_micrometer_original"] = start

        return Roi(
            name=self.fov_name,
            slices=list(roi_slices.values()),
            space="world",
            **origins,
        )
