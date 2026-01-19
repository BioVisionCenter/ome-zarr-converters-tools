"""Models for defining regions to be converted into OME-Zarr format."""

from typing import Any, Generic, Literal, NamedTuple, Self

import numpy as np
from ngio import PixelSize, Roi
from pydantic import BaseModel, ConfigDict, Field

from ome_zarr_converters_tools.models._acquisition import (
    ContextModel,
)
from ome_zarr_converters_tools.models._collection import CollectionInterfaceType
from ome_zarr_converters_tools.models._loader import (
    ImageLoaderInterfaceType,
)
from ome_zarr_converters_tools.models._roi_utils import (
    bulk_roi_union,
    roi_to_point_distance,
    shape_from_rois,
)
from ome_zarr_converters_tools.models._tile import Tile

CANONICAL_AXES_TYPE = Literal["t", "c", "z", "y", "x"]
COO_TYPE = Literal["world", "pixel"]


class TileSlice(BaseModel, Generic[ImageLoaderInterfaceType]):
    """The smallest unit of a tiled image.

    Usually corresponds to the minimal unit in which the source data
    can be loaded (e.g., a single tiff file from the microscope).

    """

    roi: Roi
    image_loader: ImageLoaderInterfaceType
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_tile(cls, tile: Tile) -> Self:
        """Create a TileSlice from a Tile."""
        return cls(
            roi=tile.to_roi(),
            # collection=tile.collection,
            image_loader=tile.image_loader,
        )

    def load_data(self, resource: Any) -> np.ndarray:
        """Load the image data for this TileSlice using the image loader."""
        return self.image_loader.load_data(resource=resource)


class TileFOVGroup(BaseModel, Generic[ImageLoaderInterfaceType]):
    """Group of TileSlices belonging to the same acquisition FOV."""

    fov_name: str
    regions: list[TileSlice[ImageLoaderInterfaceType]] = Field(default_factory=list)
    axes: list[CANONICAL_AXES_TYPE]
    pixel_size: PixelSize

    model_config = ConfigDict(extra="forbid")

    def shape(self) -> tuple[int, ...]:
        """Get the shape of the FOV group by computing the union of all regions."""
        return shape_from_rois(
            [region.roi for region in self.regions],
            self.axes,
            self.pixel_size,
        )

    def roi(self) -> Roi:
        """Get the global ROI covering all TileSlices in the FOV group."""
        union_roi = bulk_roi_union([region.roi for region in self.regions])
        union_roi.name = self.fov_name
        return union_roi

    def ref_slice(self) -> TileSlice[ImageLoaderInterfaceType]:
        """Get a reference TileSlice for this FOV group."""
        point = {}
        for axis in self.axes:
            point[axis] = 0.0

        ref_region = self.regions[0]
        ref_distance = roi_to_point_distance(ref_region.roi, point)
        for region in self.regions[1:]:
            distance = roi_to_point_distance(region.roi, point)
            if distance < ref_distance:
                ref_region = region
                ref_distance = distance
        return ref_region


class TiledImage(BaseModel, Generic[CollectionInterfaceType, ImageLoaderInterfaceType]):
    """A TiledImage is the unit that will be converted into an OME-Zarr image.

    Can contain multiple TileFOVGroups, each containing multiple TileSlices
    or it can directly contain a single TileFOVGroup.
    """

    regions: list[TileSlice[ImageLoaderInterfaceType]] = Field(default_factory=list)
    path: str
    name: str | None = None
    pixelsize: float = 1.0
    z_spacing: float = 1.0
    t_spacing: float = 1.0
    data_type: str
    channel_names: list[str] | None = None
    wavelengths: list[float] | None = None
    axes: list[CANONICAL_AXES_TYPE]
    collection: CollectionInterfaceType
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def group_by_fov(self) -> list[TileFOVGroup[ImageLoaderInterfaceType]]:
        """Group TileSlices by field of view name."""
        fov_dict: dict[str, list[TileSlice]] = {}
        for region in self.regions:
            fov_name = region.roi.name
            if fov_name is None:
                raise ValueError("TileSlice ROI must have a name to group by FOV.")
            if fov_name not in fov_dict:
                fov_dict[fov_name] = []
            fov_dict[fov_name].append(region)
        return [
            TileFOVGroup(
                fov_name=fov_name,
                regions=regions,
                axes=self.axes,
                pixel_size=self.pixel_size,
            )
            for fov_name, regions in fov_dict.items()
        ]

    @property
    def pixel_size(self) -> PixelSize:
        """Return the PixelSize of the TiledImage."""
        return PixelSize(
            x=self.pixelsize,
            y=self.pixelsize,
            z=self.z_spacing,
            t=self.t_spacing,
        )

    def add_tile(self, tile: Tile) -> None:
        """Add a Tile to the TiledImage as a TileRegion."""
        if self.channel_names != tile.channel_names:
            raise ValueError(
                "Tile channel names do not match TiledImage channel names."
            )
        if self.wavelengths != tile.wavelengths:
            raise ValueError("Tile wavelengths do not match TiledImage wavelengths.")
        if self.axes != tile.axes:
            raise ValueError("Tile axes do not match TiledImage axes.")
        if self.pixelsize != tile.pixelsize:
            raise ValueError("Tile pixelsize does not match TiledImage pixelsize.")
        if self.z_spacing != tile.z_spacing:
            raise ValueError("Tile z_spacing does not match TiledImage z_spacing.")
        if self.t_spacing != tile.t_spacing:
            raise ValueError("Tile t_spacing does not match TiledImage t_spacing.")
        tile_region = TileSlice.from_tile(tile)
        self.regions.append(tile_region)

    def shape(self) -> tuple[int, ...]:
        """Get the shape of the TiledImage by computing the union of all regions."""
        return shape_from_rois(
            [region.roi for region in self.regions],
            self.axes,
            self.pixel_size,
        )

    def roi(self) -> Roi:
        """Get the global ROI covering all TileSlices in the TiledImage."""
        union_roi = bulk_roi_union([region.roi for region in self.regions])
        union_roi.name = self.name or self.path
        return union_roi


class TiledImageWithContext(NamedTuple):
    """A TiledImage along with its associated ContextModel."""

    tiled_image: TiledImage
    context: ContextModel
