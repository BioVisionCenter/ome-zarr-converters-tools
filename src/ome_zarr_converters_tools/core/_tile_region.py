"""Models for defining regions to be converted into OME-Zarr format."""

import math
from typing import Any, Generic, Self

import numpy as np
from ngio import PixelSize, Roi
from pydantic import BaseModel, ConfigDict, Field

from ome_zarr_converters_tools.core._roi_utils import (
    bulk_roi_union,
    move_roi_by,
    roi_to_point_distance,
    shape_from_rois,
)
from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models._acquisition import (
    CANONICAL_AXES_TYPE,
)
from ome_zarr_converters_tools.models._collection import (
    CollectionInterfaceType,
)
from ome_zarr_converters_tools.models._loader import (
    ImageLoaderInterfaceType,
)


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

    def load_data(
        self, *, axes: list[CANONICAL_AXES_TYPE], resource: Any | None = None
    ) -> np.ndarray:
        """Load the image data for this TileSlice using the image loader."""
        data = self.image_loader.load_data(resource=resource)
        # Padding data to match the ROI shape if necessary
        n_axes = len(axes)
        data_axes = data.ndim
        if data_axes > n_axes:
            raise ValueError("Data has more axes than expected.")
        if data_axes < n_axes:
            data = data.reshape((1,) * (n_axes - data_axes) + data.shape)
        return data


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

    def load_data(self, resource: Any | None = None) -> np.ndarray:
        """Load the full image data for this FOV group using the image loaders."""
        shape = self.shape()
        ref_slice = self.ref_slice()
        ref_data = ref_slice.load_data(axes=self.axes, resource=resource)
        full_image = np.zeros(shape, dtype=ref_data.dtype)

        group_roi = self.roi()
        # Find the offset between the group ROI and the origin ROI
        offset = {}
        for axis in self.axes:
            group_slice = group_roi.get(axis)
            assert group_slice is not None
            ref_slice_axis = ref_slice.roi.get(axis)
            assert ref_slice_axis is not None
            start = ref_slice_axis.start
            assert start is not None
            offset[axis] = -start

        for region in self.regions:
            roi_zeroed = move_roi_by(region.roi, offset)
            region_data = region.load_data(axes=self.axes, resource=resource)
            roi_slice = roi_zeroed.to_slicing_dict(pixel_size=self.pixel_size)
            slicing = []
            for axis in self.axes:
                _slice = roi_slice[axis]
                slicing.append(slice(math.floor(_slice.start), math.ceil(_slice.stop)))
            slicing = tuple(slicing)
            full_image[slicing] = region_data
        return full_image


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
    wavelength_ids: list[str | None] | None = None
    colors: list[str | None] | None = None
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
        if self.wavelength_ids != tile.wavelength_ids:
            raise ValueError(
                "Tile wavelength IDs do not match TiledImage wavelength IDs."
            )
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

    def load_data(self, resource: Any | None = None) -> np.ndarray:
        """Load the full image data for this TiledImage using the image loaders."""
        shape = self.shape()
        dtype = np.dtype(self.data_type)
        full_image = np.zeros(shape, dtype=dtype)

        for region in self.regions:
            region_data = region.load_data(axes=self.axes, resource=resource)
            roi_slice = region.roi.to_slicing_dict(pixel_size=self.pixel_size)
            slicing = []
            for axis in self.axes:
                _slice = roi_slice[axis]
                slicing.append(slice(math.floor(_slice.start), math.ceil(_slice.stop)))
            slicing = tuple(slicing)
            full_image[slicing] = region_data
        return full_image
