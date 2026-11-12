"""Build a dummy tile for testing purposes."""

from typing import Any, NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models import AcquisitionDetails, CollectionInterface
from ome_zarr_converters_tools.models._loader import ImageLoaderInterface


def rasterize_text_with_boundary(
    shape_x: int, shape_y: int, text: str, font_scale: float = 0.34
) -> np.ndarray:
    """Create a uint8 array with text rasterized and boundaries set to 255.

    Args:
        shape_x: The width of the output array.
        shape_y: The height of the output array.
        text: The string to rasterize.
        font_scale: Scale factor for font size relative to `min(shape_x, shape_y)`.
    """
    # Use PIL's built-in default font
    font_size = int(font_scale * min(shape_x, shape_y))
    font = ImageFont.load_default(size=font_size)

    # Render text at original size first
    temp_img = Image.new("L", (shape_x, shape_y), color=0)
    temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((shape_x // 2, shape_y // 2), text, fill=255, font=font, anchor="mm")
    arr = np.array(temp_img, dtype=np.uint8)
    arr = np.where(arr == 0, 1, arr)  # Make background 1
    arr[0, :] = 255
    arr[-1, :] = 255
    arr[:, 0] = 255
    arr[:, -1] = 255
    return arr


class StartPosition(NamedTuple):
    """Where a dummy tile starts: `x`, `y`, `z`, `c`, `t`.

    Interpreted like the matching `Tile.start_*` fields, i.e. in the spaces
    declared by the acquisition details passed to `build_dummy_tile`.
    """

    x: int | float = 0
    y: int | float = 0
    z: int | float = 0
    c: int = 0
    t: int | float = 0


class TileShape(NamedTuple):
    """Size of a dummy tile in pixels: `x`, `y`, `z`, `c`, `t`.

    Also the shape of the `uint8` array its loader returns: the FOV name
    rendered as text with a 255 border, broadcast over the other axes.
    """

    x: int = 256
    y: int = 256
    z: int = 1
    c: int = 1
    t: int = 1


class DummyLoader(ImageLoaderInterface):
    shape: TileShape
    text: str
    font_scale: float = 0.22

    def load_data(self, resource: Any | None = None) -> np.ndarray:
        """Load dummy image data as a NumPy array."""
        arr = rasterize_text_with_boundary(
            shape_x=self.shape.x,
            shape_y=self.shape.y,
            text=self.text,
            font_scale=self.font_scale,
        )
        shape = (self.shape.t, self.shape.c, self.shape.z, self.shape.y, self.shape.x)
        shape = tuple(int(s) for s in shape if s is not None)
        return np.broadcast_to(arr, shape)

    def find_data_type(self, resource: Any | None = None) -> str:
        """Find the data type of the image data."""
        return "uint8"


def build_dummy_tile(
    *,
    fov_name: str,
    start: StartPosition,
    shape: TileShape,
    collection: CollectionInterface,
    acquisition_details: AcquisitionDetails,
    font_scale: float = 0.22,
) -> Tile:
    """Build a tile whose loader renders `fov_name` as text, for tests.

    Args:
        fov_name: Name of the field of view; also the text drawn in the tile.
        start: Start position along each axis, in the spaces declared by
            `acquisition_details`.
        shape: Tile size in pixels along each axis.
        collection: Where the output image goes (`ImageInPlate` or
            `SingleImage`).
        acquisition_details: Shared acquisition details for the tile.
        font_scale: Font size relative to the shorter XY side.
    """
    return Tile(
        fov_name=fov_name,
        start_x=start.x,
        start_y=start.y,
        start_z=start.z,
        start_c=start.c,
        start_t=start.t,
        length_x=shape.x,
        length_y=shape.y,
        length_z=shape.z,
        length_c=shape.c,
        length_t=shape.t,
        collection=collection,
        image_loader=DummyLoader(shape=shape, text=fov_name, font_scale=font_scale),
        acquisition_details=acquisition_details,
    )
