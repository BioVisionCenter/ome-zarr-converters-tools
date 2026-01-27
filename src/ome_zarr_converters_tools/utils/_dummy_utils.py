"""Build a dummy tile for testing purposes."""

from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ome_zarr_converters_tools.core._tile import Tile


def rasterize_text_with_boundary(shape_x, shape_y, text, font_scale=0.34):
    """Create a uint8 array with text rasterized and boundaries set to 255.

    Args:
        shape_x (int): The width of the output array
        shape_y (int): The height of the output array
        text (str): The string to rasterize
        font_scale (float): Scale factor for font size relative to min(shape_x, shape_y)
    """
    # Use PIL's built-in default font
    font_size = int(font_scale * min(shape_x, shape_y))
    font = ImageFont.load_default(size=font_size)

    # Render text at original size first
    temp_img = Image.new("L", (shape_x, shape_y), color=0)
    temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((shape_x // 2, shape_y // 2), text, fill=255, font=font, anchor="mm")
    arr = np.array(temp_img)
    arr[0, :] = 255
    arr[-1, :] = 255
    arr[:, 0] = 255
    arr[:, -1] = 255
    return arr


class StartPosition(NamedTuple):
    x: int | float = 0
    y: int | float = 0
    z: int | float = 0
    c: int = 0
    t: int | float = 0


class TileShape(NamedTuple):
    x: int = 256
    y: int = 256
    z: int | None = 1
    c: int | None = 1
    t: int | None = 1


class DummyLoader:
    def __init__(self, shape: TileShape, text, font_scale=0.22):
        self.shape = shape
        self.text = text
        self.font_scale = font_scale

    def load_data(self, resource: None = None) -> np.ndarray:
        """Load dummy image data as a NumPy array."""
        arr = rasterize_text_with_boundary(
            shape_x=self.shape.x, shape_y=self.shape.y, text="DUMMY"
        )
        shape = (self.shape.t, self.shape.c, self.shape.z, self.shape.y, self.shape.x)
        shape = tuple(int(s) for s in shape if s is not None)
        return np.broadcast_to(arr, shape)


def build_dummy_tile(
    fov_name: str,
    start: StartPosition,
    shape: TileShape,
    collection=None,
    context=None,
    font_scale=0.22,
) -> Tile:
    """Build a dummy tile with default parameters, allowing overrides."""
    return Tile.model_validate(
        {
            "fov_name": fov_name,
            "start_x": start.x,
            "start_y": start.y,
            "start_z": start.z,
            "start_c": start.c,
            "start_t": start.t,
            "length_x": shape.x,
            "length_y": shape.y,
            "length_z": shape.z,
            "length_c": shape.c,
            "length_t": shape.t,
            "collection": collection,
            "image_loader": DummyLoader(
                shape=shape, text=fov_name, font_scale=font_scale
            ),
        },
        context=context,
    )
