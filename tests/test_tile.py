import numpy as np
from ngio.ngff_meta.fractal_image_meta import PixelSize

from fractal_converters_tools.tile import Point, Tile, Vector


def test_tile():
    class DummyLoader:
        def __init__(self, shape):
            self.shape = shape

        def load(self):
            return np.zeros(self.shape, dtype="uint8")

        @property
        def dtype(self):
            return "uint8"

    tile1 = Tile(
        top_l=Point(0, 0),
        diag=Vector(1, 1, 1, 1, 1),
        pixel_size=PixelSize(x=0.1, y=0.1, z=1),
        data_loader=DummyLoader((1, 1, 1, 10, 10)),
    )

    assert np.allclose(tile1.areaXY(), 1)
    assert len(tile1.cornersXY()) == 4
    assert tile1.cornersXY() == [Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)]
    tile1_m = tile1.move_to(Point(1, 1))
    assert tile1_m.top_l == Point(1, 1)

    tile1_m = tile1.move_by(Vector(1, 1))
    assert tile1_m.top_l == Point(1, 1)

    assert tile1_m.origin == tile1.origin
    tile1_m = tile1_m.reset_origin()
    assert tile1_m.origin != tile1.origin

    assert tile1.shape == (1, 1, 1, 1, 1)
    assert tile1.to_pixel_space().shape == (1, 1, 1, 10, 10)
    assert tile1.dtype() == "uint8"
    assert tile1.load().shape == (1, 1, 1, 10, 10)

    tile1_ps = tile1.to_pixel_space()
    tile1_ps_rs = tile1_ps.to_real_space()
    assert tile1 == tile1_ps_rs

    tile2 = Tile(
        top_l=Point(0.01, 0.01),
        diag=Vector(1, 1, 1, 1, 1),
        pixel_size=PixelSize(x=0.1, y=0.1, z=1),
    )

    assert np.allclose(tile2.iouXY(tile1), 0.9609765663300323), tile2.iouXY(tile1)

    tile2 = Tile(
        top_l=Point(0.99, 0.99),
        diag=Vector(1, 1, 1, 1, 1),
        pixel_size=PixelSize(x=0.1, y=0.1, z=1),
    )

    assert np.allclose(tile2.iouXY(tile1), 5.000250012500634e-05), tile2.iouXY(tile1)

    assert isinstance(tile1.__repr__(), str)

    tile3 = Tile.from_points(
        Point(0, 0, 0, 0, 0), Point(1, 1, 1, 1, 1), PixelSize(x=0.1, y=0.1, z=1)
    )
    assert tile3 == tile1
