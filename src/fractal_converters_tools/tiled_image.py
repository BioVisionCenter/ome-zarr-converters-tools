"""A module to represent an acquisition."""

from typing import Protocol

from ngio.ngff_meta.fractal_image_meta import PixelSize

from fractal_converters_tools.tile import Tile


class PathBuilder(Protocol):
    """A protocol to build paths."""

    @property
    def path(self) -> str:
        """Return the path the relative path for the tiled image."""
        ...


def _zarrify_path(path: str) -> str:
    """Return the zarrified path."""
    if not path.endswith(".zarr"):
        return f"{path}.zarr"
    return path


class SimplePathBuilder:
    """A class to build simple paths."""

    def __init__(self, path: str):
        """Initialize the path builder."""
        self._path = path

    @property
    def path(self) -> str:
        """Return the zarr relative path."""
        return _zarrify_path(self._path)


class PlatePathBuilder:
    """A class to build paths for a standard plate."""

    def __init__(self, plate_name: str, row: str, column: int, acquisition_id: int):
        """Initialize the path builder."""
        self._plate_name = plate_name
        self._row = row
        self._column = column
        self._acquisition_id = acquisition_id

    @property
    def plate_name(self) -> str:
        """Return the plate name."""
        return self._plate_name

    @property
    def row(self) -> str:
        """Return the row."""
        return self._row

    @property
    def column(self) -> int:
        """Return the column."""
        return self._column

    @property
    def well_id(self) -> str:
        """Return the well ID {row}/{column}."""
        return f"{self.row}/{self.column}"

    @property
    def acquisition_id(self) -> int:
        """Return the acquisition ID."""
        return self._acquisition_id

    @property
    def plate_path(self) -> str:
        """Return the relative plate path."""
        return _zarrify_path(self.plate_name)

    @property
    def well_path(self) -> str:
        """Return the relative well path."""
        return f"{self.plate_path}/{self.well_id}"

    @property
    def path(self) -> str:
        """Return the image relative path."""
        return f"{self.well_path}/{self.acquisition_id}"


class TiledImage:
    """A class to represent an acquisition."""

    def __init__(
        self,
        name: str,
        path_builder: PathBuilder,
        channel_names: list[str] | None = None,
        wavelength_ids: list[int] | None = None,
        num_levels: int = 5,
    ):
        """Initialize the acquisition."""
        self._name = name
        self._path_builder = path_builder
        self._tiles = []

        self._channel_names = channel_names
        self._wavelength_ids = wavelength_ids
        self._num_levels = num_levels

    @property
    def tiles(self) -> list[Tile]:
        """Return the tiles."""
        return self._tiles

    def add_tile(self, Tile):
        """Add a tile to the acquisition."""
        self._tiles.append(Tile)

    @property
    def path_builder(self) -> PathBuilder:
        """Return the path builder."""
        return self._path_builder

    @property
    def path(self) -> str:
        """Return the zarr relative path."""
        return self.path_builder.path

    @property
    def channel_names(self) -> list[str] | None:
        """Return the channel names."""
        return self._channel_names

    @property
    def wavelength_ids(self) -> list[int] | None:
        """Return the wavelength ids."""
        return self._wavelength_ids

    @property
    def pixel_size(self) -> PixelSize | None:
        """Return the pixel size."""
        if len(self.tiles) == 0:
            return None
        return self.tiles[0].pixel_size

    @property
    def num_levels(self) -> int:
        """Return the number of levels."""
        return self._num_levels
