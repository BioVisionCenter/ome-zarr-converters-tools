"""A module to represent an acquisition."""

from fractal_converters_tools.tiles import Tile


class ImageInWell:
    """A class to represent an acquisition."""

    def __init__(
        self,
        name: str,
        row: str,
        column: int,
        acquisition_id: int,
        tiles: list[Tile],
        channel_names: list[str] | None = None,
        num_levels: int = 5,
    ):
        """Initialize the acquisition."""
        self._name = name
        self._row = row
        self._column = column
        self._acquisition_id = acquisition_id
        self.tiles = tiles
        self._channel_names = channel_names
        self.num_levels = num_levels

    @property
    def acquisition_id(self) -> int:
        """Return the acquisition ID."""
        return self._acquisition_id

    @property
    def name(self) -> str:
        """Return the acquisition name."""
        return self._name

    @property
    def row(self) -> str:
        """Return the row."""
        return self._row

    @property
    def column(self) -> int:
        """Return the column."""
        return self._column

    @property
    def well_path(self) -> str:
        """Return the well path."""
        return f"{self._row}/{self._column}"

    @property
    def acquisition_path(self) -> str:
        """Return the acquisition path."""
        return f"{self.well_path}/{self._acquisition_id}"

    @property
    def channel_names(self) -> list[str]:
        """Return the channel names."""
        if self._channel_names is None:
            tile_channel_names = self.tiles[0].channel_names
            if tile_channel_names is None:
                raise ValueError(
                    "Channel names not found. " "Please provide channel names manually."
                )
            self._channel_names = tile_channel_names
        return self._channel_names

    @property
    def xy_scale(self) -> float:
        """Return the XY scale."""
        return self.tiles[0].xy_scale

    @property
    def z_scale(self) -> float:
        """Return the Z scale."""
        return self.tiles[0].z_scale
