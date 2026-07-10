import re
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, ParamSpec, Protocol

from pydantic import BaseModel, Field

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models._collection import ImageInPlate
from ome_zarr_converters_tools.pipelines._registry import Registry


class FilterModel(BaseModel):
    name: Any


class RegexIncludeFilter(FilterModel):
    """Regex include filter model."""

    name: Literal["Path Regex Include Filter"] = "Path Regex Include Filter"
    """Name of the filter."""
    regex: str
    """
    Regex pattern to include. If the tile's base path matches this regex,
    it will be included, otherwise it will be excluded.
    """


def _regex_bases_match(tile: Tile, regex: str) -> bool:
    base_path = tile.collection.path()
    if re.search(regex, base_path):
        return True
    return False


def apply_path_include_regex_filter(
    tile: Tile, filter_params: RegexIncludeFilter
) -> bool:
    return _regex_bases_match(tile, filter_params.regex)


class RegexExcludeFilter(FilterModel):
    """Regex exclude filter model."""

    name: Literal["Path Regex Exclude Filter"] = "Path Regex Exclude Filter"
    """Name of the filter."""
    regex: str
    """
    Regex pattern to exclude. If the tile's base path matches this regex,
    it will be excluded, otherwise it will be included.
    """


def apply_path_exclude_regex_filter(
    tile: Tile, filter_params: RegexExcludeFilter
) -> bool:
    return not _regex_bases_match(tile, filter_params.regex)


class WellExcludeFilter(FilterModel):
    """Well exclude filter model."""

    name: Literal["Well Exclude Filter"] = "Well Exclude Filter"
    """Name of the filter."""
    wells_to_remove: list[str]
    """List of well identifiers to remove. E.g., ["A01", "B02"]"""


def apply_well_filter(tile: Tile, filter_params: WellExcludeFilter) -> bool:
    if not isinstance(tile.collection, ImageInPlate):
        raise ValueError(
            "Well filter can only be applied to a tile with ImageInPlate collection."
        )
    if tile.collection.well in filter_params.wells_to_remove:
        return False
    return True


class WellIncludeFilter(FilterModel):
    """Well include filter model."""

    name: Literal["Well Include Filter"] = "Well Include Filter"
    """Name of the filter."""
    wells_to_include: list[str]
    """List of well identifiers to keep. E.g., ["A01", "B02"]"""


def apply_well_include_filter(tile: Tile, filter_params: WellIncludeFilter) -> bool:
    if not isinstance(tile.collection, ImageInPlate):
        raise ValueError(
            "Well include filter can only be applied to a tile with"
            " ImageInPlate collection."
        )
    return tile.collection.well in filter_params.wells_to_include


class FovNameIncludeFilter(FilterModel):
    """FOV name include filter model."""

    name: Literal["FOV Name Include Filter"] = "FOV Name Include Filter"
    """Name of the filter."""
    regex: str
    """
    Regex pattern to include. If the tile's `fov_name` matches this regex,
    it will be included, otherwise it will be excluded.
    """


def apply_fov_name_include_filter(
    tile: Tile, filter_params: FovNameIncludeFilter
) -> bool:
    return re.search(filter_params.regex, tile.fov_name) is not None


class FovNameExcludeFilter(FilterModel):
    """FOV name exclude filter model."""

    name: Literal["FOV Name Exclude Filter"] = "FOV Name Exclude Filter"
    """Name of the filter."""
    regex: str
    """
    Regex pattern to exclude. If the tile's `fov_name` matches this regex,
    it will be excluded, otherwise it will be included.
    """


def apply_fov_name_exclude_filter(
    tile: Tile, filter_params: FovNameExcludeFilter
) -> bool:
    return re.search(filter_params.regex, tile.fov_name) is None


def _plate_acquisition(tile: Tile, filter_name: str) -> int:
    if not isinstance(tile.collection, ImageInPlate):
        raise ValueError(
            f"{filter_name} can only be applied to a tile with ImageInPlate collection."
        )
    return tile.collection.acquisition


class AcquisitionIncludeFilter(FilterModel):
    """Acquisition include filter model."""

    name: Literal["Acquisition Include Filter"] = "Acquisition Include Filter"
    """Name of the filter."""
    acquisitions: list[int]
    """List of acquisition indices to keep. E.g., [0, 1]"""


def apply_acquisition_include_filter(
    tile: Tile, filter_params: AcquisitionIncludeFilter
) -> bool:
    acquisition = _plate_acquisition(tile, filter_params.name)
    return acquisition in filter_params.acquisitions


class AcquisitionExcludeFilter(FilterModel):
    """Acquisition exclude filter model."""

    name: Literal["Acquisition Exclude Filter"] = "Acquisition Exclude Filter"
    """Name of the filter."""
    acquisitions: list[int]
    """List of acquisition indices to remove. E.g., [0, 1]"""


def apply_acquisition_exclude_filter(
    tile: Tile, filter_params: AcquisitionExcludeFilter
) -> bool:
    acquisition = _plate_acquisition(tile, filter_params.name)
    return acquisition not in filter_params.acquisitions


def _attribute_matches(
    tile: Tile, key: str, values: list[str | int | float | bool], filter_name: str
) -> bool:
    if key not in tile.attributes:
        available = sorted(tile.attributes)
        raise ValueError(
            f"{filter_name} references attribute '{key}' but tile "
            f"'{tile.fov_name}' has no such attribute; available attributes: "
            f"{available}. Fix the filter key or make the parser set the "
            "attribute on every tile."
        )
    return any(value in values for value in tile.attributes[key])


class AttributeIncludeFilter(FilterModel):
    """Attribute include filter model."""

    name: Literal["Attribute Include Filter"] = "Attribute Include Filter"
    """Name of the filter."""
    key: str
    """Attribute key to match. The attribute must be present on every tile."""
    values: list[str | int | float | bool]
    """Values to match against. A tile is included if any element of its
    attribute value matches one of these."""


def apply_attribute_include_filter(
    tile: Tile, filter_params: AttributeIncludeFilter
) -> bool:
    return _attribute_matches(
        tile, filter_params.key, filter_params.values, filter_params.name
    )


class AttributeExcludeFilter(FilterModel):
    """Attribute exclude filter model."""

    name: Literal["Attribute Exclude Filter"] = "Attribute Exclude Filter"
    """Name of the filter."""
    key: str
    """Attribute key to match. The attribute must be present on every tile."""
    values: list[str | int | float | bool]
    """Values to match against. A tile is excluded if any element of its
    attribute value matches one of these."""


def apply_attribute_exclude_filter(
    tile: Tile, filter_params: AttributeExcludeFilter
) -> bool:
    return not _attribute_matches(
        tile, filter_params.key, filter_params.values, filter_params.name
    )


def _channel_labels_match(
    tile: Tile, channel_labels: list[str], filter_name: str
) -> bool:
    """Return whether the tile's channels match `channel_labels`.

    Raises on missing channel metadata or on a partial match of a
    multi-channel tile: a filter can only drop whole tiles, so a partial
    match cannot be honored either way without silently losing or keeping
    channels.
    """
    channels = tile.acquisition_details.channels
    if channels is None:
        raise ValueError(
            f"{filter_name} requires channel metadata, but tile "
            f"'{tile.fov_name}' has acquisition_details.channels=None. "
            "Provide ChannelInfo entries in AcquisitionDetails to use "
            "channel filters."
        )
    tile_labels = [
        channels[i].channel_label
        for i in range(tile.start_c, tile.start_c + tile.length_c)
    ]
    matches = [label in channel_labels for label in tile_labels]
    if all(matches):
        return True
    if not any(matches):
        return False
    matched = [label for label, m in zip(tile_labels, matches, strict=True) if m]
    raise ValueError(
        f"{filter_name} matches channels {matched} of tile "
        f"'{tile.fov_name}', which spans channels {tile_labels}. Filters can "
        "only drop whole tiles, never crop them; split tiles per channel in "
        "the parser or adjust the filter to match all or none of the tile's "
        "channels."
    )


class ChannelIncludeFilter(FilterModel):
    """Channel include filter model."""

    name: Literal["Channel Include Filter"] = "Channel Include Filter"
    """Name of the filter."""
    channel_labels: list[str]
    """Channel labels to keep. E.g., ["DAPI", "GFP"]. A tile is kept if all
    of its channels are in this list; a partial match raises."""


def apply_channel_include_filter(
    tile: Tile, filter_params: ChannelIncludeFilter
) -> bool:
    return _channel_labels_match(tile, filter_params.channel_labels, filter_params.name)


class ChannelExcludeFilter(FilterModel):
    """Channel exclude filter model."""

    name: Literal["Channel Exclude Filter"] = "Channel Exclude Filter"
    """Name of the filter."""
    channel_labels: list[str]
    """Channel labels to remove. E.g., ["DAPI", "GFP"]. A tile is removed if
    all of its channels are in this list; a partial match raises."""


def apply_channel_exclude_filter(
    tile: Tile, filter_params: ChannelExcludeFilter
) -> bool:
    return not _channel_labels_match(
        tile, filter_params.channel_labels, filter_params.name
    )


class ZRangeFilter(FilterModel):
    """Z range filter model.

    Note:
        A tile is judged by its `start_z` only: tiles are dropped whole,
        never cropped, so a single tile spanning the full Z stack passes
        whenever its start is in range. Bounds are compared in the same
        coordinate space the tile's `start_z` is defined in
        (`acquisition_details.start_z_space`).
    """

    name: Literal["Z Range Filter"] = "Z Range Filter"
    """Name of the filter."""
    min_z: float | None = None
    """Minimum `start_z` (inclusive). `None` means unbounded."""
    max_z: float | None = None
    """Maximum `start_z` (inclusive). `None` means unbounded."""


def apply_z_range_filter(tile: Tile, filter_params: ZRangeFilter) -> bool:
    if filter_params.min_z is not None and tile.start_z < filter_params.min_z:
        return False
    if filter_params.max_z is not None and tile.start_z > filter_params.max_z:
        return False
    return True


class TRangeFilter(FilterModel):
    """Time range filter model.

    Note:
        A tile is judged by its `start_t` only: tiles are dropped whole,
        never cropped, so a single tile spanning the full time series passes
        whenever its start is in range. Bounds are compared in the same
        coordinate space the tile's `start_t` is defined in
        (`acquisition_details.start_t_space`).
    """

    name: Literal["Time Range Filter"] = "Time Range Filter"
    """Name of the filter."""
    min_t: float | None = None
    """Minimum `start_t` (inclusive). `None` means unbounded."""
    max_t: float | None = None
    """Maximum `start_t` (inclusive). `None` means unbounded."""


def apply_t_range_filter(tile: Tile, filter_params: TRangeFilter) -> bool:
    if filter_params.min_t is not None and tile.start_t < filter_params.min_t:
        return False
    if filter_params.max_t is not None and tile.start_t > filter_params.max_t:
        return False
    return True


P = ParamSpec("P")


class FilterFunctionProtocol(Protocol[P]):
    __name__: str

    def __call__(self, tile: Tile, *args: P.args, **kwargs: P.kwargs) -> bool: ...


_filter_registry: Registry[Callable[..., bool]] = Registry(
    "Filter step",
    "add_filter",
    {
        "Path Regex Include Filter": apply_path_include_regex_filter,
        "Path Regex Exclude Filter": apply_path_exclude_regex_filter,
        "Well Exclude Filter": apply_well_filter,
        "Well Include Filter": apply_well_include_filter,
        "FOV Name Include Filter": apply_fov_name_include_filter,
        "FOV Name Exclude Filter": apply_fov_name_exclude_filter,
        "Acquisition Include Filter": apply_acquisition_include_filter,
        "Acquisition Exclude Filter": apply_acquisition_exclude_filter,
        "Attribute Include Filter": apply_attribute_include_filter,
        "Attribute Exclude Filter": apply_attribute_exclude_filter,
        "Channel Include Filter": apply_channel_include_filter,
        "Channel Exclude Filter": apply_channel_exclude_filter,
        "Z Range Filter": apply_z_range_filter,
        "Time Range Filter": apply_t_range_filter,
    },
)


def add_filter(
    *,
    function: FilterFunctionProtocol,
    name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new filter.

    Note:
        Registrations are process-global: under `MultiprocessingRunner`,
        worker processes re-import the consumer's modules, so custom filters
        must be registered at import time of the module that defines them to
        be visible in workers.

    Args:
        function: Function that performs the filter step.
        name: Name of the filter step. Defaults to `function.__name__`.
        overwrite: Whether to overwrite an existing filter step
            with the same name.
    """
    _filter_registry.add(function=function, name=name, overwrite=overwrite)


def apply_filter_pipeline(
    tiles: list[Tile], *, filters_config: Sequence[FilterModel]
) -> list[Tile]:
    for step in filters_config:
        step_function = _filter_registry.get(step.name)
        tiles = [tile for tile in tiles if step_function(tile, filter_params=step)]
    return tiles


ImplementedFilters = Annotated[
    RegexExcludeFilter
    | RegexIncludeFilter
    | WellExcludeFilter
    | WellIncludeFilter
    | FovNameIncludeFilter
    | FovNameExcludeFilter
    | AcquisitionIncludeFilter
    | AcquisitionExcludeFilter
    | AttributeIncludeFilter
    | AttributeExcludeFilter
    | ChannelIncludeFilter
    | ChannelExcludeFilter
    | ZRangeFilter
    | TRangeFilter,
    Field(discriminator="name"),
]
