import re
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, ParamSpec, Protocol

from pydantic import BaseModel, Field

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models._collection import ImageInPlate
from ome_zarr_converters_tools.pipelines._registry import Registry

FilterMode = Literal["Include", "Exclude"]


class FilterModel(BaseModel):
    name: Any


def _apply_mode(matched: bool, mode: FilterMode) -> bool:
    return matched if mode == "Include" else not matched


class RegexFilter(FilterModel):
    """Path regex filter model."""

    name: Literal["Path Regex Filter"] = "Path Regex Filter"
    """Name of the filter."""
    mode: FilterMode = "Include"
    """Whether matching tiles are included (kept) or excluded (removed)."""
    regex: str
    """
    Regex pattern matched against the tile's base path. In `Include` mode
    matching tiles are kept and all others removed; in `Exclude` mode
    matching tiles are removed and all others kept.
    """


def _regex_bases_match(tile: Tile, regex: str) -> bool:
    base_path = tile.collection.path()
    if re.search(regex, base_path):
        return True
    return False


def apply_path_regex_filter(tile: Tile, filter_params: RegexFilter) -> bool:
    matched = _regex_bases_match(tile, filter_params.regex)
    return _apply_mode(matched, filter_params.mode)


class WellFilter(FilterModel):
    """Well filter model."""

    name: Literal["Well Filter"] = "Well Filter"
    """Name of the filter."""
    mode: FilterMode = "Include"
    """Whether matching tiles are included (kept) or excluded (removed)."""
    wells: list[str]
    """List of well identifiers to keep (`Include` mode) or remove
    (`Exclude` mode). E.g., ["A01", "B02"]"""


def apply_well_filter(tile: Tile, filter_params: WellFilter) -> bool:
    if not isinstance(tile.collection, ImageInPlate):
        raise ValueError(
            "Well filter can only be applied to a tile with ImageInPlate collection."
        )
    matched = tile.collection.well in filter_params.wells
    return _apply_mode(matched, filter_params.mode)


class FovNameFilter(FilterModel):
    """FOV name filter model."""

    name: Literal["FOV Name Filter"] = "FOV Name Filter"
    """Name of the filter."""
    mode: FilterMode = "Include"
    """Whether matching tiles are included (kept) or excluded (removed)."""
    regex: str
    """
    Regex pattern matched against the tile's `fov_name`. In `Include` mode
    matching tiles are kept and all others removed; in `Exclude` mode
    matching tiles are removed and all others kept.
    """


def apply_fov_name_filter(tile: Tile, filter_params: FovNameFilter) -> bool:
    matched = re.search(filter_params.regex, tile.fov_name) is not None
    return _apply_mode(matched, filter_params.mode)


def _plate_acquisition(tile: Tile, filter_name: str) -> int:
    if not isinstance(tile.collection, ImageInPlate):
        raise ValueError(
            f"{filter_name} can only be applied to a tile with ImageInPlate collection."
        )
    return tile.collection.acquisition


class AcquisitionFilter(FilterModel):
    """Acquisition filter model."""

    name: Literal["Acquisition Filter"] = "Acquisition Filter"
    """Name of the filter."""
    mode: FilterMode = "Include"
    """Whether matching tiles are included (kept) or excluded (removed)."""
    acquisitions: list[int]
    """List of acquisition indices to keep (`Include` mode) or remove
    (`Exclude` mode). E.g., [0, 1]"""


def apply_acquisition_filter(tile: Tile, filter_params: AcquisitionFilter) -> bool:
    acquisition = _plate_acquisition(tile, filter_params.name)
    matched = acquisition in filter_params.acquisitions
    return _apply_mode(matched, filter_params.mode)


class BoolValue(BaseModel):
    """Bool value for attribute filters."""

    type: Literal["Bool"] = "Bool"
    value: bool = True


class StringValue(BaseModel):
    """String value for attribute filters."""

    type: Literal["String"] = "String"
    value: str


class IntValue(BaseModel):
    """Int value for attribute filters."""

    type: Literal["Integer"] = "Integer"
    value: int


class FloatValue(BaseModel):
    """Float value for attribute filters."""

    type: Literal["Float"] = "Float"
    value: float


class IsNoneValue(BaseModel):
    """Matches attribute elements that are `None`."""

    type: Literal["Is None"] = "Is None"


class IsNotNoneValue(BaseModel):
    """Matches attribute elements that are not `None`."""

    type: Literal["Is Not None"] = "Is Not None"


# Typed wrappers (instead of a bare `str | int | float | bool` union) so the
# manifest schema stays consumable by the Fractal web UI: fractal-task-tools
# rejects a boolean schema node without a default ([E05]), which a bare union
# member inside a list cannot carry.
AttributeValue = Annotated[
    BoolValue | StringValue | IntValue | FloatValue | IsNoneValue | IsNotNoneValue,
    Field(discriminator="type"),
]


def _matches_value(
    element: str | int | float | bool | None, value: AttributeValue
) -> bool:
    if isinstance(value, IsNoneValue):
        return element is None
    if isinstance(value, IsNotNoneValue):
        return element is not None
    return element == value.value


def _attribute_matches(
    tile: Tile, key: str, values: list[AttributeValue], filter_name: str
) -> bool:
    if key not in tile.attributes:
        available = sorted(tile.attributes)
        raise ValueError(
            f"{filter_name} references attribute '{key}' but tile "
            f"'{tile.fov_name}' has no such attribute; available attributes: "
            f"{available}. Fix the filter key or make the parser set the "
            "attribute on every tile."
        )
    return any(
        _matches_value(element, value)
        for element in tile.attributes[key]
        for value in values
    )


class AttributeFilter(FilterModel):
    """Attribute filter model."""

    name: Literal["Attribute Filter"] = "Attribute Filter"
    """Name of the filter."""
    mode: FilterMode = "Include"
    """Whether matching tiles are included (kept) or excluded (removed)."""
    key: str
    """Attribute key to match. The attribute must be present on every tile."""
    values: list[AttributeValue]
    """Values to match against. A tile matches if any element of its
    attribute value matches one of these; matching tiles are kept in
    `Include` mode and removed in `Exclude` mode."""


def apply_attribute_filter(tile: Tile, filter_params: AttributeFilter) -> bool:
    matched = _attribute_matches(
        tile, filter_params.key, filter_params.values, filter_params.name
    )
    return _apply_mode(matched, filter_params.mode)


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


class ChannelFilter(FilterModel):
    """Channel filter model."""

    name: Literal["Channel Filter"] = "Channel Filter"
    """Name of the filter."""
    mode: FilterMode = "Include"
    """Whether matching tiles are included (kept) or excluded (removed)."""
    channel_labels: list[str]
    """Channel labels to keep (`Include` mode) or remove (`Exclude` mode).
    E.g., ["DAPI", "GFP"]. A tile matches if all of its channels are in this
    list; a partial match raises."""


def apply_channel_filter(tile: Tile, filter_params: ChannelFilter) -> bool:
    matched = _channel_labels_match(
        tile, filter_params.channel_labels, filter_params.name
    )
    return _apply_mode(matched, filter_params.mode)


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
        "Path Regex Filter": apply_path_regex_filter,
        "Well Filter": apply_well_filter,
        "FOV Name Filter": apply_fov_name_filter,
        "Acquisition Filter": apply_acquisition_filter,
        "Attribute Filter": apply_attribute_filter,
        "Channel Filter": apply_channel_filter,
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
    RegexFilter
    | WellFilter
    | FovNameFilter
    | AcquisitionFilter
    | AttributeFilter
    | ChannelFilter
    | ZRangeFilter
    | TRangeFilter,
    Field(discriminator="name"),
]
