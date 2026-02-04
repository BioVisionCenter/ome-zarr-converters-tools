import re
from collections.abc import Sequence
from typing import Annotated, Any, Literal, ParamSpec, Protocol

from pydantic import BaseModel, Field

from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models._collection import ImageInPlate


class FilterModel(BaseModel):
    name: Any


class RegexIncludeFilter(FilterModel):
    """Regex include filter model.

    Attributes:
        name: Name of the filter.
        regex: Regex pattern to include. If the tile's base path matches this regex,
            it will be included, otherwise it will be excluded.
    """

    name: Literal["Path Regex Include Filter"] = "Path Regex Include Filter"
    regex: str


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
    """Regex exclude filter model.

    Attributes:
        name: Name of the filter.
        regex: Regex pattern to exclude. If the tile's base path matches this regex,
            it will be excluded, otherwise it will be included.
    """

    name: Literal["Path Regex Exclude Filter"] = "Path Regex Exclude Filter"
    regex: str


def apply_path_exclude_regex_filter(
    tile: Tile, filter_params: RegexExcludeFilter
) -> bool:
    return not _regex_bases_match(tile, filter_params.regex)


class WellFilter(FilterModel):
    """Well filter model.

    Attributes:
        name: Name of the filter.
        wells_to_remove: List of well identifiers to remove.
            E.g., ["A1", "B2"]
    """

    name: Literal["Well Filter"] = "Well Filter"
    wells_to_remove: list[str]


def apply_well_filter(tile: Tile, filter_params: WellFilter) -> bool:
    if not isinstance(tile.collection, ImageInPlate):
        raise ValueError(
            "Well filter can only be applied to To tile with ImageInPlate collection."
        )
    if tile.collection.well in filter_params.wells_to_remove:
        return False
    return True


P = ParamSpec("P")


class FilterFunctionProtocol(Protocol[P]):
    __name__: str

    def __call__(self, tile: Tile, *args: P.args, **kwargs: P.kwargs) -> bool: ...


_filter_registry: dict[str, FilterFunctionProtocol] = {
    "Path Regex Include Filter": apply_path_include_regex_filter,
    "Path Regex Exclude Filter": apply_path_exclude_regex_filter,
    "Well Filter": apply_well_filter,
}


def add_filter(
    *,
    function: FilterFunctionProtocol,
    name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new filter.

    Args:
        name: Name of the registration step.
        function: Function that performs the registration step.
        overwrite: Whether to overwrite an existing registration step
            with the same name.
    """
    if name is None:
        name = function.__name__
    if not overwrite and name in _filter_registry:
        raise ValueError(f"Filter step '{name}' is already registered.")
    _filter_registry[name] = function


def apply_filter_pipeline(
    tiles: list[Tile], *, filters_config: Sequence[FilterModel]
) -> list[Tile]:
    for step in filters_config:
        step_name = step.name
        if step_name not in _filter_registry:
            raise ValueError(f"Filter step '{step_name}' is not registered.")
        step_function = _filter_registry[step_name]
        tiles = [tile for tile in tiles if step_function(tile, filter_params=step)]
    return tiles


ImplementedFilters = Annotated[
    RegexExcludeFilter | RegexIncludeFilter | WellFilter, Field(discriminator="name")
]
