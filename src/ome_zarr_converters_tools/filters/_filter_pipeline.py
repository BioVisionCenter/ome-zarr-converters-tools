import re
from typing import Any, ParamSpec, Protocol, TypedDict

from ome_zarr_converters_tools.models._collection import ImageInPlate
from ome_zarr_converters_tools.models._tile import Tile


def apply_path_include_regex_filter(tile: Tile, regex: str) -> bool:
    base_path = tile.collection.path()
    if re.search(regex, base_path):
        return True
    return False


def apply_path_exclude_regex_filter(tile: Tile, regex: str) -> bool:
    return not apply_path_include_regex_filter(tile, regex)


def apply_well_filter(tile: Tile, wells_to_remove: list[str]) -> bool:
    if not isinstance(tile.collection, ImageInPlate):
        raise ValueError(
            "Well filter can only be applied to To tile with ImageInPlate collection."
        )
    if tile.collection.well in wells_to_remove:
        return False
    return True


P = ParamSpec("P")


class FilterStep(TypedDict):
    name: str
    params: dict[str, Any]


class FilterFunctionProtocol(Protocol[P]):
    __name__: str

    def __call__(self, tile: Tile, *args: P.args, **kwargs: P.kwargs) -> bool: ...


_filter_registry: dict[str, FilterFunctionProtocol] = {
    "path_include_regex": apply_path_include_regex_filter,
    "path_exclude_regex": apply_path_exclude_regex_filter,
    "well_filter": apply_well_filter,
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
    tiles: list[Tile], *, filters_config: list[FilterStep]
) -> list[Tile]:
    for step in filters_config:
        step_name = step.get("name")
        step_params = step.get("params", {})
        if step_name not in _filter_registry:
            raise ValueError(f"Filter step '{step_name}' is not registered.")
        step_function = _filter_registry[step_name]
        tiles = [tile for tile in tiles if step_function(tile, **step_params)]
    return tiles
