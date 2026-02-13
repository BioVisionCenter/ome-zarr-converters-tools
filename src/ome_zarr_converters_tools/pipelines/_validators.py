from collections.abc import Sequence
from typing import Any, ParamSpec, Protocol, TypedDict

from ome_zarr_converters_tools.core import TiledImage

P = ParamSpec("P")


class ValidatorStep(TypedDict):
    name: str
    params: dict[str, Any]


class ValidatorFunctionProtocol(Protocol[P]):
    __name__: str

    def __call__(self, tile: TiledImage, *args: P.args, **kwargs: P.kwargs) -> None: ...


_validator_registry: dict[str, ValidatorFunctionProtocol] = {}


def add_validator(
    function: ValidatorFunctionProtocol,
    name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new validator function.

    Args:
        name: Name of the registration step.
        function: Function that performs the registration step.
        overwrite: Whether to overwrite an existing registration step
            with the same name.
    """
    if name is None:
        name = function.__name__
    if not overwrite and name in _validator_registry:
        raise ValueError(f"Validator step '{name}' is already registered.")
    _validator_registry[name] = function


def apply_validator_pipeline(
    tiles: list[TiledImage], validators_config: Sequence[ValidatorStep]
) -> list[TiledImage]:
    for step in validators_config:
        step_name = step.get("name")
        step_params = step.get("params", {})
        if step_name not in _validator_registry:
            raise ValueError(f"Validator step '{step_name}' is not registered.")
        step_function = _validator_registry[step_name]
        for tile in tiles:
            step_function(tile, **step_params)
    return tiles
