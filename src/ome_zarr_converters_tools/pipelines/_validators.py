from collections.abc import Sequence
from typing import Any, ParamSpec, Protocol, TypedDict

from ome_zarr_converters_tools.core import TiledImage
from ome_zarr_converters_tools.pipelines._registry import Registry

P = ParamSpec("P")


class ValidatorStep(TypedDict):
    name: str
    params: dict[str, Any]


class ValidatorFunctionProtocol(Protocol[P]):
    __name__: str

    def __call__(self, tile: TiledImage, *args: P.args, **kwargs: P.kwargs) -> None: ...


_validator_registry: Registry[ValidatorFunctionProtocol] = Registry(
    "Validator step", "add_validator"
)


def add_validator(
    *,
    function: ValidatorFunctionProtocol,
    name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new validator function.

    Note:
        Registrations are process-global: under `MultiprocessingRunner`,
        worker processes re-import the consumer's modules, so custom
        validators must be registered at import time of the module that
        defines them to be visible in workers.

    Args:
        function: Function that performs the validation step.
        name: Name of the validation step. Defaults to `function.__name__`.
        overwrite: Whether to overwrite an existing validation step
            with the same name.
    """
    _validator_registry.add(function=function, name=name, overwrite=overwrite)


def apply_validator_pipeline(
    tiles: list[TiledImage], validators_config: Sequence[ValidatorStep]
) -> list[TiledImage]:
    for step in validators_config:
        step_function = _validator_registry.get(step.get("name"))
        step_params = step.get("params", {})
        for tile in tiles:
            step_function(tile, **step_params)
    return tiles
