from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypedDict

from ome_zarr_converters_tools.core import TiledImage
from ome_zarr_converters_tools.models import StagePositionCorrections, TilingStrategy
from ome_zarr_converters_tools.pipelines._alignment import (
    apply_align_to_pixel_grid,
    apply_offset_removal,
    apply_reindex_channels,
    apply_xy_jitter_correction,
)
from ome_zarr_converters_tools.pipelines._registry import Registry
from ome_zarr_converters_tools.pipelines._tiling import apply_mosaic_tiling

P = ParamSpec("P")


class RegistrationFunctionProtocol(Protocol[P]):
    __name__: str

    def __call__(
        self, tiled_image: TiledImage, *args: P.args, **kwargs: P.kwargs
    ) -> TiledImage: ...


class RegistrationStep(TypedDict):
    name: str
    params: dict[str, Any]


_registration_registry: Registry[Callable[..., TiledImage]] = Registry(
    "Registration step",
    "add_registration_func",
    {
        "align_to_pixel_grid": apply_align_to_pixel_grid,
        "offset_removal": apply_offset_removal,
        "xy_jitter_correction": apply_xy_jitter_correction,
        "reindex_channels": apply_reindex_channels,
        "tile_regions": apply_mosaic_tiling,
    },
)


def add_registration_func(
    *,
    function: Callable[..., TiledImage],
    name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new registration step function.

    Note:
        Registrations are process-global: under `MultiprocessingRunner`,
        worker processes re-import the consumer's modules, so custom
        registration steps must be registered at import time of the module
        that defines them to be visible in workers.

    Args:
        function: Function that performs the registration step.
        name: Name of the registration step. Defaults to `function.__name__`.
        overwrite: Whether to overwrite an existing registration step.
    """
    _registration_registry.add(function=function, name=name, overwrite=overwrite)


def apply_registration_pipeline(
    tiled_image: TiledImage, pipeline_config: list[RegistrationStep]
) -> TiledImage:
    for step in pipeline_config:
        step_function = _registration_registry.get(step.get("name"))
        step_params = step.get("params", {})
        tiled_image = step_function(tiled_image, **step_params)
    return tiled_image


def build_default_registration_pipeline(
    alignment_corrections: StagePositionCorrections, tiling_strategy: TilingStrategy
) -> list[RegistrationStep]:
    return [
        RegistrationStep(
            name="offset_removal", params={"corrections": alignment_corrections}
        ),
        RegistrationStep(name="align_to_pixel_grid", params={}),
        RegistrationStep(
            name="xy_jitter_correction",
            params={"corrections": alignment_corrections},
        ),
        RegistrationStep(
            name="reindex_channels",
            params={"corrections": alignment_corrections},
        ),
        RegistrationStep(
            name="tile_regions",
            params={"tiling_strategy": tiling_strategy},
        ),
    ]
