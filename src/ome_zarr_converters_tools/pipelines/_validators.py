import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, ParamSpec, Protocol

from pydantic import BaseModel

from ome_zarr_converters_tools.core import TiledImage
from ome_zarr_converters_tools.core._roi_utils import move_roi_by
from ome_zarr_converters_tools.pipelines._registry import Registry


class ValidatorModel(BaseModel):
    name: Any


class ShapeDtypeProbeValidator(ValidatorModel):
    """Pre-flight probe of one tile per image against its declared geometry.

    Runs `preflight` on every tile's image loader (a cheap reachability
    check, e.g. file existence, which warns on problems), then fully loads a
    single sample tile and raises if its shape or dtype does not match the
    declared tile geometry. This front-loads parser bugs (wrong `length_*`
    or `data_type`) that would otherwise only fail after compute jobs are
    dispatched.

    Note:
        Loading the sample tile costs one full tile read per image; enable
        this validator when that is cheaper than a failed compute job.
    """

    name: Literal["Shape and Dtype Probe"] = "Shape and Dtype Probe"
    """Name of the validator."""


def _expected_region_shape(tiled_image: TiledImage, region: Any) -> tuple[int, ...]:
    """Pixel shape a region's data must have, derived as compute derives it."""
    offset = {}
    for axis in tiled_image.axes:
        axis_slice = region.roi.get(axis)
        start = axis_slice.start if axis_slice is not None else None
        offset[axis] = -(start if start is not None else 0.0)
    roi_zeroed = move_roi_by(region.roi, offset)
    roi_slice = roi_zeroed.to_slicing_dict(pixel_size=tiled_image.pixel_size)
    return tuple(
        math.ceil(roi_slice[axis].stop) - math.floor(roi_slice[axis].start)
        for axis in tiled_image.axes
    )


def apply_shape_dtype_probe(
    tiled_image: TiledImage,
    validator_params: ShapeDtypeProbeValidator,
    resource: Any | None = None,
) -> None:
    for region in tiled_image.regions:
        region.image_loader.preflight(resource=resource)

    sample = tiled_image.regions[0]
    data = sample.load_data(axes=tiled_image.axes, resource=resource)
    expected_shape = _expected_region_shape(tiled_image, sample)
    if tuple(data.shape) != expected_shape:
        raise ValueError(
            f"TiledImage '{tiled_image.path}' tile '{sample.roi.name}' "
            f"declares shape {expected_shape} (axes {tiled_image.axes}) but "
            f"its loader returned shape {tuple(data.shape)}. Fix the "
            "`length_*` values the parser sets on its tiles to match the "
            "actual data."
        )
    if str(data.dtype) != tiled_image.data_type:
        raise ValueError(
            f"TiledImage '{tiled_image.path}' declares data type "
            f"'{tiled_image.data_type}' but tile '{sample.roi.name}' loaded "
            f"as '{data.dtype}'. Fix the `data_type` in AcquisitionDetails, "
            "or leave it as AUTODETECT."
        )


P = ParamSpec("P")


class ValidatorFunctionProtocol(Protocol[P]):
    __name__: str

    def __call__(
        self, tiled_image: TiledImage, *args: P.args, **kwargs: P.kwargs
    ) -> None: ...


_validator_registry: Registry[Callable[..., None]] = Registry(
    "Validator step",
    "add_validator",
    {
        "Shape and Dtype Probe": apply_shape_dtype_probe,
    },
)


def add_validator(
    *,
    function: ValidatorFunctionProtocol,
    name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new validator function.

    Validators are pre-flight checks that run on each `TiledImage` after
    aggregation and raise on failure; use them to front-load errors that
    would otherwise only surface during compute.

    Note:
        Registrations are process-global: under `MultiprocessingRunner`,
        worker processes re-import the consumer's modules, so custom
        validators must be registered at import time of the module that
        defines them to be visible in workers.

    Args:
        function: Function that performs the validation step. It is called
            as `function(tiled_image, validator_params=step, resource=resource)`
            and must raise on failure.
        name: Name of the validation step. Defaults to `function.__name__`.
        overwrite: Whether to overwrite an existing validation step
            with the same name.
    """
    _validator_registry.add(function=function, name=name, overwrite=overwrite)


def apply_validator_pipeline(
    tiled_images: list[TiledImage],
    *,
    validators_config: Sequence[ValidatorModel],
    resource: Any | None = None,
) -> list[TiledImage]:
    for step in validators_config:
        step_function = _validator_registry.get(step.name)
        for tiled_image in tiled_images:
            step_function(tiled_image, validator_params=step, resource=resource)
    return tiled_images


# Will become a discriminated union (like `ImplementedFilters`) once more
# built-in validators exist.
ImplementedValidators = ShapeDtypeProbeValidator
