"""Unit tests for validator pipeline."""

import warnings
from typing import Any

import numpy as np
import pytest

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.core._tile_to_tiled_images import (
    tiled_image_from_tiles,
)
from ome_zarr_converters_tools.models import AcquisitionDetails, SingleImage
from ome_zarr_converters_tools.models._loader import ImageLoaderInterface
from ome_zarr_converters_tools.pipelines._validators import (
    ShapeDtypeProbeValidator,
    ValidatorModel,
    _validator_registry,
    add_validator,
    apply_validator_pipeline,
)


class _FixedShapeLoader(ImageLoaderInterface):
    """Loader returning a fixed-shape array, independent of tile geometry."""

    data_shape: tuple[int, ...]
    data_dtype: str = "uint8"

    def load_data(self, resource: Any | None = None) -> np.ndarray:
        return np.zeros(self.data_shape, dtype=self.data_dtype)


class _WarningPreflightLoader(_FixedShapeLoader):
    def preflight(self, resource: Any | None = None) -> None:
        warnings.warn("source file missing", stacklevel=2)


def _single_tile_image(loader: ImageLoaderInterface) -> TiledImage:
    tile = build_dummy_tile(
        fov_name="FOV_0",
        start=StartPosition(x=0, y=0),
        shape=TileShape(x=64, y=64),
        collection=SingleImage(image_path="img"),
        acquisition_details=AcquisitionDetails(),
    )
    tile.image_loader = loader
    images = tiled_image_from_tiles(tiles=[tile])
    assert len(images) == 1
    return images[0]


class TestValidatorModel:
    def test_validator_model_creation(self) -> None:
        step = ValidatorModel(name="my_validator")
        assert step.name == "my_validator"

    def test_shape_dtype_probe_creation(self) -> None:
        step = ShapeDtypeProbeValidator()
        assert step.name == "Shape and Dtype Probe"


class TestValidatorRegistry:
    def test_add_validator(self) -> None:
        def my_validator(tiled_image: TiledImage, **kwargs: Any) -> None:
            pass

        name = "test_validator_unique"
        try:
            add_validator(function=my_validator, name=name)
            assert name in _validator_registry
        finally:
            _validator_registry.pop(name, None)

    def test_add_validator_duplicate_error(self) -> None:
        def my_validator(tiled_image: TiledImage, **kwargs: Any) -> None:
            pass

        name = "test_dup_validator"
        try:
            add_validator(function=my_validator, name=name)
            with pytest.raises(ValueError, match="already registered"):
                add_validator(function=my_validator, name=name)
        finally:
            _validator_registry.pop(name, None)

    def test_add_validator_overwrite(self) -> None:
        def my_validator(tiled_image: TiledImage, **kwargs: Any) -> None:
            pass

        name = "test_overwrite_validator"
        try:
            add_validator(function=my_validator, name=name)
            add_validator(function=my_validator, name=name, overwrite=True)
            assert name in _validator_registry
        finally:
            _validator_registry.pop(name, None)


class TestValidatorPipeline:
    def test_empty_pipeline(self, tiled_image_from_grid: TiledImage) -> None:
        result = apply_validator_pipeline([tiled_image_from_grid], validators_config=[])
        assert len(result) == 1

    def test_apply_validator_receives_model_and_resource(
        self, tiled_image_from_grid: TiledImage
    ) -> None:
        calls: list[tuple[str, Any, Any]] = []

        def tracking_validator(
            tiled_image: TiledImage,
            validator_params: ValidatorModel,
            resource: Any | None = None,
        ) -> None:
            calls.append((tiled_image.path, validator_params, resource))

        name = "test_tracking_validator"
        try:
            add_validator(function=tracking_validator, name=name)
            step = ValidatorModel(name=name)
            apply_validator_pipeline(
                [tiled_image_from_grid], validators_config=[step], resource="res"
            )
            assert len(calls) == 1
            assert calls[0][1] is step
            assert calls[0][2] == "res"
        finally:
            _validator_registry.pop(name, None)

    def test_unknown_validator_error(self, tiled_image_from_grid: TiledImage) -> None:
        step = ValidatorModel(name="nonexistent_validator")
        with pytest.raises(ValueError, match="not registered"):
            apply_validator_pipeline([tiled_image_from_grid], validators_config=[step])


class TestShapeDtypeProbe:
    def test_probe_passes_on_consistent_image(
        self, tiled_image_from_grid: TiledImage
    ) -> None:
        result = apply_validator_pipeline(
            [tiled_image_from_grid], validators_config=[ShapeDtypeProbeValidator()]
        )
        assert len(result) == 1

    def test_probe_detects_shape_mismatch(self) -> None:
        # Tile declares 64x64 but its loader returns 64x63.
        loader = _FixedShapeLoader(data_shape=(1, 1, 1, 64, 63))
        image = _single_tile_image(loader)
        with pytest.raises(ValueError, match="returned shape"):
            apply_validator_pipeline(
                [image], validators_config=[ShapeDtypeProbeValidator()]
            )

    def test_probe_detects_dtype_mismatch(self) -> None:
        loader = _FixedShapeLoader(data_shape=(1, 1, 1, 64, 64))
        image = _single_tile_image(loader)
        image.data_type = "uint16"
        with pytest.raises(ValueError, match="data type"):
            apply_validator_pipeline(
                [image], validators_config=[ShapeDtypeProbeValidator()]
            )

    def test_probe_runs_preflight_on_all_tiles(self) -> None:
        loader = _WarningPreflightLoader(data_shape=(1, 1, 1, 64, 64))
        image = _single_tile_image(loader)
        with pytest.warns(UserWarning, match="source file missing"):
            apply_validator_pipeline(
                [image], validators_config=[ShapeDtypeProbeValidator()]
            )
