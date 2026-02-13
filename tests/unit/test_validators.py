"""Unit tests for validator pipeline."""

from typing import Any

import pytest

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.pipelines._validators import (
    ValidatorStep,
    _validator_registry,
    add_validator,
    apply_validator_pipeline,
)


class TestValidatorStep:
    def test_validator_step_creation(self) -> None:
        step = ValidatorStep(name="my_validator", params={"threshold": 0.5})
        assert step["name"] == "my_validator"
        assert step["params"] == {"threshold": 0.5}


class TestValidatorRegistry:
    def test_add_validator(self) -> None:
        def my_validator(tile: TiledImage, **kwargs: Any) -> None:
            pass

        name = "test_validator_unique"
        try:
            add_validator(my_validator, name=name)
            assert name in _validator_registry
        finally:
            _validator_registry.pop(name, None)

    def test_add_validator_duplicate_error(self) -> None:
        def my_validator(tile: TiledImage, **kwargs: Any) -> None:
            pass

        name = "test_dup_validator"
        try:
            add_validator(my_validator, name=name)
            with pytest.raises(ValueError, match="already registered"):
                add_validator(my_validator, name=name)
        finally:
            _validator_registry.pop(name, None)

    def test_add_validator_overwrite(self) -> None:
        def my_validator(tile: TiledImage, **kwargs: Any) -> None:
            pass

        name = "test_overwrite_validator"
        try:
            add_validator(my_validator, name=name)
            add_validator(my_validator, name=name, overwrite=True)
            assert name in _validator_registry
        finally:
            _validator_registry.pop(name, None)


class TestValidatorPipeline:
    def test_empty_pipeline(self, tiled_image_from_grid: TiledImage) -> None:
        result = apply_validator_pipeline([tiled_image_from_grid], [])
        assert len(result) == 1

    def test_apply_validator_calls_function(
        self, tiled_image_from_grid: TiledImage
    ) -> None:
        called: list[str] = []

        def tracking_validator(tile: TiledImage, **kwargs: Any) -> None:
            called.append(tile.path)

        name = "test_tracking_validator"
        try:
            add_validator(tracking_validator, name=name)
            step = ValidatorStep(name=name, params={})
            apply_validator_pipeline([tiled_image_from_grid], [step])
            assert len(called) == 1
        finally:
            _validator_registry.pop(name, None)

    def test_unknown_validator_error(self, tiled_image_from_grid: TiledImage) -> None:
        step = ValidatorStep(name="nonexistent_validator", params={})
        with pytest.raises(ValueError, match="not registered"):
            apply_validator_pipeline([tiled_image_from_grid], [step])
