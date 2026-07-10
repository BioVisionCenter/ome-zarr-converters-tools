"""Unit tests for registration pipeline orchestration."""

import pytest

from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.models import AutoTiling, StagePositionCorrections
from ome_zarr_converters_tools.pipelines._registration_pipeline import (
    RegistrationStep,
    _registration_registry,
    add_registration_func,
    apply_registration_pipeline,
    build_default_registration_pipeline,
)


class TestRegistrationRegistry:
    def test_add_registration_func(self) -> None:
        def my_step(tiled_image: TiledImage) -> TiledImage:
            return tiled_image

        name = "test_reg_step_unique"
        try:
            add_registration_func(function=my_step, name=name)
            assert name in _registration_registry
        finally:
            _registration_registry.pop(name, None)

    def test_add_registration_func_duplicate_error(self) -> None:
        def my_step(tiled_image: TiledImage) -> TiledImage:
            return tiled_image

        name = "test_dup_reg_step"
        try:
            add_registration_func(function=my_step, name=name)
            with pytest.raises(ValueError, match="already registered"):
                add_registration_func(function=my_step, name=name)
        finally:
            _registration_registry.pop(name, None)


class TestRegistrationPipeline:
    def test_apply_registration_pipeline(
        self, tiled_image_from_grid: TiledImage
    ) -> None:
        calls: list[str] = []

        def step_a(tiled_image: TiledImage) -> TiledImage:
            calls.append("a")
            return tiled_image

        def step_b(tiled_image: TiledImage) -> TiledImage:
            calls.append("b")
            return tiled_image

        try:
            add_registration_func(function=step_a, name="test_step_a")
            add_registration_func(function=step_b, name="test_step_b")
            config = [
                RegistrationStep(name="test_step_a", params={}),
                RegistrationStep(name="test_step_b", params={}),
            ]
            result = apply_registration_pipeline(tiled_image_from_grid, config)
            assert result is not None
            assert calls == ["a", "b"]
        finally:
            _registration_registry.pop("test_step_a", None)
            _registration_registry.pop("test_step_b", None)

    def test_unknown_step_error(self, tiled_image_from_grid: TiledImage) -> None:
        config = [RegistrationStep(name="nonexistent_step", params={})]
        with pytest.raises(ValueError, match="not registered"):
            apply_registration_pipeline(tiled_image_from_grid, config)

    def test_build_default_registration_pipeline(self) -> None:
        corrections = StagePositionCorrections()
        pipeline = build_default_registration_pipeline(corrections, AutoTiling())
        assert len(pipeline) == 5
        names = [step["name"] for step in pipeline]
        assert "offset_removal" in names
        assert "align_to_pixel_grid" in names
        assert "xy_jitter_correction" in names
        assert "reindex_channels" in names
        assert "tile_regions" in names

    def test_build_default_registration_pipeline_tolerance(self) -> None:
        corrections = StagePositionCorrections()
        strategy = AutoTiling(tolerance=2.5)
        pipeline = build_default_registration_pipeline(corrections, strategy)
        tile_regions_step = next(s for s in pipeline if s["name"] == "tile_regions")
        assert tile_regions_step["params"]["tiling_strategy"].tolerance == 2.5
