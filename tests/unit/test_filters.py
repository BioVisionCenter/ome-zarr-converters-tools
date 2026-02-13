"""Unit tests for filter pipeline."""

from typing import Any

import pytest

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile import Tile
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    ImageInPlate,
    SingleImage,
)
from ome_zarr_converters_tools.pipelines._filters import (
    RegexExcludeFilter,
    RegexIncludeFilter,
    WellFilter,
    _filter_registry,
    add_filter,
    apply_filter_pipeline,
)


def _acq() -> AcquisitionDetails:
    return AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        pixelsize=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )


def _tile_with_path(image_path: str, fov: str = "FOV_0") -> Tile[Any, Any]:
    coll = SingleImage(image_path=image_path)
    return build_dummy_tile(
        fov_name=fov,
        start=StartPosition(x=0, y=0),
        shape=TileShape(x=64, y=64, z=1, c=1, t=1),
        collection=coll,
        acquisition_details=_acq(),
    )


def _tile_in_plate(row: str, column: int, well_fov: str = "FOV_0") -> Tile[Any, Any]:
    coll = ImageInPlate(
        plate_name="plate",
        row=row,
        column=column,
        acquisition=0,
    )
    return build_dummy_tile(
        fov_name=well_fov,
        start=StartPosition(x=0, y=0),
        shape=TileShape(x=64, y=64, z=1, c=1, t=1),
        collection=coll,
        acquisition_details=_acq(),
    )


class TestFilterModels:
    def test_regex_include_filter_creation(self) -> None:
        f = RegexIncludeFilter(regex=".*test.*")
        assert f.name == "Path Regex Include Filter"
        assert f.regex == ".*test.*"

    def test_regex_exclude_filter_creation(self) -> None:
        f = RegexExcludeFilter(regex=".*exclude.*")
        assert f.name == "Path Regex Exclude Filter"
        assert f.regex == ".*exclude.*"

    def test_well_filter_creation(self) -> None:
        f = WellFilter(wells_to_remove=["A01", "B02"])
        assert f.name == "Well Filter"
        assert f.wells_to_remove == ["A01", "B02"]


class TestFilterRegistry:
    def test_add_custom_filter(self) -> None:
        def my_filter(tile: Tile[Any, Any], **kwargs: Any) -> bool:
            return True

        name = "test_custom_filter_unique"
        try:
            add_filter(function=my_filter, name=name)
            assert name in _filter_registry
        finally:
            _filter_registry.pop(name, None)

    def test_add_filter_duplicate_error(self) -> None:
        def my_filter(tile: Tile[Any, Any], **kwargs: Any) -> bool:
            return True

        name = "test_dup_filter"
        try:
            add_filter(function=my_filter, name=name)
            with pytest.raises(ValueError, match="already registered"):
                add_filter(function=my_filter, name=name)
        finally:
            _filter_registry.pop(name, None)

    def test_add_filter_overwrite(self) -> None:
        def my_filter(tile: Tile[Any, Any], **kwargs: Any) -> bool:
            return True

        name = "test_overwrite_filter"
        try:
            add_filter(function=my_filter, name=name)
            add_filter(function=my_filter, name=name, overwrite=True)
            assert name in _filter_registry
        finally:
            _filter_registry.pop(name, None)


class TestFilterPipeline:
    def test_empty_pipeline_returns_all(self) -> None:
        tiles = [_tile_with_path("img_a"), _tile_with_path("img_b")]
        result = apply_filter_pipeline(tiles, filters_config=[])
        assert len(result) == 2

    def test_regex_include_keeps_matching(self) -> None:
        tiles = [
            _tile_with_path("img_alpha"),
            _tile_with_path("img_beta"),
            _tile_with_path("img_alpha2"),
        ]
        f = RegexIncludeFilter(regex=".*alpha.*")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 2
        for t in result:
            assert "alpha" in t.collection.path()

    def test_regex_exclude_removes_matching(self) -> None:
        tiles = [
            _tile_with_path("img_alpha"),
            _tile_with_path("img_beta"),
        ]
        f = RegexExcludeFilter(regex=".*alpha.*")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert "beta" in result[0].collection.path()

    def test_well_filter_removes_wells(self) -> None:
        tiles = [
            _tile_in_plate("A", 1),
            _tile_in_plate("A", 2),
            _tile_in_plate("B", 1),
        ]
        f = WellFilter(wells_to_remove=["A01"])
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 2
        wells = [t.collection.well for t in result]
        assert "A01" not in wells

    def test_well_filter_non_plate_error(self) -> None:
        tiles = [_tile_with_path("img_a")]
        f = WellFilter(wells_to_remove=["A01"])
        with pytest.raises(ValueError, match="ImageInPlate"):
            apply_filter_pipeline(tiles, filters_config=[f])

    def test_multiple_filters_chain(self) -> None:
        tiles = [
            _tile_with_path("img_alpha"),
            _tile_with_path("img_beta"),
            _tile_with_path("img_gamma"),
        ]
        filters = [
            RegexIncludeFilter(regex=".*alpha|.*gamma"),  # keeps alpha, gamma
            RegexExcludeFilter(regex=".*gamma.*"),  # removes gamma
        ]
        result = apply_filter_pipeline(tiles, filters_config=filters)
        assert len(result) == 1
        assert "alpha" in result[0].collection.path()
