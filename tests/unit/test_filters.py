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
    AcquisitionFilter,
    AttributeFilter,
    BoolValue,
    ChannelFilter,
    FovNameFilter,
    IsNoneValue,
    IsNotNoneValue,
    RegexFilter,
    StringValue,
    TRangeFilter,
    WellFilter,
    ZRangeFilter,
    _filter_registry,
    add_filter,
    apply_filter_pipeline,
)


def _acq() -> AcquisitionDetails:
    return AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        xy_pixel_size=1.0,
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
    def test_regex_filter_creation(self) -> None:
        f = RegexFilter(regex=".*test.*")
        assert f.name == "Path Regex Filter"
        assert f.mode == "Include"
        assert f.regex == ".*test.*"

    def test_regex_filter_exclude_mode(self) -> None:
        f = RegexFilter(regex=".*exclude.*", mode="Exclude")
        assert f.name == "Path Regex Filter"
        assert f.mode == "Exclude"

    def test_well_filter_creation(self) -> None:
        f = WellFilter(wells=["A01", "B02"])
        assert f.name == "Well Filter"
        assert f.mode == "Include"
        assert f.wells == ["A01", "B02"]

    def test_filter_mode_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError, match="Include"):
            WellFilter(wells=["A01"], mode="Drop")


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
        f = RegexFilter(regex=".*alpha.*")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 2
        for t in result:
            assert "alpha" in t.collection.path()

    def test_regex_exclude_removes_matching(self) -> None:
        tiles = [
            _tile_with_path("img_alpha"),
            _tile_with_path("img_beta"),
        ]
        f = RegexFilter(regex=".*alpha.*", mode="Exclude")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert "beta" in result[0].collection.path()

    def test_well_exclude_removes_wells(self) -> None:
        tiles = [
            _tile_in_plate("A", 1),
            _tile_in_plate("A", 2),
            _tile_in_plate("B", 1),
        ]
        f = WellFilter(wells=["A01"], mode="Exclude")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 2
        wells = [t.collection.well for t in result]
        assert "A01" not in wells

    def test_well_filter_non_plate_error(self) -> None:
        tiles = [_tile_with_path("img_a")]
        f = WellFilter(wells=["A01"], mode="Exclude")
        with pytest.raises(ValueError, match="ImageInPlate"):
            apply_filter_pipeline(tiles, filters_config=[f])

    def test_well_include_keeps_wells(self) -> None:
        tiles = [
            _tile_in_plate("A", 1),
            _tile_in_plate("A", 2),
            _tile_in_plate("B", 1),
        ]
        f = WellFilter(wells=["A01", "B01"])
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 2
        wells = [t.collection.well for t in result]
        assert "A02" not in wells
        assert "A01" in wells
        assert "B01" in wells

    def test_well_include_non_plate_error(self) -> None:
        tiles = [_tile_with_path("img_a")]
        f = WellFilter(wells=["A01"])
        with pytest.raises(ValueError, match="ImageInPlate"):
            apply_filter_pipeline(tiles, filters_config=[f])

    def test_multiple_filters_chain(self) -> None:
        tiles = [
            _tile_with_path("img_alpha"),
            _tile_with_path("img_beta"),
            _tile_with_path("img_gamma"),
        ]
        filters = [
            RegexFilter(regex=".*alpha|.*gamma"),  # keeps alpha, gamma
            RegexFilter(regex=".*gamma.*", mode="Exclude"),  # removes gamma
        ]
        result = apply_filter_pipeline(tiles, filters_config=filters)
        assert len(result) == 1
        assert "alpha" in result[0].collection.path()


class TestFovNameFilters:
    def test_fov_include_keeps_matching(self) -> None:
        tiles = [
            _tile_with_path("img", fov="FOV_1"),
            _tile_with_path("img", fov="FOV_2"),
            _tile_with_path("img", fov="POS_1"),
        ]
        f = FovNameFilter(regex="^FOV_")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert [t.fov_name for t in result] == ["FOV_1", "FOV_2"]

    def test_fov_exclude_removes_matching(self) -> None:
        tiles = [
            _tile_with_path("img", fov="FOV_1"),
            _tile_with_path("img", fov="POS_1"),
        ]
        f = FovNameFilter(regex="^FOV_", mode="Exclude")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert [t.fov_name for t in result] == ["POS_1"]


def _tile_with_acquisition(acquisition: int) -> Tile[Any, Any]:
    coll = ImageInPlate(plate_name="plate", row="A", column=1, acquisition=acquisition)
    return build_dummy_tile(
        fov_name="FOV_0",
        start=StartPosition(x=0, y=0),
        shape=TileShape(x=64, y=64, z=1, c=1, t=1),
        collection=coll,
        acquisition_details=_acq(),
    )


class TestAcquisitionFilters:
    def test_acquisition_include(self) -> None:
        tiles = [_tile_with_acquisition(a) for a in (0, 1, 2)]
        f = AcquisitionFilter(acquisitions=[0, 2])
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert [t.collection.acquisition for t in result] == [0, 2]

    def test_acquisition_exclude(self) -> None:
        tiles = [_tile_with_acquisition(a) for a in (0, 1, 2)]
        f = AcquisitionFilter(acquisitions=[1], mode="Exclude")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert [t.collection.acquisition for t in result] == [0, 2]

    def test_acquisition_non_plate_error(self) -> None:
        tiles = [_tile_with_path("img_a")]
        f = AcquisitionFilter(acquisitions=[0])
        with pytest.raises(ValueError, match="ImageInPlate"):
            apply_filter_pipeline(tiles, filters_config=[f])


def _tile_with_attributes(attributes: dict[str, Any]) -> Tile[Any, Any]:
    tile = _tile_with_path("img")
    tile.attributes = attributes
    return tile


class TestAttributeFilters:
    def test_attribute_include(self) -> None:
        tiles = [
            _tile_with_attributes({"condition": ["control"]}),
            _tile_with_attributes({"condition": ["treated"]}),
        ]
        f = AttributeFilter(key="condition", values=[StringValue(value="control")])
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert result[0].attributes["condition"] == ["control"]

    def test_attribute_exclude(self) -> None:
        tiles = [
            _tile_with_attributes({"condition": ["control"]}),
            _tile_with_attributes({"condition": ["treated"]}),
        ]
        f = AttributeFilter(
            key="condition", values=[StringValue(value="control")], mode="Exclude"
        )
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert result[0].attributes["condition"] == ["treated"]

    def test_attribute_include_bool_value(self) -> None:
        tiles = [
            _tile_with_attributes({"flag": [True]}),
            _tile_with_attributes({"flag": [False]}),
        ]
        f = AttributeFilter(key="flag", values=[BoolValue()])
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert result[0].attributes["flag"] == [True]

    def test_attribute_is_none_value(self) -> None:
        tiles = [
            _tile_with_attributes({"condition": [None]}),
            _tile_with_attributes({"condition": ["treated"]}),
        ]
        f = AttributeFilter(key="condition", values=[IsNoneValue()])
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert result[0].attributes["condition"] == [None]

    def test_attribute_is_not_none_value(self) -> None:
        tiles = [
            _tile_with_attributes({"condition": [None]}),
            _tile_with_attributes({"condition": ["treated"]}),
        ]
        f = AttributeFilter(key="condition", values=[IsNotNoneValue()], mode="Exclude")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert result[0].attributes["condition"] == [None]

    def test_attribute_missing_key_error(self) -> None:
        tiles = [_tile_with_attributes({"other": ["x"]})]
        f = AttributeFilter(key="condition", values=[StringValue(value="control")])
        with pytest.raises(ValueError, match="no such attribute"):
            apply_filter_pipeline(tiles, filters_config=[f])


def _channel_tile(start_c: int, length_c: int, labels: list[str]) -> Tile[Any, Any]:
    details = AcquisitionDetails(
        channels=[ChannelInfo(channel_label=label) for label in labels],
        xy_pixel_size=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )
    return build_dummy_tile(
        fov_name="FOV_0",
        start=StartPosition(x=0, y=0, c=start_c),
        shape=TileShape(x=64, y=64, z=1, c=length_c, t=1),
        collection=SingleImage(image_path="img"),
        acquisition_details=details,
    )


class TestChannelFilters:
    def test_channel_include_keeps_matching(self) -> None:
        tiles = [
            _channel_tile(0, 1, ["DAPI", "GFP"]),
            _channel_tile(1, 1, ["DAPI", "GFP"]),
        ]
        f = ChannelFilter(channel_labels=["DAPI"])
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert result[0].start_c == 0

    def test_channel_exclude_removes_matching(self) -> None:
        tiles = [
            _channel_tile(0, 1, ["DAPI", "GFP"]),
            _channel_tile(1, 1, ["DAPI", "GFP"]),
        ]
        f = ChannelFilter(channel_labels=["DAPI"], mode="Exclude")
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert len(result) == 1
        assert result[0].start_c == 1

    def test_channel_filter_no_channels_error(self) -> None:
        tile = build_dummy_tile(
            fov_name="FOV_0",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=SingleImage(image_path="img"),
            acquisition_details=AcquisitionDetails(),
        )
        f = ChannelFilter(channel_labels=["DAPI"])
        with pytest.raises(ValueError, match="channels=None"):
            apply_filter_pipeline([tile], filters_config=[f])

    def test_channel_filter_partial_match_error(self) -> None:
        tiles = [_channel_tile(0, 2, ["DAPI", "GFP"])]
        f = ChannelFilter(channel_labels=["DAPI"])
        with pytest.raises(ValueError, match="whole tiles"):
            apply_filter_pipeline(tiles, filters_config=[f])


def _tile_at(z: float = 0.0, t: float = 0.0) -> Tile[Any, Any]:
    return build_dummy_tile(
        fov_name="FOV_0",
        start=StartPosition(x=0, y=0, z=z, t=t),
        shape=TileShape(x=64, y=64, z=1, c=1, t=1),
        collection=SingleImage(image_path="img"),
        acquisition_details=_acq(),
    )


class TestRangeFilters:
    def test_z_range_keeps_inside(self) -> None:
        tiles = [_tile_at(z=0.0), _tile_at(z=5.0), _tile_at(z=10.0)]
        f = ZRangeFilter(min_z=1.0, max_z=9.0)
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert [t.start_z for t in result] == [5.0]

    def test_z_range_unbounded_sides(self) -> None:
        tiles = [_tile_at(z=0.0), _tile_at(z=10.0)]
        assert (
            len(apply_filter_pipeline(tiles, filters_config=[ZRangeFilter(min_z=5.0)]))
            == 1
        )
        assert (
            len(apply_filter_pipeline(tiles, filters_config=[ZRangeFilter(max_z=5.0)]))
            == 1
        )
        assert len(apply_filter_pipeline(tiles, filters_config=[ZRangeFilter()])) == 2

    def test_t_range_keeps_inside(self) -> None:
        tiles = [_tile_at(t=0.0), _tile_at(t=5.0), _tile_at(t=10.0)]
        f = TRangeFilter(min_t=0.0, max_t=5.0)
        result = apply_filter_pipeline(tiles, filters_config=[f])
        assert [t.start_t for t in result] == [0.0, 5.0]

    def test_range_bounds_inclusive(self) -> None:
        tiles = [_tile_at(z=1.0), _tile_at(z=9.0)]
        f = ZRangeFilter(min_z=1.0, max_z=9.0)
        assert len(apply_filter_pipeline(tiles, filters_config=[f])) == 2
