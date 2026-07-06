"""Unit tests for the shared snapshot testing helpers."""

import numpy as np
import pytest

from ome_zarr_converters_tools.testing import (
    FingerprintModel,
    MultiPlateAssertionModel,
    MultiSingleImageAssertionModel,
    PlateAssertionModel,
    compare_snapshots,
    plugin,
)
from ome_zarr_converters_tools.testing._snapshot import _load_snapshot, _write_snapshot


def _plate_with(**image_overrides) -> MultiPlateAssertionModel:
    return MultiPlateAssertionModel(
        plates={
            "plate.zarr": {
                "wells": ["B/03"],
                "images": {"B/03/0": _image(**image_overrides)},
            }
        }
    )


def _single_model(**image_overrides) -> MultiSingleImageAssertionModel:
    return MultiSingleImageAssertionModel(
        images={"img.zarr": _image(**image_overrides)}
    )


def _roi(image: dict) -> dict:
    return image["tables"]["well_ROI_table"]["rois"]["image"]


def _image(**overrides) -> dict:
    entry = {
        "axes": ["c", "z", "y", "x"],
        "shape": [2, 1, 512, 1024],
        "pixelsize": [1.0, 1.0, 0.65, 0.65],
        "channel_labels": ["DAPI", "GFP"],
        "wavelength_ids": ["438.0", "511.0"],
        "types": {"is_3D": False},
        "attributes": {},
        "tables": {
            "well_ROI_table": {
                "rois": {
                    "image": {
                        "slice_repr": "[x: 0.0->1024.0, y: 0.0->512.0]",
                        "finger_print": {
                            "mean": 10.0,
                            "std": 2.0,
                            "min": 0.0,
                            "max": 20.0,
                            "hash": "abc123",
                        },
                        "yx_origin": None,
                    }
                }
            }
        },
    }
    entry.update(overrides)
    return entry


def _plate_model() -> MultiPlateAssertionModel:
    return MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": _image()}}}
    )


def test_fingerprint_rounds_stats_and_is_stable():
    arr = np.arange(100, dtype="uint16").reshape(10, 10)
    fp1 = FingerprintModel.from_array(arr)
    fp2 = FingerprintModel.from_array(arr)
    assert fp1.hash == fp2.hash
    assert fp1.mean == 49.5


def test_identical_snapshots_have_no_diffs():
    assert compare_snapshots(_plate_model(), _plate_model()) == []


def test_json_round_trip(tmp_path):
    model = _plate_model()
    path = tmp_path / "snap.json"
    _write_snapshot(model, path)
    assert path.read_text().endswith("\n")
    loaded = _load_snapshot(path, output_type="plate")
    assert compare_snapshots(model, loaded) == []


def test_shape_mismatch_is_reported_with_path():
    expected = _plate_model()
    actual = MultiPlateAssertionModel(
        plates={
            "plate.zarr": {
                "wells": ["B/03"],
                "images": {"B/03/0": _image(shape=[2, 1, 512, 1000])},
            }
        }
    )
    diffs = compare_snapshots(expected, actual)
    assert len(diffs) == 1
    assert "plates['plate.zarr'].images['B/03/0'].shape" in diffs[0]


def test_stats_within_tolerance_pass_but_hash_is_exact():
    expected = _plate_model()
    # Perturb the stored stats below tolerance but change the hash.
    img = _image()
    img["tables"]["well_ROI_table"]["rois"]["image"]["finger_print"].update(
        {"mean": 10.0000001, "hash": "different"}
    )
    actual = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": img}}}
    )
    diffs = compare_snapshots(expected, actual)
    assert len(diffs) == 1
    assert ".finger_print.hash" in diffs[0]


def test_missing_and_extra_images_reported():
    expected = _plate_model()
    actual = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/1": _image()}}}
    )
    diffs = compare_snapshots(expected, actual)
    assert any(".images:" in d for d in diffs)


def test_type_mismatch_between_plate_and_single_image():
    diffs = compare_snapshots(_plate_model(), MultiSingleImageAssertionModel(images={}))
    assert len(diffs) == 1
    assert "snapshot type mismatch" in diffs[0]


def test_images_common_is_deep_merged():
    plate = PlateAssertionModel(
        **{
            "wells": ["B/03"],
            "images_common": {"types": {"is_3D": True}},
            "images": {"B/03/0": _image(types={"is_3D": True})},
        }
    )
    assert plate.images["B/03/0"].types == {"is_3D": True}


# --------------------------------------------------------------------------- #
# Comparison branches
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "override, needle",
    [
        ({"axes": ["x", "y", "z", "c"]}, ".axes"),
        ({"pixelsize": [1.0, 1.0, 0.99, 0.65]}, ".pixelsize"),
        ({"channel_labels": ["DAPI", "RFP"]}, ".channel_labels"),
        ({"wavelength_ids": ["438.0", "999.0"]}, ".wavelength_ids"),
        ({"types": {"is_3D": True}}, ".types"),
        ({"attributes": {"x": 1}}, ".attributes"),
    ],
)
def test_image_field_mismatch_reported(override, needle):
    diffs = compare_snapshots(_plate_model(), _plate_with(**override))
    assert any(needle in d for d in diffs), diffs


def test_wells_mismatch_reported():
    actual = MultiPlateAssertionModel(
        plates={
            "plate.zarr": {
                "wells": ["B/03", "B/04"],
                "images": {"B/03/0": _image()},
            }
        }
    )
    diffs = compare_snapshots(_plate_model(), actual)
    assert any(".wells" in d for d in diffs), diffs


def test_table_set_mismatch_reported():
    diffs = compare_snapshots(_plate_model(), _plate_with(tables={}))
    assert any(".tables:" in d for d in diffs), diffs


def test_roi_set_mismatch_reported():
    img = _image()
    img["tables"]["well_ROI_table"]["rois"] = {"other": _roi(_image())}
    actual = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": img}}}
    )
    diffs = compare_snapshots(_plate_model(), actual)
    assert any(".rois:" in d for d in diffs), diffs


def test_expected_none_table_is_skipped():
    # Expected table is None (unreadable) -> comparison is skipped, no diff.
    assert (
        compare_snapshots(_plate_with(tables={"well_ROI_table": None}), _plate_model())
        == []
    )


def test_actual_none_table_is_reported():
    diffs = compare_snapshots(
        _plate_model(), _plate_with(tables={"well_ROI_table": None})
    )
    assert any("could not be read" in d for d in diffs), diffs


def test_slice_repr_mismatch_reported():
    img = _image()
    _roi(img)["slice_repr"] = "different"
    actual = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": img}}}
    )
    diffs = compare_snapshots(_plate_model(), actual)
    assert any(".slice_repr" in d for d in diffs), diffs


def test_fingerprint_stat_outside_tolerance_reported():
    img = _image()
    _roi(img)["finger_print"]["mean"] = 10.5
    actual = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": img}}}
    )
    diffs = compare_snapshots(_plate_model(), actual)
    assert any(".finger_print.mean" in d for d in diffs), diffs


def test_yx_origin_mismatch_reported():
    exp_img, act_img = _image(), _image()
    _roi(exp_img)["yx_origin"] = [1.0, 2.0]
    _roi(act_img)["yx_origin"] = [3.0, 4.0]
    exp = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": exp_img}}}
    )
    act = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": act_img}}}
    )
    diffs = compare_snapshots(exp, act)
    assert any(".yx_origin" in d for d in diffs), diffs


def test_yx_origin_none_expected_is_skipped():
    act_img = _image()
    _roi(act_img)["yx_origin"] = [1.0, 2.0]
    act = MultiPlateAssertionModel(
        plates={"plate.zarr": {"wells": ["B/03"], "images": {"B/03/0": act_img}}}
    )
    assert compare_snapshots(_plate_model(), act) == []


def test_single_image_set_mismatch_reported():
    actual = MultiSingleImageAssertionModel(images={"other.zarr": _image()})
    diffs = compare_snapshots(_single_model(), actual)
    assert any("images:" in d for d in diffs), diffs


def test_single_image_field_mismatch_reported():
    actual = MultiSingleImageAssertionModel(
        images={"img.zarr": _image(shape=[2, 1, 512, 1000])}
    )
    diffs = compare_snapshots(_single_model(), actual)
    assert any("images['img.zarr'].shape" in d for d in diffs), diffs


# --------------------------------------------------------------------------- #
# Pytest plugin hooks
# --------------------------------------------------------------------------- #


class _FakeParser:
    def __init__(self):
        self.options: list[str] = []

    def addoption(self, name, **kwargs):
        self.options.append(name)


class _FakeConfig:
    def __init__(self, *, extended: bool):
        self._extended = extended
        self.marker_lines: list[str] = []

    def getoption(self, name):
        return self._extended

    def addinivalue_line(self, section, line):
        self.marker_lines.append(line)


class _FakeItem:
    def __init__(self, keywords):
        self.keywords = keywords
        self.markers: list = []

    def add_marker(self, marker):
        self.markers.append(marker)


def test_plugin_addoption_registers_options():
    parser = _FakeParser()
    plugin.pytest_addoption(parser)
    assert "--update-snapshots" in parser.options
    assert "--extended" in parser.options


def test_plugin_configure_registers_marker():
    config = _FakeConfig(extended=False)
    plugin.pytest_configure(config)
    assert any("extended" in line for line in config.marker_lines)


def test_plugin_skips_extended_without_flag():
    item = _FakeItem(keywords={"extended"})
    plugin.pytest_collection_modifyitems(_FakeConfig(extended=False), [item])
    assert len(item.markers) == 1


def test_plugin_keeps_extended_with_flag():
    item = _FakeItem(keywords={"extended"})
    plugin.pytest_collection_modifyitems(_FakeConfig(extended=True), [item])
    assert item.markers == []


def test_update_snapshots_fixture_default(update_snapshots):
    # The fixture comes from the plugin (pytest11 entry point); default is False.
    assert update_snapshots is False
