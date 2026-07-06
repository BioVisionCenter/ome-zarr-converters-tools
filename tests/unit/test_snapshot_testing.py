"""Unit tests for the shared snapshot testing helpers."""

import numpy as np
import pytest

from ome_zarr_converters_tools.testing import (
    FingerprintModel,
    MultiPlateAssertionModel,
    MultiSingleImageAssertionModel,
    PlateAssertionModel,
    compare_snapshots,
)
from ome_zarr_converters_tools.testing._snapshot import _load_snapshot, _write_snapshot


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
