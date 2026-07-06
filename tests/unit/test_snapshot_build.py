"""Tests for build_snapshot / run_converter_test against tiny on-disk OME-Zarrs.

Fixtures are built directly with ngio (no microscopy files, no converter
pipeline), and `run_converter_test` is driven by a stub `api_fn` that writes an
OME-Zarr and returns an init-task-style `image_list_updates`.
"""

import json
from pathlib import Path

import ngio
import numpy as np
import pytest

from ome_zarr_converters_tools import ConverterOptions
from ome_zarr_converters_tools.testing import (
    MultiPlateAssertionModel,
    MultiSingleImageAssertionModel,
    build_snapshot,
    run_converter_test,
)
from ome_zarr_converters_tools.testing._snapshot import _load_snapshot


def _make_container(
    store, *, channels=("DAPI", "GFP"), with_table=True, overwrite=False
):
    arr = np.arange(len(channels) * 32 * 32, dtype="uint16").reshape(
        len(channels), 32, 32
    )
    container = ngio.create_ome_zarr_from_array(
        store=store,
        array=arr,
        pixelsize=0.5,
        levels=1,
        axes_names=["c", "y", "x"],
        channels_meta=list(channels),
        overwrite=overwrite,
    )
    if with_table:
        container.add_table(
            "well_ROI_table", container.build_image_roi_table("well_ROI_table")
        )
    return container


def _make_plate(store, *, wells=(("B", "3", "0"),), overwrite=False):
    plate = ngio.create_empty_plate(store=store, name="P", overwrite=overwrite)
    for row, column, image_path in wells:
        plate.add_image(row=row, column=column, image_path=image_path)
        image_store = plate.get_image_store(
            row=row, column=column, image_path=image_path
        )
        _make_container(image_store, channels=("DAPI",))
    return plate


def _stub_single(*, zarr_dir, **_):
    img = Path(zarr_dir) / "img.zarr"
    _make_container(str(img), overwrite=True)
    return [
        {
            "image_list_updates": [
                {
                    "zarr_url": str(img),
                    "types": {"is_3D": False},
                    "attributes": {"foo": "bar"},
                }
            ]
        }
    ]


def _stub_plate(*, zarr_dir, **_):
    plate = Path(zarr_dir) / "plate.zarr"
    _make_plate(str(plate), overwrite=True)
    url = str(plate / "B" / "03" / "0")
    return [
        {
            "image_list_updates": [
                {
                    "zarr_url": url,
                    "types": {"is_3D": False},
                    "attributes": {"plate": "plate.zarr"},
                }
            ]
        }
    ]


# --------------------------------------------------------------------------- #
# build_snapshot
# --------------------------------------------------------------------------- #


def test_build_snapshot_single_image(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    _make_container(str(out / "img.zarr"))
    updates = [
        {"image_list_updates": [{"zarr_url": str(out / "img.zarr"), "types": {}}]}
    ]
    model = build_snapshot(
        zarr_dir=out, image_list_updates=updates, output_type="single_image"
    )
    assert isinstance(model, MultiSingleImageAssertionModel)
    img = model.images["img.zarr"]
    assert img.axes == ("c", "y", "x")
    assert img.shape == (2, 32, 32)
    assert img.pixelsize == (1.0, 1.0, 0.5, 0.5)
    assert img.channel_labels == ["DAPI", "GFP"]
    roi = img.tables["well_ROI_table"].rois["well_ROI_table"]
    assert len(roi.finger_print.hash) == 64


def test_build_snapshot_plate(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    _make_plate(str(out / "plate.zarr"))
    url = str(out / "plate.zarr" / "B" / "03" / "0")
    updates = [{"image_list_updates": [{"zarr_url": url, "types": {}}]}]
    model = build_snapshot(
        zarr_dir=out, image_list_updates=updates, output_type="plate"
    )
    assert isinstance(model, MultiPlateAssertionModel)
    plate = model.plates["plate.zarr"]
    assert set(plate.wells) == {"B/03"}
    assert "B/03/0" in plate.images


# --------------------------------------------------------------------------- #
# run_converter_test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "api_fn, output_type, image_key",
    [
        (_stub_single, "single_image", "img.zarr"),
        (_stub_plate, "plate", "B/03/0"),
    ],
)
def test_run_converter_test_roundtrip(tmp_path, api_fn, output_type, image_key):
    snap = tmp_path / "data" / "snapshots" / "snap.json"

    # update mode writes the JSON snapshot
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=api_fn,
        api_kwargs={},
        snapshot_path=snap,
        update_snapshots=True,
        output_type=output_type,
    )
    assert snap.exists()
    loaded = _load_snapshot(snap, output_type=output_type)
    assert loaded is not None

    # validate mode against the just-written snapshot passes
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=api_fn,
        api_kwargs={},
        snapshot_path=snap,
        update_snapshots=False,
        output_type=output_type,
    )


def test_run_converter_test_reports_mismatch(tmp_path):
    snap = tmp_path / "data" / "snapshots" / "snap.json"
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=_stub_single,
        api_kwargs={},
        snapshot_path=snap,
        update_snapshots=True,
        output_type="single_image",
    )
    data = json.loads(snap.read_text())
    key = next(iter(data["images"]))
    data["images"][key]["shape"][1] = 999
    snap.write_text(json.dumps(data))

    with pytest.raises(AssertionError) as exc:
        run_converter_test(
            tmp_path=tmp_path,
            api_fn=_stub_single,
            api_kwargs={},
            snapshot_path=snap,
            update_snapshots=False,
            output_type="single_image",
        )
    assert ".shape" in str(exc.value)


def test_run_converter_test_forwards_converter_options(tmp_path):
    # When converter_options is provided it is forwarded to api_fn; assert the
    # stub received it and the snapshot is written.
    received = {}

    def api_fn(*, zarr_dir, converter_options=None, **_):
        received["converter_options"] = converter_options
        return _stub_single(zarr_dir=zarr_dir)

    snap = tmp_path / "data" / "snapshots" / "snap.json"
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=api_fn,
        api_kwargs={},
        snapshot_path=snap,
        update_snapshots=True,
        converter_options=ConverterOptions(),
        output_type="single_image",
    )
    assert isinstance(received["converter_options"], ConverterOptions)
    assert snap.exists()


def test_run_converter_test_missing_snapshot(tmp_path):
    snap = tmp_path / "data" / "snapshots" / "nope.json"
    with pytest.raises(FileNotFoundError):
        run_converter_test(
            tmp_path=tmp_path,
            api_fn=_stub_single,
            api_kwargs={},
            snapshot_path=snap,
            update_snapshots=False,
            output_type="single_image",
        )
