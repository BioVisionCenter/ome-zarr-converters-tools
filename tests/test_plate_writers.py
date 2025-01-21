import json

import pytest
from utils import generate_tiled_images

from fractal_converters_tools.omezarr_plate_writers import initiate_ome_zarr_plates


def test_init_plate(tmp_path):
    plate_path = tmp_path / "test_write_plate"
    tiled_images = generate_tiled_images(
        plate_name="plate_1", rows=["A", "B"], columns=[1, 2], acquisition_ids=[0, 0]
    )

    initiate_ome_zarr_plates(plate_path, tiled_images=tiled_images, overwrite=True)

    plate_path_attr = plate_path / "plate_1.zarr" / ".zattrs"
    assert plate_path_attr.exists()
    with open(plate_path_attr) as f:
        attrs = json.load(f)

    expected_attrs = {
        "plate": {
            "acquisitions": [{"id": 0, "name": "plate_1_id0"}],
            "columns": [{"name": "1"}, {"name": "2"}],
            "name": "plate_1",
            "rows": [{"name": "A"}, {"name": "B"}],
            "version": "0.4.0",
            "wells": [
                {"columnIndex": 0, "path": "A/1", "rowIndex": 0},
                {"columnIndex": 1, "path": "B/2", "rowIndex": 1},
            ],
        }
    }
    assert attrs == expected_attrs

    well_path_attr = plate_path / "plate_1.zarr" / "A" / "1" / ".zattrs"
    assert well_path_attr.exists()
    with open(well_path_attr) as f:
        attrs = json.load(f)

    expected_attrs = {
        "well": {"images": [{"acquisition": 0, "path": "0"}], "version": "0.4.0"}
    }
    assert attrs == expected_attrs


def test_init_multi_plates(tmp_path):
    plate_path = tmp_path / "test_write_multi_plate"
    tiled_images = generate_tiled_images(
        plate_name="plate_1", rows=["A", "B"], columns=[1, 2], acquisition_ids=[0, 0]
    )
    tiled_images2 = generate_tiled_images(
        plate_name="plate_2", rows=["A", "B"], columns=[1, 2], acquisition_ids=[0, 0]
    )

    tiled_images = tiled_images + tiled_images2
    initiate_ome_zarr_plates(plate_path, tiled_images=tiled_images, overwrite=True)

    for plate_name in ["plate_1", "plate_2"]:
        plate_path_attr = plate_path / f"{plate_name}.zarr" / ".zattrs"
        with open(plate_path_attr) as f:
            attrs = json.load(f)

        expected_attrs = {
            "plate": {
                "acquisitions": [{"id": 0, "name": f"{plate_name}_id0"}],
                "columns": [{"name": "1"}, {"name": "2"}],
                "name": plate_name,
                "rows": [{"name": "A"}, {"name": "B"}],
                "version": "0.4.0",
                "wells": [
                    {"columnIndex": 0, "path": "A/1", "rowIndex": 0},
                    {"columnIndex": 1, "path": "B/2", "rowIndex": 1},
                ],
            }
        }
        assert attrs == expected_attrs


def test_overwrite_fail(tmp_path):
    plate_path = tmp_path / "test_write_plate.zarr"
    tiled_images = generate_tiled_images(
        plate_name="plate_1", rows=["A", "B"], columns=[1, 2], acquisition_ids=[0, 0]
    )

    initiate_ome_zarr_plates(plate_path, tiled_images=tiled_images, overwrite=True)

    with pytest.raises(FileExistsError):
        initiate_ome_zarr_plates(plate_path, tiled_images=tiled_images, overwrite=False)
