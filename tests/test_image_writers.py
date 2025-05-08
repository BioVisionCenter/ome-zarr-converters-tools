from pathlib import Path

import pytest
from ngio import open_ome_zarr_container
from ngio.utils import NgioFileExistsError
from utils import generate_tiled_image

from fractal_converters_tools.omezarr_image_writers import write_tiled_image
from fractal_converters_tools.stitching import standard_stitching_pipe


def test_write_image(tmp_path):
    images_path = tmp_path / "test_write_images"
    tiled_image = generate_tiled_image(
        plate_name="plate_1",
        row="A",
        column=1,
        acquisition_id=0,
        tiled_image_name="image_1",
    )

    path, _, _ = write_tiled_image(
        zarr_dir=images_path,
        tiled_image=tiled_image,
        stiching_pipe=standard_stitching_pipe,
    )
    assert Path(path).exists()

    ome_zarr_container = open_ome_zarr_container(path)
    assert len(ome_zarr_container.list_tables()) == 2
    assert set(ome_zarr_container.list_tables()) == {"well_ROI_table", "FOV_ROI_table"}

    image = ome_zarr_container.get_image()
    assert image.shape == (1, 1, 11 * 2, 10 * 2)

    roi_table = ome_zarr_container.get_table("FOV_ROI_table")
    assert len(roi_table.rois()) == 4
    for roi in roi_table.rois():
        roi_array = image.get_roi(roi)
        assert roi_array.shape == (1, 1, 11, 10)


def test_write_advanced_params(tmp_path):
    images_path = tmp_path / "test_write_images"
    tiled_image = generate_tiled_image(
        plate_name="plate_1",
        row="A",
        column=1,
        acquisition_id=0,
        tiled_image_name="image_1",
    )

    path, _, _ = write_tiled_image(
        zarr_dir=images_path,
        tiled_image=tiled_image,
        stiching_pipe=standard_stitching_pipe,
        num_levels=2,
        max_xy_chunk=2,
        z_chunk=11,
        c_chunk=4,
        t_chunk=3,
    )
    assert Path(path).exists()

    ome_zarr_container = open_ome_zarr_container(path)
    assert ome_zarr_container.levels == 2
    image = ome_zarr_container.get_image()
    assert image.chunks == (1, 1, 2, 2)


def test_write_fail_overwrite(tmp_path):
    images_path = tmp_path / "test_write_images"
    tiled_image = generate_tiled_image(
        plate_name="plate_1",
        row="A",
        column=1,
        acquisition_id=0,
        tiled_image_name="image_1",
    )

    path, _, _ = write_tiled_image(
        zarr_dir=images_path,
        tiled_image=tiled_image,
        stiching_pipe=standard_stitching_pipe,
        num_levels=2,
    )
    assert Path(path).exists()

    with pytest.raises(NgioFileExistsError):
        path, _, _ = write_tiled_image(
            zarr_dir=images_path,
            tiled_image=tiled_image,
            stiching_pipe=standard_stitching_pipe,
            num_levels=2,
            overwrite=False,
        )
