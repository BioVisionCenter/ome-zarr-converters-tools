from pathlib import Path

import pytest
from ngio import PixelSize, open_ome_zarr_container
from ngio.utils import NgioFileExistsError
from utils import (
    DummyLoader,
    PlatePathBuilder,
    TiledImage,
    generate_tiled_image,
)

from fractal_converters_tools import Point, Tile
from fractal_converters_tools._omezarr_image_writers import write_tiled_image
from fractal_converters_tools._stitching import standard_stitching_pipe


def test_write_image(tmp_path):
    plate_path = tmp_path / "test_write_images"
    tiled_image = generate_tiled_image(
        plate_name="plate_1",
        row="A",
        column=1,
        acquisition_id=0,
        tiled_image_name="image_1",
        z_offset=2.5,
        t_offset=1,
    )
    image_url = plate_path / tiled_image.path
    im_list_types = write_tiled_image(
        zarr_url=str(image_url),
        tiled_image=tiled_image,
        stiching_pipe=standard_stitching_pipe,
    )
    assert image_url.exists()
    assert len(im_list_types) == 2
    assert im_list_types["is_3D"] is False
    assert im_list_types["has_time"] is False

    ome_zarr_container = open_ome_zarr_container(image_url)
    assert len(ome_zarr_container.list_tables()) == 2
    assert set(ome_zarr_container.list_tables()) == {"well_ROI_table", "FOV_ROI_table"}

    image = ome_zarr_container.get_image()
    assert image.shape == (1, 1, 11 * 2, 10 * 2)

    roi_table = ome_zarr_container.get_table("FOV_ROI_table", check_type="roi_table")
    assert len(roi_table.rois()) == 4

    for roi in roi_table.rois():
        roi_array = image.get_roi(roi)
        assert roi.z == 0
        assert roi.z_micrometer_original == 2.5
        assert roi_array.shape == (1, 1, 11, 10)

    roi_df = roi_table.dataframe
    assert set(roi_df.columns) == {
        "len_x_micrometer",
        "len_y_micrometer",
        "len_z_micrometer",
        "x_micrometer",
        "y_micrometer",
        "z_micrometer",
        "x_micrometer_original",
        "y_micrometer_original",
        "z_micrometer_original",
    }, roi_df.columns


def test_write_image_with_rounding_errors(tmp_path):
    """Test if the shape can fix rounding errors in the pixel size/position."""
    plate_path = tmp_path / "test_write_images"
    path_builder = PlatePathBuilder(
        plate_name="plate_1",
        row="A",
        column=1,
        acquisition_id=0,
    )
    tiled_image = TiledImage(
        name="image_1",
        path_builder=path_builder,
        channel_names=["channel1"],
        wavelength_ids=["wavelength1"],
        attributes={"cell_line": "cell_line_1"},
    )
    top_l = Point(x=645.814, y=-645.814, z=0, c=0, t=0)
    bot_r = Point(
        x=+645.814 + 1291.628,
        y=-645.814 + 1291.6283348666052,
        z=1.0,
        c=1,
        t=1,
    )
    tiled_image.add_tile(
        Tile.from_points(
            top_l=top_l,
            bot_r=bot_r,
            pixel_size=PixelSize(x=1.195952, y=1.195952, z=1.0),
            shape=(1, 1, 1, 1080, 1080),
            data_loader=DummyLoader(shape=(1, 1, 1, 1080, 1080)),
        )
    )

    image_url = plate_path / tiled_image.path
    write_tiled_image(
        zarr_url=str(image_url),
        tiled_image=tiled_image,
        stiching_pipe=standard_stitching_pipe,
    )


def test_write_advanced_params(tmp_path):
    plate_path = tmp_path / "test_write_images"
    tiled_image = generate_tiled_image(
        plate_name="plate_1",
        row="A",
        column=1,
        acquisition_id=0,
        tiled_image_name="image_1",
    )

    image_url = plate_path / tiled_image.path
    _ = write_tiled_image(
        zarr_url=str(image_url),
        tiled_image=tiled_image,
        stiching_pipe=standard_stitching_pipe,
        num_levels=2,
        max_xy_chunk=2,
        z_chunk=11,
        c_chunk=4,
        t_chunk=3,
    )
    assert image_url.exists()

    ome_zarr_container = open_ome_zarr_container(image_url)
    assert ome_zarr_container.levels == 2
    image = ome_zarr_container.get_image()
    assert image.chunks == (1, 1, 2, 2)


def test_write_fail_overwrite(tmp_path):
    plate_dir = tmp_path / "test_write_images"
    tiled_image = generate_tiled_image(
        plate_name="plate_1",
        row="A",
        column=1,
        acquisition_id=0,
        tiled_image_name="image_1",
    )

    image_url = plate_dir / tiled_image.path

    _ = write_tiled_image(
        zarr_url=str(image_url),
        tiled_image=tiled_image,
        stiching_pipe=standard_stitching_pipe,
        num_levels=2,
    )
    assert Path(image_url).exists()

    with pytest.raises(NgioFileExistsError):
        _ = write_tiled_image(
            zarr_url=str(image_url),
            tiled_image=tiled_image,
            stiching_pipe=standard_stitching_pipe,
            num_levels=2,
            overwrite=False,
        )
