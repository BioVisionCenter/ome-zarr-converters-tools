from pathlib import Path

from ngio import NgffImage
from utils import generate_tiled_image

from fractal_converters_tools.omezarr_image_writers import write_tiled_image
from fractal_converters_tools.stitching import standard_stitching_pipe


def test_init_plate(tmp_path):
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

    ngff_image = NgffImage(path)
    assert len(ngff_image.tables.list()) == 2
    assert set(ngff_image.tables.list()) == {"well_ROI_table", "FOV_ROI_table"}

    image = ngff_image.get_image()
    assert image.shape == (1, 1, 11 * 2, 10 * 2)

    roi_table = ngff_image.tables.get_table("FOV_ROI_table")
    assert len(roi_table.rois) == 4
    for roi in roi_table.rois:
        assert image.get_array_from_roi(roi).shape == (1, 1, 11, 10)
