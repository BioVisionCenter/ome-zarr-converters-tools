from pathlib import Path

import pytest
from ngio.utils import NgioFileExistsError
from utils import generate_tiled_image

from fractal_converters_tools.task_common_models import (
    AdvancedComputeOptions,
    ConvertParallelInitArgs,
)
from fractal_converters_tools.task_compute_tools import generic_compute_task
from fractal_converters_tools.task_init_tools import build_parallelization_list


def test_compute(tmp_path):
    images_path = tmp_path / "test_write_images"

    tiled_images = [
        generate_tiled_image(
            plate_name="plate_1",
            row="A",
            column=0,
            acquisition_id=0,
            tiled_image_name="image_1",
        )
    ]

    adv_comp_model = AdvancedComputeOptions()

    par_args = build_parallelization_list(
        zarr_dir=images_path,
        tiled_images=tiled_images,
        overwrite=False,
        advanced_compute_options=adv_comp_model,
    )[0]

    zarr_url = par_args["zarr_url"]
    init_args = ConvertParallelInitArgs(**par_args["init_args"])
    image_list_updates = generic_compute_task(zarr_url=zarr_url, init_args=init_args)

    assert "image_list_updates" in image_list_updates
    assert len(image_list_updates["image_list_updates"]) == 1

    new_zarr_url = image_list_updates["image_list_updates"][0]["zarr_url"]
    p_types = image_list_updates["image_list_updates"][0]["types"]
    attributes = image_list_updates["image_list_updates"][0]["attributes"]

    assert Path(new_zarr_url).exists()
    assert p_types == {"is_3D": False}
    assert attributes == {"well": "A0", "plate": "plate_1.zarr"}

    par_args = build_parallelization_list(
        zarr_dir=images_path,
        tiled_images=tiled_images,
        overwrite=False,
        advanced_compute_options=adv_comp_model,
    )[0]
    zarr_url = par_args["zarr_url"]
    init_args = ConvertParallelInitArgs(**par_args["init_args"])
    with pytest.raises(NgioFileExistsError):
        generic_compute_task(zarr_url=zarr_url, init_args=init_args)
