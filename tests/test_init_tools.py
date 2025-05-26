import pickle
from pathlib import Path

import pytest
from utils import generate_tiled_image

from fractal_converters_tools._task_common_models import (
    AdvancedComputeOptions,
    ConvertParallelInitArgs,
)
from fractal_converters_tools._task_init_tools import build_parallelization_list
from fractal_converters_tools._tiled_image import TiledImage


@pytest.mark.parametrize(
    "overwrite, tm_dir_name",
    [
        (True, "_tmp_converter_dir"),
        (False, "_tmp_converter_dir_test"),
    ],
)
def test_build_par_list(tmp_path, overwrite, tm_dir_name):
    images_path = tmp_path / "test_write_images"

    tiled_images = []
    for i in range(1, 3):
        tiled_image = generate_tiled_image(
            plate_name="plate_1",
            row="A",
            column=i,
            acquisition_id=0,
            tiled_image_name="image_1",
        )
        tiled_images.append(tiled_image)

    adv_comp_model = AdvancedComputeOptions()

    par_list = build_parallelization_list(
        zarr_dir=images_path,
        tiled_images=tiled_images,
        overwrite=overwrite,
        advanced_compute_options=adv_comp_model,
        tmp_dir_name=tm_dir_name,
    )

    for tiled_image, par_args in zip(tiled_images, par_list, strict=True):
        init_args = par_args["init_args"]
        init_args = ConvertParallelInitArgs(**init_args)
        assert Path(init_args.tiled_image_pickled_path).exists()
        assert init_args.overwrite == overwrite
        assert init_args.advanced_compute_options == adv_comp_model

        with open(init_args.tiled_image_pickled_path, "rb") as f:
            tiled_image = pickle.load(f)
            assert isinstance(tiled_image, TiledImage)
            # This is just a proxy to check the equality of the object
            assert str(tiled_image) == str(tiled_image)

        if tm_dir_name is not None:
            assert (
                images_path / tm_dir_name
                == Path(init_args.tiled_image_pickled_path).parent
            )


def test_clenup_par_list(tmp_path):
    images_path = tmp_path / "test_write_images"

    tiled_images = []
    for i in range(1, 3):
        tiled_image = generate_tiled_image(
            plate_name="plate_1",
            row="A",
            column=i,
            acquisition_id=0,
            tiled_image_name="image_1",
        )
        tiled_images.append(tiled_image)

    adv_comp_model = AdvancedComputeOptions()
    _ = build_parallelization_list(
        zarr_dir=images_path,
        tiled_images=tiled_images,
        overwrite=False,
        advanced_compute_options=adv_comp_model,
    )
    # Recomupute should clean up the temp dir
    par_list = build_parallelization_list(
        zarr_dir=images_path,
        tiled_images=tiled_images,
        overwrite=False,
        advanced_compute_options=adv_comp_model,
    )

    assert (images_path / "_tmp_converter_dir").exists()
    assert len(list((images_path / "_tmp_converter_dir").iterdir())) == len(par_list)
