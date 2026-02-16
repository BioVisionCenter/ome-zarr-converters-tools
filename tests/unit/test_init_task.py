"""Unit tests for fractal._init_task."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.fractal._init_task import (
    build_parallelization_list,
    setup_images_for_conversion,
)
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    ConverterOptions,
    OverwriteMode,
    SingleImage,
)


def _make_tiled_images(num: int = 1) -> list:
    acq = AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        pixelsize=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )
    tiles = []
    for i in range(num):
        coll = SingleImage(image_path=f"image_{i}")
        tile = build_dummy_tile(
            fov_name=f"FOV_{i}",
            start=StartPosition(x=i * 100, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=coll,
            acquisition_details=acq,
        )
        tiles.append(tile)

    return tiled_image_from_tiles(tiles=tiles, converter_options=ConverterOptions())


class TestBuildParallelizationList:
    def test_single_image(self, tmp_path: Path) -> None:
        images = _make_tiled_images(1)
        zarr_dir = str(tmp_path / "output.zarr")

        result = build_parallelization_list(
            images,
            zarr_dir=zarr_dir,
            converter_options=ConverterOptions(),
        )

        assert len(result) == 1
        assert "zarr_url" in result[0]
        assert "init_args" in result[0]

    def test_multiple_images(self, tmp_path: Path) -> None:
        images = _make_tiled_images(3)
        zarr_dir = str(tmp_path / "output.zarr")

        result = build_parallelization_list(
            images,
            zarr_dir=zarr_dir,
            converter_options=ConverterOptions(),
        )

        assert len(result) == 3
        urls = [r["zarr_url"] for r in result]
        assert len(set(urls)) == 3  # All unique

    def test_zarr_url_contains_image_path(self, tmp_path: Path) -> None:
        images = _make_tiled_images(1)
        zarr_dir = str(tmp_path / "output.zarr")

        result = build_parallelization_list(
            images,
            zarr_dir=zarr_dir,
            converter_options=ConverterOptions(),
        )

        assert images[0].path in result[0]["zarr_url"]

    def test_init_args_has_json_dump_url(self, tmp_path: Path) -> None:
        images = _make_tiled_images(1)
        zarr_dir = str(tmp_path / "output.zarr")

        result = build_parallelization_list(
            images,
            zarr_dir=zarr_dir,
            converter_options=ConverterOptions(),
        )

        init_args = result[0]["init_args"]
        assert "tiled_image_json_dump_url" in init_args
        assert init_args["tiled_image_json_dump_url"].endswith(".json")

    def test_init_args_contains_converter_options(self, tmp_path: Path) -> None:
        images = _make_tiled_images(1)
        zarr_dir = str(tmp_path / "output.zarr")

        result = build_parallelization_list(
            images,
            zarr_dir=zarr_dir,
            converter_options=ConverterOptions(),
        )

        init_args = result[0]["init_args"]
        assert "converter_options" in init_args
        assert "overwrite_mode" in init_args

    def test_overwrite_mode_passed_through(self, tmp_path: Path) -> None:
        images = _make_tiled_images(1)
        zarr_dir = str(tmp_path / "output.zarr")

        result = build_parallelization_list(
            images,
            zarr_dir=zarr_dir,
            converter_options=ConverterOptions(),
            overwrite_mode=OverwriteMode.OVERWRITE,
        )

        init_args = result[0]["init_args"]
        assert init_args["overwrite_mode"] == OverwriteMode.OVERWRITE

    def test_json_files_created(self, tmp_path: Path) -> None:
        images = _make_tiled_images(2)
        zarr_dir = str(tmp_path / "output.zarr")

        result = build_parallelization_list(
            images,
            zarr_dir=zarr_dir,
            converter_options=ConverterOptions(),
        )

        for entry in result:
            json_path = entry["init_args"]["tiled_image_json_dump_url"]
            assert Path(json_path).exists()


class TestSetupImagesForConversion:
    @patch("ome_zarr_converters_tools.fractal._init_task.setup_ome_zarr_collection")
    def test_calls_setup_collection(
        self, mock_setup: MagicMock, tmp_path: Path
    ) -> None:
        images = _make_tiled_images(1)
        zarr_dir = str(tmp_path / "output.zarr")

        setup_images_for_conversion(
            images,
            zarr_dir=zarr_dir,
            collection_type="SingleImage",
            converter_options=ConverterOptions(),
        )

        mock_setup.assert_called_once()
        call_kwargs = mock_setup.call_args[1]
        assert call_kwargs["collection_type"] == "SingleImage"
        assert call_kwargs["zarr_dir"] == zarr_dir

    @patch("ome_zarr_converters_tools.fractal._init_task.setup_ome_zarr_collection")
    def test_returns_parallelization_list(
        self, mock_setup: MagicMock, tmp_path: Path
    ) -> None:
        images = _make_tiled_images(2)
        zarr_dir = str(tmp_path / "output.zarr")

        result = setup_images_for_conversion(
            images,
            zarr_dir=zarr_dir,
            collection_type="SingleImage",
            converter_options=ConverterOptions(),
        )

        assert len(result) == 2
        for entry in result:
            assert "zarr_url" in entry
            assert "init_args" in entry

    @patch("ome_zarr_converters_tools.fractal._init_task.setup_ome_zarr_collection")
    def test_overwrite_mode_forwarded(
        self, mock_setup: MagicMock, tmp_path: Path
    ) -> None:
        images = _make_tiled_images(1)
        zarr_dir = str(tmp_path / "output.zarr")

        setup_images_for_conversion(
            images,
            zarr_dir=zarr_dir,
            collection_type="SingleImage",
            converter_options=ConverterOptions(),
            overwrite_mode=OverwriteMode.EXTEND,
        )

        call_kwargs = mock_setup.call_args[1]
        assert call_kwargs["overwrite_mode"] == OverwriteMode.EXTEND
