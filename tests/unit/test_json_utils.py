"""Unit tests for fractal._json_utils: JSON serialization of TiledImages."""

import json
from pathlib import Path

import pytest

from ome_zarr_converters_tools.core._dummy_tiles import (
    DummyLoader,
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile_region import TiledImage
from ome_zarr_converters_tools.core._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.fractal._json_utils import (
    cleanup_if_exists,
    dump_to_json,
    remove_json,
    tiled_image_from_json,
)
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    ConverterOptions,
    SingleImage,
)


@pytest.fixture
def sample_tiled_image() -> TiledImage:
    acq = AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        pixelsize=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )
    coll = SingleImage(image_path="test_image")
    tiles = [
        build_dummy_tile(
            fov_name="FOV_0",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=coll,
            acquisition_details=acq,
        ),
    ]
    return tiled_image_from_tiles(tiles=tiles, converter_options=ConverterOptions())[0]


class TestDumpToJson:
    def test_creates_json_file(
        self, sample_tiled_image: TiledImage, tmp_path: Path
    ) -> None:
        json_url = str(tmp_path / "json_store")
        result = dump_to_json(json_url, sample_tiled_image)
        assert result.endswith(".json")
        assert Path(result).exists()

    def test_json_content_is_valid(
        self, sample_tiled_image: TiledImage, tmp_path: Path
    ) -> None:
        json_url = str(tmp_path / "json_store")
        result = dump_to_json(json_url, sample_tiled_image)
        with open(result) as f:
            data = json.load(f)
        assert "path" in data
        assert "regions" in data

    def test_multiple_dumps_create_unique_files(
        self, sample_tiled_image: TiledImage, tmp_path: Path
    ) -> None:
        json_url = str(tmp_path / "json_store")
        result1 = dump_to_json(json_url, sample_tiled_image)
        result2 = dump_to_json(json_url, sample_tiled_image)
        assert result1 != result2
        assert Path(result1).exists()
        assert Path(result2).exists()

    def test_s3_url_raises_not_implemented(
        self, sample_tiled_image: TiledImage
    ) -> None:
        with pytest.raises(NotImplementedError, match="S3"):
            dump_to_json("s3://bucket/store", sample_tiled_image)


class TestTiledImageFromJson:
    def test_roundtrip(self, sample_tiled_image: TiledImage, tmp_path: Path) -> None:
        json_url = str(tmp_path / "json_store")
        json_path = dump_to_json(json_url, sample_tiled_image)
        loaded = tiled_image_from_json(
            json_path,
            collection_type=SingleImage,
            image_loader_type=DummyLoader,
        )
        assert loaded.path == sample_tiled_image.path
        assert len(loaded.regions) == len(sample_tiled_image.regions)
        assert loaded.axes == sample_tiled_image.axes

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            tiled_image_from_json(
                str(tmp_path / "nonexistent.json"),
                collection_type=SingleImage,
                image_loader_type=DummyLoader,
            )

    def test_invalid_num_retries_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONVERTERS_TOOLS_NUM_RETRIES", "0")
        with pytest.raises(ValueError, match="NUM_RETRIES"):
            tiled_image_from_json(
                str(tmp_path / "test.json"),
                collection_type=SingleImage,
                image_loader_type=DummyLoader,
            )

    def test_s3_url_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="S3"):
            tiled_image_from_json(
                "s3://bucket/test.json",
                collection_type=SingleImage,
                image_loader_type=DummyLoader,
            )


class TestRemoveJson:
    def test_removes_file(self, sample_tiled_image: TiledImage, tmp_path: Path) -> None:
        json_url = str(tmp_path / "json_store")
        json_path = dump_to_json(json_url, sample_tiled_image)
        assert Path(json_path).exists()
        remove_json(json_path)
        assert not Path(json_path).exists()

    def test_removes_empty_parent_dir(
        self, sample_tiled_image: TiledImage, tmp_path: Path
    ) -> None:
        json_url = str(tmp_path / "json_store")
        json_path = dump_to_json(json_url, sample_tiled_image)
        parent = Path(json_path).parent
        remove_json(json_path)
        assert not parent.exists()

    def test_nonexistent_file_does_not_raise(self, tmp_path: Path) -> None:
        # Should log an error but not raise
        remove_json(str(tmp_path / "nonexistent.json"))

    def test_s3_url_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        remove_json("s3://bucket/test.json")
        assert "not implemented" in caplog.text.lower()

    def test_unknown_url_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        remove_json("gcs://bucket/test.json")
        assert "not implemented" in caplog.text.lower()


class TestCleanupIfExists:
    def test_cleans_directory(
        self, sample_tiled_image: TiledImage, tmp_path: Path
    ) -> None:
        json_url = str(tmp_path / "json_store")
        dump_to_json(json_url, sample_tiled_image)
        dump_to_json(json_url, sample_tiled_image)
        json_store = tmp_path / "json_store"
        assert json_store.exists()
        cleanup_if_exists(json_url)
        assert not json_store.exists()

    def test_nonexistent_directory_does_not_raise(self, tmp_path: Path) -> None:
        cleanup_if_exists(str(tmp_path / "nonexistent_store"))

    def test_cleanup_with_permission_error_logs(
        self,
        sample_tiled_image: TiledImage,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        json_url = str(tmp_path / "json_store")
        dump_to_json(json_url, sample_tiled_image)

        def _failing_iterdir(self):
            raise PermissionError("no access")

        monkeypatch.setattr(Path, "iterdir", _failing_iterdir)
        cleanup_if_exists(json_url)
        assert "error" in caplog.text.lower()

    def test_s3_url_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        cleanup_if_exists("s3://bucket/store")
        assert "not implemented" in caplog.text.lower()

    def test_unknown_url_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        cleanup_if_exists("gcs://bucket/store")
        assert "not implemented" in caplog.text.lower()
