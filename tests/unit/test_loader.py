"""Unit tests for models._loader."""

from pathlib import Path

import numpy as np
import pytest

from ome_zarr_converters_tools.models._loader import (
    DefaultImageLoader,
    ImageLoaderInterface,
)


class TestImageLoaderInterface:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            ImageLoaderInterface()  # type: ignore[abstract]

    def test_find_data_type_delegates_to_load_data(self) -> None:
        class StubLoader(ImageLoaderInterface):
            def load_data(self, resource=None):
                return np.zeros((2, 2), dtype=np.uint16)

        loader = StubLoader()
        assert loader.find_data_type() == "uint16"

    def test_find_data_type_float(self) -> None:
        class StubLoader(ImageLoaderInterface):
            def load_data(self, resource=None):
                return np.zeros((2, 2), dtype=np.float32)

        loader = StubLoader()
        assert loader.find_data_type() == "float32"


class TestDefaultImageLoader:
    def test_load_npy(self, tmp_path: Path) -> None:
        data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        npy_path = tmp_path / "test.npy"
        np.save(npy_path, data)

        loader = DefaultImageLoader(file_path=str(npy_path))
        loaded = loader.load_data()
        np.testing.assert_array_equal(loaded, data)

    def test_load_tiff(self, tmp_path: Path) -> None:
        tifffile = pytest.importorskip("tifffile")
        data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        tiff_path = tmp_path / "test.tiff"
        tifffile.imwrite(str(tiff_path), data)

        loader = DefaultImageLoader(file_path=str(tiff_path))
        loaded = loader.load_data()
        np.testing.assert_array_equal(loaded, data)

    def test_load_tif_extension(self, tmp_path: Path) -> None:
        tifffile = pytest.importorskip("tifffile")
        data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        tif_path = tmp_path / "test.tif"
        tifffile.imwrite(str(tif_path), data)

        loader = DefaultImageLoader(file_path=str(tif_path))
        loaded = loader.load_data()
        np.testing.assert_array_equal(loaded, data)

    def test_load_png(self, tmp_path: Path) -> None:
        from PIL import Image

        data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        png_path = tmp_path / "test.png"
        Image.fromarray(data).save(png_path)

        loader = DefaultImageLoader(file_path=str(png_path))
        loaded = loader.load_data()
        np.testing.assert_array_equal(loaded, data)

    def test_load_jpg(self, tmp_path: Path) -> None:
        from PIL import Image

        # JPG is lossy, so just check shape
        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        jpg_path = tmp_path / "test.jpg"
        Image.fromarray(data).save(jpg_path)

        loader = DefaultImageLoader(file_path=str(jpg_path))
        loaded = loader.load_data()
        assert loaded.shape[:2] == (10, 10)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        fake_path = str(tmp_path / "test.xyz")
        loader = DefaultImageLoader(file_path=fake_path)
        with pytest.raises(ValueError, match="cannot handle file type"):
            loader.load_data()

    def test_resource_prepends_path(self, tmp_path: Path) -> None:
        subdir = tmp_path / "images"
        subdir.mkdir()
        data = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
        np.save(subdir / "img.npy", data)

        loader = DefaultImageLoader(file_path="img.npy")
        loaded = loader.load_data(resource=str(subdir))
        np.testing.assert_array_equal(loaded, data)

    def test_resource_as_path_object(self, tmp_path: Path) -> None:
        data = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
        np.save(tmp_path / "img.npy", data)

        loader = DefaultImageLoader(file_path="img.npy")
        loaded = loader.load_data(resource=tmp_path)
        np.testing.assert_array_equal(loaded, data)

    def test_find_data_type(self, tmp_path: Path) -> None:
        data = np.zeros((5, 5), dtype=np.float64)
        np.save(tmp_path / "data.npy", data)

        loader = DefaultImageLoader(file_path=str(tmp_path / "data.npy"))
        assert loader.find_data_type() == "float64"

    def test_extra_fields_ignored(self) -> None:
        # ImageLoaderInterface has extra="ignore"
        loader = DefaultImageLoader(file_path="test.npy", unknown_field="value")  # type: ignore
        assert loader.file_path == "test.npy"
