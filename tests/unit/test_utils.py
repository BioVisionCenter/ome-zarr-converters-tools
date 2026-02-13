"""Unit tests for utility functions."""

import numpy as np

from ome_zarr_converters_tools.core._dask_lazy_loader import lazy_array_from_regions
from ome_zarr_converters_tools.models._url_utils import (
    UrlType,
    find_url_type,
    join_url_paths,
)


class TestUrlUtils:
    def test_find_url_type_local(self) -> None:
        assert find_url_type("/some/path") == UrlType.LOCAL

    def test_find_url_type_s3(self) -> None:
        assert find_url_type("s3://my-bucket/key") == UrlType.S3

    def test_find_url_type_unsupported(self) -> None:
        assert find_url_type("http://example.com") == UrlType.NOT_SUPPORTED

    def test_join_url_paths(self) -> None:
        result = join_url_paths("/base", "sub", "file.txt")
        assert result == "/base/sub/file.txt"

    def test_join_url_paths_strips_leading_slashes(self) -> None:
        result = join_url_paths("/base", "/sub", "/file.txt")
        assert result == "/base/sub/file.txt"

    def test_join_url_paths_s3(self) -> None:
        result = join_url_paths("s3://bucket", "prefix", "key")
        assert result == "s3://bucket/prefix/key"


class TestDaskLazyLoader:
    def test_lazy_array_single_region(self) -> None:
        data = np.ones((10, 10), dtype="uint8") * 42
        regions = [
            (
                (slice(0, 10), slice(0, 10)),
                lambda: data,
            )
        ]
        arr = lazy_array_from_regions(
            regions,
            shape=(10, 10),
            chunks=(10, 10),
            dtype="uint8",  # type: ignore[arg-type]
        )
        result = arr.compute()  # type: ignore[no-untyped-call]
        np.testing.assert_array_equal(result, data)

    def test_lazy_array_multiple_regions(self) -> None:
        data_a = np.ones((5, 10), dtype="uint8") * 1
        data_b = np.ones((5, 10), dtype="uint8") * 2
        regions = [
            ((slice(0, 5), slice(0, 10)), lambda: data_a),
            ((slice(5, 10), slice(0, 10)), lambda: data_b),
        ]
        arr = lazy_array_from_regions(
            regions,
            shape=(10, 10),
            chunks=(5, 10),
            dtype="uint8",  # type: ignore[arg-type]
        )
        result = arr.compute()  # type: ignore[no-untyped-call]
        np.testing.assert_array_equal(result[:5], 1)
        np.testing.assert_array_equal(result[5:], 2)

    def test_lazy_array_fill_value(self) -> None:
        data = np.ones((5, 5), dtype="float32") * 99
        regions = [
            ((slice(0, 5), slice(0, 5)), lambda: data),
        ]
        arr = lazy_array_from_regions(
            regions,
            shape=(10, 10),
            chunks=(5, 5),
            dtype="float32",
            fill_value=-1.0,  # type: ignore[arg-type]
        )
        result = arr.compute()  # type: ignore[no-untyped-call]
        np.testing.assert_array_equal(result[:5, :5], 99)
        np.testing.assert_array_equal(result[5:, :], -1.0)
        np.testing.assert_array_equal(result[:5, 5:], -1.0)
