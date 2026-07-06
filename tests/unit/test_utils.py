"""Unit tests for utility functions."""

from pathlib import Path

import numpy as np
import pytest

from ome_zarr_converters_tools.core._dask_lazy_loader import lazy_array_from_regions
from ome_zarr_converters_tools.models._url_utils import (
    UrlType,
    basename_url,
    find_url_type,
    glob_url_paths,
    is_absolute_url,
    join_url_paths,
    local_url_to_path,
    parent_url,
)


class TestUrlUtils:
    def test_find_url_type_local(self) -> None:
        assert find_url_type("/some/path") == UrlType.LOCAL

    def test_find_url_type_s3(self) -> None:
        assert find_url_type("s3://my-bucket/key") == UrlType.S3

    def test_find_url_type_unsupported(self) -> None:
        assert find_url_type("http://example.com") == UrlType.NOT_SUPPORTED

    def test_find_url_type_windows_drive_backslash(self) -> None:
        assert find_url_type("C:\\path\\to\\file") == UrlType.LOCAL

    def test_find_url_type_windows_drive_forward_slash(self) -> None:
        assert find_url_type("C:/path/to/file") == UrlType.LOCAL

    def test_find_url_type_unc_path(self) -> None:
        assert find_url_type("\\\\server\\share\\path") == UrlType.LOCAL

    def test_find_url_type_home(self) -> None:
        assert find_url_type("~/data") == UrlType.LOCAL
        assert find_url_type("~user/data") == UrlType.LOCAL

    def test_find_url_type_relative_unsupported(self) -> None:
        assert find_url_type("rel/path") == UrlType.NOT_SUPPORTED
        assert find_url_type("../up/path") == UrlType.NOT_SUPPORTED

    def test_join_url_paths(self) -> None:
        result = join_url_paths("/base", "sub", "file.txt")
        assert result == "/base/sub/file.txt"

    def test_join_url_paths_strips_leading_slashes(self) -> None:
        result = join_url_paths("/base", "/sub", "/file.txt")
        assert result == "/base/sub/file.txt"

    def test_join_url_paths_s3(self) -> None:
        result = join_url_paths("s3://bucket", "prefix", "key")
        assert result == "s3://bucket/prefix/key"

    def test_join_url_paths_s3_nested(self) -> None:
        assert join_url_paths("s3://bucket/dir", "a", "b") == "s3://bucket/dir/a/b"

    def test_join_url_paths_resolves_dotdot(self) -> None:
        assert join_url_paths("s3://bucket/dir", "..", "x") == "s3://bucket/x"

    def test_join_url_paths_local_file(self) -> None:
        assert join_url_paths("/data/exp", "tiles.csv") == "/data/exp/tiles.csv"

    def test_join_url_paths_raises_when_escaping_bucket(self) -> None:
        # `..` must not silently drop the bucket (would target a different one).
        with pytest.raises(ValueError, match="escapes network location"):
            join_url_paths("s3://bucket", "..", "x")

    def test_join_url_paths_dotdot_within_bucket_ok(self) -> None:
        # Ascending within the bucket key is fine.
        assert join_url_paths("s3://bucket/dir", "..", "x") == "s3://bucket/x"

    def test_join_url_paths_local_absolute_clamps_at_root(self) -> None:
        # posixpath.normpath clamps at `/`; a local absolute base cannot escape.
        assert join_url_paths("/data/exp", "..", "..", "..", "x") == "/x"

    @pytest.mark.parametrize(
        ("base_url", "paths"),
        [
            ("s3://bucket/dir", ("a", "b")),
            ("s3://bucket/dir", ("..", "x")),
            ("s3://bucket", ("sub\\dir", "file.txt")),
        ],
    )
    def test_join_url_paths_no_backslash_in_remote_output(
        self, base_url: str, paths: tuple[str, ...]
    ) -> None:
        # Guards against the os.path.normpath Windows regression that turns
        # `/` into `\` on remote URLs (an invalid S3 key).
        assert "\\" not in join_url_paths(base_url, *paths)

    def test_parent_url_local_file(self) -> None:
        assert parent_url("/path/to/file.txt") == "/path/to"

    def test_parent_url_local_dir_trailing_slash(self) -> None:
        assert parent_url("/path/to/dir/") == "/path/to"

    def test_parent_url_windows_drive_posix_output(self) -> None:
        # Always POSIX `/` separators, even for a Windows backslash path.
        assert parent_url("C:\\data\\exp\\file.tif") == "C:/data/exp"

    def test_parent_url_s3(self) -> None:
        assert parent_url("s3://bucket/dir/file.txt") == "s3://bucket/dir"

    def test_parent_url_s3_dir(self) -> None:
        assert parent_url("s3://bucket/dir/") == "s3://bucket"

    @pytest.mark.parametrize("url", ["/", "", "s3://bucket"])
    def test_parent_url_raises_at_root(self, url: str) -> None:
        with pytest.raises(ValueError, match="No parent directory"):
            parent_url(url)

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("/path/to/dir/", "dir"),
            ("/path/to/dir", "dir"),
            ("/path/to/file.txt", "file.txt"),
            ("s3://bucket/a/b", "b"),
            # Backslash paths must yield the trailing component (loader relies
            # on basename_url instead of a raw split("/")).
            ("C:\\data\\exp\\file.tif", "file.tif"),
        ],
    )
    def test_basename_url(self, url: str, expected: str) -> None:
        assert basename_url(url) == expected

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("/abs/path", True),
            ("relative/path", False),
            ("s3://bucket/key", True),
            # Host-independent: home-anchored and Windows paths are absolute even
            # when the suite runs on POSIX.
            ("~/data", True),
            ("C:/path", True),
            ("\\\\server\\share", True),
        ],
    )
    def test_is_absolute_url(self, url: str, expected: bool) -> None:
        assert is_absolute_url(url) is expected

    def test_local_url_to_path_expands_home_without_touching_fs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Run from an isolated cwd so a regression that creates a literal `~`
        # directory would be observable and would not pollute the repo.
        monkeypatch.chdir(tmp_path)
        result = local_url_to_path("~/some_dir/file.txt")
        assert result == (Path.home() / "some_dir" / "file.txt").resolve()
        # No filesystem side effects: neither `~` nor the home subdir is created.
        assert not (tmp_path / "~").exists()
        assert not (Path.home() / "some_dir").exists()

    def test_glob_url_paths_local(self, tmp_path: Path) -> None:
        (tmp_path / "a.jdce").touch()
        (tmp_path / "b.jdce").touch()
        (tmp_path / "c.txt").touch()
        result = glob_url_paths(base_url=str(tmp_path), pattern="*.jdce")
        # Compare via Path so separator style (POSIX vs. Windows) is irrelevant.
        assert {Path(p) for p in result} == {
            tmp_path / "a.jdce",
            tmp_path / "b.jdce",
        }

    def test_glob_url_paths_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        assert glob_url_paths(base_url=str(tmp_path), pattern="nope.jdce") == []


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
