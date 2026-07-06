import posixpath
from enum import Enum
from logging import getLogger
from pathlib import Path

import fsspec
import fsspec.core

logger = getLogger(__name__)


class UrlType(Enum):
    LOCAL = "local"
    S3 = "s3"
    NOT_SUPPORTED = "not_supported"


def _is_windows_drive(url: str) -> bool:
    r"""Return True for a Windows drive path like `C:\` or `C:/`."""
    return len(url) >= 2 and url[1] == ":" and url[0].isalpha()


def _is_local_absolute(url: str) -> bool:
    r"""Return True if `url` is an absolute local path (host-independent).

    Covers POSIX roots (`/`), `~`-anchored home paths, Windows drive paths
    (`C:\` / `C:/`), and UNC paths (`\\server\share`). Used by both
    `find_url_type` and `is_absolute_url` so their classification of
    Windows/home paths cannot drift across operating systems.
    """
    return (
        url.startswith(("/", "~"))
        or _is_windows_drive(url)
        # Windows UNC paths: \\server\share — "\\\\" is two backslash chars.
        or url.startswith("\\\\")
    )


def find_url_type(url: str) -> UrlType:
    """Classify a URL as local, S3, or unsupported (see `UrlType`)."""
    if url.startswith("s3://"):
        return UrlType.S3
    elif _is_local_absolute(url):
        return UrlType.LOCAL
    return UrlType.NOT_SUPPORTED


def local_url_to_path(url: str) -> Path:
    """Convert a local URL to an absolute Path, expanding `~`.

    Does not touch the filesystem.
    """
    return Path(url).expanduser().resolve()


def join_url_paths(base_url: str, *paths: str) -> str:
    """Join path components to a base URL, normalizing `.`/`..` segments.

    Used instead of `os.path.join` or `pathlib.Path` to support both local
    and S3 URLs. Resolves `.`/`..` and collapses redundant slashes while
    staying protocol- and Windows-safe: it uses `posixpath.normpath` (never
    `os.path.normpath`, which on Windows would rewrite forward slashes to
    backslashes and corrupt S3 keys).

    Raises:
        ValueError: if `..` segments would ascend above the network location
            of a protocol URL (e.g. `join_url_paths("s3://bucket", "..", "x")`
            would otherwise silently drop the bucket).
    """
    protocol, base = fsspec.core.split_protocol(base_url)
    combined = f"{base}/{'/'.join(str(p) for p in paths)}".replace("\\", "/")
    joined = posixpath.normpath(combined)
    if protocol is None:
        return joined
    # The first component after the protocol is the network location (e.g. the
    # S3 bucket) and is inviolable — reject any `..` that escapes it.
    netloc = base.replace("\\", "/").split("/", 1)[0]
    if joined != netloc and not joined.startswith(f"{netloc}/"):
        raise ValueError(
            f"Path escapes network location for URL: {base_url!r} + {paths!r}"
        )
    return f"{protocol}://{joined}"


def parent_url(url: str) -> str:
    r"""Return the parent directory of a URL path.

    Robust to forward/back slashes and URL protocols, and always returns POSIX
    `/` separators regardless of host OS (uses `posixpath` for the local
    branch too, never OS-native `pathlib`, which would emit `\` on Windows).
    For remote protocols the first component after the protocol is the network
    location (e.g. the S3 bucket), which has no parent.

    Raises:
        ValueError: if `url` is a filesystem/network-location root with no
            parent (e.g. `/`, `s3://bucket`, or an empty string).
    """
    protocol, path = fsspec.core.split_protocol(url)
    path = path.replace("\\", "/").rstrip("/")
    parent = posixpath.dirname(path)
    if parent == "":
        raise ValueError(f"No parent directory for URL: {url}")
    return parent if protocol is None else f"{protocol}://{parent}"


def basename_url(url: str) -> str:
    """Return the last path component of a URL.

    A trailing slash is stripped so a directory yields its own name.
    """
    return posixpath.basename(url.replace("\\", "/").rstrip("/"))


def is_absolute_url(url: str) -> bool:
    """Return True if the URL is absolute (protocol, root, home, or Windows).

    Host-independent: shares `_is_local_absolute` with `find_url_type` so a
    Windows drive/UNC path or a `~`-anchored home path is treated as absolute
    even when the suite runs on POSIX. Falls back to `Path(path).is_absolute()`
    for anything else. Used to decide whether a path from a manifest/CSV needs
    to be resolved against a base directory.
    """
    protocol, path = fsspec.core.split_protocol(url)
    if protocol is not None or _is_local_absolute(path):
        return True
    return Path(path).is_absolute()


def filesystem_for_url(
    url: str, error_msg_prefix: str = "File handling"
) -> fsspec.AbstractFileSystem:
    """Return the fsspec filesystem for a URL; raise if its type is unsupported."""
    url_type = find_url_type(url)
    if url_type == UrlType.NOT_SUPPORTED:
        raise NotImplementedError(
            f"{error_msg_prefix} for URL {url} "
            f"with detected type {url_type} is not implemented yet."
        )
    return fsspec.filesystem(url_type.value)


def glob_url_paths(*, base_url: str | None, pattern: str) -> list[str]:
    """Glob a URL pattern, re-prefixing the protocol on each match.

    Uses `filesystem_for_url` so it works for local and S3 URLs, and
    re-prefixes the protocol on each result so matches stay usable URLs. If
    `base_url` is `None`, `pattern` is treated as absolute — it must then
    itself be absolute (`/…`, `~/…`, `s3://…`, or a Windows path), since a
    relative pattern classifies as `NOT_SUPPORTED` and raises.

    A literal, non-existent pattern yields an empty list (fsspec behaviour that
    downstream code relies on for existence checks).
    """
    if base_url is not None:
        pattern = join_url_paths(base_url, pattern)
    fs = filesystem_for_url(pattern)
    protocol = fsspec.core.split_protocol(pattern)[0]
    matched = fs.glob(pattern)
    if protocol is not None:
        return [f"{protocol}://{p}" for p in matched]
    return matched
