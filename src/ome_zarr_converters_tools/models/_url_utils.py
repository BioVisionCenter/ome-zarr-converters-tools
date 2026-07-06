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


def find_url_type(url: str) -> UrlType:
    if url.startswith("/"):
        return UrlType.LOCAL
    elif url.startswith("s3://"):
        return UrlType.S3
    # Windows drive paths: C:\ or C:/
    elif len(url) >= 2 and url[1] == ":" and url[0].isalpha():
        return UrlType.LOCAL
    # Windows UNC paths: \\server\share — "\\\\" is two backslash chars in Python
    elif url.startswith("\\\\"):
        return UrlType.LOCAL
    return UrlType.NOT_SUPPORTED


def local_url_to_path(url: str) -> Path:
    """Convert a local URL to a Path object."""
    path = Path(url)
    path = path.resolve().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def join_url_paths(base_url: str, *paths: str) -> str:
    """Join path components to a base URL, normalizing ``.``/``..`` segments.

    Used instead of ``os.path.join`` or ``pathlib.Path`` to support both local
    and S3 URLs. Resolves ``.``/``..`` and collapses redundant slashes while
    staying protocol- and Windows-safe: it uses ``posixpath.normpath`` (never
    ``os.path.normpath``, which on Windows would rewrite forward slashes to
    backslashes and corrupt S3 keys).
    """
    protocol, base = fsspec.core.split_protocol(base_url)
    combined = f"{base}/{'/'.join(str(p) for p in paths)}".replace("\\", "/")
    joined = posixpath.normpath(combined)
    return joined if protocol is None else f"{protocol}://{joined}"


def parent_url(url: str) -> str:
    """Return the parent directory of a URL path.

    Robust to forward/back slashes and URL protocols. For remote protocols the
    first component after the protocol is the network location (e.g. the S3
    bucket), which has no parent.

    Raises:
        ValueError: if ``url`` is a filesystem/network-location root with no
            parent (e.g. ``/``, ``s3://bucket``, or an empty string).
    """
    protocol, path = fsspec.core.split_protocol(url)
    path = path.rstrip("\\").rstrip("/")
    if protocol is None:
        parent = str(Path(path).parent)
        # Path("").parent == "." so the naive `parent == path` guard never
        # fires for "/"; catch the empty / "." / root case explicitly.
        if parent == path or path in ("", "."):
            raise ValueError(f"No parent directory for URL: {url}")
        return parent
    parent = posixpath.dirname(path.replace("\\", "/"))
    if parent == "":
        raise ValueError(f"No parent directory for URL: {url}")
    return f"{protocol}://{parent}"


def basename_url(url: str) -> str:
    """Return the last path component of a URL.

    A trailing slash is stripped so a directory yields its own name.
    """
    return posixpath.basename(url.replace("\\", "/").rstrip("/"))


def is_absolute_url(url: str) -> bool:
    """Return True if the URL has a protocol, else ``Path(path).is_absolute()``.

    Used to decide whether a path from a manifest/CSV needs to be resolved
    against a base directory.
    """
    protocol, path = fsspec.core.split_protocol(url)
    return True if protocol is not None else Path(path).is_absolute()


def filesystem_for_url(
    url: str, error_msg_prefix: str = "File handling"
) -> fsspec.AbstractFileSystem:
    url_type = find_url_type(url)
    if url_type == UrlType.NOT_SUPPORTED:
        raise NotImplementedError(
            f"{error_msg_prefix} for URL {url} "
            f"with detected type {url_type} is not implemented yet."
        )
    return fsspec.filesystem(url_type.value)


def glob_url_paths(*, base_url: str | None, pattern: str) -> list[str]:
    """Glob a URL pattern, re-prefixing the protocol on each match.

    Uses :func:`filesystem_for_url` so it works for local and S3 URLs, and
    re-prefixes the protocol on each result so matches stay usable URLs. If
    ``base_url`` is ``None``, ``pattern`` is treated as absolute.

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
