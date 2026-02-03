from enum import Enum
from logging import getLogger
from pathlib import Path

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
    logger.error(
        f"Unsupported URL type for {url}. "
        "Only absolute local paths and S3 URLs are supported."
    )
    return UrlType.NOT_SUPPORTED


def local_url_to_path(url: str) -> Path:
    """Convert a local URL to a Path object."""
    path = Path(url)
    path = path.resolve().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def join_url_paths(base_url: str, *paths: str) -> str:
    """Join multiple path components to a base URL.

    This is used instead of os.path.join or pathlib.Path to ensure
    support for both local and S3 URLs.
    """
    # Ensure base_url does not end with a slash
    base_url = base_url.rstrip("/")
    # Iterate for all but the last path to avoid adding a trailing slash
    for path in paths:
        # Strip leading slashes from path components
        path = str(path).lstrip("/")
        base_url = f"{base_url}/{path}"
    return base_url
