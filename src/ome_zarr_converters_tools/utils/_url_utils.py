from enum import Enum
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


class URLType(Enum):
    LOCAL = "local"
    S3 = "s3"
    NOT_SUPPORTED = "not_supported"


def find_url_type(url: str) -> URLType:
    path = Path(url)
    if path.exists():
        return URLType.LOCAL
    elif url.startswith("s3://"):
        return URLType.S3
    else:
        logger.error(f"Unsupported URL type for {url}")
        return URLType.NOT_SUPPORTED


def local_url_to_path(url: str) -> Path:
    """Convert a local URL to a Path object."""
    path = Path(url)
    path = path.resolve().absolute()
    return path
