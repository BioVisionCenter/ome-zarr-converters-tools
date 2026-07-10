"""Models for defining regions to be converted into OME-Zarr format."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
import tifffile
from PIL import Image
from pydantic import BaseModel, ConfigDict

from ome_zarr_converters_tools.models._url_utils import (
    basename_url,
    filesystem_for_url,
    join_url_paths,
)


class ImageLoaderInterface(BaseModel, ABC):
    """Base class for image loaders; subclass it to support a custom format.

    Implement `load_data` to return the tile pixels as a NumPy array. The
    optional `resource` carries per-call context (e.g. a base directory or an
    open handle) and is threaded through the conversion pipeline unchanged.
    """

    model_config = ConfigDict(extra="ignore")

    @abstractmethod
    def load_data(self, resource: Any | None = None) -> np.ndarray:
        """Load the image data as a NumPy array."""
        pass

    def preflight(self, resource: Any | None = None) -> None:
        """Cheaply verify the source data is reachable, without loading it.

        The default implementation is a no-op. Override it to emit a warning
        on missing or unreadable sources (e.g. a file-existence check) so
        that pre-flight validators can surface such problems at init time,
        before compute jobs are dispatched. Implementations should warn, not
        raise: only loading the data decides whether it is truly unreadable.
        """

    def find_data_type(self, resource: Any | None = None) -> str:
        """Find the data type of the image data."""
        return str(self.load_data(resource).dtype)


ImageLoaderInterfaceType = TypeVar(
    "ImageLoaderInterfaceType", bound=ImageLoaderInterface
)


class DefaultImageLoader(ImageLoaderInterface):
    """File-based image loader for common formats (TIFF, PNG/JPEG/BMP, NPY).

    The file type is inferred from the extension; unrecognized extensions are
    attempted as TIFF with a warning.
    """

    file_path: str
    """Path to the image file. If relative, it is resolved against the
    `resource` passed to `load_data` (usually the acquisition base directory)."""

    def _resolve_path(self, resource: Any | None) -> str:
        """Resolve `file_path` against the optional `resource` base directory."""
        try:
            if resource is not None:
                # Ensure we can convert to str
                resource = str(resource)
        except Exception:
            raise ValueError(  # noqa: B904
                "DefaultImageLoader expects resource to be of type str, Path, or None."
            )
        if resource and isinstance(resource, str):
            return join_url_paths(resource, self.file_path)
        return self.file_path

    def preflight(self, resource: Any | None = None) -> None:
        """Warn if the source file does not exist, without reading it."""
        path = self._resolve_path(resource)
        try:
            fs = filesystem_for_url(path, error_msg_prefix="Preflight check")
            exists = fs.exists(path)
        except Exception as e:
            warnings.warn(
                f"Preflight check could not verify source file '{path}': {e}",
                stacklevel=2,
            )
            return
        if not exists:
            warnings.warn(
                f"Source file '{path}' does not exist. Check that the file "
                "was not moved or deleted, and that `file_path` (combined "
                "with the `resource` base directory, if any) points to it.",
                stacklevel=2,
            )

    def load_data(self, resource: Any | None = None) -> np.ndarray:
        """Load the image data as a NumPy array."""
        path = self._resolve_path(resource)

        suffix = basename_url(path).split(".")[-1].lower()
        if suffix in ["tiff", "tif", "tf2", "tf8", "btf"]:
            return self.load_tiff(path)
        elif suffix in ["png", "jpg", "jpeg", "bmp"]:
            return self.load_png(path)
        elif suffix == "npy":
            return self.load_npy(path)
        else:
            # Unknown extension: many files (e.g. custom/uncommon TIFF variants) are
            # still readable by tifffile, so warn and attempt a best-effort TIFF read.
            warnings.warn(
                f"DefaultImageLoader does not recognize file type {suffix!r}; "
                "attempting to load it as a TIFF file.",
                stacklevel=2,
            )
            try:
                return self.load_tiff(path)
            except Exception as e:
                raise ValueError(
                    f"DefaultImageLoader cannot handle file type {suffix!r}: "
                    "the TIFF fallback failed to read the file. Supported types are "
                    ".tiff, .tif, .tf2, .tf8, .btf, .png, .jpg, .jpeg, .bmp, .npy."
                ) from e

    def load_tiff(self, path: str) -> np.ndarray:
        fs = filesystem_for_url(path, error_msg_prefix="Loading image")
        with fs.open(path, "rb") as f:
            with tifffile.TiffFile(f) as tif:
                return tif.asarray()

    def load_png(self, path: str) -> np.ndarray:
        fs = filesystem_for_url(path, error_msg_prefix="Loading image")
        with fs.open(path, "rb") as f:
            return np.array(Image.open(f))

    def load_npy(self, path: str) -> np.ndarray:
        fs = filesystem_for_url(path, error_msg_prefix="Loading image")
        with fs.open(path, "rb") as f:
            return np.load(f)
