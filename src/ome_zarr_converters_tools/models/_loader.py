"""Models for defining regions to be converted into OME-Zarr format."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
import tifffile
from PIL import Image
from pydantic import BaseModel, ConfigDict

from ome_zarr_converters_tools.models._url_utils import (
    filesystem_for_url,
    join_url_paths,
)


class ImageLoaderInterface(BaseModel, ABC):
    model_config = ConfigDict(extra="ignore")

    @abstractmethod
    def load_data(self, resource: Any = None) -> np.ndarray:
        """Load the image data as a NumPy array."""
        pass

    def find_data_type(self, resource: Any = None) -> str:
        """Find the data type of the image data."""
        return str(self.load_data(resource).dtype)


ImageLoaderInterfaceType = TypeVar(
    "ImageLoaderInterfaceType", bound=ImageLoaderInterface
)


class DefaultImageLoader(ImageLoaderInterface):
    file_path: str

    def load_data(self, resource: Any = None) -> np.ndarray:
        """Load the image data as a NumPy array."""
        try:
            if resource is not None:
                # Ensure we can convert to str
                resource = str(resource)
        except Exception:
            raise ValueError(  # noqa: B904
                "DefaultImageLoader expects resource to be of type str, Path, or None."
            )
        if resource and isinstance(resource, str):
            path = join_url_paths(resource, self.file_path)
        else:
            path = self.file_path

        suffix = path.split("/")[-1].split(".")[-1].lower()
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
