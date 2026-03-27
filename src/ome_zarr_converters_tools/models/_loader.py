"""Models for defining regions to be converted into OME-Zarr format."""

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
        fs = filesystem_for_url(path, error_msg_prefix="Loading image")

        with fs.open(path, "rb") as f:
            if suffix in ["tiff", "tif"]:
                with tifffile.TiffFile(f) as tif:
                    image = tif.asarray()
            elif suffix in ["png", "jpg", "jpeg", "bmp"]:
                image = np.array(Image.open(f))

            elif suffix == "npy":
                image = np.load(f)
            else:
                raise ValueError(
                    f"DefaultImageLoader cannot handle file type {suffix}, "
                    "supported types are .tiff, .tif, .png, .jpg, .jpeg, .bmp, .npy"
                )
            return image
