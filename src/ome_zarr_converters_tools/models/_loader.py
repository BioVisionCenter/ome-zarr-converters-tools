"""Models for defining regions to be converted into OME-Zarr format."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import tifffile
from PIL import Image
from pydantic import BaseModel, ConfigDict


class ImageLoaderInterface(BaseModel, ABC):
    model_config = ConfigDict(extra="ignore")

    @abstractmethod
    def load_data(self, resource: Any = None) -> np.ndarray:
        """Load the image data as a Dask array."""
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
        if resource and isinstance(resource, (Path, str)):
            path = Path(resource) / self.file_path
        elif resource is None:
            path = Path(self.file_path)
        else:
            raise ValueError(
                "DefaultImageLoader cannot handle resource of "
                f"type {type(resource)}, expected Path or str or None."
            )

        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        if path.suffix.lower() in [".tiff", ".tif"]:
            with tifffile.TiffFile(path) as tif:
                image = tif.asarray()
        elif path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            image = np.array(Image.open(path))

        elif path.suffix.lower() == ".npy":
            image = np.load(path)
        else:
            raise ValueError(
                f"DefaultImageLoader cannot handle file type {path.suffix}, "
                "supported types are .tiff, .tif, .png, .jpg, .jpeg, .bmp, .npy"
            )
        return image
