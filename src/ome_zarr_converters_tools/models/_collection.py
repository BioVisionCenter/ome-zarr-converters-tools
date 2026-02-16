"""Models for defining regions to be converted into OME-Zarr format."""

import re
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from ome_zarr_converters_tools.models._url_utils import join_url_paths

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

VALID_ZARR_NAME_PATTERN = r"^(?!__)(?! )((?!^\.+$)[a-zA-Z0-9_. -]+)(?<! )$"


def validate_zarr_name(name: str) -> str:
    """Validate a name to be used as a Zarr group or dataset name."""
    if not re.match(VALID_ZARR_NAME_PATTERN, name):
        raise ValueError(
            f"Invalid Zarr name '{name}'. "
            "Names must only contain A-Z, a-z, 0-9, -, _, space, and . characters. "
            "Additionally, names cannot have leading or trailing spaces, "
            "start with '__', or consist only of dots."
        )
    return name


class CollectionInterface(BaseModel):
    model_config = ConfigDict(extra="ignore")

    def path(self) -> str:
        raise NotImplementedError("Subclasses must implement path method.")


CollectionInterfaceType = TypeVar("CollectionInterfaceType", bound=CollectionInterface)


def sanitize_path(path: str) -> str:
    """Make sure path ends with .zarr and is a valid Zarr name."""
    validate_zarr_name(path)
    if not path.endswith(".zarr"):
        path = f"{path}.zarr"
    return path


class SingleImage(CollectionInterface):
    image_path: str
    _suffix: str = PrivateAttr("")

    def path(self) -> str:
        return sanitize_path(f"{self.image_path}{self._suffix}")


class ImageInPlate(CollectionInterface):
    plate_name: str
    row: str
    column: int = Field(ge=1)
    acquisition: int = Field(default=0, ge=0)
    # Auto-generated suffix for tiling (do not set manually)
    _suffix: str = PrivateAttr("")

    @property
    def well(self) -> str:
        return f"{self.row}{self.column:02d}"

    def plate_path(self) -> str:
        return sanitize_path(self.plate_name)

    def well_path(self) -> str:
        return join_url_paths(self.row, f"{self.column:02d}")

    def path_in_well(self) -> str:
        return f"{self.acquisition}{self._suffix}"

    def image_in_well_path(self) -> str:
        return join_url_paths(self.well_path(), self.path_in_well())

    def path(self) -> str:
        return join_url_paths(self.plate_path(), self.well_path(), self.path_in_well())

    @field_validator("row", mode="before")
    @classmethod
    def row_to_str(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v
        v = int(v)
        if v < 1 or v >= len(ALPHABET):
            raise ValueError(
                f"Row index {v} out of range. Must be between 1 and {len(ALPHABET) - 1}"
            )
        return ALPHABET[v - 1]
