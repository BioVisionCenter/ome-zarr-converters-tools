"""Models for defining regions to be converted into OME-Zarr format."""

from typing import Any, TypeVar
from warnings import warn

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CollectionInterface(BaseModel):
    model_config = ConfigDict(extra="ignore")

    def path(self) -> str:
        raise NotImplementedError("Subclasses must implement path method.")


CollectionInterfaceType = TypeVar("CollectionInterfaceType", bound=CollectionInterface)


def sanitize_path(path: str) -> str:
    """Sanitize the plate name to be used as a Zarr group path."""
    characters_to_replace = [" ", "/"]
    for char in characters_to_replace:
        if char in path:
            warn(
                f"Path '{path}' contains '{char}', "
                "which will be replaced with underscores.",
                UserWarning,
                stacklevel=2,
            )
        path = path.replace(char, "_")
    # Make sure it ends with .zarr
    if not path.endswith(".zarr"):
        path = f"{path}.zarr"
    return path


class SingleImage(CollectionInterface):
    image_path: str
    suffix: str = ""

    def path(self) -> str:
        return sanitize_path(f"{self.image_path}{self.suffix}")


class ImageInPlate(CollectionInterface):
    plate_name: str
    row: str
    column: int = Field(ge=1)
    acquisition: int = Field(default=0, ge=0)
    # Auto-generated suffix for tiling (do not set manually)
    suffix: str = ""

    @property
    def well(self) -> str:
        return f"{self.row}{self.column}"

    def plate_path(self) -> str:
        return sanitize_path(self.plate_name)

    def well_path(self) -> str:
        return f"{self.plate_path()}/{self.row}/{self.column}"

    def path_in_well(self) -> str:
        return f"{self.acquisition}{self.suffix}"

    def path(self) -> str:
        return f"{self.well_path()}/{self.path_in_well()}"

    @classmethod
    @field_validator("row", mode="before")
    def row_to_str(cls, v: Any) -> Any:
        if isinstance(v, int):
            if v < 1 or v >= len(ALPHABET):
                raise ValueError(
                    f"Row index {v} out of range. "
                    f"Must be between 1 and {len(ALPHABET) - 1}"
                )
            return ALPHABET[v - 1]
        return v
