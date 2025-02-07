"""Shared models for the fractal_converters_tools tasks."""

from typing import Literal

from pydantic import BaseModel, Field


class AdvancedComputeOptions(BaseModel):
    """Advanced options for the conversion.

    Attributes:
        num_levels (int): The number of resolution levels in the pyramid.
        tiling_mode (Literal["auto", "grid", "free", "none"]): Specify the tiling mode.
            "auto" will automatically determine the tiling mode.
            "grid" if the input data is a grid, it will be tiled using snap-to-grid.
            "free" will remove any overlap between tiles using a snap-to-corner
            approach.
            "none" will write the positions as is, using the microscope metadata.
        swap_xy (bool): Swap x and y axes coordinates in the metadata. This is sometimes
            necessary to ensure correct image tiling and registration.
        invert_x (bool): Invert x axis coordinates in the metadata. This is
            sometimes necessary to ensure correct image tiling and registration.
        invert_y (bool): Invert y axis coordinates in the metadata. This is
            sometimes necessary to ensure correct image tiling and registration.
        max_xy_chunk (int): XY chunk size is set as the minimum of this value and the
            microscope tile size.
        z_chunk (int): Z chunk size.
        c_chunk (int): C chunk size.
        t_chunk (int): T chunk size.
    """

    num_levels: int = Field(default=5, ge=1)
    tiling_mode: Literal["auto", "grid", "free", "none"] = "auto"
    swap_xy: bool = False
    invert_x: bool = False
    invert_y: bool = False
    max_xy_chunk: int = Field(default=4096, ge=1)
    z_chunk: int = Field(default=10, ge=1)
    c_chunk: int = Field(default=1, ge=1)
    t_chunk: int = Field(default=1, ge=1)


class ConvertParallelInitArgs(BaseModel):
    """Arguments for the compute task."""

    tiled_image_pickled_path: str
    overwrite: bool
    advanced_compute_options: AdvancedComputeOptions
