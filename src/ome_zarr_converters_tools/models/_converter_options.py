from enum import StrEnum
from typing import Annotated, Literal

from ngio import DefaultNgffVersion, NgffVersions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ome_zarr_converters_tools.models._runtime_settings import RuntimeSettings


class OverwriteMode(StrEnum):
    NO_OVERWRITE = "No Overwrite"
    OVERWRITE = "Overwrite"
    EXTEND = "Extend"


class AutoTiling(BaseModel):
    mode: Literal["Auto"] = "Auto"
    """
    Automatically determine if Snap to Grid is possible, otherwise use Snap to Corners.
    """
    tolerance: float = Field(default=1, ge=0, title="Tiling Tolerance (in pixels)")


class SnapToGridTiling(BaseModel):
    mode: Literal["Snap to Grid"] = "Snap to Grid"
    """
    Tile images to fit a regular grid. This is only possible if image positions align
    to a grid (potentially with overlap).
    """
    tolerance: float = Field(default=1, ge=0, title="Tiling Tolerance (in pixels)")


class SnapToCornersTiling(BaseModel):
    mode: Literal["Snap to Corners"] = "Snap to Corners"
    """Tile images to fit a grid defined by the corner positions."""


class InplaceTiling(BaseModel):
    mode: Literal["Inplace"] = "Inplace"
    """
    Write tiles in their original stage positions.
    This may lead to artifacts if microscope stage positions are not precise,
    when tiles overlap the last written tile will overwrite previous tiles in the
    overlapping region.
    """


class NoTiling(BaseModel):
    mode: Literal["No Tiling"] = "No Tiling"
    """Each field of view is written as a single OME-Zarr."""


TilingStrategy = Annotated[
    AutoTiling | SnapToGridTiling | SnapToCornersTiling | InplaceTiling | NoTiling,
    Field(discriminator="mode"),
]


class BackendType(StrEnum):
    ANNDATA = "anndata"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


class Scalings(StrEnum):
    QUARTER = "0.25"
    HALF = "0.5"
    ONE = "1"
    DOUBLE = "2"
    QUADRUPLE = "4"

    def to_float(self) -> float:
        return float(self.value)


class WriterMode(StrEnum):
    BY_TILE = "By Tile"
    BY_FOV = "By FOV"
    BY_FOV_DASK = "By FOV (Using Dask)"
    BY_TILE_DASK = "By Tile (Using Dask)"
    IN_MEMORY = "In Memory"


class StagePositionCorrections(BaseModel):
    """Stage position corrections applied during registration."""

    remove_xy_offset: Literal["False", "Global"] = Field(
        default="Global", title="Remove XY Offset"
    )
    """
    Translate the mosaic so its XY origin is 0.
    Cases:
    - `False`: No translation is applied, failing if the stage
    position are negative, if stage position are positive results
    in left padded images.
    - `Global`: The mosaic is translated so its XY origin is 0.
    """
    remove_z_offset: Literal["False", "Per-FOV", "Global"] = Field(
        default="Global", title="Remove Z Offset"
    )
    """
    Remove Z offset from the mosaic.
    Cases:
    - `False`: No Z offset is removed, failing if the stage
    position are negative, if stage position are positive results
    in left padded images.
    - `Per-FOV`: The Z offset is removed per FOV.
    - `Global`: The mosaic is translated so its Z origin is 0.
    """
    remove_t_offset: Literal["False", "Global"] = Field(
        default="Global", title="Remove T Offset"
    )
    """
    Remove T offset from the mosaic.
    Cases:
    - `False`: No T offset is removed, failing if the stage
    position are negative, if stage position are positive results
    in left padded images.
    - `Global`: The mosaic is translated so its T origin is 0.
    """
    remove_xy_jitter: bool = Field(default=True, title="Remove XY Jitter")
    """
    Remove intra-FOV stage position inconsistencies (snap a FOV's sub-tiles to a
    shared origin).
    """
    reindex_channels: bool = Field(default=True, title="Reindex Channels")
    """
    If True only existing channels will be converted, if False missing channels will
    be stored as empty array.
    """
    model_config = ConfigDict(extra="forbid")


class FovBasedChunking(BaseModel):
    """Chunking strategy that matches the field of view."""

    mode: Literal["Same as FOV"] = "Same as FOV"
    """Chunking based on FOV size."""
    xy_scaling: Scalings = Field(default=Scalings.ONE, title="XY Scaling Factor")
    """
    Scaling factor for XY chunk size. If set to 1, chunk size matches FOV size.
    If set to 0.5, chunk size is half the FOV size (smaller chunks, more files).
    If set to 2, chunk size is double the FOV size (larger chunks, fewer files).
    """
    z_chunk: int = Field(default=10, ge=1, title="Chunk Size for Z")
    """Chunk size for Z dimension."""
    c_chunk: int = Field(default=1, ge=1, title="Chunk Size for C")
    """Chunk size for C dimension."""
    t_chunk: int = Field(default=1, ge=1, title="Chunk Size for T")
    """Chunk size for T dimension."""

    def get_xy_chunk(self, fov_xy_shape: int) -> int:
        scaling_factor = self.xy_scaling.to_float()
        chunk_size = int(fov_xy_shape * scaling_factor)
        return max(1, chunk_size)


class FixedSizeChunking(BaseModel):
    """Chunking strategy with fixed chunk sizes."""

    mode: Literal["Fixed Size"] = "Fixed Size"
    """Fixed size chunking."""
    xy_chunk: int = Field(default=4096, ge=1, title="Chunk Size for XY")
    """Chunk size for XY dimensions."""
    z_chunk: int = Field(default=10, ge=1, title="Chunk Size for Z")
    """Chunk size for Z dimension."""
    c_chunk: int = Field(default=1, ge=1, title="Chunk Size for C")
    """Chunk size for C dimension."""
    t_chunk: int = Field(default=1, ge=1, title="Chunk Size for T")
    """Chunk size for T dimension."""

    def get_xy_chunk(self, fov_shape: int) -> int:
        return self.xy_chunk


ChunkingStrategy = Annotated[
    FovBasedChunking | FixedSizeChunking, Field(discriminator="mode")
]


class OmeZarrOptions(BaseModel):
    """Options specific to OME-Zarr writing."""

    num_levels: int = Field(default=5, ge=1)
    """Number of resolution levels to create."""
    chunks: ChunkingStrategy = Field(
        default_factory=FovBasedChunking, title="Chunking Strategy"
    )
    """Chunking strategy to use."""
    ngff_version: NgffVersions = DefaultNgffVersion
    """Version of the OME-NGFF specification to target."""
    table_backend: BackendType = Field(
        default=BackendType.ANNDATA, title="Table Backend"
    )
    """Backend type for storing tables."""
    model_config = ConfigDict(extra="forbid")


class ConverterOptions(BaseModel):
    """Options for the OME-Zarr conversion process."""

    writer_mode: WriterMode = Field(default=WriterMode.BY_FOV, title="Writer Mode")
    """
    Mode for writing data during conversion.

    - By Tile: Write data one tile at a time. This consumes less memory, but may be
      slower.
    - By Tile (Using Dask): Write tiles in parallel using Dask. This is usually faster
      than writing by tile sequentially, but may consume more memory.
    - By FOV: Write data one field of view at a time. This may the best compromise
      between speed and memory usage in most cases.
    - By FOV (Using Dask): Write fields of view in parallel using Dask. This is usually
      faster than writing by FOV sequentially, but may consume more memory.
    - In Memory: Load all data into memory before writing.
    """
    tiling_strategy: TilingStrategy = Field(
        default_factory=AutoTiling, title="Tiling Strategy"
    )
    """
    Tiling strategy to use during conversion.

    - Auto: Automatically determine if Snap to Grid is possible, otherwise use Snap to
      Corners. Accepts an optional tolerance (in pixels) for grid alignment.
    - Snap to Grid: Tile images to fit a regular grid. This is only possible if image
      positions align to a grid (potentially with overlap). Accepts an optional
      tolerance (in pixels).
    - Snap to Corners: Tile images to fit a grid defined by the corner positions.
    - Inplace: Write tiles in their original positions without tiling. This may lead to
      artifacts if microscope stage positions are not precise.
    - No Tiling: Each field of view is written as a single OME-Zarr.
    """
    stage_position_corrections: StagePositionCorrections = Field(
        default_factory=StagePositionCorrections,
        title="Stage Position Corrections",
    )
    """Stage position correction options."""
    omezarr_options: OmeZarrOptions = Field(
        default_factory=OmeZarrOptions, title="OME-Zarr Options"
    )
    """Options specific to OME-Zarr writing."""
    runtime_settings: RuntimeSettings = Field(
        default_factory=RuntimeSettings, title="Runtime Settings"
    )
    """Runtime knobs (zarr codec, dask scheduler) applied via a scoped
    context manager during conversion."""
    model_config = ConfigDict(extra="forbid")
