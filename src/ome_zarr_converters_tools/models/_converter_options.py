from enum import StrEnum
from typing import Annotated, Literal, Self

from ngio import DefaultNgffVersion, NgffVersions
from pydantic import (
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ome_zarr_converters_tools.models._base import UserFacingModel
from ome_zarr_converters_tools.models._runtime_settings import RuntimeSettings


class OverwriteMode(StrEnum):
    """How to handle existing output data: fail, replace, or extend it.

    `No Overwrite` fails instead of touching existing data, `Overwrite`
    removes and replaces it, and `Extend` adds new wells or acquisitions
    while leaving existing ones untouched.
    """

    NO_OVERWRITE = "No Overwrite"
    OVERWRITE = "Overwrite"
    EXTEND = "Extend"


class AutoTiling(UserFacingModel):
    """Automatically pick between `Snap to Grid` and `Snap to Corners`."""

    model_config = ConfigDict(title="Auto")

    mode: Literal["Auto"] = "Auto"
    """Automatically use `Snap to Grid` when the field of view positions
    align to a regular grid, `Snap to Corners` otherwise."""
    tolerance: float = Field(default=1, ge=0, title="Tiling Tolerance (in pixels)")
    """Maximum misalignment (in pixels) tolerated when checking whether the
    field of view positions fit a regular grid."""


class SnapToGridTiling(UserFacingModel):
    """Arrange the fields of view on a regular grid."""

    model_config = ConfigDict(title="Snap to Grid")

    mode: Literal["Snap to Grid"] = "Snap to Grid"
    """Arrange the fields of view on a regular grid. Only possible when
    their positions already align to a grid (potentially with overlap)."""
    tolerance: float = Field(default=1, ge=0, title="Tiling Tolerance (in pixels)")
    """Maximum misalignment (in pixels) tolerated when checking whether the
    field of view positions fit a regular grid."""


class SnapToCornersTiling(UserFacingModel):
    """Arrange the fields of view on a grid defined by their corners."""

    model_config = ConfigDict(title="Snap to Corners")

    mode: Literal["Snap to Corners"] = "Snap to Corners"
    """Arrange the fields of view on a grid defined by their corner
    positions."""


class InplaceTiling(UserFacingModel):
    """Keep every field of view at its original stage position."""

    model_config = ConfigDict(title="Inplace")

    mode: Literal["Inplace"] = "Inplace"
    """Keep every field of view at its original stage position. Imprecise
    stage positions **may produce artifacts**: where tiles overlap, the last
    written tile wins."""


TilingStrategy = Annotated[
    AutoTiling | SnapToGridTiling | SnapToCornersTiling | InplaceTiling,
    Field(discriminator="mode"),
]


class MosaicGrouping(UserFacingModel):
    """Aggregate all fields of view of an acquisition into one mosaic OME-Zarr."""

    mode: Literal["Mosaic"] = "Mosaic"
    """How fields of view are grouped into output images."""
    tiling_strategy: TilingStrategy = Field(
        default_factory=AutoTiling, title="Tiling Strategy"
    )
    """
    How the fields of view are arranged within the mosaic.

    - `Auto`: Automatically determine if `Snap to Grid` is possible,
      otherwise use `Snap to Corners`. Accepts an optional tolerance
      (in pixels) for grid alignment.
    - `Snap to Grid`: Tile images to fit a regular grid. This is only
      possible if image positions align to a grid (potentially with
      overlap). Accepts an optional tolerance (in pixels).
    - `Snap to Corners`: Tile images to fit a grid defined by the corner
      positions.
    - `Inplace`: Write tiles in their original positions without tiling.
      This may lead to artifacts if microscope stage positions are not
      precise.
    """
    model_config = ConfigDict(title="Mosaic")

    @property
    def split_per_fov(self) -> bool:
        """Whether each field of view becomes its own OME-Zarr image."""
        return False

    def tiling_for_registration(self) -> TilingStrategy:
        """Within-image arrangement strategy for the registration pipeline."""
        return self.tiling_strategy


class PerFovGrouping(UserFacingModel):
    """Write each field of view as its own OME-Zarr image (no mosaic)."""

    mode: Literal["Per-FOV"] = "Per-FOV"
    """How fields of view are grouped into output images."""
    model_config = ConfigDict(title="Per-FOV")

    @property
    def split_per_fov(self) -> bool:
        """Whether each field of view becomes its own OME-Zarr image."""
        return True

    def tiling_for_registration(self) -> TilingStrategy:
        """Within-image arrangement strategy for the registration pipeline.

        Per-FOV images contain a single field of view, so there is nothing to
        arrange; `InplaceTiling` is a no-op here (identical to the removed
        `NoTiling` path in the tiling step).
        """
        return InplaceTiling()


Grouping = Annotated[
    MosaicGrouping | PerFovGrouping,
    Field(discriminator="mode"),
]


class BackendType(StrEnum):
    """File format used to store tables (e.g. ROI tables) in the OME-Zarr."""

    ANNDATA = "anndata"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


class Scalings(StrEnum):
    """Scaling factor applied to the field of view size to get the chunk size."""

    QUARTER = "0.25"
    HALF = "0.5"
    ONE = "1"
    DOUBLE = "2"
    QUADRUPLE = "4"

    def to_float(self) -> float:
        return float(self.value)


class WriterMode(StrEnum):
    """How image data is written, trading off speed against memory usage."""

    BY_TILE = "By Tile"
    BY_FOV = "By FOV"
    BY_FOV_DASK = "By FOV (Using Dask)"
    BY_TILE_DASK = "By Tile (Using Dask)"
    IN_MEMORY = "In Memory"


class StagePositionCorrections(UserFacingModel):
    """Corrections applied to the stage positions before writing the image."""

    remove_xy_offset: Literal["Keep", "Global"] = Field(
        default="Global", title="Remove XY Offset"
    )
    """
    Whether to shift the image so its XY origin is 0.

    - `Global`: Shift all positions together so the image origin is 0.
    - `Keep`: Use the stage positions as-is. **Fails** if any position is
      negative; positive positions produce empty padding at the image origin.
    """
    remove_z_offset: Literal["Keep", "Per-FOV", "Global"] = Field(
        default="Global", title="Remove Z Offset"
    )
    """
    Whether to shift the image so its Z origin is 0.

    - `Global`: Shift all positions together so the Z origin is 0.
    - `Per-FOV`: Shift each field of view independently to Z origin 0.
    - `Keep`: Use the stage positions as-is. **Fails** if any position is
      negative; positive positions produce empty padding at the image origin.
    """
    remove_t_offset: Literal["Keep", "Global"] = Field(
        default="Global", title="Remove T Offset"
    )
    """
    Whether to shift the image so its time origin is 0.

    - `Global`: Shift all positions together so the time origin is 0.
    - `Keep`: Use the stage positions as-is. **Fails** if any position is
      negative; positive positions produce empty padding at the image origin.
    """
    remove_xy_jitter: bool = Field(default=True, title="Remove XY Jitter")
    """
    Remove small stage position inconsistencies within a field of view by
    snapping its sub-tiles to a shared origin.
    """
    reindex_channels: bool = Field(default=True, title="Reindex Channels")
    """
    If enabled, only the channels actually present are converted; if
    disabled, missing channels are stored as empty images.
    """


class FovBasedChunking(UserFacingModel):
    """Store the image in chunks sized like the fields of view."""

    model_config = ConfigDict(title="Same as FOV")

    mode: Literal["Same as FOV"] = "Same as FOV"
    """How the image is split into storage chunks."""
    xy_scaling: Scalings = Field(default=Scalings.ONE, title="XY Scaling Factor")
    """
    Scaling factor for the XY chunk size, relative to the field of view size.

    - `1`: chunks match the field of view size.
    - `0.5`: chunks are half the field of view (smaller chunks, more files).
    - `2`: chunks are double the field of view (larger chunks, fewer files).
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


class FixedSizeChunking(UserFacingModel):
    """Store the image in chunks of a fixed size."""

    model_config = ConfigDict(title="Fixed Size")

    mode: Literal["Fixed Size"] = "Fixed Size"
    """How the image is split into storage chunks."""
    xy_chunk: int = Field(default=4096, ge=1, title="Chunk Size for XY")
    """Chunk size for XY dimensions."""
    z_chunk: int = Field(default=10, ge=1, title="Chunk Size for Z")
    """Chunk size for Z dimension."""
    c_chunk: int = Field(default=1, ge=1, title="Chunk Size for C")
    """Chunk size for C dimension."""
    t_chunk: int = Field(default=1, ge=1, title="Chunk Size for T")
    """Chunk size for T dimension."""

    def get_xy_chunk(self, fov_xy_shape: int) -> int:
        return self.xy_chunk


ChunkingStrategy = Annotated[
    FovBasedChunking | FixedSizeChunking, Field(discriminator="mode")
]


class NoSharding(UserFacingModel):
    """Store every chunk as its own object (no sharding)."""

    model_config = ConfigDict(title="No Sharding")

    mode: Literal["None"] = "None"
    """How storage chunks are grouped into shards."""


class ChunkMultipleSharding(UserFacingModel):
    """Group a fixed number of chunks per axis into one shard."""

    model_config = ConfigDict(title="Multiple of Chunks")

    mode: Literal["Multiple of Chunks"] = "Multiple of Chunks"
    """How storage chunks are grouped into shards."""
    xy_factor: int = Field(default=4, ge=1, title="Chunks per Shard in XY")
    """Number of chunks per shard along X and along Y."""
    z_factor: int = Field(default=1, ge=1, title="Chunks per Shard in Z")
    """Number of chunks per shard along Z."""
    c_factor: int = Field(default=1, ge=1, title="Chunks per Shard in C")
    """Number of chunks per shard along C."""
    t_factor: int = Field(default=1, ge=1, title="Chunks per Shard in T")
    """Number of chunks per shard along T."""

    def get_factor(self, axis: str) -> int:
        """Chunks per shard along `axis` (`1` for axes without a setting)."""
        if axis in ("x", "y"):
            return self.xy_factor
        if axis == "z":
            return self.z_factor
        if axis == "c":
            return self.c_factor
        if axis == "t":
            return self.t_factor
        return 1


ShardingStrategy = Annotated[
    NoSharding | ChunkMultipleSharding, Field(discriminator="mode")
]


class NumberOfLevels(UserFacingModel):
    """Create a number of resolution levels with default names (`0`, `1`, ...)."""

    model_config = ConfigDict(title="Number of Levels")

    mode: Literal["Number of Levels"] = "Number of Levels"
    """How the resolution levels of the pyramid are defined."""
    num_levels: int = Field(default=5, ge=1, title="Number of Resolution Levels")
    """Number of resolution levels in the pyramid of the output image."""

    def to_ngio_levels(self) -> int | list[str]:
        """Levels in the form ngio's `create_empty_ome_zarr` accepts."""
        return self.num_levels


class NamedLevels(UserFacingModel):
    """Name each resolution level explicitly, from highest to lowest resolution."""

    model_config = ConfigDict(title="Custom Names")

    mode: Literal["Custom Names"] = "Custom Names"
    """How the resolution levels of the pyramid are defined."""
    level_names: list[str] = Field(min_length=1, title="Level Names")
    """Names of the resolution levels, from highest to lowest resolution,
    e.g. `["s0", "s1", "s2"]`. Each name becomes a path inside the OME-Zarr."""

    @field_validator("level_names")
    @classmethod
    def _validate_level_names(cls, level_names: list[str]) -> list[str]:
        for name in level_names:
            if not name or "/" in name or name != name.strip():
                raise ValueError(
                    f"Invalid level name {name!r}: each level name must be a "
                    "non-empty path segment without '/' or leading/trailing "
                    "whitespace, e.g. ['s0', 's1', 's2']."
                )
        duplicates = sorted({n for n in level_names if level_names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate level names {duplicates}: each resolution level "
                "needs a unique name, e.g. ['s0', 's1', 's2']."
            )
        return level_names

    def to_ngio_levels(self) -> int | list[str]:
        """Levels in the form ngio's `create_empty_ome_zarr` accepts."""
        return self.level_names


PyramidLevels = Annotated[NumberOfLevels | NamedLevels, Field(discriminator="mode")]


class OmeZarrOptions(UserFacingModel):
    """Options controlling the layout of the output OME-Zarr."""

    levels: PyramidLevels = Field(
        default_factory=NumberOfLevels, title="Resolution Levels"
    )
    """
    How many resolution levels the output pyramid has and how they are named.

    - `Number of Levels`: a number of levels with the default names
      (`0`, `1`, ...).
    - `Custom Names`: an explicit list of level names, e.g. `["s0", "s1"]`.
    """
    chunks: ChunkingStrategy = Field(
        default_factory=FovBasedChunking, title="Chunking Strategy"
    )
    """How the image is split into storage chunks: sized like the fields of
    view (`Same as FOV`) or with fixed sizes (`Fixed Size`)."""
    shards: ShardingStrategy = Field(
        default_factory=NoSharding, title="Sharding Strategy"
    )
    """
    How storage chunks are grouped into shards (one object per shard).
    Requires NGFF 0.5 (zarr format 3).

    - `None`: one object per chunk.
    - `Multiple of Chunks`: a fixed number of chunks per axis form one shard.

    A shard is the atomic write unit, so prefer the Dask writer modes (which
    write one shard at a time) or shards no larger than a field of view;
    otherwise every write re-encodes the shards it touches.
    """
    ngff_version: NgffVersions = Field(default=DefaultNgffVersion, title="NGFF Version")
    """Version of the OME-NGFF specification to target."""
    table_backend: BackendType = Field(default=BackendType.CSV, title="Table Backend")
    """File format used to store tables (e.g. ROI tables) inside the
    OME-Zarr."""

    @model_validator(mode="after")
    def _shards_require_ngff_0_5(self) -> Self:
        if not isinstance(self.shards, NoSharding) and self.ngff_version == "0.4":
            raise ValueError(
                "Sharding requires OME-Zarr 0.5 (zarr format 3), but "
                "ngff_version='0.4' was requested. Set ngff_version='0.5' or "
                "use the `None` sharding strategy."
            )
        return self


class ConverterOptions(UserFacingModel):
    """Options for the OME-Zarr conversion process."""

    writer_mode: WriterMode = Field(default=WriterMode.BY_FOV, title="Writer Mode")
    """
    Mode for writing data during conversion.

    - `By Tile`: Write data one tile at a time. This consumes less memory, but may be
      slower.
    - `By Tile (Using Dask)`: Build the whole image lazily with Dask and stream
      it a few write blocks at a time, so memory does not grow with the image.
      Usually faster than writing by tile sequentially; each tile is loaded once.
    - `By FOV`: Write data one field of view at a time. Often the best compromise
      between speed and memory usage in most cases.
    - `By FOV (Using Dask)`: Build each field of view lazily with Dask and
      stream it a few write blocks at a time. Usually faster than writing by
      FOV sequentially.
    - `In Memory`: Load all data into memory before writing.
    """
    grouping: Grouping = Field(default_factory=MosaicGrouping, title="Grouping")
    """
    How fields of view are grouped into output images.

    - `Mosaic`: Aggregate all fields of view of an acquisition into one OME-Zarr,
      arranged by the nested `tiling_strategy`.
    - `Per-FOV`: Write each field of view as its own OME-Zarr image (no mosaic,
      no tiling strategy).
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
    """Advanced performance settings: parallelism, storage codec, and
    temporary storage. The defaults are safe for most conversions."""
