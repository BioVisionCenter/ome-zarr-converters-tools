from enum import StrEnum
from typing import Annotated, Literal

from ngio import DefaultNgffVersion, NgffVersions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class OverwriteMode(StrEnum):
    NO_OVERWRITE = "No Overwrite"
    OVERWRITE = "Overwrite"
    EXTEND = "Extend"


class TilingMode(StrEnum):
    AUTO = "Auto"
    SNAP_TO_GRID = "Snap to Grid"
    SNAP_TO_CORNERS = "Snap to Corners"
    INPLACE = "Inplace"
    NO_TILING = "No Tiling"


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


class AlignmentCorrections(BaseModel):
    """Alignment correction for stage positions.

    Attributes:
        align_xy: Whether to align the positions in the XY plane by FOV.
            This addresses minor imprecision that often occurs during
            image acquisition.
        align_z: Whether to align the positions in the Z axis by FOV.
            This addresses minor imprecision that often occurs during
            image acquisition.
        align_t: Whether to align the positions in the T axis by FOV.
            This addresses minor imprecision that often occurs during
            image acquisition.
    """

    align_xy: bool = Field(default=False, title="Align XY")
    align_z: bool = Field(default=False, title="Align Z")
    align_t: bool = Field(default=False, title="Align T")
    model_config = ConfigDict(extra="forbid")


class FovBasedChunking(BaseModel):
    """Chunking strategy that matches the field of view.

    Attributes:
        mode: Chunking based on FOV size.
        xy_scaling: Scaling factor for XY chunk size. If set to 1, chunk size matches
            FOV size. If set to 0.5, chunk size is half the FOV size
            (smaller chunks, more files). If set to 2, chunk size is double the FOV
            size (larger chunks, less files).
        z_chunk: Chunk size for Z dimension.
        c_chunk: Chunk size for C dimension.
        t_chunk: Chunk size for T dimension.
    """

    mode: Literal["Same as FOV"] = "Same as FOV"
    xy_scaling: Scalings = Field(default=Scalings.ONE, title="XY Scaling Factor")
    z_chunk: int = Field(default=10, ge=1, title="Chunk Size for Z")
    c_chunk: int = Field(default=1, ge=1, title="Chunk Size for C")
    t_chunk: int = Field(default=1, ge=1, title="Chunk Size for T")

    def get_xy_chunk(self, fov_xy_shape: int) -> int:
        scaling_factor = self.xy_scaling.to_float()
        chunk_size = int(fov_xy_shape * scaling_factor)
        return max(1, chunk_size)


class FixedSizeChunking(BaseModel):
    """Chunking strategy with fixed chunk sizes.

    Attributes:
        mode: Fixed size chunking.
        xy_chunk: Chunk size for XY dimensions.
        z_chunk: Chunk size for Z dimension.
        c_chunk: Chunk size for C dimension.
        t_chunk: Chunk size for T dimension.
    """

    mode: Literal["Fixed Size"] = "Fixed Size"
    xy_chunk: int = Field(default=4096, ge=1, title="Chunk Size for XY")
    z_chunk: int = Field(default=10, ge=1, title="Chunk Size for Z")
    c_chunk: int = Field(default=1, ge=1, title="Chunk Size for C")
    t_chunk: int = Field(default=1, ge=1, title="Chunk Size for T")

    def get_xy_chunk(self, fov_shape: int) -> int:
        return self.xy_chunk


ChunkingStrategy = Annotated[
    FovBasedChunking | FixedSizeChunking, Field(discriminator="mode")
]


class OmeZarrOptions(BaseModel):
    """Options specific to OME-Zarr writing.

    Attributes:
        num_levels: Number of resolution levels to create.
        chunks: Chunking strategy to use.
        ngff_version: Version of the OME-NGFF specification to target.
        table_backend: Backend type for storing tables.
    """

    num_levels: int = Field(default=5, ge=1)
    chunks: ChunkingStrategy = Field(
        default_factory=FovBasedChunking, title="Chunking Strategy"
    )
    ngff_version: NgffVersions = DefaultNgffVersion
    table_backend: BackendType = Field(
        default=BackendType.ANNDATA, title="Table Backend"
    )
    model_config = ConfigDict(extra="forbid")


class TempJsonOptions(BaseModel):
    """Options for temporary JSON storage during conversion.

    Attributes:
        temp_url: Template for the temporary JSON URL.
    """

    temp_url: str = "{zarr_dir}/_tmp_json"

    def format_temp_url(self, zarr_dir: str) -> str:
        return self.temp_url.format(zarr_dir=zarr_dir)


class ConverterOptions(BaseModel):
    """Options for the OME-Zarr conversion process.

    Attributes:
        tiling_mode: Tiling mode to use during conversion.
            - Auto: Automatically determine if Snap to Grid is possible,
            otherwise use Snap to Corners.
            - Snap to Grid: Tile images to fit a regular grid. This is
            only possible if image positions align to a grid (potentially with overlap).
            - Snap to Corners: Tile images to fit a grid defined by the corner
            positions.
            - Inplace: Write tiles in their original positions without tiling. This
            may lead to artifacts if microscope stage positions are not precise.
            - No Tiling: Each field of view is written as a single OME-Zarr.
        writer_mode: Mode for writing data during conversion.
            - By Tile: Write data one tile at a time. This consumes less memory,
            but may be slower.
            - By Tile (Using Dask): Write tiles in parallel using Dask. This is
            usually faster than writing by tile sequentially, but may consume more
            memory.
            - By FOV: Write data one field of view at a time. This may the best
            compromise between speed and memory usage in most cases.
            - By FOV (Using Dask): Write fields of view in parallel using Dask.
            This is usually faster than writing by FOV sequentially,
            but may consume more memory.
            - In Memory: Load all data into memory before writing.
        alignment_correction: Alignment correction options.
        omezarr_options: Options specific to OME-Zarr writing.
        temp_json_options: Options for temporary JSON storage.

    """

    tiling_mode: TilingMode = Field(default=TilingMode.AUTO, title="Tiling Mode")
    writer_mode: WriterMode = Field(default=WriterMode.BY_FOV, title="Writer Mode")
    alignment_correction: AlignmentCorrections = Field(
        default_factory=AlignmentCorrections,
        title="Alignment Corrections",
    )
    omezarr_options: OmeZarrOptions = Field(
        default_factory=OmeZarrOptions, title="OME-Zarr Options"
    )
    temp_json_options: TempJsonOptions = Field(
        default_factory=TempJsonOptions, title="Temporary JSON Options"
    )
    model_config = ConfigDict(extra="forbid")


# class ContextModel(NamedTuple):
#    """Base model for context information during conversion.
#
#    This models holds the all context information needed during the conversion
#    process, including acquisition details and converter options.
#    """
#
#    acquisition_details: AcquisitionDetails
#    converter_options: ConverterOptions
