# Pipeline Configuration

The conversion pipeline processes tiles through several stages before writing the final OME-Zarr dataset. This page documents the configurable components: **acquisition details**, **converter options**, **filters**, **registration steps**, **tiling modes**, **writer modes**, and **overwrite modes**.

## ConverterOptions

`ConverterOptions` is the central configuration object that bundles the most common settings:

```python
from ome_zarr_converters_tools import ConverterOptions, MosaicGrouping

opts = ConverterOptions(
    grouping=MosaicGrouping(),             # How FOVs are grouped into images
    writer_mode=WriterMode.BY_FOV,         # How data is loaded and written
    stage_position_corrections=StagePositionCorrections(),  # Stage position corrections
    omezarr_options=OmeZarrOptions(),      # OME-Zarr writing options (levels, chunks, etc.)
)
```

When passed to `tiles_aggregation_pipeline()` and `tiled_image_creation_pipeline()`, its fields are used as defaults. You can also override specific settings (like `writer_mode` or `grouping`) by passing them directly to the pipeline functions.

## AcquisitionDetails

`AcquisitionDetails` describes the physical properties of the acquisition: pixel sizes, coordinate systems, channel metadata, and stage corrections. It is passed when building `Tile` objects (via `hcs_images_from_dataframe()` or `single_images_from_dataframe()`) and is shared by all tiles in the same acquisition.

```python
from ome_zarr_converters_tools import AcquisitionDetails, ChannelInfo

acq = AcquisitionDetails(
    xy_pixel_size=0.65,          # XY pixel size in micrometers
    z_spacing=5.0,           # Z-step size in micrometers
    t_spacing=1.0,           # Time interval in seconds
    channels=[
        ChannelInfo(channel_label="DAPI", wavelength_id="405"),
        ChannelInfo(channel_label="GFP", wavelength_id="488"),
    ],
    axes=["c", "z", "y", "x"],  # Subset of t, c, z, y, x in canonical order
    start_x_space="world",    # How to interpret start_x values
    start_y_space="world",
    start_z_space="pixel",
    start_t_space="pixel",
)
```

### Coordinate Systems

Each `start_*` and `length_*` dimension has a coordinate system setting (`"world"` or `"pixel"`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `start_x_space`, `start_y_space` | `"world"` | Interpret start positions as physical units (micrometers). The library divides by `xy_pixel_size` to convert to pixels. |
| `start_z_space` | `"world"` | Interpret Z start as physical units. Divided by `z_spacing` to convert to pixels. |
| `start_t_space` | `"world"` | Interpret T start as physical units (seconds). Divided by `t_spacing`. |
| `length_x_space`, `length_y_space` | `"pixel"` | Interpret lengths as pixel counts (no conversion). |
| `length_z_space` | `"pixel"` | Interpret Z length as number of slices. |
| `length_t_space` | `"pixel"` | Interpret T length as number of time points. |

!!! tip
    Most microscopes report stage positions in physical units (micrometers) and image dimensions in pixels. The defaults (`start_*_space="world"`, `length_*_space="pixel"`) match this convention. If your metadata already provides pixel coordinates for positions, set `start_x_space="pixel"` etc.

### Channels

Channel metadata is defined with `ChannelInfo`:

```python
from ome_zarr_converters_tools import ChannelInfo

channels = [
    ChannelInfo(channel_label="DAPI", wavelength_id="405", colors="Blue (0000FF)"),
    ChannelInfo(channel_label="GFP", wavelength_id="488", colors="Green (00FF00)"),
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `channel_label` | Yes | Display name for the channel |
| `wavelength_id` | No | Alternative identifier, useful for illumination correction in multiplexed acquisitions |
| `colors` | No | Visualization color (default: `Blue`). Available: `Blue`, `Red`, `Yellow`, `Magenta`, `Cyan`, `Gray`, `Green`, `Orange`, `Purple`, `Teal`, `Lime`, `Amber`, `Pink`, `Navy`, `Maroon`, `Olive`, `Coral`, `Violet` |

### Axes

The `axes` parameter defines which dimensions the output image will have. It must be a subset of `["t", "c", "z", "y", "x"]` in canonical order:

```python
axes=["t", "c", "z", "y", "x"]  # 5D time-series (default)
axes=["c", "z", "y", "x"]       # 4D stack (no time)
axes=["z", "y", "x"]            # 3D single-channel
axes=["y", "x"]                 # 2D (minimum)
```

### Stage Corrections

Some microscopes have inverted or swapped stage axes. Use `StageOrientation` to fix this:

```python
from ome_zarr_converters_tools import AcquisitionDetails, StageOrientation

acq = AcquisitionDetails(
    xy_pixel_size=0.65,
    stage_orientation=StageOrientation(
        flip_x=True,    # Invert X positions
        flip_y=False,   # Keep Y as-is
        swap_xy=False,  # Don't swap X and Y
    ),
)
```

These corrections are applied when converting `Tile` positions to ROIs during the registration pipeline.

### Other Fields

| Field | Default | Description |
|-------|---------|-------------|
| `data_type` | `None` | Force the output data type (`"uint8"`, `"uint16"`, or `"uint32"`). If `None`, inferred from the first tile's image data. |
| `condition_table_path` | `None` | Path to an external condition table CSV. When set, this is stored in the plate metadata. |

## Filters

Filters are applied during tile aggregation to include or exclude tiles before they are grouped into `TiledImage` objects. Filters operate on individual `Tile` objects and return `True` (keep) or `False` (discard).

Filters match against the tile's **collection path** -- the output path derived from the collection type. For `ImageInPlate`, this is the plate/well/acquisition path (e.g., `"MyPlate.zarr/A/1/0"`). For `SingleImage`, this is `"{image_path}.zarr"`.

### Built-in Filters

#### RegexIncludeFilter

Keeps only tiles whose collection path matches a regex pattern. All non-matching tiles are discarded.

```python
from ome_zarr_converters_tools.pipelines import apply_filter_pipeline, FilterModel

# Import the specific filter from the internal module
from ome_zarr_converters_tools.pipelines._filters import RegexIncludeFilter

f = RegexIncludeFilter(regex=".*PlateA.*")
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])
```

#### RegexExcludeFilter

Removes tiles whose collection path matches a regex pattern. All non-matching tiles are kept.

```python
from ome_zarr_converters_tools.pipelines._filters import RegexExcludeFilter

f = RegexExcludeFilter(regex=".*control.*")
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])
```

#### WellExcludeFilter

Removes tiles belonging to specific wells. Only works with `ImageInPlate` collections.

```python
from ome_zarr_converters_tools.pipelines._filters import WellExcludeFilter

f = WellExcludeFilter(wells_to_remove=["A1", "B2"])
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])
```

#### WellIncludeFilter

Keeps only tiles belonging to specific wells, removing all others. Only works with `ImageInPlate` collections.

```python
from ome_zarr_converters_tools.pipelines._filters import WellIncludeFilter

f = WellIncludeFilter(wells_to_include=["A1", "B2"])
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])
```

!!! note
    The individual filter classes (`RegexIncludeFilter`, `RegexExcludeFilter`, `WellExcludeFilter`, `WellIncludeFilter`) are imported from `ome_zarr_converters_tools.pipelines._filters`. The public API exports `FilterModel` (the base class), `ImplementedFilters` (the union type), `apply_filter_pipeline`, and `add_filter`.

### Using Filters in the Pipeline

Filters can be passed directly to `tiles_aggregation_pipeline()`:

```python
from ome_zarr_converters_tools import tiles_aggregation_pipeline, ConverterOptions
from ome_zarr_converters_tools.pipelines._filters import RegexIncludeFilter

images = tiles_aggregation_pipeline(
    tiles=tiles,
    converter_options=ConverterOptions(),
    filters=[RegexIncludeFilter(regex=".*keep.*")],
)
```

### Custom Filters

Create a custom filter by subclassing `FilterModel` and registering the filter function with `add_filter()`:

```python
from ome_zarr_converters_tools.pipelines import FilterModel, add_filter
from ome_zarr_converters_tools.core import Tile


class LargeTilesFilter(FilterModel):
    """Filter that keeps only tiles wider than a threshold."""

    name: str = "large_tiles_only"
    min_width: int = 1000


def apply_large_tiles_filter(tile: Tile, filter_params: LargeTilesFilter) -> bool:
    """Keep only tiles with length_x > min_width."""
    return tile.length_x > filter_params.min_width


add_filter(function=apply_large_tiles_filter, name="large_tiles_only")
```

The filter can then be used with `apply_filter_pipeline()`:

```python
filtered = apply_filter_pipeline(
    tiles, filters_config=[LargeTilesFilter(min_width=512)]
)
```

## Registration Steps

The registration pipeline transforms tile positions to prepare them for writing. It runs as a sequence of steps, each modifying the `TiledImage` in place. These steps convert raw microscope stage positions into clean, non-overlapping pixel coordinates suitable for writing into a Zarr array.

### Default Registration Pipeline

The default pipeline is built with `build_default_registration_pipeline()` and runs four steps in order:

```python
from ome_zarr_converters_tools.models import StagePositionCorrections, AutoTiling
from ome_zarr_converters_tools.pipelines import build_default_registration_pipeline

pipeline = build_default_registration_pipeline(
    alignment_corrections=StagePositionCorrections(),
    tiling_strategy=AutoTiling(),
)
```

This creates:

1. **`offset_removal`** -- Removes per-axis stage offsets according to `StagePositionCorrections` (see below). By default (`"Global"`) each axis is shifted so its minimum position is 0.

2. **`align_to_pixel_grid`** -- Snaps start positions and lengths to integer pixel coordinates. After this step, all positions and sizes are exact pixel values (no sub-pixel offsets), including z/t which become integer plane/timepoint indices.

3. **`xy_jitter_correction`** -- When `remove_xy_jitter=True` (default), tiles within the same FOV that have slightly different XY positions (due to stage drift between Z-slices or channels) are snapped to the FOV's reference position.

4. **`reindex_channels`** -- When `reindex_channels=True` (default), the channel indices actually present are compacted to a dense `0, 1, 2, …` range and channel metadata is reconciled, so a filtered channel does not leave an empty channel in the output.

5. **`tile_regions`** -- Applies tiling/snapping to remove overlaps between FOVs (see [Tiling Strategies](#tiling-strategies) below). This is the step that determines the final non-overlapping layout.

### StagePositionCorrections

Controls the per-axis position handling applied during registration:

```python
from ome_zarr_converters_tools.models import StagePositionCorrections

corrections = StagePositionCorrections(
    remove_xy_offset="Global",   # "Global" (zero origin) or "Keep" (keep absolute)
    remove_z_offset="Global",    # "Global", "Per-FOV" (zero each FOV's z), or "Keep"
    remove_t_offset="Global",    # "Global" or "Keep"
    remove_xy_jitter=True,       # snap a FOV's sub-tiles to a shared XY origin
    reindex_channels=True,       # compact present channels to a dense range
)
```

Offset modes: `"Global"` translates the axis so its origin is 0; `"Keep"` keeps absolute positions (raising on negatives, left-padding on positives); `"Per-FOV"` (z only) zeros each FOV's z independently, useful when FOVs are focused independently. `remove_xy_jitter` corrects minor XY stage drift between the sub-acquisitions (Z-slices/channels) of a single FOV.

### Custom Registration Steps

Register custom steps with `add_registration_func()`:

```python
from ome_zarr_converters_tools.pipelines import add_registration_func
from ome_zarr_converters_tools.core import TiledImage


def my_custom_step(tiled_image: TiledImage, **kwargs) -> TiledImage:
    """Custom registration step that modifies tile positions."""
    for region in tiled_image.regions:
        # Modify region.roi as needed
        pass
    return tiled_image


add_registration_func(function=my_custom_step, name="my_step")
```

Then include it in a pipeline using `RegistrationStep`:

```python
from ome_zarr_converters_tools.pipelines import RegistrationStep

pipeline = [
    RegistrationStep(name="offset_removal", params={"corrections": corrections}),
    RegistrationStep(name="my_step", params={"some_param": 42}),
    RegistrationStep(name="align_to_pixel_grid", params={}),
]
```

## Grouping

Grouping (`ConverterOptions.grouping`) is the topology decision: how the fields of view of an acquisition map to output images. It is a discriminated union with two variants.

| Grouping | Description |
|----------|-------------|
| `MosaicGrouping(tiling_strategy=...)` | Aggregate all FOVs of an acquisition into a single mosaic OME-Zarr, arranged by the nested `tiling_strategy` (see below). This is the default. |
| `PerFovGrouping()` | Write each FOV as its own OME-Zarr image. There is no tiling strategy -- a single-FOV image has nothing to arrange. |

```python
from ome_zarr_converters_tools import (
    ConverterOptions,
    MosaicGrouping,
    PerFovGrouping,
    SnapToGridTiling,
)

# One mosaic image per acquisition (default), snapped to a grid
ConverterOptions(grouping=MosaicGrouping(tiling_strategy=SnapToGridTiling()))

# One OME-Zarr per field of view
ConverterOptions(grouping=PerFovGrouping())
```

## Tiling Strategies

Tiling controls how overlapping FOVs are arranged relative to each other *within a mosaic*. It lives on `MosaicGrouping.tiling_strategy` and is applied as the last step in the default registration pipeline. Each strategy is a Pydantic model, so tolerance and other parameters are passed directly on construction.

| Strategy | Description |
|----------|-------------|
| `AutoTiling(tolerance=0)` | Tries `SnapToGridTiling` first; falls back to `SnapToCornersTiling` if tiles don't form a regular grid |
| `SnapToGridTiling(tolerance=0)` | Snaps FOV positions to a regular grid, removing overlaps. Requires tiles to be arranged in a grid pattern |
| `SnapToCornersTiling()` | Snaps each FOV to the nearest corner, removing overlaps without requiring a grid structure |
| `InplaceTiling()` | No tiling -- keeps original positions as-is |

```python
from ome_zarr_converters_tools.models import (
    AutoTiling,
    SnapToCornersTiling,
    SnapToGridTiling,
)

# For regular grid acquisitions (e.g., snake scan)
pipeline = build_default_registration_pipeline(
    StagePositionCorrections(), SnapToGridTiling()
)

# For irregular FOV arrangements
pipeline = build_default_registration_pipeline(
    StagePositionCorrections(), SnapToCornersTiling()
)

# Let the library decide (with tolerance for minor stage jitter)
pipeline = build_default_registration_pipeline(
    StagePositionCorrections(), AutoTiling(tolerance=2.5)
)
```

## RuntimeSettings

`RuntimeSettings` configures low-level runtime behaviour for a conversion call. It is applied as a scoped context manager, so settings are restored when the block exits.

```python
from ome_zarr_converters_tools import RuntimeSettings
from ome_zarr_converters_tools.models import ThreadScheduler

settings = RuntimeSettings(
    use_zarrs_codec=True,                     # Use the Zarrs Rust codec backend
    dask_scheduler=ThreadScheduler(num_workers=4),  # Run Dask with 4 threads
)

with settings.apply():
    # Conversion code here runs under the configured scheduler and codec
    ...
```

### Dask Schedulers

| Scheduler | Description |
|-----------|-------------|
| `DefaultScheduler()` | Do not change Dask's scheduler (default) |
| `ThreadScheduler(num_workers=8)` | Use Dask's threaded scheduler |
| `ProcessScheduler(num_workers=8)` | Use Dask's multiprocessing scheduler |
| `SynchronousScheduler()` | Use Dask's synchronous scheduler (single-threaded, useful for debugging) |

### Zarrs Codec Backend

Setting `use_zarrs_codec=True` switches Zarr to the `zarrs.ZarrsCodecPipeline` Rust backend, which can significantly speed up encoding/decoding. This requires the optional `zarrs` package:

```bash
pip install ome-zarr-converters-tools[zarrs]
```

### Temporary JSON Storage

`RuntimeSettings` also controls where temporary JSON metadata is stored during conversion via `temp_json_options`. The default stores it alongside the Zarr output:

```python
from ome_zarr_converters_tools.models._runtime_settings import TempJsonOptions

settings = RuntimeSettings(
    temp_json_options=TempJsonOptions(temp_url="{zarr_dir}/_my_tmp"),
)
```

## Writer Modes

Writer modes control how image data is loaded and written to the OME-Zarr dataset. The choice affects memory usage and performance.

| Mode | Description | Memory | Speed |
|------|-------------|--------|-------|
| `WriterMode.BY_TILE` | Loads and writes one tile at a time, sequentially | Low | Slow |
| `WriterMode.BY_FOV` | Loads and writes one FOV at a time, sequentially | Medium | Medium |
| `WriterMode.BY_FOV_DASK` | Loads FOVs lazily via Dask, writes one at a time | Medium | Fast |
| `WriterMode.BY_TILE_DASK` | Loads all tiles lazily via Dask, writes at once | High | Fast |
| `WriterMode.IN_MEMORY` | Loads entire image into memory, writes at once | High | Fastest |

```python
from ome_zarr_converters_tools.models import WriterMode, OverwriteMode
from ome_zarr_converters_tools.pipelines import tiled_image_creation_pipeline

# For large datasets with limited memory
omezarr = tiled_image_creation_pipeline(
    zarr_url=zarr_url,
    tiled_image=tiled_image,
    registration_pipeline=pipeline,
    converter_options=opts,
    writer_mode=WriterMode.BY_FOV,
    overwrite_mode=OverwriteMode.OVERWRITE,
    resource=data_dir,
)

# For small datasets where speed matters
omezarr = tiled_image_creation_pipeline(
    ...,
    writer_mode=WriterMode.IN_MEMORY,
)
```

### Choosing a Writer Mode

- **`BY_FOV`** (default recommendation): good balance of memory usage and performance. Loads one FOV at a time.
- **`BY_FOV_DASK`**: same as `BY_FOV` but uses Dask for lazy loading, which can be faster for large tiles.
- **`BY_TILE`**: lowest memory usage, useful when individual tiles are very large.
- **`IN_MEMORY`**: fastest for small datasets that fit entirely in memory.
- **`BY_TILE_DASK`**: loads everything lazily and writes at once. Good when Dask is already part of your workflow.

## Overwrite Modes

Overwrite modes control what happens when the target OME-Zarr dataset already exists.

| Mode | Description |
|------|-------------|
| `OverwriteMode.NO_OVERWRITE` | Raises an error if the dataset already exists |
| `OverwriteMode.OVERWRITE` | Deletes the existing dataset and creates a new one |
| `OverwriteMode.EXTEND` | Adds new data to the existing dataset without deleting it |

```python
from ome_zarr_converters_tools.models import OverwriteMode

# Fail if output already exists (safe default for production)
overwrite_mode = OverwriteMode.NO_OVERWRITE

# Replace existing output (useful during development)
overwrite_mode = OverwriteMode.OVERWRITE

# Append to existing dataset
overwrite_mode = OverwriteMode.EXTEND
```
