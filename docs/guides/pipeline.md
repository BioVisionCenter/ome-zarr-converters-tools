---
description: "Configure the conversion pipeline: acquisition details, filters, validators, registration steps, tiling strategies, writer and overwrite modes."
---

# Pipeline configuration

**Every configurable stage of the pipeline, from parsed tiles to written store.**

The conversion pipeline processes tiles through several stages before writing the final OME-Zarr store. This page documents the configurable components in the order they appear below: **converter options**, **acquisition details**, **filters**, **validators**, **registration steps**, **grouping** and **tiling strategies**, **runtime settings**, **writer modes**, **overwrite modes**, and how to **raise errors** from your own code.

## Converter options

`ConverterOptions` is the central configuration object that bundles the most common settings:

```python
from ome_zarr_converters_tools import (
    ConverterOptions,
    MosaicGrouping,
    OmeZarrOptions,
    StagePositionCorrections,
    WriterMode,
)

opts = ConverterOptions(
    grouping=MosaicGrouping(),  # How FOVs are grouped into images
    writer_mode=WriterMode.BY_FOV,  # How data is loaded and written
    stage_position_corrections=StagePositionCorrections(),  # Offset and jitter handling
    omezarr_options=OmeZarrOptions(),  # OME-Zarr writing options (levels, chunks, etc.)
)
```

Each pipeline function reads the fields it needs: `tiles_aggregation_pipeline()` reads `grouping`; `tiled_image_creation_pipeline()` reads `omezarr_options` and `runtime_settings`, and takes `writer_mode` and `overwrite_mode` as explicit arguments. Only the Fractal compute task (`generic_compute_task()`) takes `writer_mode` from the options, so outside Fractal you pass it yourself, as the tutorials do.

### OME-Zarr options

`OmeZarrOptions` controls the on-disk layout of each OME-Zarr image:

- `levels`: how many resolution levels the pyramid has (`NumberOfLevels`) or their explicit names (`NamedLevels`).
- `chunks`: the storage chunk shape, sized like a field of view (`FovBasedChunking`, the default) or fixed (`FixedSizeChunking`).
- `shards`: how chunks are grouped into shards, i.e. one stored object holding several chunks. `NoSharding` (the default) stores one object per chunk; `ChunkMultipleSharding` groups a fixed number of chunks per axis. Sharding requires `ngff_version="0.5"` (Zarr v3); requesting it with `"0.4"` fails when the options are built.
- `ngff_version` and `table_backend`: the OME-NGFF version to write (`"0.4"` by default, or `"0.5"`) and the file format of the ROI/condition tables.

```python
from ome_zarr_converters_tools import (
    ChunkMultipleSharding,
    FixedSizeChunking,
    OmeZarrOptions,
)

opts = OmeZarrOptions(
    ngff_version="0.5",
    chunks=FixedSizeChunking(xy_chunk=512),
    shards=ChunkMultipleSharding(xy_factor=4),  # 2048 x 2048 pixel shards
)
```

A shard is the atomic write unit: every write re-encodes the shards it touches. The Dask writer modes stream a few write units at a time, so they pair well with large shards; with the NumPy writer modes keep shards no larger than a field of view.

## Acquisition details

`AcquisitionDetails` describes the physical properties of the acquisition: pixel sizes, coordinate systems, channel metadata, and stage orientation. It is passed when building `Tile` objects (via `hcs_images_from_dataframe()` or `single_images_from_dataframe()`) and is shared by all tiles in the same acquisition.

```python
from ome_zarr_converters_tools import AcquisitionDetails, ChannelInfo

acq = AcquisitionDetails(
    xy_pixel_size=0.65,  # XY pixel size in micrometers
    z_spacing=5.0,  # Z-step size in micrometers
    t_spacing=1.0,  # Time interval in seconds
    channels=[
        ChannelInfo(channel_label="DAPI", wavelength_id="405"),
        ChannelInfo(channel_label="GFP", wavelength_id="488"),
    ],
    axes=["c", "z", "y", "x"],  # Subset of t, c, z, y, x in canonical order
    start_x_space="world",  # How to interpret start_x values
    start_y_space="world",
    start_z_space="pixel",
    start_t_space="pixel",
)
```

### Coordinate systems

Each `start_*` and `length_*` dimension has a coordinate system setting (`"world"` or `"pixel"`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `start_x_space`, `start_y_space` | `"world"` | Interpret start positions as physical units (micrometers). The library divides by `xy_pixel_size` to convert to pixels. |
| `start_z_space` | `"world"` | Interpret Z start as physical units. Divided by `z_spacing` to convert to pixels. |
| `start_t_space` | `"world"` | Interpret T start as physical units (seconds). Divided by `t_spacing`. |
| `length_x_space`, `length_y_space` | `"pixel"` | Interpret lengths as pixel counts (no conversion). |
| `length_z_space` | `"pixel"` | Interpret Z length as number of slices. |
| `length_t_space` | `"pixel"` | Interpret T length as number of time points. |

!!! tip "The defaults match what most microscopes report"
    Most microscopes report stage positions in physical units (micrometers) and image dimensions in pixels. The defaults (`start_*_space="world"`, `length_*_space="pixel"`) match this convention. If your metadata already provides pixel coordinates for positions, set `start_x_space="pixel"` etc.

!!! warning "Positions are the top-left corner"

    `start_x` and `start_y` are the corner of the tile with the smallest X and Y, in the
    same frame as the image axes: the row index grows with Y. If your metadata reports
    the centre of each field of view, subtract `length / 2` in the parser. A centre
    value shifts every FOV by half a tile, and snapping to a grid still "works", so the
    error goes unnoticed. Use `StageOrientation` when the finished mosaic comes out
    mirrored or transposed.

### Channels

Channel metadata is defined with `ChannelInfo`:

```python
from ome_zarr_converters_tools import ChannelInfo

channels = [
    ChannelInfo(channel_label="DAPI", wavelength_id="405", color="#0000FF"),
    ChannelInfo(channel_label="GFP", wavelength_id="488", color="#00FF00"),
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `channel_label` | Yes | Display name for the channel |
| `wavelength_id` | No | Alternative identifier, useful for illumination correction in multiplexed acquisitions |
| `color` | No | Display colour as a hex string (`#RRGGBB` or `#RGB`). When omitted it is chosen from `wavelength_id`, or by matching the label against common fluorophore names. Fractal task users pick from a named menu instead; see [acquisition options](fractal_tasks.md#acquisition-options). |

### Axes

The `axes` parameter defines which dimensions the output image will have. It must be a subset of `["t", "c", "z", "y", "x"]` in canonical order:

```python
axes = ["t", "c", "z", "y", "x"]  # 5D time-series (default)
axes = ["c", "z", "y", "x"]  # 4D stack (no time)
axes = ["z", "y", "x"]  # 3D single-channel
axes = ["y", "x"]  # 2D (minimum)
```

Omitting `c` means the output has no channel axis, so `channels` must be left as `None`; providing channel metadata for a `c`-less image is an error.

### Channel indices

`start_c` and `length_c` on a `Tile` index into `acquisition_details.channels`: a tile covers channels `start_c … start_c + length_c - 1`. Most parsers emit one single-channel tile per channel, but `length_c > 1` is supported for acquisitions that capture several channels at once (e.g. a two-camera system). Because the span is a contiguous range, channels acquired together must be **adjacent** in `channels` — if they are not, order the channel list so they are, or emit one tile per channel.

### Stage orientation

Some microscopes have inverted or swapped stage axes. Use `StageOrientation` to fix this:

```python
from ome_zarr_converters_tools import AcquisitionDetails, StageOrientation

acq = AcquisitionDetails(
    xy_pixel_size=0.65,
    stage_orientation=StageOrientation(
        flip_x=True,  # Invert X positions
        flip_y=False,  # Keep Y as-is
        swap_xy=False,  # Don't swap X and Y
    ),
)
```

These corrections are applied when converting `Tile` positions to ROIs during the registration pipeline.

### Other fields

| Field | Default | Description |
|-------|---------|-------------|
| `data_type` | `"autodetect"` | Force the output data type (`"uint8"`, `"uint16"`, or `"uint32"`). With `"autodetect"` (`DataTypeEnum.AUTODETECT`) it is inferred from the first tile's image data, which means that tile is read during aggregation. |
| `condition_table_path` | `None` | Path of a condition table supplied by the task user, carried over from `AcquisitionOptions` for converters that attach one themselves. The built-in pipelines do not read it. |

### Attributes, condition tables and the image list

`Tile.attributes` carries per-image metadata the parser knows and the output should
keep: a plate map's compound and concentration, say. The rules, which the tutorials'
odd-looking `drug: ["DMSO"]` follows:

- Values are **lists**, one element per row of the condition table: `{"drug": ["DMSO"],
  "conc_um": [10.0]}` is one row, two-element lists are two rows. Every attribute of one
  image must have the same length; pad with `None`.
- Every image that has attributes must have the **same keys** (`None` values are
  allowed), and `row`, `column`, `acquisition` and `path_in_well` are **reserved**.
  Violations raise `ConvertersToolsUserError` when the collection is set up.
- Attributes are read when an image is created from its first tile, so give every tile
  of an image the same attributes.
- What is written: a `condition_table` next to each image with the attribute columns,
  and for plates one more at plate level with the reserved columns added, so that each
  row names its well and acquisition.
- The Fractal image list receives the attributes as scalars (a one-element list becomes
  its element, longer lists are joined with ` & `), plus `plate`, `well` and
  `acquisition` for plates, and the types `is_3D` and, for time series,
  `is_time_series`.
- Filter tiles on them with `AttributeFilter(key="drug", values=[StringValue(value="DMSO")])`;
  see [filters](#filters).

## Filters

Filters are applied during tile aggregation to include or exclude tiles before they are grouped into `TiledImage` objects. Filters operate on individual `Tile` objects and return `True` (keep) or `False` (discard).

Each filter looks at one property of the tile: its output path, well, acquisition, FOV name, channels, attributes, or Z/T start. The **output path** is derived from the collection type: for `ImageInPlate` it is the plate/well/acquisition path (e.g. `"MyPlate.zarr/A/01/0"`), for `SingleImage` it is `"{image_path}.zarr"`.

### Built-in filters

| Filter | Matches | `mode` |
|--------|---------|--------|
| `RegexFilter(regex=...)` | Output path against a regular expression | Yes |
| `WellFilter(wells=[...])` | Well id, as row letter plus two-digit column (`"A01"`); plates only | Yes |
| `AcquisitionFilter(acquisitions=[...])` | Acquisition index inside the well; plates only | Yes |
| `FovNameFilter(regex=...)` | `fov_name` against a regular expression | Yes |
| `ChannelFilter(channel_labels=[...])` | Tiles whose channel labels are all in the list; a partial match raises | Yes |
| `AttributeFilter(key=..., values=[...])` | One of the tile's `attributes[key]` values equals a listed typed value | Yes |
| `ZRangeFilter(min_z=..., max_z=...)` | `start_z` within the inclusive range | No |
| `TRangeFilter(min_t=..., max_t=...)` | `start_t` within the inclusive range | No |

Every matching filter carries a `mode` field (`"Include"` or `"Exclude"`, default `"Include"`): in `Include` mode, matching tiles are kept and all others discarded; in `Exclude` mode, matching tiles are discarded and all others kept. The two range filters always keep what is inside the range. Two of them in use:

```python
from ome_zarr_converters_tools import RegexFilter, apply_filter_pipeline

# Keep only tiles from PlateA
f = RegexFilter(regex=".*PlateA.*")
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])

# Remove control tiles
f = RegexFilter(regex=".*control.*", mode="Exclude")
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])
```

```python
from ome_zarr_converters_tools import WellFilter, apply_filter_pipeline

# Keep only wells A01 and B02
f = WellFilter(wells=["A01", "B02"])
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])

# Remove wells A01 and B02
f = WellFilter(wells=["A01", "B02"], mode="Exclude")
filtered_tiles = apply_filter_pipeline(tiles, filters_config=[f])
```

!!! note "Where the filters live"
    Every filter class is exported from the package root and from `ome_zarr_converters_tools.pipelines`, alongside `FilterModel` (the base class), `ImplementedFilters` (the union type), `apply_filter_pipeline` and `add_filter`. See the [pipelines API reference](../api/pipelines.md#filters) for every field.

### Using filters in the pipeline

Filters can be passed directly to `tiles_aggregation_pipeline()`:

```python
from ome_zarr_converters_tools import (
    ConverterOptions,
    RegexFilter,
    tiles_aggregation_pipeline,
)

images = tiles_aggregation_pipeline(
    tiles=tiles,
    converter_options=ConverterOptions(),
    filters=[RegexFilter(regex=".*keep.*")],
)
```

### Custom filters

Create a custom filter by subclassing `FilterModel` and registering the filter function with `add_filter()`:

```python
from ome_zarr_converters_tools import FilterModel, Tile, add_filter


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
from ome_zarr_converters_tools import apply_filter_pipeline

filtered = apply_filter_pipeline(
    tiles, filters_config=[LargeTilesFilter(min_width=512)]
)
```

## Validators

Validators run after aggregation, once per `TiledImage`, and raise when something about the image is wrong. They are opt-in because they cost I/O: a probe reads real data.

### Built-in validators

`ShapeDtypeProbeValidator` runs `preflight()` on every tile's loader (a cheap reachability check that warns on problems), then fully loads one sample tile per image and raises if its shape or dtype does not match the declared geometry. Enable it while developing a converter, where a loader returning the wrong axes or a wrong `length_*` or `data_type` would otherwise only surface at write time:

```python
from ome_zarr_converters_tools import (
    ConverterOptions,
    ShapeDtypeProbeValidator,
    tiles_aggregation_pipeline,
)

images = tiles_aggregation_pipeline(
    tiles=tiles,
    converter_options=ConverterOptions(),
    validators=[ShapeDtypeProbeValidator()],
    resource=data_dir,
)
```

### Custom validators

Create a custom validator by subclassing `ValidatorModel` and registering the function with `add_validator()`. The function is called as `function(tiled_image, validator_params=step, resource=resource)` and must raise on failure:

```python
from typing import Any

from ome_zarr_converters_tools import (
    ConvertersToolsUserError,
    TiledImage,
    ValidatorModel,
    add_validator,
)


class MaxRegionsValidator(ValidatorModel):
    """Fail when an image would be assembled from too many tiles."""

    name: str = "max_regions"
    max_regions: int = 10_000


def apply_max_regions(
    tiled_image: TiledImage,
    validator_params: MaxRegionsValidator,
    resource: Any | None = None,
) -> None:
    n = len(tiled_image.regions)
    if n > validator_params.max_regions:
        raise ConvertersToolsUserError(
            f"Image '{tiled_image.path}' has {n} regions, more than the "
            f"{validator_params.max_regions} allowed. Filter the tiles or "
            "raise max_regions."
        )


add_validator(function=apply_max_regions, name="max_regions")
```

Pass it through `validators=[MaxRegionsValidator(max_regions=500)]` like the built-in one. `ValidatorModel`, `ShapeDtypeProbeValidator`, `apply_validator_pipeline` and `add_validator` are exported from the package root and from `ome_zarr_converters_tools.pipelines`; see the [pipelines API reference](../api/pipelines.md#validators).

## Registration steps

The registration pipeline transforms tile positions to prepare them for writing. It runs as a sequence of steps, each modifying the `TiledImage` in place. These steps convert raw microscope stage positions into clean, non-overlapping pixel coordinates suitable for writing into a Zarr array.

### Default registration pipeline

The default pipeline is built with `build_default_registration_pipeline()` and runs five steps in order:

```python
from ome_zarr_converters_tools import (
    AutoTiling,
    StagePositionCorrections,
    build_default_registration_pipeline,
)

pipeline = build_default_registration_pipeline(
    alignment_corrections=StagePositionCorrections(),
    tiling_strategy=AutoTiling(),
)
```

This creates:

1. **`offset_removal`** — Removes per-axis stage offsets according to `StagePositionCorrections` (see below). By default (`"Global"`) each axis is shifted so its minimum position is 0.
2. **`align_to_pixel_grid`** — Snaps start positions and lengths to integer pixel coordinates. After this step, all positions and sizes are exact pixel values (no sub-pixel offsets), including z/t which become integer plane/time-point indices.
3. **`xy_jitter_correction`** — When `remove_xy_jitter=True` (default), tiles within the same FOV that have slightly different XY positions (due to stage drift between Z slices or channels) are snapped to the FOV's reference position.
4. **`reindex_channels`** — When `reindex_channels=True` (default), the channel indices actually present are compacted to a dense `0, 1, 2, …` range and channel metadata is reconciled, so a filtered channel does not leave an empty channel in the output. See [channel reindexing](#channel-reindexing) for what that means across images.
5. **`tile_regions`** — Applies tiling/snapping to remove overlaps between FOVs (see [tiling strategies](#tiling-strategies) below). This is the step that determines the final non-overlapping layout.

### Channel reindexing

Compaction happens per `TiledImage`, so the resulting channel set — and the label each channel index maps to — depends on how tiles were grouped: with `split_per_fov=True` a FOV that acquired only `GFP` becomes a single-channel image whose index 0 is `GFP`, while a neighbouring FOV's index 0 may be `DAPI`. The same applies across the wells of a plate. Downstream tasks that address channels by label are unaffected; tasks that address them by position across several images are not.

Tiles spanning several channels (`length_c > 1`, e.g. a two-camera acquisition) are supported: their start index is remapped and their width preserved. Compaction can never split such a tile, because the channels it covers are consecutive and always all present.

When `reindex_channels=False`, the declared channel layout is preserved instead: the output always has one plane per entry in `channels`, and any channel no tile covers — whether it was never acquired or was removed by a `ChannelFilter` — is written as an empty plane keeping its label. Channel index `n` then means the same channel in every image, which is what to use when a downstream task addresses channels by position rather than by label. The empty planes cost no disk space under the default [chunking](../api/models.md#pyramids-and-chunking) — a chunk that is entirely fill value is never written — only a longer `c` axis and channel list for downstream tasks to iterate over.

### Stage position corrections

`StagePositionCorrections` controls the per-axis position handling applied during registration:

```python
from ome_zarr_converters_tools import StagePositionCorrections

corrections = StagePositionCorrections(
    remove_xy_offset="Global",  # "Global" (zero origin) or "Keep" (keep absolute)
    remove_z_offset="Global",  # "Global", "Per-FOV" (zero each FOV's z), or "Keep"
    remove_t_offset="Global",  # "Global" or "Keep"
    remove_xy_jitter=True,  # snap a FOV's sub-tiles to a shared XY origin
    reindex_channels=True,  # compact present channels to a dense range
)
```

Offset modes: `"Global"` translates the axis so its origin is 0; `"Keep"` keeps absolute positions (raising on negatives, left-padding on positives); `"Per-FOV"` (z only) zeros each FOV's z independently, useful when FOVs are focused independently. `remove_xy_jitter` corrects minor XY stage drift between the sub-acquisitions (Z slices/channels) of a single FOV.

### Custom registration steps

Register custom steps with `add_registration_func()`:

```python
from ome_zarr_converters_tools import TiledImage, add_registration_func


def my_custom_step(tiled_image: TiledImage, **kwargs) -> TiledImage:
    """Custom registration step that modifies tile positions."""
    for region in tiled_image.regions:
        # Modify region.roi as needed
        pass
    return tiled_image


add_registration_func(function=my_custom_step, name="my_step")
```

A step receives the `TiledImage` and returns it. Each `region.roi` is an ngio `Roi`, a frozen model, so derive a changed copy with `model_copy` rather than assigning fields. Its units depend on where the step sits: before `align_to_pixel_grid` the ROIs are in world units (micrometres), after it in pixels.

Then include it in a pipeline using `RegistrationStep`; the `params` of each step are forwarded to the function as keyword arguments:

```python
from ome_zarr_converters_tools import RegistrationStep, StagePositionCorrections

corrections = StagePositionCorrections()

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
| `MosaicGrouping(tiling_strategy=...)` | Aggregate all FOVs of an acquisition into a single mosaic OME-Zarr image, arranged by the nested `tiling_strategy` (see below). This is the default. |
| `PerFovGrouping()` | Write each FOV as its own OME-Zarr image. There is no tiling strategy — a single-FOV image has nothing to arrange. |

```python
from ome_zarr_converters_tools import (
    ConverterOptions,
    MosaicGrouping,
    PerFovGrouping,
    SnapToGridTiling,
)

# One mosaic image per acquisition (default), snapped to a grid
ConverterOptions(grouping=MosaicGrouping(tiling_strategy=SnapToGridTiling()))

# One OME-Zarr image per field of view
ConverterOptions(grouping=PerFovGrouping())
```

## Tiling strategies

Tiling controls how overlapping FOVs are arranged relative to each other within a mosaic. It lives on `MosaicGrouping.tiling_strategy` and is applied as the last step in the default registration pipeline. Each strategy is a Pydantic model, so tolerance and other parameters are passed directly on construction; `tolerance` is in pixels.

| Strategy | Description |
|----------|-------------|
| `AutoTiling(tolerance=1)` | Tries `SnapToGridTiling` first; falls back to `SnapToCornersTiling` if tiles don't form a regular grid |
| `SnapToGridTiling(tolerance=1)` | Snaps FOV positions to a regular grid, removing overlaps. Requires tiles to be arranged in a grid pattern |
| `SnapToCornersTiling()` | Snaps each FOV to the nearest corner, removing overlaps without requiring a grid structure |
| `InplaceTiling()` | No tiling — keeps original positions as-is |

**What snapping does to your pixels.** The snapping strategies move whole fields of
view: each FOV is translated to a point of a grid whose pitch is the tile size, so after
snapping neighbours abut exactly. Nothing is cropped or blended. The strip the microscope
imaged twice (a 10 % stage overlap, say) therefore appears twice in the mosaic, once at
the edge of each neighbour, and the mosaic is wider than the physical extent by the sum
of the overlaps. `InplaceTiling` keeps the stage positions instead, and where tiles
overlap the last written one wins. In both cases two tiles of one FOV that land on the
same `(z, c, t)` overwrite each other silently: that is a parser bug the library does not
flag, so enable `ShapeDtypeProbeValidator` during development and compare
`len(tiled_image.regions)` with what the parser produced.

```python
from ome_zarr_converters_tools import (
    AutoTiling,
    SnapToCornersTiling,
    SnapToGridTiling,
    StagePositionCorrections,
    build_default_registration_pipeline,
)

# For regular grid acquisitions (e.g. snake scan)
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

## Runtime settings

`RuntimeSettings` configures low-level runtime behaviour for a conversion call. It is applied as a scoped context manager, so settings are restored when the block exits.

```python
from ome_zarr_converters_tools import RuntimeSettings, ThreadScheduler

settings = RuntimeSettings(
    use_zarrs_codec=True,  # Use the zarrs Rust codec backend
    dask_scheduler=ThreadScheduler(num_workers=4),  # Run Dask with 4 threads
)

with settings.apply():
    # Conversion code here runs under the configured scheduler and codec
    ...
```

### Dask schedulers

| Scheduler | Description |
|-----------|-------------|
| `DefaultScheduler()` | Do not change Dask's scheduler (default) |
| `ThreadScheduler(num_workers=8)` | Use Dask's threaded scheduler |
| `ProcessScheduler(num_workers=8)` | Use Dask's multiprocessing scheduler |
| `SynchronousScheduler()` | Use Dask's synchronous scheduler (single-threaded, useful for debugging) |

### zarrs codec backend

Setting `use_zarrs_codec=True` switches Zarr to the `zarrs.ZarrsCodecPipeline` Rust backend, which can significantly speed up encoding/decoding. This requires the optional `zarrs` package:

```bash
pip install ome-zarr-converters-tools[zarrs]
```

### Temporary JSON storage

`RuntimeSettings` also controls how `TiledImage` objects travel from a Fractal init task to its compute tasks, via `temp_json_options`. By default they are passed in memory when small and written to a temporary JSON file alongside the Zarr output otherwise:

```python
from ome_zarr_converters_tools import RuntimeSettings, TempJsonOptions

settings = RuntimeSettings(
    temp_json_options=TempJsonOptions(temp_url="{zarr_dir}/_my_tmp"),
)
```

## Writer modes

Writer modes control how image data is loaded and written to the OME-Zarr store. The choice affects memory usage and performance.

| Mode | Description | Memory | Speed |
|------|-------------|--------|-------|
| `WriterMode.BY_TILE` | Loads and writes one tile at a time, sequentially | Low | Slow |
| `WriterMode.BY_FOV` | Loads and writes one FOV at a time, sequentially | Medium | Medium |
| `WriterMode.BY_FOV_DASK` | Loads FOVs lazily via Dask, streams each FOV a few write blocks at a time | Low–Medium | Fast |
| `WriterMode.BY_TILE_DASK` | Loads all tiles lazily via Dask, streams a few write blocks at a time | Low–Medium | Fast |
| `WriterMode.IN_MEMORY` | Loads entire image into memory, writes at once | High | Fastest |

The Dask modes write each region in parallel, a few storage chunks or shards per Dask block; ngio cuts the blocks on whole write units, so no two writers share one.

```python
from ome_zarr_converters_tools import (
    OverwriteMode,
    WriterMode,
    tiled_image_creation_pipeline,
)

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

### Choosing a writer mode

- **`BY_FOV`** (default recommendation): good balance of memory usage and performance. Loads one FOV at a time.
- **`BY_FOV_DASK`**: same as `BY_FOV` but uses Dask for lazy loading, which can be faster for large tiles.
- **`BY_TILE`**: lowest memory usage, useful when individual tiles are very large.
- **`IN_MEMORY`**: fastest for small datasets that fit entirely in memory.
- **`BY_TILE_DASK`**: builds the whole image lazily and streams it a few write blocks at a time, so memory does not grow with the image; each tile is loaded once. Good when Dask is already part of your workflow.

To size the modes against your data: a 2048 × 2048 `uint16` field of view with 20 Z
slices and 4 channels is 2048 × 2048 × 20 × 4 × 2 bytes, about 670 MB. `BY_FOV` holds one
of those at a time; `BY_TILE` one tile, here one Z slice of one channel at 8 MB;
`IN_MEMORY` every FOV of the image at once; the Dask modes a few write blocks.

## Overwrite modes

Overwrite modes control what happens when the target OME-Zarr image or plate already exists.

| Mode | Description |
|------|-------------|
| `OverwriteMode.NO_OVERWRITE` | Raises an error if the output already exists |
| `OverwriteMode.OVERWRITE` | Deletes the existing output and creates a new one |
| `OverwriteMode.EXTEND` | Adds new data to the existing output without deleting it |

```python
from ome_zarr_converters_tools import OverwriteMode

# Fail if output already exists (safe default for production)
overwrite_mode = OverwriteMode.NO_OVERWRITE

# Replace existing output (useful during development)
overwrite_mode = OverwriteMode.OVERWRITE

# Append to existing output
overwrite_mode = OverwriteMode.EXTEND
```

## Raising errors from a converter

Every error the package raises subclasses `ConvertersToolsError`, and the
message is written for the person who can act on it; the
[Models API reference](../api/models.md#errors) lists the classes. Custom
parsers, image loaders, filters, validators and registration steps should raise
the same classes, so that task users can tell a problem they can fix from a bug:

- `ConvertersToolsUserError` (a `ValueError`) for anything the task user or the
  converter author can fix: wrong parameters, inconsistent or unreadable input
  data, a missing dependency. Say what is wrong, name the image, the tile
  (`fov_name`) and the source file, give the expected and actual values, then
  say how to fix it.
- `ConvertersToolsInternalError` (a `RuntimeError`) for an invariant that a
  working converter can never violate. Write only the diagnostic sentence with
  the values in scope; the message automatically ends with where to report the
  bug.
- `ConvertersToolsWarning` (a `UserWarning`) with `warnings.warn(message,
  ConvertersToolsWarning, stacklevel=2)` for problems that do not stop the
  conversion, e.g. a file that is missing at pre-flight time.

Errors raised inside Pydantic validators stay plain `ValueError`: Pydantic wraps
them in a `ValidationError` and the class is lost anyway.

Override `ImageLoaderInterface.describe_source()` so that every error naming a
tile shows where its data comes from (`DefaultImageLoader` returns its file
path):

```python
from typing import Any

import numpy as np

from ome_zarr_converters_tools import ConvertersToolsUserError, ImageLoaderInterface


class MyLoader(ImageLoaderInterface):
    file_path: str
    series: int

    def describe_source(self, resource: Any | None = None) -> str:
        return f"{self.file_path} (series {self.series})"

    def load_data(self, resource: Any | None = None) -> np.ndarray:
        try:
            return read_my_format(self.file_path, series=self.series)
        except MyFormatError as e:
            raise ConvertersToolsUserError(
                f"Cannot read series {self.series} of '{self.file_path}': {e}. "
                "Check that the file is a complete export from the microscope."
            ) from e
```

## Next

- [Converters as Fractal tasks](fractal_tasks.md) — package the pipeline as an init/compute pair
- [Models API](../api/models.md) — every option and its default
- [Pipelines API](../api/pipelines.md) — filters, validators, registration and writing
