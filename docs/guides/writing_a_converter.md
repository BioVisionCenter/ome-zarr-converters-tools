---
description: What a converter built on this library consists of, in what order to build it, and how to run, test and package it.
---

# Writing a converter

**From a microscope's metadata to a tested, packaged converter, in seven steps.**

The tutorials walk one dataset through the pipeline. This page is the map for building
your own converter for a format the [existing converters](converters.md) do not cover:
what you write, what the library provides, and the contracts between the two.

## When to use this

Read this first if you have a parser for your microscope's metadata, or can write one,
and want a converter that other people can run, as plain Python or as a Fractal task.
The tutorials remain the place to see the pipeline run; this page links to them at each
step.

## Step 1: What a converter is

A converter built on this library is a few things you write around a pipeline the library
owns:

| You write | The library owns |
|---|---|
| A **parser** that reads the metadata and returns `AcquisitionDetails` plus one `Tile` per unit of raw data | Aggregation into `TiledImage` objects, filters and validators |
| An **image loader**: `DefaultImageLoader` for one image per file, or your own `ImageLoaderInterface` subclass | Registration: stage orientation, snapping, tiling |
| An **options model** for what the task user may change (optional) | Collection setup: plate skeleton, ROI tables, condition table |
| An **init task** and a **compute task** wrapping the above (Fractal only) | Writing: pyramids, chunking, OME-Zarr metadata |

The parser has no interface of its own. Its shape is up to you; the one the existing
converters use is

```python
def parse_my_format(path: str) -> tuple[AcquisitionDetails, list[Tile]]: ...
```

From your tiles the pipeline derives three objects, each adding one thing:

- `Tile`: one array on disk and where it belongs, in your coordinates.
- `TileSlice`: the same region as an ngio `Roi` in world units, with the stage
  orientation applied.
- `TileFOVGroup`: every slice of one field of view (FOV), i.e. its Z slices, channels
  and time points.
- `TiledImage`: every FOV group that becomes one output image, with its output path,
  axes, data type and channels.

Which tiles share a `TiledImage` is decided by `collection.path()`: for `ImageInPlate`
that is plate, row, column and acquisition; for `SingleImage` it is the image path.
`fov_name` groups tiles *within* that image, so FOV names only have to be unique within
one well and acquisition.

## Step 2: Map the metadata onto `AcquisitionDetails` and `Tile`

| Your metadata says | Goes to | Notes |
|---|---|---|
| Pixel size | `AcquisitionDetails.xy_pixel_size` | µm |
| Z step | `AcquisitionDetails.z_spacing` | µm |
| Time interval | `AcquisitionDetails.t_spacing` | seconds |
| Channel names, wavelengths, colours | `AcquisitionDetails.channels`, one `ChannelInfo` per channel index | `color` is a hex string; omitted colours are chosen from the wavelength or the label |
| Which of t, c, z the data has | `AcquisitionDetails.axes` | subset of `t, c, z, y, x` in that order; `c` is required when `channels` is set |
| Units of positions and sizes | `AcquisitionDetails.start_*_space`, `length_*_space` | defaults: positions in µm, sizes in pixels; set `start_z_space="pixel"` when Z positions are slice indices |
| Stage position of a field of view | `Tile.start_x`, `Tile.start_y` | the **top-left corner**, see below |
| Z slice, channel and time point of a file | `Tile.start_z`, `Tile.start_c`, `Tile.start_t` | `start_c` indexes into `channels` |
| Image size; slices, channels, time points per file | `Tile.length_x` … `Tile.length_t` | pixels and counts by default |
| Well and multiplexing cycle | `ImageInPlate(plate_name=…, row=…, column=…, acquisition=…)` as `Tile.collection` | `acquisition` is the cycle index inside the well, `0` when there is one |
| Field of view | `Tile.fov_name` | unique within one well and acquisition |
| Plate-map metadata (compound, concentration, …) | `Tile.attributes`, values as lists | see [attributes and condition tables](pipeline.md#attributes-condition-tables-and-the-image-list) |
| Pixel data type, when known | `AcquisitionDetails.data_type` | saves reading the first tile of every image |

!!! warning "Positions are the top-left corner"

    `start_x` and `start_y` are the corner of the tile with the smallest X and Y, in the
    same frame as the image axes: the row index grows with Y. If your metadata reports
    the centre of each field of view, subtract `length / 2` in the parser. A centre
    value shifts every FOV by half a tile, and snapping to a grid still "works", so the
    error goes unnoticed. Use `StageOrientation` when the finished mosaic comes out
    mirrored or transposed.

## Step 3: Map your acquisition onto tiles

The tutorials show one channel, one Z slice per file, no time. Real formats come in a
handful of shapes; each row says what the tiles carry and what the loader must return.
The rule behind the last column: `load_data()` returns an array laid out along
`AcquisitionDetails.axes` (trailing `y, x`), and may drop *leading* singleton axes only.
A mismatch fails at write time, or earlier with `ShapeDtypeProbeValidator`.

| Acquisition | Tiles | `axes` | `load_data()` returns |
|---|---|---|---|
| One file per channel and Z slice | one `Tile` per file: `start_c=i`, `length_c=1`, `start_z=k`, `length_z=1`, with `start_z_space="pixel"` | `["c", "z", "y", "x"]` | `(y, x)` |
| One multi-page TIFF per Z stack | one `Tile` per file: `start_z=0`, `length_z=n` | `["z", "y", "x"]` or `["c", "z", "y", "x"]` | `(z, y, x)` |
| Two-camera file holding two channels | one `Tile` per file: `start_c=i`, `length_c=2`; the two channels **adjacent** in `channels` | must contain `c` | `(c, y, x)` only for `axes=["c", "y", "x"]`; otherwise the full layout, e.g. `(c, 1, y, x)` for `["c", "z", "y", "x"]`, or the channels silently become Z slices |
| Time series, one file per time point | one `Tile` per file: `start_t=j` with `start_t_space="pixel"`, `length_t=1` | must contain `t` | `(y, x)` |
| Several wells and multiplexing cycles | from a table: one `hcs_images_from_dataframe(…, acquisition_id=cycle)` call per cycle, lists concatenated; by hand: set `acquisition` on each `ImageInPlate` | any | as above |

Keep `start_z_space` at its default `"world"` only when the parser has physical Z
positions in µm (a motorised stage reading, say); slice indices need `"pixel"`.

## Step 4: Choose or write an image loader

`DefaultImageLoader(file_path=…)` reads one file (TIFF, PNG/JPEG/BMP, NPY) relative to
the `resource` directory and returns the array as stored, so a multi-page TIFF comes
back with its pages as the leading axis. If every raw file is one 2D image, or one Z
stack, it is all you need.

Write your own loader when a file holds more than that, needs decoding, or lives behind
an API. A loader is a Pydantic model whose fields are what the compute task needs to find
the data again, so keep them JSON-serialisable (paths, indices, names): `TiledImage`
objects travel from the init task to the compute tasks as JSON. One loader class per
converter: `generic_compute_task()` takes a single `image_loader_type`.

The surface, with when the pipeline calls each method:

- `load_data(resource)`, required: the array, following the layout rule above. Called
  once per tile slice at write time.
- `find_data_type(resource)`: the dtype as a string. Called once per image during
  aggregation unless `AcquisitionDetails.data_type` is set; the default loads the whole
  tile, so override it to read a header instead.
- `preflight(resource)`: warn, without loading, when the source is missing. Called by
  `ShapeDtypeProbeValidator` before it probes the data.
- `describe_source(resource)`: where this loader reads from, for error messages.

[Custom file formats](../getting_started/2_programmatic_tiles.md) walks through a
complete loader; the reference for the contract is `ImageLoaderInterface.load_data` on
the [Models API page](../api/models.md#image-loaders).

## Step 5: Run it locally

Even before Fractal is involved, the existing converters structure the conversion as an
init task that builds the `TiledImage` objects and a compute task that writes one of
them, and expose a Python entry point that runs the pair with `exec_compound_task()`:

```python
from ome_zarr_converters_tools import (
    ConverterOptions,
    ImageListUpdateDict,
    OverwriteMode,
    RunnerType,
    exec_compound_task,
)


def convert_my_format(
    *,
    zarr_dir: str,
    acquisitions: list[MyAcquisition],
    converter_options: ConverterOptions | None = None,
    overwrite: OverwriteMode = OverwriteMode.NO_OVERWRITE,
    runner: RunnerType | None = None,
) -> list[ImageListUpdateDict]:
    return exec_compound_task(
        init_task_fn=my_init_task,
        compute_task_fn=my_compute_task,
        init_task_kwargs={
            "zarr_dir": zarr_dir,
            "acquisitions": acquisitions,
            "converter_options": converter_options or ConverterOptions(),
            "overwrite": overwrite,
        },
        runner=runner,
    )
```

The init task returns `{"parallelization_list": [...]}` (what
`setup_images_for_conversion()` builds), and the compute task is called once per entry.
`runner=None` or `SequentialRunner()` runs everything in one process, which is the
easiest to debug; `ThreadedRunner(num_threads=4)` converts images in parallel threads;
`MultiprocessingRunner(num_processes=4)` uses processes, so the compute task must be a
picklable module-level function. See [converters as Fractal tasks](fractal_tasks.md)
for the two task functions themselves.

Without tasks at all, the sequence in the tutorials (aggregate, set up the collection,
`tiled_image_creation_pipeline()` per image) is the same pipeline called directly.

## Step 6: Test with snapshots

The testing helpers fingerprint a converted store (array hashes, ROI and condition
tables, image-list updates) into a JSON snapshot that lives in the repository. One test
per small dataset, and the suite tells you when any conversion result changes.

Layout:

```text
tests/
  conftest.py
  data/my_format/
    raw/small/              # a tiny acquisition: a few small tiles
    snapshots/small.json    # committed
    output/                 # written by --update-snapshots; add to .gitignore
  test_my_format.py
```

`conftest.py` loads the plugin, which adds the `--update-snapshots` and `--extended`
options, the `extended` marker and the `update_snapshots` fixture:

```python
pytest_plugins = ["ome_zarr_converters_tools.testing.plugin"]
```

The test calls the entry point from step 5; `run_converter_test()` calls it as
`api_fn(zarr_dir=str(path), **api_kwargs)`, adds `converter_options=` when one is
given, and expects the list of image-list updates back:

```python
from pathlib import Path

from ome_zarr_converters_tools.testing import run_converter_test

from my_converter import convert_my_format

DATA_DIR = Path(__file__).parent / "data" / "my_format"


def test_small(tmp_path: Path, update_snapshots: bool):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_my_format,
        api_kwargs={"acquisitions": [{"path": str(DATA_DIR / "raw" / "small")}]},
        snapshot_path=DATA_DIR / "snapshots" / "small.json",
        update_snapshots=update_snapshots,
    )
```

Run `pytest --update-snapshots` once to write `small.json` (the store goes to
`data/my_format/output/`, with `overwrite=OverwriteMode.OVERWRITE` injected), review the
JSON, commit it. A plain `pytest` then converts into `tmp_path` and compares; a mismatch
raises one `AssertionError` listing every difference. Mark tests on large datasets with
`@pytest.mark.extended`; they are skipped unless `--extended` is passed. Pass
`output_type="single_image"` for converters that write stand-alone images.

For unit tests that need tiles without raw data, `build_dummy_tile()` makes a tile whose
loader draws its FOV name into a `uint8` array with a bright border, so a wrong
placement is visible in the output:

```python
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    SingleImage,
    StartPosition,
    TileShape,
    build_dummy_tile,
)

acq = AcquisitionDetails(start_x_space="pixel", start_y_space="pixel")
tiles = [
    build_dummy_tile(
        fov_name=f"FOV_{i}",
        start=StartPosition(x=256 * i, y=0),
        shape=TileShape(x=256, y=256),
        collection=SingleImage(image_path="dummy"),
        acquisition_details=acq,
    )
    for i in range(3)
]
```

## Step 7: Package as a Fractal task

Wrap the init and compute functions as Fractal tasks, generate the manifest, and
publish the package; see [packaging](fractal_tasks.md#packaging) in the Fractal guide.
The converters listed under [converters built on this library](converters.md) are
complete examples of this layout.

## Next

- [Converters as Fractal tasks](fractal_tasks.md): the init and compute tasks, running
  them locally, packaging
- [Pipeline configuration](pipeline.md): filters, validators, registration, tiling,
  writer modes
- [Testing API reference](../api/testing.md): the snapshot and assertion models
