---
description: Convert a table of microscopy tiles into an OME-Zarr HCS plate, step by step.
---

# HCS plates

**Turn a table of image files into an OME-Zarr high-content screening plate.**

The input to the library is a pandas `DataFrame` describing your tiles — one row per
image file on disk. You can build that frame from any source: a CSV, a database, a
custom parser for your microscope's metadata. This page loads it from a CSV for
convenience, then walks the full pipeline through to a written plate.

The example data is a small copy of the [cardiomyocyte differentiation
dataset](https://zenodo.org/records/8287221): one well (`A/01`), three fields of view,
two Z slices each, one channel.

## Step 1: Prepare the tile table

The frame must contain one row per tile. Columns fall into three groups.

**Tile position and size**

| Column | Description |
|--------|-------------|
| `file_path` | Path to the raw image file, relative to the `resource` directory or absolute |
| `fov_name` | Field-of-view identifier — tiles sharing a `fov_name` are stitched together |
| `start_x`, `start_y` | XY position, in the coordinate system declared on `AcquisitionDetails` |
| `start_z` | Z position: a slice index or a physical position |
| `start_c` | Channel index, 0-based |
| `start_t` | Time-point index |
| `length_x`, `length_y` | Tile dimensions, in µm or pixels |
| `length_z` | Number of Z slices in this tile (usually 1), or its physical size in µm |
| `length_c` | Number of channels in this tile (usually 1) |
| `length_t` | Number of time points in this tile (usually 1), or its physical size in s |

**HCS plate**

| Column | Description |
|--------|-------------|
| `row` | Well row — a letter (`A`, `B`, …) or a 1-based index |
| `column` | Well column, a 1-based integer |

**Extra columns** — any additional column is carried through onto the tile's
`attributes` and can be written out as a condition table. Here that is `drug`.

```python exec="true" source="material-block" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:load_table"
```

```python exec="true" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:table_helpers"
```

```python exec="true" html="1" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:show_table"
```

## Step 2: Describe the acquisition

`AcquisitionDetails` carries everything that is shared across tiles: pixel sizes, channel
metadata, axis order, and the coordinate systems the `start_*` and `length_*` columns are
expressed in.

!!! note "Coordinate systems"

    The `start_*_space` and `length_*_space` parameters say how to read the numbers in
    your table. Use `"world"` for physical units (micrometers) and `"pixel"` for
    indices. Below, `start_x`/`start_y` are micrometers while `start_z`/`start_t` are
    indices. Lengths default to pixels.

    Getting these wrong is quiet rather than loud: a `start_z` read as micrometers when
    it is an index collapses every slice onto the same plane.

```python exec="true" source="material-block" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:acquisition_details"
```

## Step 3: Parse tiles from the table

`hcs_images_from_dataframe()` turns each row into a `Tile`. `plate_name` and
`acquisition_id` are function arguments rather than columns, because they apply to the
whole table.

```python exec="true" source="material-block" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:parse_tiles"
```

Each `Tile` bundles:

- **Position and size** — `start_x`, `start_y`, `start_z`, … and `length_x`, `length_y`, …
- **Collection** (`ImageInPlate`) — where this tile lands in the plate hierarchy
- **Image loader** (`DefaultImageLoader`) — how to read the file from disk
- **Acquisition details** — the shared pixel sizes, channels and coordinate systems
- **Attributes** — the extra columns, here `drug: ["DMSO"]`

Drawing the tiles in stage coordinates shows what the pipeline is actually given: three
overlapping rectangles that have to be resolved into one image. The two Z slices of each
FOV sit at the same XY position, so they share a rectangle (dashed).

```python exec="true" html="1" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:plot_layout"
```

## Step 4: Aggregate tiles into TiledImages

`tiles_aggregation_pipeline()` groups tiles that belong to the same output image and
produces `TiledImage` objects.

`resource` is the base directory used to resolve relative `file_path` values. It must be
an absolute path or an `s3://` URL. If your paths are already absolute you can omit it.

```python exec="true" source="material-block" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:aggregate"
```

The path (`CardiomyocytePlate.zarr/A/01/0`) is where this image lands inside the store:
plate name, well row and column, acquisition index. All six tiles — three FOVs times two
Z slices — grouped into a single `TiledImage`, because they belong to the same well and
acquisition.

## Step 5: Create the plate structure

Plate metadata, wells and acquisitions have to exist before individual images are
written into them. `setup_ome_zarr_collection()` creates that skeleton.

```python exec="true" source="material-block" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:setup_plate"
```

## Step 6: Register and write

The registration pipeline resolves tile positions — snapping to the pixel grid, removing
overlaps — and `tiled_image_creation_pipeline()` writes each `TiledImage` into the store.
The `zarr_url` for each image is `zarr_dir` joined with `tiled_image.path`.

See [pipeline configuration](../guides/pipeline.md) for the registration steps, tiling
strategies and writer modes available here.

```python exec="true" source="material-block" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:write"
```

## Step 7: Verify the result

`tiled_image_creation_pipeline()` returns an `OmeZarrContainer`, so the written data can
be inspected with [ngio](https://biovisioncenter.github.io/ngio/) directly.

```python exec="true" source="material-block" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:verify"
```

```python exec="true" html="1" session="hcs"
--8<-- "docs/snippets/getting_started/hcs_plates.py:plot_result"
```

The three fields of view are now one image. The empty quadrant is real rather than a
rendering artefact: three FOVs do not cover the bounding box they span, and everything
outside them is written as background.

## Next

- [Single images](1_single_images.md) — the same pipeline without a plate hierarchy
- [Programmatic tiles](2_programmatic_tiles.md) — build `Tile` objects without a table
- [Pipeline configuration](../guides/pipeline.md) — filters, registration, writer modes
