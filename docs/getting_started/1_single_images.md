---
description: Convert microscopy tiles into standalone OME-Zarr images, without a plate hierarchy.
---

# Single images

**The same pipeline as [HCS plates](0_hcs_plates.md), without the plate hierarchy.**

Use this when your data does not follow a plate layout — individual tissue scans, brain
slices, or anything where wells and acquisitions do not apply. The input is still a
pandas `DataFrame`, one row per image file, and the pipeline after parsing is identical.

The example reuses images from the [cardiomyocyte differentiation
dataset](https://zenodo.org/records/8287221): one image, two fields of view, two Z slices
each, one channel. The set-up (clone, data location, extra packages) is described at the
top of the [plate tutorial](0_hcs_plates.md).

## Step 1: Prepare the tile table

The position and size columns are the same as in the [plate
tutorial](0_hcs_plates.md#step-1-prepare-the-tile-table). The `row` and `column` plate
columns are replaced by one:

| Column | Description |
|--------|-------------|
| `image_path` | Name of the output dataset — `"brain_scan"` becomes `brain_scan.zarr` |

```python exec="true" source="material-block" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:load_table"
```

```python exec="true" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:table_helpers"
```

```python exec="true" html="1" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:show_table"
```

## Step 2: Describe the acquisition

Unchanged from the plate case — `AcquisitionDetails` does not know about plates.

```python exec="true" source="material-block" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:acquisition_details"
```

## Step 3: Parse tiles from the table

`single_images_from_dataframe()` replaces `hcs_images_from_dataframe()`. The tiles it
produces carry a `SingleImage` collection instead of an `ImageInPlate` one; that
collection is the only difference between the two paths.

```python exec="true" source="material-block" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:parse_tiles"
```

The `image_path` value (`cardiomyocyte_scan`) becomes the output store name,
`cardiomyocyte_scan.zarr`.

```python exec="true" html="1" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:plot_layout"
```

## Step 4: Aggregate tiles into TiledImages

Identical to the plate case.

```python exec="true" source="material-block" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:aggregate"
```

## Step 5: Register and write

There is no plate structure to create, so this goes straight from the registration
pipeline to writing. Everything else matches [step 6 of the plate
tutorial](0_hcs_plates.md#step-6-register-and-write).

```python exec="true" source="material-block" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:write"
```

## Step 6: Verify the result

```python exec="true" source="material-block" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:verify"
```

Two tables are written alongside the image: `FOV_ROI_table`, one region of interest per
field of view, and `well_ROI_table`, covering the whole image.

```python exec="true" html="1" session="single"
--8<-- "docs/snippets/getting_started/single_images.py:plot_result"
```

## Next

- [Custom file formats](2_programmatic_tiles.md) — build `Tile` objects without a table,
  with your own image loader
- [Pipeline configuration](../guides/pipeline.md) — filters, registration, writer modes
