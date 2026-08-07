---
description: Build Tile objects and a custom image loader directly, without going through a DataFrame.
---

# Programmatic tiles

**Build `Tile` objects in code, with a loader that reads your own file format.**

The DataFrame entry points are a convenience, not the interface. Once you have a list of
`Tile` objects the rest of the pipeline is the same, so building them by hand is the way
in when:

- your format needs a **custom loader** — a proprietary microscope file, a remote API
- you need **fine-grained control** over how tiles are constructed
- your data source does not map cleanly onto a table

## Step 1: Implement a custom image loader

A loader extends `ImageLoaderInterface` (a Pydantic model) and implements `load_data()`.
The method receives the optional `resource` passed to the pipeline — typically a base
directory — and returns a NumPy array.

```python exec="true" source="material-block" session="programmatic"
--8<-- "docs/snippets/getting_started/programmatic_tiles.py:loader"
```

```python exec="true" session="programmatic"
--8<-- "docs/snippets/getting_started/programmatic_tiles.py:plot_helpers"
```

## Step 2: Build the tiles

Each `Tile` needs a position and size, a `collection` describing where the output goes
(`SingleImage` or `ImageInPlate`), an `image_loader`, and the shared
`acquisition_details`.

!!! warning "`start_z` is read in world coordinates by default"

    `AcquisitionDetails.start_z_space` defaults to `"world"`, so a `start_z` meant as a
    slice index is interpreted as micrometers. At `z_spacing=5.0` that rounds `start_z=1`
    back to slice 0, and the second tile silently overwrites the first. Declare
    `start_z_space="pixel"` when your positions are indices.

```python exec="true" source="material-block" session="programmatic"
--8<-- "docs/snippets/getting_started/programmatic_tiles.py:build_tiles"
```

## Step 3: Aggregate, register and write

From here nothing is specific to the programmatic path — this is the same sequence as
[step 4 onwards of the plate tutorial](0_hcs_plates.md#step-4-aggregate-tiles-into-tiledimages).

```python exec="true" source="material-block" session="programmatic"
--8<-- "docs/snippets/getting_started/programmatic_tiles.py:write"
```

## Step 4: Verify the result

```python exec="true" source="material-block" session="programmatic"
--8<-- "docs/snippets/getting_started/programmatic_tiles.py:verify"
```

```python exec="true" html="1" session="programmatic"
--8<-- "docs/snippets/getting_started/programmatic_tiles.py:plot_result"
```

## Next

- [Pipeline configuration](../guides/pipeline.md) — filters, registration, writer modes
- [Converters as Fractal tasks](../guides/fractal_tasks.md) — package this as a task
- [Core API](../api/core.md) — the `Tile` and `TiledImage` reference
