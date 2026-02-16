# Welcome to OME-Zarr Converters Tools

OME-Zarr Converters Tools is a Python library that provides shared utilities for building OME-Zarr image converters. It handles tile management, image registration, filtering, validation, and writing OME-Zarr datasets, with optional [Fractal](https://fractal-analytics-platform.github.io/fractal-server/) integration for parallel processing.

## Features

1. **Abstraction layer** for building OME-Zarr images and HCS plates from microscope metadata and image data
2. **Customizable pipeline** for filtering, validating, registering, and tiling images
3. **Python API** for building custom converters, with optional Fractal integration for parallel processing
4. **Flexible input**: parse tiles from DataFrame tables or construct them programmatically

### Architecture Diagram

![OME-Zarr Converters Tools Architecture](ome-zarr-converter-tools.png)

## Main Concepts

In general a single microscopy image is not acquired as a single big array in a single file, but rather as multiple smaller tiles. How atomic these tiles are depends on the specific microscope and the acquisition settings.

To make building converters easier, OME-Zarr Converters Tools provides an abstraction layer that allows you to map these on-disk raw data to an image object which we call a `Tile`.

Usually a single microscopy image is not composed of a single tile, but rather multiple tiles that are stitched together to form a complete image. We call these composite objects `TiledImage`.

![OME-Zarr Converters Tools Architecture](ome-zarr-converter-tools.png)

## Collection Types

OME-Zarr Converters Tools is particularly designed to convter complex microscopy acquisitions. The library provides two main collection types to handle different acquisition structures:

- **HCS Plates**: for high-content screening applications where multiple images are organized in a multi-well plate layout. Each image is placed in a specific well (row/column) of the plate, following the OME-Zarr HCS specification.
- **Single Images**: for standalone OME-Zarr images conversions without plate structure.

See the [Tutorial](tutorial.ipynb) for a hands-on walkthrough on how to use the library to build a converter for both collection types. 

## Pipeline Overview

The typical conversion pipeline follows these steps:

1. **Parse metadata** into `Tile` objects, this step maps the raw images (e.g. TIFF files) to `Tile` objects with associated metadata (e.g. position, channel, timepoint)
2. **Filter** tiles can be processed using custom filters to exclude unwanted tiles (e.g. exclude failed acquisitions, filter certain channels, etc.)
3. **Aggregate** tiles `Tiles` into `TiledImage` objects. `Tiles` define partial slices of the final image, so we need to aggregate them into `TiledImage` objects that represent the complete image. This step also defines the final layout of the OME-Zarr dataset (e.g. how channels, timepoints, and z-slices are organized)
4. **Register** `TiledImage` objects to correct for any misalignment between tiles (e.g. due to inacurate stage positions, inhomogeneous z-step, etc.). In this ste we can also tile fields of view into a single mosaic image if needed.
5. **Setup Collection**: define the collection type (HCS plate or single image) and set up the OME-Zarr metadata accordingly
6. **Write** OME-Zarr images to disk.

To know more about the different [Pipeline Configuration](pipeline.md) options, see the page for details on filters, registration steps, tiling modes, and writer modes.

## Extensibility

The library is designed to be extended:

- **Custom image loaders**: implement `ImageLoaderInterface` to load any image format (see [Tutorial](tutorial.ipynb#step-2-manual-tile-construction-advanced))
- **Custom pipeline steps**: add [registration](pipeline.md#custom-registration-steps), [filtering](pipeline.md#custom-filters), or validation steps
- **Custom collection types**: register new collection handlers via `add_collection_handler()`

## Installation

Install via pip:

```bash
pip install ome-zarr-converters-tools
```
