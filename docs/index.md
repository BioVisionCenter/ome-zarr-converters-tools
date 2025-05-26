# Welcome to Fractal Converters Tools

Fractal Converters Tools is a Python package that provides tooling for building OME-Zarr converters for the Fractal platform.

It includes three main components:

1. Abstraction layer for mapping the on-disk raw data to Image objects
2. Common tooling to build converters as Fractal Compound Tasks

## Main Concepts

In general a single microscopy image is not acquired in a single big array in a single file, but rather in multiple smaller tiles. How atomic these tiles are depends on the specific microscope and the acquisition settings.

To make building converters easier, Fractal Converters Tools provides an abstraction layer that allows you to map these on-disk raw data to an Image object which we call `Tile`.

Moreover, usually a single microscopy image is not composed of a single tile, but rather multiple tiles that are stitched together to form a complete image. We call these objects `TiledImage`.

```mermaid
flowchart LR
    subgraph A[Metadata Parsing]
    A100[img_B3_fov1_c0_z0.tif] --> B1[Tile1]
    A101[img_B3_fov1_c0_z1.tif] --> B1
    A200[img_B3_fov2_c0_z0.tif] --> B2[Tile2]
    A201[img_B3_fov2_c0_z1.tif] --> B2
    A20x[img_...] --> B3[Tile...]

    B1 --> C1[TiledImage1]
    B2 --> C1

    B3 --> C2[TiledImage2]
    end

    C1 --> D[Init - Task]
    C2 -->|"Many..."| D
    D --> E[Compute Tile1]
    D --> E1[Compute Tile2]
    D -->|Many...| E2[Compute ...]
```

Additional Fractal Converters Tools supports high-content screening HCS applications. In the context of HCS it is common to have multiple images that are related to each other in a single plate collection. Plates are standardized in OME-Zarr and Fractal Converters Tools provides the necessary tools to correctly place the images in a plate collection.

## Installation

To get started with Fractal Converters Tools, you can install it via pip:

```bash
pip install fractal-converters-tools
```
