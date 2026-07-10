# ome-zarr-converters-tools

[![License](https://img.shields.io/pypi/l/ome-zarr-converters-tools.svg?color=green)](https://github.com/BioVisionCenter/ome-zarr-converters-tools/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ome-zarr-converters-tools.svg?color=green)](https://pypi.org/project/ome-zarr-converters-tools)
[![Python Version](https://img.shields.io/pypi/pyversions/ome-zarr-converters-tools.svg?color=green)](https://python.org)
[![CI](https://github.com/BioVisionCenter/ome-zarr-converters-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/BioVisionCenter/ome-zarr-converters-tools/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BioVisionCenter/ome-zarr-converters-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/BioVisionCenter/ome-zarr-converters-tools)

A Python library that provides shared utilities for building OME-Zarr image converters. It handles tile management, image registration, filtering, validation, and writing OME-Zarr datasets.

## Features

1. **Abstraction layer** for building OME-Zarr images and HCS plates from microscope metadata and image data
2. **Customizable pipeline** for filtering, validating, registering, and tiling images
3. **Python API** for building custom converters, with optional [Fractal](https://fractal-analytics-platform.github.io/fractal-server/) integration for parallel processing
4. **Flexible input**: parse tiles from DataFrames or construct them programmatically

### Architecture Diagram

![OME-Zarr Converters Tools Architecture](docs/ome-zarr-converter-tools.png)

## Getting Started

Install via pip:

```bash
pip install ome-zarr-converters-tools
```

For converting data stored on S3 (`s3://` URLs), install the `s3` extra:

```bash
pip install "ome-zarr-converters-tools[s3]"
```

### Quickstart

Convert a tiles table (one row per tile: stage position, size, and image file)
into a single OME-Zarr image:

```python
import pandas as pd

from ome_zarr_converters_tools import (
    AcquisitionDetails,
    AutoTiling,
    ChannelInfo,
    ConverterOptions,
    OverwriteMode,
    StagePositionCorrections,
    WriterMode,
    build_default_registration_pipeline,
    single_images_from_dataframe,
    tiled_image_creation_pipeline,
    tiles_aggregation_pipeline,
)

acq = AcquisitionDetails(
    channels=[ChannelInfo(channel_label="DAPI")],
    xy_pixel_size=0.65,  # micrometers
    z_spacing=1.0,
)
tiles = single_images_from_dataframe(
    tiles_table=pd.read_csv("tiles.csv"), acquisition_details=acq
)
options = ConverterOptions()
tiled_image = tiles_aggregation_pipeline(
    tiles=tiles, converter_options=options, resource="/path/to/image/files"
)[0]
tiled_image_creation_pipeline(
    zarr_url="/path/to/output.zarr",
    tiled_image=tiled_image,
    registration_pipeline=build_default_registration_pipeline(
        StagePositionCorrections(), AutoTiling()
    ),
    converter_options=options,
    writer_mode=WriterMode.BY_FOV,
    overwrite_mode=OverwriteMode.NO_OVERWRITE,
    resource="/path/to/image/files",
)
```

See the [tutorials](https://BioVisionCenter.github.io/ome-zarr-converters-tools/) for HCS plates, custom image loaders, and Fractal tasks.

## Documentation

For detailed documentation, tutorials, and API reference, visit the [official documentation](https://BioVisionCenter.github.io/ome-zarr-converters-tools/).

The documentation includes:

- [HCS Plate Tutorial](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/hcs_tutorial/) -- converting plate-based microscopy data
- [Single Images Tutorial](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/images_tutorial/) -- converting standalone images
- [Advanced Tutorial](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/advanced_tutorial/) -- programmatic tile construction with custom loaders
- [Fractal Tasks Guide](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/converters_as_fractal_tasks/) -- building parallel converters with the Fractal platform
- [Pipeline Configuration](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/pipeline/) -- filters, registration, tiling, and writer modes
- [API Reference](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/api/)
