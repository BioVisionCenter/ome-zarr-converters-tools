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
4. **Flexible input**: parse tiles from CSV/DataFrame tables or construct them programmatically

### Architecture Diagram

![OME-Zarr Converters Tools Architecture](docs/ome-zarr-converter-tools.png)

## Getting Started

Install via pip:

```bash
pip install ome-zarr-converters-tools
```

## Documentation

For detailed documentation, tutorials, and API reference, visit the [official documentation](https://BioVisionCenter.github.io/ome-zarr-converters-tools/).

The documentation includes:

- A step-by-step [tutorial](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/tutorial/) covering both table-based and manual tile construction
- [Pipeline configuration](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/pipeline/) guide for filters, registration, tiling, and writer modes
- Full [API reference](https://BioVisionCenter.github.io/ome-zarr-converters-tools/stable/api/)
