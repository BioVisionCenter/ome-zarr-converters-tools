# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ome-zarr-converters-tools** is a Python library providing shared utilities for building OME-Zarr image converters. It handles tile management, image registration, filtering, validation, and writing OME-Zarr datasets, with optional Fractal platform integration for parallel processing.

## Development Environment

Uses **pixi** as the package manager. Always use `pixi run -e dev` instead of bare commands:

```bash
pixi run -e dev test              # Run full test suite (pytest with coverage)
pixi run -e dev ruff              # Format code
pixi run -e dev ruff-fix-imports  # Fix import sorting
pixi run -e dev chores            # Run all pre-commit hooks
```

Run a single test file or test:
```bash
pixi run -e dev pytest tests/unit/test_core.py
pixi run -e dev pytest tests/unit/test_core.py::test_name -v
```

Type checking:
```bash
pixi run -e dev mypy src/
```

## Code Style

- **Ruff** for linting and formatting (line length: 88)
- **Google-style docstrings**
- Strict **mypy** type checking
- All internal modules are prefixed with `_` (private)

## Architecture

### Core Pipeline Flow

Tiles → Registration → Filtering → Validation → Aggregation → OME-Zarr Writing

### Key Abstractions

- **`Tile`** (`core/_tile.py`): Fundamental unit — a region of an image with position, size, loader, acquisition metadata, and collection model.
- **`TiledImage` / `TileFOVGroup` / `TileSlice`** (`core/_tile_region.py`): Group tiles into complete images by FOV, handling stitching and multi-dimensional data.
- **`CollectionInterface`** (`models/_collection.py`): Defines how to build paths to source images (`ImageInPlate`, `SingleImage`).
- **`ImageLoaderInterfaceType`** (`models/_loader.py`): Abstract interface for loading image data from various formats.
- **`AcquisitionDetails` / `ChannelInfo`** (`models/_acquisition.py`): Acquisition metadata (channels, wavelengths, etc.).
- **`ConverterOptions`** (`models/_converter_options.py`): Configuration for the conversion process.

### Module Responsibilities

- **`core/`**: Tile/TiledImage data models, lazy loading, ROI utilities, tile-to-image aggregation, HCS table helpers, and plotting.
- **`models/`**: Configuration and interfaces — acquisition metadata, collection types, converter options, image loader protocol.
- **`pipelines/`**: All orchestration — registration (alignment, tiling, snap-to-grid), filtering, validation, collection setup, OME-Zarr writing, and image creation/aggregation pipelines.
- **`fractal/`**: Fractal platform integration — `setup_images_for_conversion()` (init task), `generic_compute_task()` (compute task factory), and JSON serialization for inter-task communication.

### Key Dependencies

- **ngio** (>=0.5.3,<0.6.0): OME-Zarr I/O — tightly coupled
- **Dask**: Lazy/parallel image loading and writing
- **Pydantic**: All models use Pydantic for validation

## Testing

- Tests live in `tests/unit/` and `tests/integration/`
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`
- Python support: 3.11–3.14
- CI runs matrix of Python versions on Ubuntu + macOS
