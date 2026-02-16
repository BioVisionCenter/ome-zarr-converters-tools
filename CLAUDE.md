# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ome-zarr-converters-tools** is a Python library providing shared utilities for building OME-Zarr image converters. It handles tile management, image registration, filtering, validation, and writing OME-Zarr datasets, with optional Fractal platform integration for parallel processing.

## Development Environment

Uses **pixi** as the package manager. Always use `pixi run -e dev` instead of bare commands:

```bash
pixi run -e dev pytest tests/                          # Full test suite (with coverage)
pixi run -e dev pytest tests/unit/test_core.py -v      # Single test file
pixi run -e dev pytest tests/unit/test_core.py::TestClass::test_method -v  # Single test
pixi run -e dev pre-commit run --all-files             # All hooks (typos, ruff, ruff-format, nbstripout)
pixi run -e dev mypy src/                              # Type checking
```

**Important**: Never use bare `python`, `pytest`, `ruff`, or `mypy` — always prefix with `pixi run -e dev`.

## Code Style

- **Ruff** for linting and formatting (line length: 88, target: py311)
- **Google-style docstrings** (enforced via `D` rules, disabled for tests)
- Strict **mypy** type checking
- All internal modules are prefixed with `_` (private)
- **typos** for spell checking — false positives go in `_typos.toml` (`extend-identifiers` for tokens like FOV, `extend-words` for substrings)
- **Pydantic v2**: `@field_validator` must come before `@classmethod` (decorator order matters)

## Architecture

### Core Pipeline Flow

```
CSV/DataFrame → Tile building → Aggregation → Registration → Filtering → Validation → OME-Zarr Writing
```

### Source Layout

```
src/ome_zarr_converters_tools/
├── core/           # Data models and tile operations
│   ├── _tile.py              # Tile model (position, size, loader, metadata)
│   ├── _tile_region.py       # TiledImage, TileFOVGroup, TileSlice
│   ├── _tile_to_tiled_images.py  # tiled_image_from_tiles()
│   ├── _table.py             # hcs_images_from_dataframe(), single_images_from_dataframe()
│   ├── _roi_utils.py         # ROI slice manipulation
│   ├── _dask_lazy_loader.py  # Dask-based lazy image loading
│   └── _dummy_tiles.py       # DummyLoader + build_dummy_tile() for testing
├── models/         # Pydantic configuration and interfaces
│   ├── _acquisition.py       # AcquisitionDetails, ChannelInfo
│   ├── _collection.py        # CollectionInterface, ImageInPlate, SingleImage
│   ├── _converter_options.py # ConverterOptions, AlignmentCorrections, TilingMode, WriterMode, OverwriteMode
│   ├── _loader.py            # ImageLoaderInterface, DefaultImageLoader (PNG/TIFF/NPY)
│   └── _url_utils.py         # URL type detection (local/S3), path joining
├── pipelines/      # Orchestration and I/O
│   ├── _tiled_image_creation_pipeline.py  # tiled_image_creation_pipeline()
│   ├── _tiles_aggregation_pipeline.py     # tiles_aggregation_pipeline()
│   ├── _registration_pipeline.py          # build_default_registration_pipeline()
│   ├── _alignment.py         # FOV alignment corrections
│   ├── _snap_utils.py        # Snap-to-grid overlap removal
│   ├── _tiling.py            # Tile splitting modes
│   ├── _filters.py           # RegexIncludeFilter, RegexExcludeFilter
│   ├── _validators.py        # Tile validation
│   ├── _collection_setup.py  # setup_plates(), collection handler registry
│   ├── _write_ome_zarr.py    # OME-Zarr metadata + array writing
│   └── _to_zarr.py           # WriterMode dispatch (BY_TILE/BY_FOV/BY_FOV_DASK/IN_MEMORY)
└── fractal/        # Fractal platform integration
    ├── _init_task.py          # setup_images_for_conversion()
    ├── _compute_task.py       # generic_compute_task() factory
    ├── _json_utils.py         # TiledImage JSON serialization with retry logic
    └── _models.py             # Fractal-specific Pydantic models
```

### Key Patterns

- **`TiledImage` is generic**: `TiledImage[CollectionType, LoaderType]` — when deserializing from JSON, concrete types must be specified.
- **`DefaultImageLoader` needs `resource`**: When `file_name` paths are relative, pass `resource=str(data_dir)` to pipeline functions so the loader can locate files.
- **Collection handler registry**: `_collection_setup.py` uses a registry — `setup_plates()` is registered as `"ImageInPlate"`, extend via `add_collection_handler()`.
- **Writer modes**: `BY_TILE` (sequential per tile), `BY_FOV` (sequential per FOV), `BY_FOV_DASK` (parallel via Dask), `IN_MEMORY` (load all then write).

### Key Dependencies

- **ngio** (>=0.5.3,<0.6.0): OME-Zarr I/O — `OmeZarrContainer`, `Image`, HCS plate operations. Key API: `img.get_array()`, `img.num_channels`, `img.channel_labels`, `container.list_tables()`.
- **Dask**: Lazy/parallel image loading and writing
- **Pydantic v2**: All models use Pydantic for validation
- **polars**: Condition tables in `_collection_setup.py`

## Testing

### Structure

- **Unit tests**: `tests/unit/` — individual functions/classes with mocks and `DummyLoader`
- **Integration tests**: `tests/integration/test_pipelines.py` — end-to-end with real PNG images from `examples/`
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`
- Python support: 3.11–3.14, CI runs on Ubuntu + macOS

### Test Data

- `examples/hcs_plate/` — HCS plate: `tiles.csv` + `data/` with 4 PNGs (2560x2160), well A/1, 3 FOVs, 2 Z-slices, drug=DMSO
- `examples/single_acquisitions/` — Single image: `tiles.csv`, reuses same PNGs via `resource` parameter
- `tests/data/hiPSC_Tiny/` — Alternative test data with `metadata.json` + `tiles.csv`

### Writing Tests

- Use `build_dummy_tile()` from `core/_dummy_tiles.py` for synthetic tiles (fast, no I/O)
- Use `tiled_image_from_tiles()` to aggregate tiles into `TiledImage` objects
- For real image tests, pass `resource=str(data_dir)` to `tiles_aggregation_pipeline()` and `tiled_image_creation_pipeline()`
- Use `tmp_path` fixture for OME-Zarr output
- Validate OME-Zarr with ngio: `OmeZarrContainer` → `get_image()` → `get_array()`, `list_tables()`

### Common Full-Pipeline Test Pattern

```python
tiles = hcs_images_from_dataframe(tiles_table=df, acquisition_details=acq, plate_name="P")
images = tiles_aggregation_pipeline(tiles=tiles, converter_options=opts, resource=str(data_dir))
pipeline = build_default_registration_pipeline(AlignmentCorrections(), TilingMode.AUTO)
omezarr = tiled_image_creation_pipeline(
    zarr_url=zarr_url, tiled_image=images[0], registration_pipeline=pipeline,
    converter_options=opts, writer_mode=WriterMode.BY_FOV,
    overwrite_mode=OverwriteMode.OVERWRITE, resource=str(data_dir),
)
```
