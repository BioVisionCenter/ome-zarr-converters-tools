# Changelog

## Unreleased

### Documentation

- Rewrote tutorials as three separate notebooks:
  - **HCS Plate Tutorial** -- table-based workflow for plate data with correct DataFrame column descriptions and proper plate setup via `setup_ome_zarr_collection()`
  - **Single Images Tutorial** -- standalone image workflow using `single_images_from_dataframe()`
  - **Advanced Tutorial** -- programmatic `Tile` construction with custom `ImageLoaderInterface`
- Added **Fractal Tasks Guide** (`converters_as_fractal_tasks.md`) covering init/compute task model, `setup_images_for_conversion()`, `generic_compute_task()`, `AcquisitionOptions`, and manifest generation
- Fixed incorrect DataFrame column names in tutorial (e.g., `fov_id` -> `fov_name`, `channel_id` -> `start_c`, `plate_row` -> `row`)
- Clarified that the library input is a pandas DataFrame, not necessarily a CSV file
- Fixed missing `setup_ome_zarr_collection()` call in HCS plate tutorial
- Fixed incorrect `zarr_url` construction for HCS plate images
- Updated `mkdocs.yml` navigation and `index.md` links
- Updated README documentation links
