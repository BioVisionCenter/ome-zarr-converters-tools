# Example of Single Image Acquisitions from Table

This example demonstrates how to structure a CSV (or Parquet) table to define standalone (non-HCS) image acquisitions for conversion into OME-Zarr format using the `ome-zarr-converters-tools` library. Unlike the HCS plate example, single acquisitions do not have a plate/well/acquisition hierarchy.

## Table Structure
The table should contain the following columns:
- `file_path`: Path to each image file that needs to be converted. Supported formats include TIFF, PNG, JPEG.
- `image_path`: Name used as the output OME-Zarr path (a `.zarr` suffix is appended automatically).
- `fov_name`: Name of the field of view.
- `start_x`, `start_y`, `start_z`, `start_c`, `start_t`: Starting positions for each dimension. Only `start_x` and `start_y` are required; others default to 0 if not provided.
- `length_x`, `length_y`, `length_z`, `length_c`, `length_t`: Size of each dimension. Only `length_x` and `length_y` are required; others default to 1 if not provided.
- `...`: Additional metadata columns can be included as needed. For fractal users these will be added as key-value pairs in the image list.

For example, see the [`tiles.csv`](./tiles.csv) file.

## Companion `acquisition_details.toml` File
The same companion TOML file format as described in the [HCS plate example](../hcs_plate/README.md) is used. See the [`acquisition_details.toml`](./acquisition_details.toml) file.

## Key Differences from HCS Plate
- No `row` or `column` columns — there is no plate/well layout.
- The `image_path` column replaces the well-based path construction.
- All tiles sharing the same `image_path` are grouped into a single output OME-Zarr.
- The `plate_name` and `acquisition_id` parameters are not needed.
