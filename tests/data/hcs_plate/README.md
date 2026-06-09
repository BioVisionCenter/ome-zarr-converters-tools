# Example of HCS Plate conversion from Table

This example demonstrates how to structure a CSV (or Parquet) table to define a high-content screening (HCS) plate layout for conversion into an OME-Zarr format using the `ome-zarr-converters-tools` library.

## Table Structure
The table should contain the following columns:
- `file_path`: Path to each image file that needs to be converted. Supported formats include TIFF, PNG, JPEG.
- `row`: The row index of the well in the plate (e.g., A, B, C, ...) or (1, 2, 3, ...). If a index is used this will be converted to letters automatically.
- `column`: The column index of the well in the plate (e.g., 1, 2, 3, ...).
- `fov_name`: (Optional) Name of the field of view.
- `start_x`, `start_y`, `start_z`, `start_c`, `start_t`: Starting positions for each dimension. Only `start_x` and `start_y` are required; others default to 0 if not provided.
- `length_x`, `length_y`, `length_z`, `length_c`, `length_t`: Size of each dimension. Only `length_x` and `length_y` are required; others default to 1 if not provided.
- `pixelsize`: (Optional) Pixel size in micrometers. This can be alternatively specified in the companion `acquisition_details.toml` file.
- `z_spacing`: (Optional) Spacing between Z slices in micrometers. This can be alternatively specified in the companion `acquisition_details.toml` file.
- `t_spacing`: (Optional) Spacing between time points in seconds. This can be alternatively specified in the companion `acquisition_details.toml` file.
- `channel_names`: (Optional) Slash-separated names of the channels. These can be alternatively specified in the companion `acquisition_details.toml` file.
- `wavelengths`: (Optional) Slash-separated wavelengths for each channel in nanometers. These can be alternatively specified in the companion `acquisition_details.toml` file.
- `...`: Additional metadata columns can be included as needed. For fractal users these will be added as key-value pairs in the image list.

Note: The `plate_name` and `acquisition_id` parameters are not specified in the table — they are passed directly to the `hcs_images_from_dataframe()` function.

For example, see the [`tiles.csv`](./tiles.csv) file.


## Companion `acquisition_details.toml` File
In addition to the CSV table, you can provide a `acquisition_details.toml` file to specify default acquisition parameters that apply to all images. This file can include:
- `pixelsize`: (Optional) Default pixel size in micrometers. If specified in both places, the table value takes precedence. If neither is specified, a default of 1.0 micrometers is used.
- `z_spacing`: (Optional) Default spacing between Z slices in micrometers. If specified in both places, the table value takes precedence. If neither is specified, a default of 1.0 micrometers is used.
- `t_spacing`: (Optional) Default spacing between time points in seconds. If specified in both places, the table value takes precedence. If neither is specified, a default of 1.0 seconds is used.
- `channel_names`: (Optional) Default list of the channels. If specified in both places, the table value takes precedence. If neither is specified, default names like "channel_0", "channel_1", etc. are used.
- `wavelengths`: (Optional) Default list of wavelengths for each channel in nanometers. If specified in both places, the table value takes precedence. If neither is specified, the channel names are used as default.
- Coordinate space options (`start_x_coo`, `start_y_coo`, etc.): Sometimes it might be useful to define the coordinate space of the acquisition differently for the start positions and the lengths. For example, the start positions might be given in world coordinates (micrometers), while the lengths are given in pixel coordinates.
  Implicit casting rules:
  - all `start_*_coo` fields default to `"world"` if not specified
  - all `length_*_coo` fields default to `"pixel"` if not specified
  - `start_c` and `length_c` are always in pixel coordinates and do not need to be specified here.
- `axes`: (Optional) List of strings defining the order of axes in the images. The elements should be an ordered subset of ["t", "c", "z", "y", "x"]. If not specified, the default order is assumed to be ["c", "z", "y", "x"] or if length_t > 1 then ["t", "c", "z", "y", "x"].

## Example `acquisition_details.toml`

See the [`acquisition_details.toml`](./acquisition_details.toml) file, or the example below:

```toml
# Coordinate space definitions
# Implicit casting rules:
# - all start_* fields default to "world" if not specified
# - all length_* fields default to "pixel" if not specified
# start_c and length_c are always in pixel coordinates
# and do not need to be specified here.
start_x_coo = "world"  # can either be "world" or "pixel"
start_y_coo = "world"  # can either be "world" or "pixel"
start_z_coo = "pixel"  # can either be "world" or "pixel"
start_t_coo = "pixel"  # can either be "world" or "pixel"
length_x_coo = "pixel" # can either be "world" or "pixel"
length_y_coo = "pixel" # can either be "world" or "pixel"
length_z_coo = "pixel" # can either be "world" or "pixel"
length_t_coo = "pixel" # can either be "world" or "pixel"

# Pixel size in micrometers
pixelsize = 0.65
# Z spacing in micrometers
z_spacing = 5.0
# T spacing in seconds
t_spacing = 1.0
# Channel names
channel_names = ["DAPI"]
# Wavelengths id (e.g., string of the excitation wavelength in nm)
wavelengths = ["405"]
# Axes order in the images
axes = ["t", "c", "z", "y", "x"]
```
