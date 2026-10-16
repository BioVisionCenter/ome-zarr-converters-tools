"""Snippets for docs/getting_started/1_single_images.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own, from the repo root:

    python docs/snippets/getting_started/single_images.py
"""

# --8<-- [start:load_table]
import pandas as pd

tiles_table = pd.read_csv("tests/data/single_acquisitions/tiles.csv")
print("Columns:", list(tiles_table.columns))
# --8<-- [end:load_table]

# --8<-- [start:table_helpers]
import sys

sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, show_image, show_tile_layout, table_html
# --8<-- [end:table_helpers]

# --8<-- [start:show_table]
print(table_html(tiles_table))
# --8<-- [end:show_table]

# --8<-- [start:acquisition_details]
from ome_zarr_converters_tools import AcquisitionDetails, ChannelInfo

acq = AcquisitionDetails(
    channels=[ChannelInfo(channel_label="DAPI", wavelength_id="405")],
    xy_pixel_size=0.65,
    z_spacing=5.0,
    t_spacing=1.0,
    axes=["t", "c", "z", "y", "x"],
    start_x_space="world",
    start_y_space="world",
    start_z_space="pixel",
    start_t_space="pixel",
)
# --8<-- [end:acquisition_details]

# --8<-- [start:parse_tiles]
from ome_zarr_converters_tools import single_images_from_dataframe

tiles = single_images_from_dataframe(
    tiles_table=tiles_table,
    acquisition_details=acq,
)

print(f"Number of tiles: {len(tiles)}")
print(f"Collection type: {type(tiles[0].collection).__name__}")
print(f"Image path: {tiles[0].collection.image_path}")
# --8<-- [end:parse_tiles]

# --8<-- [start:plot_layout]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
show_tile_layout(ax, tiles, title="Stage layout · cardiomyocyte_scan")
fig.tight_layout()
print(figure_html(fig, alt="Two fields of view in stage coordinates."))
# --8<-- [end:plot_layout]

# --8<-- [start:aggregate]
from pathlib import Path

from ome_zarr_converters_tools import ConverterOptions, tiles_aggregation_pipeline

# This table describes a standalone scan, but it reuses the same PNG files as the HCS
# example — they live only under hcs_plate/data, which is what `resource` points at.
data_dir = str(Path("tests/data/hcs_plate/data").resolve())
opts = ConverterOptions()

tiled_images = tiles_aggregation_pipeline(
    tiles=tiles,
    converter_options=opts,
    resource=data_dir,
)

print(f"Number of TiledImages: {len(tiled_images)}")
for ti in tiled_images:
    print(
        f"  Path: {ti.path}, regions: {len(ti.regions)}, FOVs: {len(ti.group_by_fov())}"
    )
# --8<-- [end:aggregate]

# --8<-- [start:write]
from ome_zarr_converters_tools import (
    AutoTiling,
    OverwriteMode,
    StagePositionCorrections,
    WriterMode,
    build_default_registration_pipeline,
    tiled_image_creation_pipeline,
)

pipeline = build_default_registration_pipeline(
    alignment_corrections=StagePositionCorrections(),
    tiling_strategy=AutoTiling(),
)

# ./data is gitignored and cleaned by `pixi run -e docs clean_docs_data`.
zarr_dir = str(Path("data/getting_started_single_images").resolve())

for tiled_image in tiled_images:
    zarr_url = f"{zarr_dir}/{tiled_image.path}"
    omezarr = tiled_image_creation_pipeline(
        zarr_url=zarr_url,
        tiled_image=tiled_image,
        registration_pipeline=pipeline,
        converter_options=opts,
        writer_mode=WriterMode.BY_FOV,
        overwrite_mode=OverwriteMode.OVERWRITE,
        resource=data_dir,
    )
    print(f"Written: {tiled_image.path}")
# --8<-- [end:write]

# --8<-- [start:verify]
image = omezarr.get_image()

print(f"Image: {image}")
print(f"Channels: {image.channel_labels}")
print(f"Tables: {omezarr.list_tables()}")
# --8<-- [end:verify]

# --8<-- [start:plot_result]
fig, ax = plt.subplots(figsize=(6.5, 6.5))
show_image(
    ax,
    image.get_as_numpy(channel_selection="DAPI", z=0),
    title="DAPI · z=0 · cardiomyocyte_scan",
    ignore_zeros=True,
    pixel_size=image.pixel_size,
)
fig.tight_layout()
print(figure_html(fig, alt="The two fields of view stitched into one DAPI image."))
# --8<-- [end:plot_result]
