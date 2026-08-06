"""Snippets for docs/getting_started/0_hcs_plates.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own, from the repo root:

    python docs/snippets/getting_started/hcs_plates.py
"""

# --8<-- [start:load_table]
import pandas as pd

# One row per image file on disk. Paths are relative to `resource`, set further down.
tiles_table = pd.read_csv("tests/data/hcs_plate/tiles.csv")
# --8<-- [end:load_table]

# --8<-- [start:table_helpers]
import sys

# markdown-exec execs this block, so `__file__` does not exist; a standalone run puts
# only this script's own directory on sys.path. Both run from the repo root, which is
# what the rest of the snippets already assume, so resolve the shared module against it.
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
    xy_pixel_size=0.65,  # micrometers
    z_spacing=5.0,  # micrometers
    t_spacing=1.0,  # seconds
    axes=["t", "c", "z", "y", "x"],
    # Coordinate systems: start positions are in world coordinates,
    # lengths are in pixel coordinates
    start_x_space="world",
    start_y_space="world",
    start_z_space="pixel",
    start_t_space="pixel",
)
print(acq)
# --8<-- [end:acquisition_details]

# --8<-- [start:parse_tiles]
from ome_zarr_converters_tools import hcs_images_from_dataframe

tiles = hcs_images_from_dataframe(
    tiles_table=tiles_table,
    acquisition_details=acq,
    plate_name="CardiomyocytePlate",
    acquisition_id=0,
)

fov_names = {t.fov_name for t in tiles}
print(f"Number of tiles: {len(tiles)}")
print(f"FOV names: {sorted(fov_names)}")
print(f"Collection type: {type(tiles[0].collection).__name__}")
# --8<-- [end:parse_tiles]

# --8<-- [start:plot_layout]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
show_tile_layout(ax, tiles, title="Stage layout · well A/01")
fig.tight_layout()
print(figure_html(fig, alt="Three overlapping fields of view in stage coordinates."))
# --8<-- [end:plot_layout]

# --8<-- [start:aggregate]
from pathlib import Path

from ome_zarr_converters_tools import ConverterOptions, tiles_aggregation_pipeline

# The DefaultImageLoader resolves file paths against `resource`, which must be an
# absolute path (or an s3:// URL). Resolve the example data dir to an absolute path.
data_dir = str(Path("tests/data/hcs_plate/data").resolve())
opts = ConverterOptions()

tiled_images = tiles_aggregation_pipeline(
    tiles=tiles,
    converter_options=opts,
    resource=data_dir,
)

print(f"Number of TiledImages: {len(tiled_images)}")
for ti in tiled_images:
    print(f"  Path: {ti.path}, regions: {len(ti.regions)}, FOVs: {len(ti.group_by_fov())}")
# --8<-- [end:aggregate]

# --8<-- [start:setup_plate]
from ome_zarr_converters_tools import OverwriteMode, setup_ome_zarr_collection

# ./data is gitignored and cleaned by `pixi run -e docs clean_docs_data`.
zarr_dir = str(Path("data/getting_started_hcs").resolve())

setup_ome_zarr_collection(
    tiled_images=tiled_images,
    collection_type="ImageInPlate",
    zarr_dir=zarr_dir,
    overwrite_mode=OverwriteMode.OVERWRITE,
)
print("Plate structure created.")
# --8<-- [end:setup_plate]

# --8<-- [start:write]
from ome_zarr_converters_tools import (
    AutoTiling,
    StagePositionCorrections,
    WriterMode,
    build_default_registration_pipeline,
    tiled_image_creation_pipeline,
)

pipeline = build_default_registration_pipeline(
    alignment_corrections=StagePositionCorrections(),
    tiling_strategy=AutoTiling(),
)

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
from ngio import open_ome_zarr_plate

ome_zarr_plate = open_ome_zarr_plate(f"{zarr_dir}/CardiomyocytePlate.zarr")

print(f"Plate: {ome_zarr_plate}")
print(f"Images: {ome_zarr_plate.get_images()}")

ome_zarr_container = ome_zarr_plate.get_image(row="A", column=1, image_path="0")
image = ome_zarr_container.get_image()
print(f"Image: {image}")
# --8<-- [end:verify]

# --8<-- [start:plot_result]
fig, ax = plt.subplots(figsize=(6.5, 6.5))
show_image(
    ax,
    image.get_as_numpy(channel_selection="DAPI", z=0),
    title="DAPI · z=0 · stitched well A/01",
    ignore_zeros=True,
    pixel_size=image.pixel_size,
)
fig.tight_layout()
print(figure_html(fig, alt="The three fields of view stitched into one DAPI image."))
# --8<-- [end:plot_result]
