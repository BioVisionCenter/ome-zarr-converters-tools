"""Snippets for docs/getting_started/2_programmatic_tiles.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own, from the repo root:

    python docs/snippets/getting_started/programmatic_tiles.py
"""

# --8<-- [start:loader]
from typing import Any

import numpy as np
from PIL import Image

from ome_zarr_converters_tools import ImageLoaderInterface


class PngLoader(ImageLoaderInterface):
    """Custom loader that loads a single PNG file."""

    file_path: str

    def load_data(self, resource: Any = None) -> np.ndarray:
        """Load the PNG file as a NumPy array."""
        if resource is not None:
            path = f"{resource}/{self.file_path}"
        else:
            path = self.file_path
        return np.array(Image.open(path))
# --8<-- [end:loader]

# --8<-- [start:plot_helpers]
import sys

sys.path.append("docs/snippets")

from matplotlib import pyplot as plt

from _render import figure_html, show_image
# --8<-- [end:plot_helpers]

# --8<-- [start:build_tiles]
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    ChannelInfo,
    SingleImage,
    Tile,
)

acq = AcquisitionDetails(
    channels=[ChannelInfo(channel_label="DAPI")],
    xy_pixel_size=0.65,
    z_spacing=5.0,
    t_spacing=1.0,
    # `start_z` below is a slice index, not a position in micrometers. Without this
    # the default ("world") reads start_z=1 as 1 um, which at z_spacing=5.0 rounds
    # back to slice 0 — both tiles would land on the same slice, and the second
    # would silently overwrite the first.
    start_z_space="pixel",
)

collection = SingleImage(image_path="manual_example")

# One FOV, two Z slices.
tiles = [
    Tile(
        fov_name="FOV_1",
        start_x=10.0,
        start_y=10.0,
        start_z=z,
        length_x=2560,
        length_y=2160,
        length_z=1,
        length_c=1,
        length_t=1,
        collection=collection,
        image_loader=PngLoader(file_path=file_path),
        acquisition_details=acq,
    )
    for z, file_path in enumerate(
        [
            "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F001L01A01Z01C01.png",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_B03_T0001F001L01A01Z02C01.png",
        ]
    )
]

print(f"Number of tiles: {len(tiles)}")
print(f"Loader: {type(tiles[0].image_loader).__name__}")
# --8<-- [end:build_tiles]

# --8<-- [start:write]
from pathlib import Path

from ome_zarr_converters_tools import (
    AutoTiling,
    ConverterOptions,
    OverwriteMode,
    StagePositionCorrections,
    WriterMode,
    build_default_registration_pipeline,
    tiled_image_creation_pipeline,
    tiles_aggregation_pipeline,
)

data_dir = str(Path("tests/data/hcs_plate/data").resolve())
opts = ConverterOptions()

tiled_images = tiles_aggregation_pipeline(
    tiles=tiles,
    converter_options=opts,
    resource=data_dir,
)

pipeline = build_default_registration_pipeline(
    alignment_corrections=StagePositionCorrections(),
    tiling_strategy=AutoTiling(),
)

# ./data is gitignored and cleaned by `pixi run -e docs clean_docs_data`.
zarr_dir = str(Path("data/getting_started_programmatic").resolve())

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
# --8<-- [end:verify]

# --8<-- [start:plot_result]
fig, ax = plt.subplots(figsize=(5.5, 5.5))
show_image(
    ax,
    image.get_as_numpy(channel_selection="DAPI", z=0),
    title="DAPI · z=0 · written through PngLoader",
    pixel_size=image.pixel_size,
)
fig.tight_layout()
print(figure_html(fig, alt="The single field of view written through the custom loader."))
# --8<-- [end:plot_result]
