"""Collection setup functions for OME-Zarr converters tools."""

import zarr
from ngio import DefaultNgffVersion, NgffVersions
from ngio.hcs import create_empty_plate, open_ome_zarr_plate
from ngio.hcs._plate import ImageInWellPath

from ome_zarr_converters_tools.models._acquisition import OverwriteMode
from ome_zarr_converters_tools.models._collection import (
    ImageInPlate,
)
from ome_zarr_converters_tools.models._tile_region import TiledImage


def setup_plates(
    zarr_dir: str,
    tiled_images: list[TiledImage],
    ngff_version: NgffVersions = DefaultNgffVersion,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
) -> None:
    """Set up an ImageInPlate collection in the Zarr group."""
    assert isinstance(tiled_images[0].collection, ImageInPlate)
    zarr_format = 2 if ngff_version == "0.4" else 3
    if overwrite_mode == OverwriteMode.NO_OVERWRITE:
        mode = "w-"
    elif overwrite_mode == OverwriteMode.OVERWRITE:
        mode = "w"
    else:  # extend
        mode = "a"

    plates = {}
    for tile in tiled_images:
        plate_path = tile.collection.plate_path()
        if plate_path not in plates:
            plates[plate_path] = {"name": tile.collection.plate_name, "images": []}
        image_in_well = ImageInWellPath(
            row=tile.collection.row,
            column=tile.collection.column,
            path=tile.collection.path_in_well(),
            acquisition_id=tile.collection.acquisition,
            acquisition_name=str(tile.collection.acquisition),
        )
        plates[plate_path]["images"].append(image_in_well)
    for plate_path, plate_info in plates.items():
        plante_url = f"{zarr_dir}/{plate_path}"
        group = zarr.open_group(store=plante_url, mode=mode, zarr_format=zarr_format)
        try:
            # This can only succeed in "extend" mode if the group already exists
            plate = open_ome_zarr_plate(group, cache=True)
        except Exception:
            plate = create_empty_plate(
                store=group,
                name=plate_info["name"],
                ngff_version=ngff_version,
                overwrite=True,
                cache=True,
            )
        existing_image = plate.images_paths()
        for image in plate_info["images"]:
            image_path = f"{image.row}/{image.column}/{image.path}"
            if image_path in existing_image:
                # Image already exists in the plate, skip adding
                # This can only happen in 'extend' mode
                # othe other modes would have overwritten or raised an error
                continue
            plate.add_image(
                row=image.row,
                column=image.column,
                image_path=image.path,
                acquisition_id=image.acquisition_id,
                acquisition_name=image.acquisition_name,
            )
