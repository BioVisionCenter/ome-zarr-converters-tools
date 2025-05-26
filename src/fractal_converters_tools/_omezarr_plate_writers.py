"""Utility functions for building OME metadata from fractal-tasks-core models."""

from pathlib import Path

from ngio import ImageInWellPath, create_empty_plate

from fractal_converters_tools._tiled_image import PlatePathBuilder, TiledImage


def _initiate_ome_zarr_plate(
    zarr_dir: Path,
    tiled_images: list[TiledImage],
    overwrite: bool = False,
) -> None:
    """Create an OME-Zarr plate from a list of acquisitions."""
    images_in_plate = []
    plate_name = ""
    for img in tiled_images:
        if not isinstance(img.path_builder, PlatePathBuilder):
            raise ValueError(
                "Something went wrong with the parsing. "
                "Some of the metadata is missing or not correctly "
                "formatted."
            )
        path_builder = img.path_builder
        if plate_name == "":
            plate_name = path_builder.plate_name

        if plate_name != path_builder.plate_name:
            raise ValueError(
                "Something went wrong with the parsing. "
                "Some of the metadata is missing or not correctly "
                "formatted."
            )

        _image_in_plate = ImageInWellPath(
            row=path_builder.row,
            column=path_builder.column,
            path=str(path_builder.acquisition_id),
            acquisition_id=path_builder.acquisition_id,
            acquisition_name=f"{plate_name}_id{path_builder.acquisition_id}",
        )
        images_in_plate.append(_image_in_plate)

    zarr_url = zarr_dir / f"{plate_name}.zarr"
    create_empty_plate(
        store=zarr_url, name=plate_name, images=images_in_plate, overwrite=overwrite
    )


def initiate_ome_zarr_plates(
    zarr_dir: str | Path,
    tiled_images: list[TiledImage],
    overwrite: bool = False,
) -> None:
    """Create an OME-Zarr plate from a list of acquisitions."""
    zarr_dir = Path(zarr_dir)
    plates = {}
    for img in tiled_images:
        if not isinstance(img.path_builder, PlatePathBuilder):
            raise ValueError(
                "Something went wrong with the parsing. "
                "Some of the metadata is missing or not correctly "
                "formatted."
            )
        if img.path_builder.plate_name not in plates:
            plates[img.path_builder.plate_name] = []
        plates[img.path_builder.plate_name].append(img)

    for images in plates.values():
        _initiate_ome_zarr_plate(
            zarr_dir=zarr_dir,
            tiled_images=images,
            overwrite=overwrite,
        )


def update_ome_zarr_plate(
    store: str | Path,
    plate_name: str,
    tiledimages: list[TiledImage],
):
    """Update an Existing OME-Zarr plate with new TiledImages."""
    raise NotImplementedError("Not implemented yet.")
