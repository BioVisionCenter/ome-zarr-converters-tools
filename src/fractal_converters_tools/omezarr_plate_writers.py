"""Utility functions for building OME metadata from fractal-tasks-core models."""

from pathlib import Path

import zarr
from fractal_tasks_core.ngff.specs import (
    AcquisitionInPlate,
    ColumnInPlate,
    NgffPlateMeta,
    NgffWellMeta,
    Plate,
    RowInPlate,
    Well,
    WellInPlate,
)
from fractal_tasks_core.ngff.specs import ImageInWell as ImageInWellMeta

from fractal_converters_tools.tiled_image import PlatePathBuilder, TiledImage

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def validate_tiled_images(tiled_images: list[TiledImage]) -> None:
    """Validate if the tiled images.

    Check if all the required attributes are present, like:
    - path_builder == PlatePathBuilder
    """
    for img in tiled_images:
        if not isinstance(img.path_builder, PlatePathBuilder):
            raise ValueError(
                "Something went wrong with the parsing. "
                "Some of the metadata is missing or not correctly "
                "formatted."
            )


def build_plate_meta(tiled_images: list[TiledImage]) -> NgffPlateMeta:
    """Build a plate metadata object from a list of acquisitions."""
    if len(tiled_images) == 0:
        raise ValueError("Empty list of acquisitions")

    plate_name = tiled_images[0].path_builder.plate_name

    _acquisition_ids = list({img.path_builder.acquisition_id for img in tiled_images})
    acquisition_ids = []
    for acquisition_id in _acquisition_ids:
        acq_model = AcquisitionInPlate(
            id=acquisition_id,
            name=f"{plate_name}_id{acquisition_id}",
            maximumfieldcount=None,
            description=None,
        )
        acquisition_ids.append(acq_model)

    rows = []
    existing_rows = {img.path_builder.row for img in tiled_images}
    for row_name in alphabet:
        if row_name in existing_rows:
            rows.append(RowInPlate(name=row_name))

    columns = []
    existing_columns = {img.path_builder.column for img in tiled_images}
    for column_name in range(1, 100):
        if column_name in existing_columns:
            columns.append(ColumnInPlate(name=str(column_name)))

    wells = {}
    for row_id, row in enumerate(rows):
        for column_id, column in enumerate(columns):
            well_id = f"{row.name}/{column.name}"

            for img in tiled_images:
                query_well_id = img.path_builder.well_id
                if query_well_id == well_id and query_well_id not in wells:
                    wells[well_id] = WellInPlate(
                        path=well_id,
                        rowIndex=row_id,
                        columnIndex=column_id,
                    )
    wells_list = list(wells.values())

    plate = Plate(
        acquisitions=acquisition_ids,
        rows=rows,
        columns=columns,
        wells=wells_list,
        name=plate_name,
        version="0.4.0",
    )
    return NgffPlateMeta(plate=plate)


def build_well_meta(tiled_images: list[TiledImage]) -> dict[str, NgffWellMeta]:
    """Build a well metadata object from a list of acquisitions."""
    well_meta = {}

    for img in tiled_images:
        if img.path_builder.well_id not in well_meta:
            well_meta[img.path_builder.well_id] = set()

        well_meta[img.path_builder.well_id].add(img.path_builder.acquisition_id)

    _well_meta = {}
    for path, wells in well_meta.items():
        images = []
        for acquisition_id in wells:
            images.append(
                ImageInWellMeta(acquisition=acquisition_id, path=str(acquisition_id))
            )

        _well_meta[path] = NgffWellMeta(well=Well(images=images, version="0.4.0"))
    return _well_meta


def _initiate_ome_zarr_plate(
    store: Path,
    tiled_images: list[TiledImage],
    overwrite: bool = False,
) -> None:
    """Create an OME-Zarr plate from a list of acquisitions."""
    plate_meta = build_plate_meta(tiled_images)
    plate_wells_meta = build_well_meta(tiled_images)

    plate_store = store / tiled_images[0].path_builder.plate_path

    if plate_store.exists() and not overwrite:
        raise FileExistsError(
            f"Zarr file already exists at {store}. Set overwrite=True to overwrite."
        )

    plate_group = zarr.open_group(plate_store, mode="w")
    plate_group.attrs.update(plate_meta.model_dump(exclude_none=True))

    for well_id, well_meta in plate_wells_meta.items():
        well_group = plate_group.create_group(well_id)
        well_group.attrs.update(well_meta.model_dump(exclude_none=True))


def initiate_ome_zarr_plates(
    store: str | Path,
    tiled_images: list[TiledImage],
    overwrite: bool = False,
) -> None:
    """Create an OME-Zarr plate from a list of acquisitions."""
    store = Path(store)

    validate_tiled_images(tiled_images)
    plates = {}
    for img in tiled_images:
        if img.path_builder.plate_name not in plates:
            plates[img.path_builder.plate_name] = []
        plates[img.path_builder.plate_name].append(img)

    for images in plates.values():
        _initiate_ome_zarr_plate(store=store, tiled_images=images, overwrite=overwrite)


def update_ome_zarr_plate(
    store: str | Path,
    plate_name: str,
    tiledimages: list[TiledImage],
):
    """Update an Existing OME-Zarr plate with new TiledImages."""
    raise NotImplementedError("Not implemented yet.")
