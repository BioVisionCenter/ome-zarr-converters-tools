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

from fractal_converters_tools.image import ImageInWell

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_plate_meta(acquisitions: list[ImageInWell], plate_name) -> NgffPlateMeta:
    """Build a plate metadata object from a list of acquisitions."""
    if len(acquisitions) == 0:
        raise ValueError("Empty list of acquisitions")

    _acquisition_ids = list({acq.acquisition_id for acq in acquisitions})
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
    existing_rows = {acq.row for acq in acquisitions}
    for row_name in alphabet:
        if row_name in existing_rows:
            rows.append(RowInPlate(name=row_name))

    columns = []
    existing_columns = {acq.column for acq in acquisitions}
    for column_name in range(1, 100):
        if column_name in existing_columns:
            columns.append(ColumnInPlate(name=str(column_name)))

    wells = {}
    for row_id, row in enumerate(rows):
        for column_id, column in enumerate(columns):
            path = f"{row.name}/{column.name}"

            for acq in acquisitions:
                if acq.well_path == path and path not in wells:
                    wells[path] = WellInPlate(
                        path=path,
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


def build_well_meta(acquisitions: list[ImageInWell]) -> dict[str, NgffWellMeta]:
    """Build a well metadata object from a list of acquisitions."""
    well_meta = {}

    for acq in acquisitions:
        if acq.well_path not in well_meta:
            well_meta[acq.well_path] = set()

        well_meta[acq.well_path].add(acq.acquisition_id)

    _well_meta = {}
    for path, wells in well_meta.items():
        images = []
        for acquisition_id in wells:
            images.append(
                ImageInWellMeta(acquisition=acquisition_id, path=str(acquisition_id))
            )

        _well_meta[path] = NgffWellMeta(well=Well(images=images, version="0.4.0"))
    return _well_meta


def initiate_ome_zarr_plate(
    store: str | Path,
    plate_name: str,
    acquisitions: list[ImageInWell],
    overwrite: bool = False,
) -> None:
    """Create an OME-Zarr plate from a list of acquisitions."""
    plate_meta = build_plate_meta(acquisitions, plate_name)
    plate_wells_meta = build_well_meta(acquisitions)

    store = Path(store)
    if store.exists() and not overwrite:
        raise ValueError(
            f"Zarr file already exists at {store}. " "Set overwrite=True to overwrite."
        )

    plate_group = zarr.open_group(store, mode="w")
    plate_group.attrs.update(plate_meta.model_dump(exclude_none=True))

    for well_path, well_meta in plate_wells_meta.items():
        well_group = plate_group.create_group(well_path)
        well_group.attrs.update(well_meta.model_dump(exclude_none=True))
