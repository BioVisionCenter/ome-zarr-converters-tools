"""End-to-end snapshot testing utilities for OME-Zarr converters.

A converter test runs a real conversion of a raw microscopy dataset and compares
the produced OME-Zarr against a human-validated snapshot stored on disk as JSON.
`run_converter_test` is the single entry point used by every converter's test
suite. Pass `--update-snapshots` (registered by
`ome_zarr_converters_tools.testing.plugin`) to (re)generate the snapshot files.

The snapshot for each image records its `axes`, `shape`, `pixelsize`, channel
metadata, the `types`/`attributes` returned by the init task, and, per ROI in
each table, a fingerprint of the pixel data (`mean`/`std`/`min`/`max` plus a
sha256 hash of the rounded array). Generation and validation share a single code
path: `build_snapshot` reads the converted output into a model, which is either
written to disk (update mode) or compared field-by-field against the on-disk
model (`compare_snapshots`).

Each snapshot also carries a top-level `versions` block recording the versions of
key dependencies (this package, ngio, zarr, numpy, dask, tifffile, pillow,
pydantic, and Python) at generation time. It is informational only and is never
compared -- a version drift must not fail a test; it exists to help diagnose a
snapshot discrepancy caused by an upstream change.
"""

import hashlib
import json
import math
import platform
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import numpy as np
from ngio import (
    OmeZarrContainer,
    open_ome_zarr_container,
    open_ome_zarr_plate,
)
from ngio.utils import NgioValueError
from pydantic import BaseModel, Field, model_validator

from ome_zarr_converters_tools.models._converter_options import (
    ConverterOptions,
    OverwriteMode,
)

# Fingerprint stats are rounded to this many decimals on write; comparison uses
# an absolute/relative tolerance of the same order. The sha256 pixel hash stays
# the authoritative, exact pixel check.
_FINGERPRINT_DECIMALS = 6
_STAT_REL_TOL = 1e-6
_STAT_ABS_TOL = 1e-6

# Dependencies whose versions are recorded in each snapshot's informational
# `versions` block (never compared). Covers the full write/read stack that can
# shift metadata or pixel hashes between environments.
_VERSIONED_PACKAGES = (
    "ome-zarr-converters-tools",
    "ngio",
    "zarr",
    "numpy",
    "dask",
    "tifffile",
    "pillow",
    "pydantic",
)


def _collect_versions() -> dict[str, str | None]:
    """Collect versions of key dependencies for the informational snapshot block.

    A package that is not installed is recorded as `None` rather than raising.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in _VERSIONED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


class FingerprintModel(BaseModel):
    """Compact pixel-content signature for a single ROI."""

    mean: float
    std: float
    min: float
    max: float
    hash: str

    @classmethod
    def from_array(cls, arr: np.ndarray, decimals: int = _FINGERPRINT_DECIMALS):
        """Build a fingerprint from a pixel array.

        Args:
            arr: The pixel data read back from a ROI.
            decimals: Decimals to round the stored stats to, and to round the
                array to before hashing.
        """
        rounded = np.round(arr, decimals)
        return cls(
            mean=round(float(np.mean(arr)), decimals),
            std=round(float(np.std(arr)), decimals),
            min=round(float(np.min(arr)), decimals),
            max=round(float(np.max(arr)), decimals),
            hash=hashlib.sha256(rounded.tobytes()).hexdigest(),
        )


class RoiAssertionModel(BaseModel):
    """Expected fingerprint and geometry for a single ROI."""

    slice_repr: str
    finger_print: FingerprintModel
    yx_origin: list[float] | None = None


class TableAssertionModel(BaseModel):
    """Expected ROIs for a single table."""

    rois: dict[str, RoiAssertionModel] = Field(default_factory=dict)


class ImageAssertionModel(BaseModel):
    """Expected metadata and per-table ROI fingerprints for one image."""

    axes: tuple[str, ...]
    shape: tuple[int, ...]
    pixelsize: tuple[float, ...]
    channel_labels: list[str] = Field(default_factory=list)
    wavelength_ids: list[str | None] = Field(default_factory=list)
    types: dict[str, bool] = Field(default_factory=dict)
    attributes: dict[str, str | int | float] = Field(default_factory=dict)
    tables: dict[str, TableAssertionModel | None] = Field(default_factory=dict)


def _deep_merge(a: dict, b: dict) -> dict:
    result = a.copy()
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class PlateAssertionModel(BaseModel):
    """Expected wells and images for a single plate.

    A top-level `images_common` mapping may be supplied on input to factor out
    assertions shared by every image; it is deep-merged into each image and not
    retained on the model.
    """

    wells: list[str]
    images: dict[str, ImageAssertionModel]

    @model_validator(mode="before")
    @classmethod
    def validate_images(cls, values):
        """Deep-merge `images_common` into every image entry.

        `images_common` provides shared defaults; per-image values override them.
        """
        common_assertions = values.pop("images_common", {})
        images = values.get("images", {})
        updated_image_assertions = {}
        for image_path, image_assertions in images.items():
            # Common first (base), image second (overrides), for every image —
            # including keys the image does not itself specify.
            updated_image_assertions[image_path] = _deep_merge(
                common_assertions, image_assertions
            )
        values["images"] = updated_image_assertions
        return values


class MultiPlateAssertionModel(BaseModel):
    """Snapshot for HCS plate output: one or more plates keyed by zarr name."""

    # Informational only; recorded at generation time, never compared.
    versions: dict[str, str | None] = Field(default_factory=dict)
    plates: dict[str, PlateAssertionModel]


class MultiSingleImageAssertionModel(BaseModel):
    """Snapshot for single-image output: images keyed by zarr name."""

    # Informational only; recorded at generation time, never compared.
    versions: dict[str, str | None] = Field(default_factory=dict)
    images: dict[str, ImageAssertionModel]


# --------------------------------------------------------------------------- #
# Snapshot construction (single source of truth for generate + validate)
# --------------------------------------------------------------------------- #


def _build_image_entry(*, ome_zarr_image: OmeZarrContainer, upd: dict | None) -> dict:
    image = ome_zarr_image.get_image()
    entry: dict = {
        "axes": list(image.axes),
        "shape": list(image.shape),
        "pixelsize": list(image.pixel_size.tzyx),
        "channel_labels": ome_zarr_image.channel_labels,
        "wavelength_ids": ome_zarr_image.wavelength_ids,
    }
    if upd is not None:
        if "types" in upd:
            entry["types"] = upd["types"]
        if "attributes" in upd:
            entry["attributes"] = upd["attributes"]

    table_names = ome_zarr_image.list_tables()
    if table_names:
        tables_dict: dict[str, dict | None] = {}
        for table_name in sorted(table_names):
            try:
                roi_table = ome_zarr_image.get_roi_table(table_name)
            except NgioValueError:
                # Not a ROI table (e.g. a condition table) — recorded as None.
                # Anything else (corrupt store, validation failure) propagates
                # so a broken table can't silently pass snapshot validation.
                tables_dict[table_name] = None
                continue
            rois_dict = {}
            for roi in roi_table.rois():
                roi_pixel = roi.to_pixel(pixel_size=image.pixel_size)
                roi_array = image.get_roi_as_numpy(roi)
                fp = FingerprintModel.from_array(roi_array)
                y_origin = getattr(roi, "y_micrometer_original", None)
                x_origin = getattr(roi, "x_micrometer_original", None)
                if y_origin is not None and x_origin is not None:
                    yx_origin = [y_origin, x_origin]
                else:
                    yx_origin = None
                rois_dict[roi.name] = {
                    "slice_repr": str(roi_pixel.slices),
                    "finger_print": fp.model_dump(),
                    "yx_origin": yx_origin,
                }
            tables_dict[table_name] = {"rois": rois_dict}
        entry["tables"] = tables_dict
    return entry


def _updates_by_image(*, zarr_dir: Path, image_list_updates: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for updates in image_list_updates:
        for upd in updates.get("image_list_updates", []):
            rel = Path(upd["zarr_url"]).relative_to(zarr_dir).as_posix()
            result[rel] = upd
    return result


def _build_plate_snapshot(
    *, zarr_dir: Path, image_list_updates: list[dict]
) -> MultiPlateAssertionModel:
    updates = _updates_by_image(
        zarr_dir=zarr_dir, image_list_updates=image_list_updates
    )
    plate_names = sorted(
        {
            Path(upd["zarr_url"]).relative_to(zarr_dir).parts[0]
            for image_updates in image_list_updates
            for upd in image_updates.get("image_list_updates", [])
        }
    )
    all_plates = {}
    for plate_name in plate_names:
        ome_zarr_plate = open_ome_zarr_plate(zarr_dir / plate_name)
        # `max_workers="auto"` opts in to ngio 1.2's default now (the 1.1
        # default, serial reads, emits an `NgioFutureWarning`). Results keep
        # their input order, so snapshots stay deterministic.
        wells = list(ome_zarr_plate.get_wells(max_workers="auto").keys())
        images_dict = {}
        for img_path, ome_zarr_image in ome_zarr_plate.get_images(
            max_workers="auto"
        ).items():
            full_path = f"{plate_name}/{img_path}"
            images_dict[img_path] = _build_image_entry(
                ome_zarr_image=ome_zarr_image, upd=updates.get(full_path)
            )
        all_plates[plate_name] = {
            "wells": wells,
            "images": dict(sorted(images_dict.items())),
        }
    return MultiPlateAssertionModel.model_validate({"plates": all_plates})


def _build_single_image_snapshot(
    *, zarr_dir: Path, image_list_updates: list[dict]
) -> MultiSingleImageAssertionModel:
    updates = _updates_by_image(
        zarr_dir=zarr_dir, image_list_updates=image_list_updates
    )
    images_dict = {}
    for zarr_name in sorted(updates):
        ome_zarr_image = open_ome_zarr_container(zarr_dir / zarr_name)
        images_dict[zarr_name] = _build_image_entry(
            ome_zarr_image=ome_zarr_image, upd=updates[zarr_name]
        )
    return MultiSingleImageAssertionModel.model_validate({"images": images_dict})


def build_snapshot(
    *,
    zarr_dir: Path,
    image_list_updates: list[dict],
    output_type: Literal["plate", "single_image"] = "plate",
) -> MultiPlateAssertionModel | MultiSingleImageAssertionModel:
    """Build a snapshot model from converted OME-Zarr output.

    This is the single source of truth for both snapshot generation and
    validation: the returned model is either written to disk or compared against
    the on-disk expectation. The returned model also carries an informational
    `versions` block (dependency versions at generation time), which is never
    compared.

    Args:
        zarr_dir: Directory the converter wrote its output into.
        image_list_updates: The list returned by the converter's init task.
        output_type: `"plate"` for HCS plate output, `"single_image"` for
            individual containers at the top level of `zarr_dir`.
    """
    if output_type == "plate":
        model: MultiPlateAssertionModel | MultiSingleImageAssertionModel = (
            _build_plate_snapshot(
                zarr_dir=zarr_dir, image_list_updates=image_list_updates
            )
        )
    else:
        model = _build_single_image_snapshot(
            zarr_dir=zarr_dir, image_list_updates=image_list_updates
        )
    model.versions = _collect_versions()
    return model


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def _compare_fingerprint(
    path: str, exp: FingerprintModel, act: FingerprintModel, diffs: list[str]
) -> None:
    for field in ("mean", "std", "min", "max"):
        e, a = getattr(exp, field), getattr(act, field)
        if not math.isclose(e, a, rel_tol=_STAT_REL_TOL, abs_tol=_STAT_ABS_TOL):
            diffs.append(f"{path}.{field}: expected {e} != actual {a}")
    if exp.hash != act.hash:
        diffs.append(f"{path}.hash: expected {exp.hash} != actual {act.hash}")


def _compare_roi(
    path: str, exp: RoiAssertionModel, act: RoiAssertionModel, diffs: list[str]
) -> None:
    if exp.slice_repr != act.slice_repr:
        diffs.append(
            f"{path}.slice_repr: expected {exp.slice_repr!r} "
            f"!= actual {act.slice_repr!r}"
        )
    _compare_fingerprint(
        f"{path}.finger_print", exp.finger_print, act.finger_print, diffs
    )
    # yx_origin is only checked when the snapshot pins it.
    if exp.yx_origin is not None:
        if act.yx_origin is None or not np.allclose(exp.yx_origin, act.yx_origin):
            diffs.append(
                f"{path}.yx_origin: expected {exp.yx_origin} != actual {act.yx_origin}"
            )


def _compare_table(
    path: str,
    exp: TableAssertionModel | None,
    act: TableAssertionModel | None,
    diffs: list[str],
) -> None:
    if exp is None:
        return
    if act is None:
        diffs.append(f"{path}: expected ROIs but the table could not be read")
        return
    exp_rois, act_rois = set(exp.rois), set(act.rois)
    if exp_rois != act_rois:
        diffs.append(
            f"{path}.rois: expected {sorted(exp_rois)} != actual {sorted(act_rois)}"
        )
    for name in sorted(exp_rois & act_rois):
        _compare_roi(f"{path}.rois['{name}']", exp.rois[name], act.rois[name], diffs)


def _compare_image(
    path: str, exp: ImageAssertionModel, act: ImageAssertionModel, diffs: list[str]
) -> None:
    if tuple(exp.axes) != tuple(act.axes):
        diffs.append(
            f"{path}.axes: expected {list(exp.axes)} != actual {list(act.axes)}"
        )
    if tuple(exp.shape) != tuple(act.shape):
        diffs.append(
            f"{path}.shape: expected {list(exp.shape)} != actual {list(act.shape)}"
        )
    if not np.allclose(exp.pixelsize, act.pixelsize):
        diffs.append(
            f"{path}.pixelsize: expected {list(exp.pixelsize)} "
            f"!= actual {list(act.pixelsize)}"
        )
    # Channel metadata is only checked when the snapshot pins it.
    if exp.channel_labels and list(exp.channel_labels) != list(act.channel_labels):
        diffs.append(
            f"{path}.channel_labels: expected {list(exp.channel_labels)} "
            f"!= actual {list(act.channel_labels)}"
        )
    if exp.wavelength_ids and list(exp.wavelength_ids) != list(act.wavelength_ids):
        diffs.append(
            f"{path}.wavelength_ids: expected {list(exp.wavelength_ids)} "
            f"!= actual {list(act.wavelength_ids)}"
        )
    if exp.types != act.types:
        diffs.append(f"{path}.types: expected {exp.types} != actual {act.types}")
    if exp.attributes != act.attributes:
        diffs.append(
            f"{path}.attributes: expected {exp.attributes} != actual {act.attributes}"
        )
    exp_tables, act_tables = set(exp.tables), set(act.tables)
    if exp_tables != act_tables:
        diffs.append(
            f"{path}.tables: expected {sorted(exp_tables)} "
            f"!= actual {sorted(act_tables)}"
        )
    for name in sorted(exp_tables & act_tables):
        _compare_table(
            f"{path}.tables['{name}']", exp.tables[name], act.tables[name], diffs
        )


def compare_snapshots(
    expected: MultiPlateAssertionModel | MultiSingleImageAssertionModel,
    actual: MultiPlateAssertionModel | MultiSingleImageAssertionModel,
) -> list[str]:
    """Compare an expected snapshot against a freshly built one.

    Returns a list of human-readable, fully-pathed difference messages (empty
    when the snapshots match). Floating-point stats use a tolerance; the sha256
    pixel hash and structural fields are compared exactly.

    The `versions` block is deliberately not compared: it is informational only
    (a dependency version drift must never fail a snapshot test).
    """
    diffs: list[str] = []
    if isinstance(expected, MultiPlateAssertionModel) and isinstance(
        actual, MultiPlateAssertionModel
    ):
        exp_plates, act_plates = set(expected.plates), set(actual.plates)
        if exp_plates != act_plates:
            diffs.append(
                f"plates: expected {sorted(exp_plates)} != actual {sorted(act_plates)}"
            )
        for pname in sorted(exp_plates & act_plates):
            exp_plate, act_plate = expected.plates[pname], actual.plates[pname]
            if set(exp_plate.wells) != set(act_plate.wells):
                diffs.append(
                    f"plates['{pname}'].wells: expected {sorted(exp_plate.wells)} "
                    f"!= actual {sorted(act_plate.wells)}"
                )
            exp_imgs, act_imgs = set(exp_plate.images), set(act_plate.images)
            if exp_imgs != act_imgs:
                diffs.append(
                    f"plates['{pname}'].images: expected {sorted(exp_imgs)} "
                    f"!= actual {sorted(act_imgs)}"
                )
            for iname in sorted(exp_imgs & act_imgs):
                _compare_image(
                    f"plates['{pname}'].images['{iname}']",
                    exp_plate.images[iname],
                    act_plate.images[iname],
                    diffs,
                )
    elif isinstance(expected, MultiSingleImageAssertionModel) and isinstance(
        actual, MultiSingleImageAssertionModel
    ):
        exp_imgs, act_imgs = set(expected.images), set(actual.images)
        if exp_imgs != act_imgs:
            diffs.append(
                f"images: expected {sorted(exp_imgs)} != actual {sorted(act_imgs)}"
            )
        for iname in sorted(exp_imgs & act_imgs):
            _compare_image(
                f"images['{iname}']",
                expected.images[iname],
                actual.images[iname],
                diffs,
            )
    else:
        diffs.append(
            f"snapshot type mismatch: expected {type(expected).__name__} "
            f"!= actual {type(actual).__name__}"
        )
    return diffs


# --------------------------------------------------------------------------- #
# Disk I/O + entry point
# --------------------------------------------------------------------------- #


def _write_snapshot(
    model: MultiPlateAssertionModel | MultiSingleImageAssertionModel, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(model.model_dump(mode="json"), f, indent=2, sort_keys=False)
        f.write("\n")


def _load_snapshot(
    path: Path, *, output_type: Literal["plate", "single_image"]
) -> MultiPlateAssertionModel | MultiSingleImageAssertionModel:
    with open(path) as f:
        data = json.load(f)
    if output_type == "plate":
        return MultiPlateAssertionModel(**data)
    return MultiSingleImageAssertionModel(**data)


def run_converter_test(
    *,
    tmp_path: Path,
    api_fn: Callable,
    api_kwargs: dict,
    snapshot_path: Path,
    update_snapshots: bool,
    converter_options: ConverterOptions | None = None,
    output_type: Literal["plate", "single_image"] = "plate",
) -> None:
    """Run a converter end-to-end and check it against a snapshot.

    Runs `api_fn` into a temporary (or snapshot-adjacent) zarr directory and
    builds a snapshot from the output. With `update_snapshots` the snapshot is
    written to `snapshot_path` as JSON; otherwise it is compared against the
    on-disk snapshot and a single `AssertionError` listing every difference is
    raised on mismatch.

    Args:
        tmp_path: Pytest `tmp_path` for zarr output (validation mode).
        api_fn: The high-level converter API function.
        api_kwargs: Keyword arguments for `api_fn` (e.g. `acquisitions`).
        snapshot_path: Path to the snapshot JSON file.
        update_snapshots: If True, (re)generate the snapshot instead of checking.
        converter_options: Options controlling the conversion; passed to `api_fn`
            only when provided.
        output_type: `"plate"` for HCS plate output, `"single_image"` for
            individual zarr containers at the top level of the output directory.
    """
    if update_snapshots:
        zarr_dir = snapshot_path.parent.parent / "output"
        api_kwargs = api_kwargs | {"overwrite": OverwriteMode.OVERWRITE}
    else:
        zarr_dir = tmp_path / "output"
    zarr_dir.mkdir(parents=True, exist_ok=True)

    call_kwargs = dict(api_kwargs)
    if converter_options is not None:
        call_kwargs["converter_options"] = converter_options
    updates_list = api_fn(zarr_dir=str(zarr_dir), **call_kwargs)

    actual = build_snapshot(
        zarr_dir=zarr_dir,
        image_list_updates=updates_list,
        output_type=output_type,
    )

    if update_snapshots:
        _write_snapshot(actual, snapshot_path)
        return

    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot file {snapshot_path} not found. "
            "Run with --update-snapshots to generate it."
        )

    expected = _load_snapshot(snapshot_path, output_type=output_type)
    diffs = compare_snapshots(expected, actual)
    if diffs:
        joined = "\n".join(f"  - {d}" for d in diffs)
        raise AssertionError(
            f"Snapshot mismatch for {snapshot_path.name} "
            f"({len(diffs)} difference(s)):\n{joined}"
        )
