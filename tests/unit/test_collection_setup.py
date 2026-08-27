"""Unit tests for pipelines._collection_setup."""

from pathlib import Path

import polars as pl
import pytest
from ngio.hcs import open_ome_zarr_plate

from ome_zarr_converters_tools.core._dummy_tiles import (
    StartPosition,
    TileShape,
    build_dummy_tile,
)
from ome_zarr_converters_tools.core._tile_to_tiled_images import tiled_image_from_tiles
from ome_zarr_converters_tools.models import (
    AcquisitionDetails,
    ChannelInfo,
    ImageInPlate,
    OverwriteMode,
    SingleImage,
)
from ome_zarr_converters_tools.pipelines._collection_setup import (
    _collection_setup_registry,
    _setup_condition_table,
    add_collection_handler,
    setup_ome_zarr_collection,
    setup_plates,
    setup_singleimage,
)


def _make_plate_tiled_images(
    attributes: dict[str, list] | None = None,
    num_images: int = 1,
) -> list:
    """Build TiledImages with ImageInPlate collections."""
    acq = AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        xy_pixel_size=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )
    all_tiles = []
    for i in range(num_images):
        coll = ImageInPlate(
            plate_name="TestPlate", row="A", column=i + 1, acquisition=0
        )
        tile = build_dummy_tile(
            fov_name=f"FOV_{i}",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=coll,
            acquisition_details=acq,
        )
        if attributes:
            tile.attributes = attributes
        all_tiles.append(tile)

    images = tiled_image_from_tiles(tiles=all_tiles, split_per_fov=False)
    return images


def _make_single_tiled_images(num_images: int = 1) -> list:
    """Build TiledImages with SingleImage collections."""
    acq = AcquisitionDetails(
        channels=[ChannelInfo(channel_label="DAPI")],
        xy_pixel_size=1.0,
        z_spacing=1.0,
        t_spacing=1.0,
    )
    tiles = []
    for i in range(num_images):
        coll = SingleImage(image_path=f"image_{i}")
        tile = build_dummy_tile(
            fov_name=f"FOV_{i}",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=coll,
            acquisition_details=acq,
        )
        tiles.append(tile)
    return tiled_image_from_tiles(tiles=tiles, split_per_fov=False)


class TestSetupConditionTable:
    def test_no_attributes_returns_none(self) -> None:
        images = _make_plate_tiled_images()
        result = _setup_condition_table(images)
        assert result is None

    def test_with_attributes(self) -> None:
        images = _make_plate_tiled_images(attributes={"drug": ["DMSO"], "dose": [1.0]})
        result = _setup_condition_table(images)
        assert result is not None
        assert isinstance(result, pl.DataFrame)
        assert "drug" in result.columns
        assert "dose" in result.columns
        assert "row" in result.columns
        assert "column" in result.columns

    def test_mismatched_attribute_lengths_raises(self) -> None:
        images = _make_plate_tiled_images(
            attributes={"drug": ["DMSO"], "dose": [1.0, 2.0]}
        )
        with pytest.raises(ValueError, match="same number of values"):
            _setup_condition_table(images)

    def test_multiple_images_with_attributes(self) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        tiles = []
        for i, drug in enumerate(["DMSO", "CompA"]):
            coll = ImageInPlate(
                plate_name="TestPlate", row="A", column=i + 1, acquisition=0
            )
            tile = build_dummy_tile(
                fov_name=f"FOV_{i}",
                start=StartPosition(x=0, y=0),
                shape=TileShape(x=64, y=64, z=1, c=1, t=1),
                collection=coll,
                acquisition_details=acq,
            )
            tile.attributes = {"drug": [drug]}
            tiles.append(tile)

        images = tiled_image_from_tiles(tiles=tiles, split_per_fov=False)
        result = _setup_condition_table(images)
        assert result is not None
        assert result.shape[0] == 2  # two rows

    def test_heterogeneous_attribute_keys_raises(self) -> None:
        # Images with different attribute key sets would produce unequal-length
        # columns; a clear error must name the offending images and keys.
        image_a = _make_plate_tiled_images(attributes={"drug": ["DMSO"]})[0]
        image_b = _make_plate_tiled_images(
            attributes={"drug": ["CompA"], "dose": [1.0]}
        )[0]
        with pytest.raises(ValueError, match="same attribute keys"):
            _setup_condition_table([image_a, image_b])

    def test_attribute_key_first_appearing_on_later_image_raises(self) -> None:
        image_a = _make_plate_tiled_images(attributes={"drug": ["DMSO"]})[0]
        image_b = _make_plate_tiled_images(attributes={"dose": [1.0]})[0]
        with pytest.raises(ValueError, match="same attribute keys"):
            _setup_condition_table([image_a, image_b])

    def test_image_without_attributes_is_skipped(self) -> None:
        # An image with no attributes at all contributes no rows but is legal.
        with_attrs = _make_plate_tiled_images(attributes={"drug": ["DMSO"]})[0]
        without_attrs = _make_plate_tiled_images()[0]
        result = _setup_condition_table([with_attrs, without_attrs])
        assert result is not None
        assert result.shape[0] == 1

    def test_reserved_attribute_key_raises(self) -> None:
        images = _make_plate_tiled_images(attributes={"row": ["oops"]})
        with pytest.raises(ValueError, match="reserved columns"):
            _setup_condition_table(images)


class TestSetupSingleimage:
    def test_no_overwrite_succeeds_when_zarr_absent(self, tmp_path: Path) -> None:
        images = _make_single_tiled_images()
        setup_singleimage(
            zarr_dir=str(tmp_path),
            tiled_images=images,
            overwrite_mode=OverwriteMode.NO_OVERWRITE,
        )

    def test_no_overwrite_raises_when_zarr_exists(self, tmp_path: Path) -> None:
        images = _make_single_tiled_images()
        (tmp_path / "image_0.zarr").mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            setup_singleimage(
                zarr_dir=str(tmp_path),
                tiled_images=images,
                overwrite_mode=OverwriteMode.NO_OVERWRITE,
            )

    def test_overwrite_succeeds_when_zarr_exists(self, tmp_path: Path) -> None:
        images = _make_single_tiled_images()
        (tmp_path / "image_0.zarr").mkdir()
        setup_singleimage(
            zarr_dir=str(tmp_path),
            tiled_images=images,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )

    def test_extend_succeeds_when_zarr_exists(self, tmp_path: Path) -> None:
        images = _make_single_tiled_images()
        (tmp_path / "image_0.zarr").mkdir()
        setup_singleimage(
            zarr_dir=str(tmp_path),
            tiled_images=images,
            overwrite_mode=OverwriteMode.EXTEND,
        )

    def test_wrong_collection_type_raises(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images()
        with pytest.raises(ValueError, match="Expected SingleImage"):
            setup_singleimage(
                zarr_dir=str(tmp_path),
                tiled_images=images,
                overwrite_mode=OverwriteMode.NO_OVERWRITE,
            )


class TestCollectionSetupRegistry:
    def test_default_registry_has_image_in_plate(self) -> None:
        assert "ImageInPlate" in _collection_setup_registry

    def test_default_registry_has_single_image(self) -> None:
        assert "SingleImage" in _collection_setup_registry

    def test_add_handler(self) -> None:
        def dummy_handler(
            zarr_dir: str,
            tiled_images: list,
            ngff_version: str = "0.4",
            overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
        ) -> None:
            pass

        dummy_handler.__name__ = "DummyCollection"
        add_collection_handler(
            function=dummy_handler,
            collection_type="DummyCollection",
            overwrite=True,
        )
        assert "DummyCollection" in _collection_setup_registry
        # Clean up
        _collection_setup_registry.pop("DummyCollection")

    def test_add_handler_duplicate_raises(self) -> None:
        def another_handler(
            zarr_dir: str,
            tiled_images: list,
            ngff_version: str = "0.4",
            overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
        ) -> None:
            pass

        another_handler.__name__ = "ImageInPlate"
        with pytest.raises(ValueError, match="already registered"):
            add_collection_handler(
                function=another_handler,
                collection_type="ImageInPlate",
                overwrite=False,
            )

    def test_add_handler_infers_name(self) -> None:
        def my_custom_setup(
            zarr_dir: str,
            tiled_images: list,
            ngff_version: str = "0.4",
            overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
        ) -> None:
            pass

        add_collection_handler(function=my_custom_setup, overwrite=True)
        assert "my_custom_setup" in _collection_setup_registry
        # Clean up
        _collection_setup_registry.pop("my_custom_setup")


class TestSetupPlates:
    def test_creates_plate_with_overwrite(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images()
        zarr_dir = str(tmp_path)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        plate_path = tmp_path / "TestPlate.zarr"
        assert plate_path.exists()
        plate = open_ome_zarr_plate(plate_path)
        assert len(plate.images_paths(max_workers="auto")) == 1

    def test_creates_plate_no_overwrite(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images()
        zarr_dir = str(tmp_path)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images,
            overwrite_mode=OverwriteMode.NO_OVERWRITE,
        )
        plate_path = tmp_path / "TestPlate.zarr"
        assert plate_path.exists()

    def test_extend_mode_adds_images(self, tmp_path: Path) -> None:
        images1 = _make_plate_tiled_images(num_images=1)
        zarr_dir = str(tmp_path)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images1,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        # Extend with a second image in a different well
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        coll = ImageInPlate(plate_name="TestPlate", row="B", column=1, acquisition=0)
        tile = build_dummy_tile(
            fov_name="FOV_ext",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=coll,
            acquisition_details=acq,
        )
        images2 = tiled_image_from_tiles(tiles=[tile], split_per_fov=False)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images2,
            overwrite_mode=OverwriteMode.EXTEND,
        )
        plate = open_ome_zarr_plate(tmp_path / "TestPlate.zarr")
        assert len(plate.images_paths(max_workers="auto")) == 2

    def test_extend_mode_skips_existing_image(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images()
        zarr_dir = str(tmp_path)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        # Extend with the same image — should not duplicate
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images,
            overwrite_mode=OverwriteMode.EXTEND,
        )
        plate = open_ome_zarr_plate(tmp_path / "TestPlate.zarr")
        assert len(plate.images_paths(max_workers="auto")) == 1

    def test_multiple_plates(self, tmp_path: Path) -> None:
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        tiles = []
        for plate_name in ["PlateA", "PlateB"]:
            coll = ImageInPlate(
                plate_name=plate_name,
                row="A",
                column=1,
                acquisition=0,
            )
            tile = build_dummy_tile(
                fov_name="FOV_0",
                start=StartPosition(x=0, y=0),
                shape=TileShape(x=64, y=64, z=1, c=1, t=1),
                collection=coll,
                acquisition_details=acq,
            )
            tiles.append(tile)
        images = tiled_image_from_tiles(tiles=tiles, split_per_fov=False)
        setup_plates(
            zarr_dir=str(tmp_path),
            tiled_images=images,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        assert (tmp_path / "PlateA.zarr").exists()
        assert (tmp_path / "PlateB.zarr").exists()

    def test_no_overwrite_raises_when_plate_exists(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images()
        zarr_dir = str(tmp_path)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            setup_plates(
                zarr_dir=zarr_dir,
                tiled_images=images,
                overwrite_mode=OverwriteMode.NO_OVERWRITE,
            )

    def test_overwrite_replaces_existing_plate(self, tmp_path: Path) -> None:
        # Create plate with image in well A/1
        images_a = _make_plate_tiled_images(num_images=1)
        zarr_dir = str(tmp_path)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images_a,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        # Overwrite with a different image in well B/2
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        coll = ImageInPlate(plate_name="TestPlate", row="B", column=2, acquisition=0)
        tile = build_dummy_tile(
            fov_name="FOV_new",
            start=StartPosition(x=0, y=0),
            shape=TileShape(x=64, y=64, z=1, c=1, t=1),
            collection=coll,
            acquisition_details=acq,
        )
        images_b = tiled_image_from_tiles(tiles=[tile], split_per_fov=False)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images_b,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        plate = open_ome_zarr_plate(tmp_path / "TestPlate.zarr")
        image_paths = plate.images_paths(max_workers="auto")
        # Old plate (well A/1) must be gone; only new plate (well B/2) remains
        assert len(image_paths) == 1
        assert not any("A" in p for p in image_paths), (
            f"Old well A image still present after OVERWRITE: {image_paths}"
        )

    def test_no_overwrite_succeeds_when_plate_absent(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images()
        setup_plates(
            zarr_dir=str(tmp_path),
            tiled_images=images,
            overwrite_mode=OverwriteMode.NO_OVERWRITE,
        )
        assert (tmp_path / "TestPlate.zarr").exists()

    def test_condition_table_isolated_per_plate(self, tmp_path: Path) -> None:
        # Each plate's condition table must contain only that plate's rows, not
        # every plate's rows (regression: the table was built from the full
        # cross-plate list inside the per-image loop).
        acq = AcquisitionDetails(
            channels=[ChannelInfo(channel_label="DAPI")],
            xy_pixel_size=1.0,
            z_spacing=1.0,
            t_spacing=1.0,
        )
        tiles = []
        for plate_name, drug in [("PlateA", "DrugA"), ("PlateB", "DrugB")]:
            coll = ImageInPlate(plate_name=plate_name, row="A", column=1, acquisition=0)
            tile = build_dummy_tile(
                fov_name="FOV_0",
                start=StartPosition(x=0, y=0),
                shape=TileShape(x=64, y=64, z=1, c=1, t=1),
                collection=coll,
                acquisition_details=acq,
            )
            tile.attributes = {"drug": [drug]}
            tiles.append(tile)
        images = tiled_image_from_tiles(tiles=tiles, split_per_fov=False)
        setup_plates(
            zarr_dir=str(tmp_path),
            tiled_images=images,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        for plate_name, drug in [("PlateA", "DrugA"), ("PlateB", "DrugB")]:
            plate = open_ome_zarr_plate(tmp_path / f"{plate_name}.zarr")
            table = plate.get_table("condition_table").dataframe
            assert table.shape[0] == 1
            assert table["drug"].to_list() == [drug]

    def test_writes_condition_table(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images(attributes={"drug": ["DMSO"]})
        zarr_dir = str(tmp_path)
        setup_plates(
            zarr_dir=zarr_dir,
            tiled_images=images,
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        plate = open_ome_zarr_plate(tmp_path / "TestPlate.zarr")
        table_names = plate.list_tables()
        assert "condition_table" in table_names


class TestSetupOmeZarrCollection:
    def test_unknown_collection_type_raises(self) -> None:
        images = _make_plate_tiled_images()
        with pytest.raises(ValueError, match="not registered"):
            setup_ome_zarr_collection(
                tiled_images=images,
                collection_type="UnknownType",
                zarr_dir="/tmp/test",
            )

    def test_dispatches_to_registered_handler(self, tmp_path: Path) -> None:
        images = _make_plate_tiled_images()
        setup_ome_zarr_collection(
            tiled_images=images,
            collection_type="ImageInPlate",
            zarr_dir=str(tmp_path),
            overwrite_mode=OverwriteMode.OVERWRITE,
        )
        plate_path = tmp_path / "TestPlate.zarr"
        assert plate_path.exists()

    def test_dispatches_to_singleimage_handler(self, tmp_path: Path) -> None:
        images = _make_single_tiled_images()
        setup_ome_zarr_collection(
            tiled_images=images,
            collection_type="SingleImage",
            zarr_dir=str(tmp_path),
            overwrite_mode=OverwriteMode.NO_OVERWRITE,
        )
