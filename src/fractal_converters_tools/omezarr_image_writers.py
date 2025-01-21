"""OME-Zarr Image Writers."""

import copy
from pathlib import Path
from typing import Callable

from ngio import NgffImage
from ngio.core.roi import RasterCooROI, WorldCooROI
from ngio.core.utils import create_empty_ome_zarr_image
from ngio.ngff_meta.fractal_image_meta import PixelSize

from fractal_converters_tools.tile import Tile
from fractal_converters_tools.tiled_image import TiledImage


def _find_shape(tiles: list[Tile]) -> tuple[int, int, int, int, int]:
    """Find the shape of the image."""
    shape_x = max(int(tile.bot_r.x) for tile in tiles)
    shape_y = max(int(tile.bot_r.y) for tile in tiles)
    shape_t, shape_c, shape_z, *_ = tiles[0].shape
    return shape_t, shape_c, shape_z, shape_y, shape_x


def _find_chunk_shape(
    tiles: list[Tile],
    max_xy_chunk: int = 4096,
    z_chunk: int = 1,
    c_chunk: int = 1,
    t_chunk: int = 1,
) -> tuple[int, int, int, int, int]:
    shape_t, shape_c, shape_z, shape_y, shape_x = tiles[0].shape
    chunk_y = min(shape_y, max_xy_chunk)
    chunk_x = min(shape_x, max_xy_chunk)
    chunk_z = min(shape_z, z_chunk)
    chunk_c = min(shape_c, c_chunk)
    chunk_t = min(shape_t, t_chunk)
    return chunk_t, chunk_c, chunk_z, chunk_y, chunk_x


def _find_dtype(tiles: list[Tile]) -> str:
    """Find the dtype of the image."""
    return tiles[0].dtype()


def apply_stitching_pipe(
    tiled_image: TiledImage, stiching_pipe: Callable[[list[Tile]], list[Tile]]
) -> list[Tile]:
    """Apply a stitching pipe to a list of tiles."""
    tiles = tiled_image.tiles
    if len(tiles) == 0:
        raise ValueError("No tiles in the TiledImage object.")

    tiles = copy.deepcopy(tiles)
    tiles = stiching_pipe(tiles)

    if len(tiles) != len(tiled_image.tiles):
        # Maybe we should raise a warning here
        raise ValueError("Something went wrong with the stitching pipe.")
    return tiles


def init_empty_ome_zarr_image(
    zarr_dir: Path | str,
    image_path: str,
    tiles: list[Tile],
    pixel_size: PixelSize,
    channel_names: list[str],
    wavelength_ids: list[int],
    overwrite: bool = False,
):
    """Initialize an empty OME-Zarr image."""
    on_disk_axis = ("t", "c", "z", "y", "x")
    on_disk_shape = _find_shape(tiles)
    chunk_shape = _find_chunk_shape(tiles)

    squeeze_t = False if on_disk_shape[0] > 1 else True
    if squeeze_t:
        chunk_shape = chunk_shape[1:]
        on_disk_axis = on_disk_axis[1:]
        on_disk_shape = on_disk_shape[1:]

    if pixel_size is None:
        raise ValueError("Pixel size is not defined in the TiledImage object.")

    tile_dtype = _find_dtype(tiles)

    zarr_dir = Path(zarr_dir)
    zarr_dir.mkdir(parents=True, exist_ok=True)
    new_zarr_url = str(zarr_dir / image_path)

    create_empty_ome_zarr_image(
        store=new_zarr_url,
        on_disk_shape=on_disk_shape,
        on_disk_axis=on_disk_axis,
        chunks=chunk_shape,
        dtype=tile_dtype,
        pixel_sizes=pixel_size,
        channel_labels=channel_names,
        channel_wavelengths=wavelength_ids,
        overwrite=overwrite,
    )
    return new_zarr_url


def write_tiles_as_rois(
    ngff_image: NgffImage, tiles: list[Tile], well_roi: WorldCooROI
):
    """Write the tiles as ROIs in the image."""
    image = ngff_image.get_image()
    pixel_size = image.pixel_size
    squeeze_t = True if "t" not in image.axes_names else False

    # Create the well ROI
    fov_roi_table = ngff_image.tables.new("FOV_ROI_table", table_type="roi_table")
    _fov_rois = []
    for i, tile in enumerate(tiles):
        # Create the ROI for the tile
        roi = RasterCooROI(
            x=int(tile.top_l.x),
            y=int(tile.top_l.y),
            z=int(tile.top_l.z),
            x_length=int(tile.diag.x),
            y_length=int(tile.diag.y),
            z_length=int(tile.diag.z),
            original_roi=well_roi,
        ).to_world_coo_roi(pixel_size=pixel_size)
        roi.infos = {"FieldIndex": f"FOV_{i}", **tile.origin._asdict()}
        _fov_rois.append(roi)

        # Load the whole tile and set the data in the image
        tile_data = tile.load()
        tile_data = tile_data[0] if squeeze_t else tile_data
        image.set_array_from_roi(tile_data, roi)

    # Set order to 0 if the image has the time axis
    order = 1 if squeeze_t else 0
    image.consolidate(order=order)
    ngff_image.update_omero_window(start_percentile=1, end_percentile=99.9)
    fov_roi_table.set_rois(_fov_rois)
    fov_roi_table.consolidate()
    return image


def write_well_roi_table(ngff_image: NgffImage):
    """Write the well ROI table."""
    image = ngff_image.get_image()
    dimensions = image.dimensions
    well_roi_table = ngff_image.tables.new("well_ROI_table", table_type="roi_table")

    well_roi = WorldCooROI(
        x=0,
        y=0,
        z=0,
        x_length=dimensions.get("x") * image.pixel_size.x,
        y_length=dimensions.get("y") * image.pixel_size.y,
        z_length=dimensions.get("z") * image.pixel_size.z,
        unit="micrometer",
        infos={"FieldIndex": "Well"},
    )
    well_roi_table.set_rois([well_roi])
    well_roi_table.consolidate()
    return well_roi


def write_tiled_image(
    zarr_dir: Path | str,
    tiled_image: TiledImage,
    stiching_pipe: Callable[[list[Tile]], list[Tile]],
    overwrite: bool = False,
) -> tuple[str, bool, bool]:
    """Build a tiled ome-zarr image from a TiledImage object."""
    tiles = apply_stitching_pipe(tiled_image, stiching_pipe)
    new_zarr_url = init_empty_ome_zarr_image(
        zarr_dir=zarr_dir,
        image_path=tiled_image.path,
        tiles=tiles,
        pixel_size=tiled_image.pixel_size,
        channel_names=tiled_image.channel_names,
        wavelength_ids=tiled_image.wavelength_ids,
        overwrite=overwrite,
    )
    ngff_image = NgffImage(store=new_zarr_url)
    well_roi = write_well_roi_table(ngff_image)

    # Write the tiles as ROIs in the image
    image = write_tiles_as_rois(ngff_image, tiles, well_roi)
    return new_zarr_url, image.is_3d, image.is_time_series
