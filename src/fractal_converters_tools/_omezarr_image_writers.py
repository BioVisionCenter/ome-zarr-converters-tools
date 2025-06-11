"""OME-Zarr Image Writers."""

import copy
from collections.abc import Callable
from pathlib import Path

from ngio import OmeZarrContainer, PixelSize, RoiPixels, create_empty_ome_zarr
from ngio.tables import RoiTable

from fractal_converters_tools._tile import Tile
from fractal_converters_tools._tiled_image import TiledImage


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
    zarr_url: str | Path,
    tiles: list[Tile],
    pixel_size: PixelSize,
    channel_names: list[str] | None,
    wavelength_ids: list[str] | None,
    num_levels: int = 5,
    max_xy_chunk: int = 4096,
    z_chunk: int = 10,
    c_chunk: int = 1,
    t_chunk: int = 1,
    overwrite: bool = False,
) -> OmeZarrContainer:
    """Initialize an empty OME-Zarr image."""
    on_disk_axis = ("t", "c", "z", "y", "x")
    on_disk_shape = _find_shape(tiles)
    chunk_shape = _find_chunk_shape(
        tiles,
        max_xy_chunk=max_xy_chunk,
        z_chunk=z_chunk,
        c_chunk=c_chunk,
        t_chunk=t_chunk,
    )

    # Chunk shape should be smaller or equal to the on disk shape
    chunk_shape = tuple(
        min(c, s) for c, s in zip(chunk_shape, on_disk_shape, strict=True)
    )

    squeeze_t = False if on_disk_shape[0] > 1 else True
    if squeeze_t:
        chunk_shape = chunk_shape[1:]
        on_disk_axis = on_disk_axis[1:]
        on_disk_shape = on_disk_shape[1:]

    if pixel_size is None:
        raise ValueError("Pixel size is not defined in the TiledImage object.")

    tile_dtype = _find_dtype(tiles)

    return create_empty_ome_zarr(
        store=zarr_url,
        shape=on_disk_shape,
        axes_names=on_disk_axis,
        chunks=chunk_shape,
        dtype=tile_dtype,
        xy_pixelsize=pixel_size.x,
        z_spacing=pixel_size.z,
        time_spacing=pixel_size.t,
        channel_labels=channel_names,
        channel_wavelengths=wavelength_ids,
        overwrite=overwrite,
        levels=num_levels,
    )


def write_tiles_as_rois(ome_zarr_container: OmeZarrContainer, tiles: list[Tile]):
    """Write the tiles as ROIs in the image."""
    image = ome_zarr_container.get_image()
    pixel_size = image.pixel_size

    squeeze_t = not ome_zarr_container.is_time_series

    # Create the well ROI
    _fov_rois = []
    for i, tile in enumerate(tiles):
        # Create the ROI for the tile
        # Load the whole tile and set the data in the image
        tile_data = tile.load()
        _, _, s_z, s_y, s_x = tile_data.shape

        tile_data = tile_data[0] if squeeze_t else tile_data
        roi_pix = RoiPixels(
            name=f"FOV_{i}",
            x=int(tile.top_l.x),
            y=int(tile.top_l.y),
            z=int(tile.top_l.z),
            x_length=s_x,
            y_length=s_y,
            z_length=s_z,
            **tile.origin._asdict(),
        )
        roi = roi_pix.to_roi(pixel_size=pixel_size)
        _fov_rois.append(roi)
        image.set_roi(roi=roi, patch=tile_data)

    # Set order to 0 if the image has the time axis
    order = 1 if squeeze_t else 0
    image.consolidate(order=order)
    ome_zarr_container.set_channel_percentiles(start_percentile=1, end_percentile=99.9)
    table = RoiTable(rois=_fov_rois)
    ome_zarr_container.add_table("FOV_ROI_table", table=table)
    return image


def write_tiled_image(
    zarr_url: Path | str,
    tiled_image: TiledImage,
    stiching_pipe: Callable[[list[Tile]], list[Tile]],
    num_levels: int = 5,
    max_xy_chunk: int = 4096,
    z_chunk: int = 10,
    c_chunk: int = 1,
    t_chunk: int = 1,
    overwrite: bool = False,
) -> dict[str, bool]:
    """Build a tiled ome-zarr image from a TiledImage object."""
    tiles = apply_stitching_pipe(tiled_image, stiching_pipe)

    zarr_url = Path(zarr_url)
    zarr_url.mkdir(parents=True, exist_ok=True)

    pixel_size = tiled_image.pixel_size
    if pixel_size is None:
        raise ValueError("Pixel size is not defined in the TiledImage object.")

    ome_zarr_container = init_empty_ome_zarr_image(
        zarr_url=zarr_url,
        tiles=tiles,
        pixel_size=pixel_size,
        channel_names=tiled_image.channel_names,
        wavelength_ids=tiled_image.wavelength_ids,
        num_levels=num_levels,
        max_xy_chunk=max_xy_chunk,
        z_chunk=z_chunk,
        c_chunk=c_chunk,
        t_chunk=t_chunk,
        overwrite=overwrite,
    )
    well_roi = ome_zarr_container.build_image_roi_table("Well")
    ome_zarr_container.add_table("well_ROI_table", table=well_roi)

    # Write the tiles as ROIs in the image
    image = write_tiles_as_rois(ome_zarr_container=ome_zarr_container, tiles=tiles)

    im_list_types = {"is_3D": image.is_3d, "has_time": image.is_time_series}
    return im_list_types
