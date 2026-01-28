"""Unit tests for core module (Tile, TiledImage, TileSlice)."""

import pytest


class TestTile:
    """Tests for the Tile class."""

    @pytest.mark.skip(reason="Requires complex fixture setup")
    def test_tile_creation(self):
        """Test basic tile creation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tile_roi_properties(self):
        """Test tile ROI property access."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tile_collection_path(self):
        """Test tile collection path generation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tile_to_roi(self):
        """Test tile to ROI conversion."""


class TestTileSlice:
    """Tests for the TileSlice class."""

    @pytest.mark.skip(reason="Requires complex fixture setup")
    def test_tile_slice_creation(self):
        """Test basic tile slice creation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tile_slice_roi_access(self):
        """Test tile slice ROI access."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tile_slice_load_data(self):
        """Test loading data from tile slice."""


class TestTiledImage:
    """Tests for the TiledImage class."""

    @pytest.mark.skip(reason="Requires complex fixture setup")
    def test_tiled_image_creation(self):
        """Test basic tiled image creation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tiled_image_group_by_fov(self):
        """Test grouping regions by field of view."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tiled_image_pixel_size(self):
        """Test pixel size property."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_tiled_image_from_tiles(self):
        """Test creating TiledImage from list of tiles."""
