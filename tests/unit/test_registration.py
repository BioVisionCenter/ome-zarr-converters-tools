"""Unit tests for registration module (alignment, tiling, snap utils)."""

import pytest


class TestAlignment:
    """Tests for alignment functions."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_align_xy_regions(self):
        """Test XY region alignment."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_fov_alignment_corrections(self):
        """Test FOV alignment corrections."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_align_to_pixel_grid(self):
        """Test alignment to pixel grid."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_remove_offsets(self):
        """Test offset removal."""


class TestTiling:
    """Tests for tiling functions."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_no_tiling(self):
        """Test no tiling mode."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_snap_to_corners_tiling(self):
        """Test snap to corners tiling."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_snap_to_grid_tiling(self):
        """Test snap to grid tiling."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_auto_tiling(self):
        """Test auto tiling mode selection."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_mosaic_tiling(self):
        """Test mosaic tiling application."""


class TestSnapUtils:
    """Tests for snap utility functions."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_calculate_snap_to_corner_offset(self):
        """Test corner snap offset calculation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_calculate_snap_to_grid_offset(self):
        """Test grid snap offset calculation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_not_a_grid_error(self):
        """Test NotAGridError is raised for non-grid layouts."""


class TestRegistrationPipeline:
    """Tests for registration pipeline."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_add_registration_func(self):
        """Test adding custom registration function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_registration_pipeline(self):
        """Test applying registration pipeline."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_build_default_registration_pipeline(self):
        """Test building default registration pipeline."""
