"""Integration test fixtures for file I/O and zarr operations."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_zarr_path(tmp_path: Path) -> Path:
    """Provide a temporary directory for zarr store operations."""
    zarr_path = tmp_path / "test_store.zarr"
    zarr_path.mkdir()
    return zarr_path


@pytest.fixture
def tmp_plate_path(tmp_path: Path) -> Path:
    """Provide a temporary directory for plate zarr operations."""
    plate_path = tmp_path / "test_plate.zarr"
    plate_path.mkdir()
    return plate_path


@pytest.fixture
def sample_data_path() -> Path:
    """Provide path to the sample test data directory."""
    return Path(__file__).parent.parent / "data" / "hiPSC_Tiny" / "data"
