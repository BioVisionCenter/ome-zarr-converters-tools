"""Unit tests for validator pipeline."""

import pytest

from ome_zarr_converters_tools.pipelines import (
    ValidatorStep,
    add_validator,
    apply_validator_pipeline,
)


class TestValidatorStep:
    """Tests for ValidatorStep model."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_validator_step_creation(self):
        """Test ValidatorStep creation."""


class TestValidatorRegistry:
    """Tests for validator registry operations."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_add_validator(self):
        """Test adding a custom validator."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_add_validator_overwrite(self):
        """Test overwriting an existing validator."""


class TestValidatorPipeline:
    """Tests for validator pipeline execution."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_validator_pipeline_empty(self):
        """Test applying empty validator pipeline."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_validator_pipeline(self):
        """Test applying validator pipeline."""
