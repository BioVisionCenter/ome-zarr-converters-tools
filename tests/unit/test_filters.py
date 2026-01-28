"""Unit tests for filter pipeline."""

import pytest

from ome_zarr_converters_tools.filters import (
    FilterModel,
    ImplementedFilters,
    add_filter,
    apply_filter_pipeline,
)
from ome_zarr_converters_tools.filters._filter_pipeline import (
    RegexExcludeFilter,
    RegexIncludeFilter,
    WellFilter,
)


class TestFilterModels:
    """Tests for filter model classes."""

    def test_regex_include_filter_creation(self):
        """Test RegexIncludeFilter creation."""
        filter_model = RegexIncludeFilter(regex=".*test.*")
        assert filter_model.name == "Path Regex Include Filter"
        assert filter_model.regex == ".*test.*"

    def test_regex_exclude_filter_creation(self):
        """Test RegexExcludeFilter creation."""
        filter_model = RegexExcludeFilter(regex=".*exclude.*")
        assert filter_model.name == "Path Regex Exclude Filter"
        assert filter_model.regex == ".*exclude.*"

    def test_well_filter_creation(self):
        """Test WellFilter creation."""
        filter_model = WellFilter(wells_to_remove=["A1", "B2"])
        assert filter_model.name == "Well Filter"
        assert filter_model.wells_to_remove == ["A1", "B2"]


class TestFilterRegistry:
    """Tests for filter registry operations."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_add_filter(self):
        """Test adding a custom filter to registry."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_add_filter_overwrite(self):
        """Test overwriting an existing filter."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_add_filter_duplicate_error(self):
        """Test error when adding duplicate filter without overwrite."""


class TestFilterPipeline:
    """Tests for filter pipeline execution."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_filter_pipeline_empty(self):
        """Test applying empty filter pipeline."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_regex_include_filter(self):
        """Test applying regex include filter."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_regex_exclude_filter(self):
        """Test applying regex exclude filter."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_well_filter(self):
        """Test applying well filter."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_apply_multiple_filters(self):
        """Test applying multiple filters in sequence."""
