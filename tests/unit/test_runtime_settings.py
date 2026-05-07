"""Unit tests for RuntimeSettings."""

import dask
import pytest
import zarr
from pydantic import ValidationError

from ome_zarr_converters_tools import RuntimeSettings
from ome_zarr_converters_tools.models import _runtime_settings


def test_default_is_noop() -> None:
    rs = RuntimeSettings()
    assert rs.use_zarrs_codec is False
    assert rs.dask_scheduler is None
    assert rs.dask_num_workers is None

    scheduler_before = dask.config.get("scheduler", default=None)
    codec_path_before = zarr.config.get("codec_pipeline.path", default=None)
    with rs.apply():
        assert dask.config.get("scheduler", default=None) == scheduler_before
        assert zarr.config.get("codec_pipeline.path", default=None) == codec_path_before


def test_zarrs_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_find_spec(name: str) -> None:
        return None

    monkeypatch.setattr(_runtime_settings, "find_spec", _fake_find_spec)
    with pytest.raises(ImportError, match=r"\[zarrs\]"):
        RuntimeSettings(use_zarrs_codec=True)


def test_dask_num_workers_requires_scheduler() -> None:
    with pytest.raises(ValidationError, match="dask_num_workers"):
        RuntimeSettings(dask_num_workers=4)
    with pytest.raises(ValidationError, match="dask_num_workers"):
        RuntimeSettings(dask_scheduler="synchronous", dask_num_workers=4)


def test_dask_apply_sets_and_restores_config() -> None:
    scheduler_before = dask.config.get("scheduler", default=None)
    workers_before = dask.config.get("num_workers", default=None)

    rs = RuntimeSettings(dask_scheduler="threads", dask_num_workers=2)
    with rs.apply():
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 2

    assert dask.config.get("scheduler", default=None) == scheduler_before
    assert dask.config.get("num_workers", default=None) == workers_before


def test_dask_synchronous_no_workers() -> None:
    rs = RuntimeSettings(dask_scheduler="synchronous")
    with rs.apply():
        assert dask.config.get("scheduler") == "synchronous"


def test_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        RuntimeSettings(unknown_field=True)  # type: ignore[call-arg]


def test_dask_num_workers_ge_one() -> None:
    with pytest.raises(ValidationError):
        RuntimeSettings(dask_scheduler="threads", dask_num_workers=0)


def test_runtime_settings_attached_to_converter_options() -> None:
    from ome_zarr_converters_tools import ConverterOptions

    opts = ConverterOptions()
    assert isinstance(opts.runtime_settings, RuntimeSettings)
    assert opts.runtime_settings.use_zarrs_codec is False
    assert opts.runtime_settings.dask_scheduler is None
