"""Unit tests for RuntimeSettings."""

import dask
import pytest
import zarr
from pydantic import ValidationError

from ome_zarr_converters_tools import (
    DefaultScheduler,
    ProcessScheduler,
    RuntimeSettings,
    SynchronousScheduler,
    ThreadScheduler,
)
from ome_zarr_converters_tools.models import _runtime_settings


def test_default_is_noop() -> None:
    rs = RuntimeSettings()
    assert rs.use_zarrs_codec is False
    assert isinstance(rs.dask_scheduler, DefaultScheduler)

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


def test_thread_scheduler_apply_sets_and_restores_config() -> None:
    scheduler_before = dask.config.get("scheduler", default=None)
    workers_before = dask.config.get("num_workers", default=None)

    rs = RuntimeSettings(dask_scheduler=ThreadScheduler(num_workers=2))
    with rs.apply():
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 2

    assert dask.config.get("scheduler", default=None) == scheduler_before
    assert dask.config.get("num_workers", default=None) == workers_before


def test_process_scheduler_apply_sets_and_restores_config() -> None:
    scheduler_before = dask.config.get("scheduler", default=None)
    workers_before = dask.config.get("num_workers", default=None)

    rs = RuntimeSettings(dask_scheduler=ProcessScheduler(num_workers=3))
    with rs.apply():
        assert dask.config.get("scheduler") == "processes"
        assert dask.config.get("num_workers") == 3

    assert dask.config.get("scheduler", default=None) == scheduler_before
    assert dask.config.get("num_workers", default=None) == workers_before


def test_synchronous_scheduler_apply() -> None:
    rs = RuntimeSettings(dask_scheduler=SynchronousScheduler())
    with rs.apply():
        assert dask.config.get("scheduler") == "synchronous"


def test_default_scheduler_does_not_modify_dask_config() -> None:
    scheduler_before = dask.config.get("scheduler", default=None)
    rs = RuntimeSettings(dask_scheduler=DefaultScheduler())
    with rs.apply():
        assert dask.config.get("scheduler", default=None) == scheduler_before


def test_synchronous_scheduler_rejects_num_workers() -> None:
    with pytest.raises(ValidationError):
        SynchronousScheduler(num_workers=4)  # type: ignore[call-arg]


def test_default_scheduler_rejects_extras() -> None:
    with pytest.raises(ValidationError):
        DefaultScheduler(num_workers=4)  # type: ignore[call-arg]


def test_thread_scheduler_num_workers_ge_one() -> None:
    with pytest.raises(ValidationError):
        ThreadScheduler(num_workers=0)


def test_process_scheduler_num_workers_ge_one() -> None:
    with pytest.raises(ValidationError):
        ProcessScheduler(num_workers=0)


def test_dict_construction_uses_discriminator() -> None:
    rs = RuntimeSettings(
        dask_scheduler={"type": "Threads", "num_workers": 2},  # type: ignore[arg-type]
    )
    assert isinstance(rs.dask_scheduler, ThreadScheduler)
    assert rs.dask_scheduler.num_workers == 2


def test_invalid_discriminator_value() -> None:
    with pytest.raises(ValidationError):
        RuntimeSettings(
            dask_scheduler={"type": "bogus"},  # type: ignore[arg-type]
        )


def test_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        RuntimeSettings(unknown_field=True)  # type: ignore[call-arg]


def test_runtime_settings_attached_to_converter_options() -> None:
    from ome_zarr_converters_tools import ConverterOptions

    opts = ConverterOptions()
    assert isinstance(opts.runtime_settings, RuntimeSettings)
    assert opts.runtime_settings.use_zarrs_codec is False
    assert isinstance(opts.runtime_settings.dask_scheduler, DefaultScheduler)
