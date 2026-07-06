"""Runtime settings applied via scoped context managers during conversion."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.util import find_spec
from typing import Annotated, Literal

import dask
import zarr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ome_zarr_converters_tools.models._url_utils import join_url_paths


class ThreadScheduler(BaseModel):
    """Use Dask's threaded scheduler for parallelism."""

    type: Literal["Threads"] = "Threads"
    """The dask scheduler will be set to "threads" when this scheduler is selected."""

    num_workers: int = Field(default=8, ge=1, title="Number of Threads")
    """Number of worker threads to use. Must be at least 1."""

    model_config = ConfigDict(extra="forbid")

    def get_config(self) -> dict[str, object]:
        return {"scheduler": "threads", "num_workers": self.num_workers}


class ProcessScheduler(BaseModel):
    """Use Dask's multiprocessing scheduler for parallelism."""

    type: Literal["Processes"] = "Processes"
    """The dask scheduler will be set to "processes" when this scheduler is selected."""

    num_workers: int = Field(default=8, ge=1, title="Number of Processes")
    """Number of worker processes to use. Must be at least 1."""

    model_config = ConfigDict(extra="forbid")

    def get_config(self) -> dict[str, object]:
        return {"scheduler": "processes", "num_workers": self.num_workers}


class SynchronousScheduler(BaseModel):
    """Use Dask's synchronous scheduler (no parallelism)."""

    type: Literal["Synchronous"] = "Synchronous"
    """
    The dask scheduler will be set to "synchronous" when this scheduler is selected.
    """

    model_config = ConfigDict(extra="forbid")

    def get_config(self) -> dict[str, object]:
        return {"scheduler": "synchronous"}


class DefaultScheduler(BaseModel):
    """Do not set a Dask scheduler; leave it up to the caller."""

    type: Literal["Default"] = "Default"
    """The dask scheduler will not be modified when this scheduler is selected."""

    model_config = ConfigDict(extra="forbid")

    def get_config(self) -> dict[str, object]:
        return {}


DaskScheduler = Annotated[
    ThreadScheduler | ProcessScheduler | SynchronousScheduler | DefaultScheduler,
    Field(discriminator="type"),
]


class TempJsonOptions(BaseModel):
    """Options for temporary JSON storage during conversion."""

    temp_url: str = "{zarr_dir}/_tmp_json"
    """Template for the temporary JSON URL."""
    serialization: Literal["Auto", "Memory", "JSON"] = "Auto"
    """Serialization mode for tiled image data between init and compute phases.

    - ``"Memory"``: always keep data in-memory (skips all filesystem I/O).
    - ``"JSON"``: always write to a temporary JSON file on disk (required for
      distributed Fractal runs where init and compute execute on different machines).
    - ``"Auto"``: use in-memory when the total serialized payload is ≤50 MB,
      otherwise fall back to JSON files on disk.
    """
    max_in_memory_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
        title="Max In-Memory Bytes",
    )
    """Maximum total size of serialized tiled image data to keep in-memory
    between init and compute phases when serialization="Auto".
    If the total size exceeds this threshold, data will be written to temporary
    JSON files on disk instead. Default is 10 MiB.
    """

    def format_temp_url(self, zarr_dir: str) -> str:
        # Route through join_url_paths (no extra parts) so a zarr_dir with a
        # trailing/duplicate/back-slash is normalized and the protocol preserved.
        return join_url_paths(self.temp_url.format(zarr_dir=zarr_dir))

    def use_in_memory(self, total_bytes: int) -> bool:
        """Resolve whether to skip disk I/O for the given total serialized size."""
        if self.serialization == "Memory":
            return True
        if self.serialization == "JSON":
            return False
        return total_bytes <= self.max_in_memory_bytes


class RuntimeSettings(BaseModel):
    """Runtime knobs applied during conversion via a scoped context manager.

    Defaults are no-ops: callers that don't construct a RuntimeSettings
    explicitly get unchanged behavior.
    """

    use_zarrs_codec: bool = Field(default=False, title="Use Zarrs Codec Pipeline")
    """Use the `zarrs.ZarrsCodecPipeline` Rust codec backend.

    Requires the optional `zarrs` dependency.
    """
    dask_scheduler: DaskScheduler = Field(
        default_factory=DefaultScheduler, title="Dask Scheduler"
    )
    """Dask scheduler to set via `dask.config.set` for the conversion call.
    If set to `DefaultScheduler`, the scheduler will not be modified.
    """
    temp_json_options: TempJsonOptions = Field(
        default_factory=TempJsonOptions, title="Temporary JSON Options"
    )
    """Options for temporary JSON storage."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> "RuntimeSettings":
        if self.use_zarrs_codec and find_spec("zarrs") is None:
            raise ImportError(
                "use_zarrs_codec=True but the 'zarrs' package is not installed. "
                "Install it with: pip install ome-zarr-converters-tools[zarrs]"
            )
        return self

    @contextmanager
    def apply(self) -> Iterator[None]:
        """Apply settings as a scoped context manager.

        Mutates `zarr.config` and/or `dask.config` only for the duration of
        the `with` block. Default-constructed settings produce a no-op
        (no zarr config mutation, and `dask.config.set({})` for the default
        scheduler).
        """
        with ExitStack() as stack:
            if self.use_zarrs_codec:
                stack.enter_context(
                    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
                )

            stack.enter_context(dask.config.set(self.dask_scheduler.get_config()))
            yield
