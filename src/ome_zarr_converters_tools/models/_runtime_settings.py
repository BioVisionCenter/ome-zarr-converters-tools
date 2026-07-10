"""Runtime settings applied via scoped context managers during conversion."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.util import find_spec
from typing import Annotated, Literal

import dask
import zarr
from pydantic import ConfigDict, Field, model_validator

from ome_zarr_converters_tools.models._base import UserFacingModel
from ome_zarr_converters_tools.models._url_utils import join_url_paths


class ThreadScheduler(UserFacingModel):
    """Parallelize the conversion using multiple threads."""

    mode: Literal["Threads"] = "Threads"
    """How the conversion work is parallelized."""

    num_workers: int = Field(default=8, ge=1, title="Number of Threads")
    """Number of worker threads to use. Must be at least 1."""

    model_config = ConfigDict(title="Threads")

    def get_config(self) -> dict[str, object]:
        return {"scheduler": "threads", "num_workers": self.num_workers}


class ProcessScheduler(UserFacingModel):
    """Parallelize the conversion using multiple processes."""

    mode: Literal["Processes"] = "Processes"
    """How the conversion work is parallelized."""

    num_workers: int = Field(default=8, ge=1, title="Number of Processes")
    """Number of worker processes to use. Must be at least 1."""

    model_config = ConfigDict(title="Processes")

    def get_config(self) -> dict[str, object]:
        return {"scheduler": "processes", "num_workers": self.num_workers}


class SynchronousScheduler(UserFacingModel):
    """Run the conversion sequentially, without parallelism."""

    mode: Literal["Synchronous"] = "Synchronous"
    """How the conversion work is parallelized."""

    model_config = ConfigDict(title="Synchronous")

    def get_config(self) -> dict[str, object]:
        return {"scheduler": "synchronous"}


class DefaultScheduler(UserFacingModel):
    """Keep the environment's default parallelism settings unchanged."""

    mode: Literal["Default"] = "Default"
    """How the conversion work is parallelized."""

    model_config = ConfigDict(title="Default")

    def get_config(self) -> dict[str, object]:
        return {}


DaskScheduler = Annotated[
    ThreadScheduler | ProcessScheduler | SynchronousScheduler | DefaultScheduler,
    Field(discriminator="mode"),
]


class TempJsonOptions(UserFacingModel):
    """Where and when intermediate conversion data is stored on disk.

    The data is handed over between the init and compute phases of the task.
    """

    temp_url: str = Field(default="{zarr_dir}/_tmp_json", title="Temporary Storage URL")
    """Where intermediate JSON files are written. The `{zarr_dir}`
    placeholder is replaced by the task's output directory."""
    serialization: Literal["Auto", "Memory", "JSON"] = "Auto"
    """Serialization mode for tiled image data between init and compute phases.

    - `Memory`: always keep data in-memory (skips all filesystem I/O).
    - `JSON`: always write to a temporary JSON file on disk (**required** for
      distributed Fractal runs where init and compute execute on different
      machines).
    - `Auto`: use in-memory when the total serialized payload is within
      `max_in_memory_bytes` (default 10 MiB), otherwise fall back to JSON files
      on disk.
    """
    max_in_memory_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
        title="Max In-Memory Bytes",
    )
    """Maximum total size of serialized tiled image data to keep in-memory
    between init and compute phases when `serialization` is `Auto`.
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


class RuntimeSettings(UserFacingModel):
    """Advanced performance settings; the defaults suit most conversions.

    Covers parallelism, the storage codec, and temporary storage.
    """

    use_zarrs_codec: bool = Field(default=False, title="Use Zarrs Codec Pipeline")
    """Read and write image data with the `zarrs` Rust backend, which is
    usually faster. **Requires** the optional `zarrs` dependency to be
    installed.
    """
    dask_scheduler: DaskScheduler = Field(
        default_factory=DefaultScheduler, title="Dask Scheduler"
    )
    """
    How the conversion work is parallelized.

    - `Threads`: parallelize using multiple threads.
    - `Processes`: parallelize using multiple processes.
    - `Synchronous`: run sequentially, without parallelism.
    - `Default`: keep the environment's parallelism settings unchanged.
    """
    temp_json_options: TempJsonOptions = Field(
        default_factory=TempJsonOptions, title="Temporary JSON Options"
    )
    """Where and when intermediate conversion data is stored on disk."""

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
