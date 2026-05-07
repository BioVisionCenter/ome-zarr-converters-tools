"""Runtime settings applied via scoped context managers during conversion."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.util import find_spec
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DaskScheduler = Literal["threads", "synchronous", "processes"]


class RuntimeSettings(BaseModel):
    """Runtime knobs applied during conversion via a scoped context manager.

    Defaults are no-ops: callers that don't construct a RuntimeSettings
    explicitly get unchanged behavior.
    """

    use_zarrs_codec: bool = Field(default=False, title="Use Zarrs Codec Pipeline")
    """Opt into the `zarrs.ZarrsCodecPipeline` Rust codec backend.

    Requires the optional `zarrs` extra:
    `pip install ome-zarr-converters-tools[zarrs]`.
    """
    dask_scheduler: DaskScheduler | None = Field(default=None, title="Dask Scheduler")
    """Dask scheduler to set via `dask.config.set` for the conversion call.

    `None` leaves dask config untouched (so callers managing their own
    scheduler/Client are not disrupted).
    """
    dask_num_workers: int | None = Field(default=None, ge=1, title="Dask Num Workers")
    """Worker count to set alongside the dask scheduler.

    Only meaningful for `"threads"` and `"processes"`; must be `None` when
    `dask_scheduler` is `None` or `"synchronous"`.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> "RuntimeSettings":
        if self.use_zarrs_codec and find_spec("zarrs") is None:
            raise ImportError(
                "use_zarrs_codec=True but the 'zarrs' package is not installed. "
                "Install it with: pip install ome-zarr-converters-tools[zarrs]"
            )
        if self.dask_num_workers is not None and self.dask_scheduler in (
            None,
            "synchronous",
        ):
            raise ValueError(
                "dask_num_workers is only meaningful when dask_scheduler is "
                "'threads' or 'processes'; got "
                f"dask_scheduler={self.dask_scheduler!r}."
            )
        return self

    @contextmanager
    def apply(self) -> Iterator[None]:
        """Apply settings as a scoped context manager.

        Mutates `zarr.config` and/or `dask.config` only for the duration of
        the `with` block. Default-constructed settings produce a no-op
        (no imports, no config mutations).
        """
        with ExitStack() as stack:
            if self.use_zarrs_codec:
                import zarr

                stack.enter_context(
                    zarr.config.set(
                        {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
                    )
                )
            if self.dask_scheduler is not None:
                import dask

                cfg: dict[str, object] = {"scheduler": self.dask_scheduler}
                if self.dask_num_workers is not None:
                    cfg["num_workers"] = self.dask_num_workers
                stack.enter_context(dask.config.set(cfg))
            yield
