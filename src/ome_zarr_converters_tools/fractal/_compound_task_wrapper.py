"""Execution runners for Fractal compound tasks."""

import concurrent.futures
from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ome_zarr_converters_tools.fractal._compute_task import ImageListUpdateDict


class SequentialRunner(BaseModel):
    """Runner for executing compound tasks sequentially."""

    mode: Literal["Synchronous"] = "Synchronous"


class ThreadedRunner(BaseModel):
    """Runner for executing compound tasks using threads."""

    mode: Literal["Threaded"] = "Threaded"
    num_threads: Annotated[int, Field(gt=0)] = 4


class MultiprocessingRunner(BaseModel):
    """Runner for executing compound tasks using multiprocessing."""

    mode: Literal["Multiprocessing"] = "Multiprocessing"
    num_processes: Annotated[int, Field(gt=0)] = 4


RunnerType = Annotated[
    SequentialRunner | ThreadedRunner | MultiprocessingRunner,
    Field(discriminator="mode"),
]


def _run_single_compute_task(
    fn: Callable[..., ImageListUpdateDict],
    extra_kwargs: dict,
    parallelization_kwargs: dict,
) -> ImageListUpdateDict:
    return fn(**parallelization_kwargs, **extra_kwargs)


def _exec_compound_task_synchronous(
    init_task_fn: Callable[..., dict],
    compute_task_fn: Callable[..., ImageListUpdateDict],
    init_task_kwargs: dict,
    compute_task_kwargs: dict | None = None,
) -> list[ImageListUpdateDict]:
    init_output = init_task_fn(**init_task_kwargs)
    extra_kwargs = compute_task_kwargs or {}
    return [
        compute_task_fn(**p_kwargs, **extra_kwargs)
        for p_kwargs in init_output["parallelization_list"]
    ]


def _exec_compound_task_threaded(
    init_task_fn: Callable[..., dict],
    compute_task_fn: Callable[..., ImageListUpdateDict],
    init_task_kwargs: dict,
    compute_task_kwargs: dict | None = None,
    num_threads: int = 4,
) -> list[ImageListUpdateDict]:
    init_output = init_task_fn(**init_task_kwargs)
    extra_kwargs = compute_task_kwargs or {}
    task = partial(_run_single_compute_task, compute_task_fn, extra_kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        return list(executor.map(task, init_output["parallelization_list"]))


def _exec_compound_task_multiprocessing(
    init_task_fn: Callable[..., dict],
    compute_task_fn: Callable[..., ImageListUpdateDict],
    init_task_kwargs: dict,
    compute_task_kwargs: dict | None = None,
    num_processes: int = 4,
) -> list[ImageListUpdateDict]:
    init_output = init_task_fn(**init_task_kwargs)
    extra_kwargs = compute_task_kwargs or {}
    task = partial(_run_single_compute_task, compute_task_fn, extra_kwargs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        return list(executor.map(task, init_output["parallelization_list"]))


def exec_compound_task(
    init_task_fn: Callable[..., dict],
    compute_task_fn: Callable[..., ImageListUpdateDict],
    init_task_kwargs: dict,
    compute_task_kwargs: dict | None = None,
    runner: RunnerType | None = None,
) -> list[ImageListUpdateDict]:
    """Execute a Fractal compound task.

    Runs `init_task_fn` once to obtain a `parallelization_list`, then calls
    `compute_task_fn` for each item, merging `compute_task_kwargs` into each
    call. Results are returned in the same order as `parallelization_list`.

    Args:
        init_task_fn: Returns a dict with a `"parallelization_list"` key.
        compute_task_fn: Invoked once per parallelization item. When using
            `MultiprocessingRunner`, this must be a picklable module-level
            function.
        init_task_kwargs: Keyword arguments forwarded to `init_task_fn`.
        compute_task_kwargs: Additional kwargs merged into each compute call.
        runner: Execution strategy. Defaults to sequential when `None`.

    Returns:
        List of `ImageListUpdateDict` in `parallelization_list` order.
    """
    match runner:
        case None | SequentialRunner():
            return _exec_compound_task_synchronous(
                init_task_fn=init_task_fn,
                compute_task_fn=compute_task_fn,
                init_task_kwargs=init_task_kwargs,
                compute_task_kwargs=compute_task_kwargs,
            )
        case ThreadedRunner(num_threads=n):
            return _exec_compound_task_threaded(
                init_task_fn=init_task_fn,
                compute_task_fn=compute_task_fn,
                init_task_kwargs=init_task_kwargs,
                compute_task_kwargs=compute_task_kwargs,
                num_threads=n,
            )
        case MultiprocessingRunner(num_processes=n):
            return _exec_compound_task_multiprocessing(
                init_task_fn=init_task_fn,
                compute_task_fn=compute_task_fn,
                init_task_kwargs=init_task_kwargs,
                compute_task_kwargs=compute_task_kwargs,
                num_processes=n,
            )
        case _:
            raise ValueError(
                f"Unsupported runner {runner!r}. Use one of SequentialRunner, "
                "ThreadedRunner, MultiprocessingRunner, or None for sequential "
                "execution."
            )
