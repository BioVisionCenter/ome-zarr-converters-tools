"""Unit tests for fractal._compound_task_wrapper."""

import pytest

from ome_zarr_converters_tools.fractal._compound_task_wrapper import (
    MultiprocessingRunner,
    RunnerType,
    SequentialRunner,
    ThreadedRunner,
    exec_compound_task,
)


def _make_init_fn(n: int):
    """Return an init function that yields n items."""

    def init_fn(**kwargs):
        return {"parallelization_list": [{"index": i, **kwargs} for i in range(n)]}

    return init_fn


def _compute_fn(index: int, extra: str = "", **kwargs) -> dict:
    return {"image_list_updates": [{"zarr_url": f"/zarr/{index}", "extra": extra}]}


class TestRunnerModels:
    def test_sequential_runner_defaults(self) -> None:
        r = SequentialRunner()
        assert r.mode == "Synchronous"

    def test_threaded_runner_defaults(self) -> None:
        r = ThreadedRunner()
        assert r.mode == "Threaded"
        assert r.num_threads == 4

    def test_multiprocessing_runner_defaults(self) -> None:
        r = MultiprocessingRunner()
        assert r.mode == "Multiprocessing"
        assert r.num_processes == 4

    def test_threaded_runner_invalid_num_threads(self) -> None:
        with pytest.raises(ValueError):
            ThreadedRunner(num_threads=0)

    def test_multiprocessing_runner_invalid_num_processes(self) -> None:
        with pytest.raises(ValueError):
            MultiprocessingRunner(num_processes=-1)

    def test_runner_type_discriminated_union(self) -> None:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(RunnerType)
        r = adapter.validate_python({"mode": "Threaded", "num_threads": 2})
        assert isinstance(r, ThreadedRunner)
        assert r.num_threads == 2


class TestExecCompoundTaskSequential:
    def test_none_runner_runs_sequentially(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(3),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
        )
        assert len(results) == 3

    def test_sequential_runner_explicit(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(2),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            runner=SequentialRunner(),
        )
        assert len(results) == 2

    def test_result_order_preserved(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(5),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
        )
        urls = [r["image_list_updates"][0]["zarr_url"] for r in results]
        assert urls == [f"/zarr/{i}" for i in range(5)]

    def test_compute_task_kwargs_merged(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(2),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            compute_task_kwargs={"extra": "hello"},
        )
        for r in results:
            assert r["image_list_updates"][0]["extra"] == "hello"

    def test_empty_parallelization_list(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(0),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
        )
        assert results == []

    def test_init_kwargs_forwarded(self) -> None:
        captured = {}

        def init_fn(**kwargs):
            captured.update(kwargs)
            return {"parallelization_list": [{"index": 0}]}

        exec_compound_task(
            init_task_fn=init_fn,
            compute_task_fn=_compute_fn,
            init_task_kwargs={"foo": "bar"},
        )
        assert captured == {"foo": "bar"}


class TestExecCompoundTaskThreaded:
    def test_threaded_runner_returns_correct_count(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(4),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            runner=ThreadedRunner(num_threads=2),
        )
        assert len(results) == 4

    def test_threaded_result_order_preserved(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(6),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            runner=ThreadedRunner(num_threads=3),
        )
        urls = [r["image_list_updates"][0]["zarr_url"] for r in results]
        assert urls == [f"/zarr/{i}" for i in range(6)]

    def test_threaded_compute_task_kwargs_merged(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(3),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            compute_task_kwargs={"extra": "threaded"},
            runner=ThreadedRunner(num_threads=2),
        )
        for r in results:
            assert r["image_list_updates"][0]["extra"] == "threaded"

    def test_threaded_empty_list(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(0),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            runner=ThreadedRunner(),
        )
        assert results == []


class TestExecCompoundTaskMultiprocessing:
    def test_multiprocessing_runner_returns_correct_count(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(4),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            runner=MultiprocessingRunner(num_processes=2),
        )
        assert len(results) == 4

    def test_multiprocessing_result_order_preserved(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(6),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            runner=MultiprocessingRunner(num_processes=2),
        )
        urls = [r["image_list_updates"][0]["zarr_url"] for r in results]
        assert urls == [f"/zarr/{i}" for i in range(6)]

    def test_multiprocessing_compute_task_kwargs_merged(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(3),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            compute_task_kwargs={"extra": "mp"},
            runner=MultiprocessingRunner(num_processes=2),
        )
        for r in results:
            assert r["image_list_updates"][0]["extra"] == "mp"

    def test_multiprocessing_empty_list(self) -> None:
        results = exec_compound_task(
            init_task_fn=_make_init_fn(0),
            compute_task_fn=_compute_fn,
            init_task_kwargs={},
            runner=MultiprocessingRunner(),
        )
        assert results == []
