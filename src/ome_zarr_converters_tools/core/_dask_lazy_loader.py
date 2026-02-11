"""Build a lazy dask array by compositing overlapping loader regions.

Problem
-------
We have a large N-dimensional output array and many *loader* callables, each
producing a sub-array that covers a rectangular slice of the output.  Loaders
may overlap (last writer wins) and may not cover the entire output (uncovered
areas are filled with *fill_value*).  The result must be a lazy ``dask.array``
where each loader is called **at most once** during compute, even when its
region spans multiple output chunks.

Design decisions
----------------
1. **Direct graph construction with HighLevelGraph**
   Instead of using ``dask.delayed`` + ``da.from_delayed`` + ``da.block``
   (which creates and merges one mini-graph per chunk — expensive at scale),
   we build the task graph dict directly and wrap it in a ``HighLevelGraph``.
   This is a single O(C + L) pass that produces a flat dict, avoiding the
   quadratic merge cost of ``da.block`` with many chunks.

   ``HighLevelGraph`` (rather than a raw dict) is used so that dask's
   internal optimisation passes and schedulers can reason about layer
   dependencies.

2. **Loader deduplication via graph keys**
   Each loader gets exactly one graph key ``(loader_layer_name, i)``.
   Output chunk tasks reference these keys; the scheduler ensures each key
   is computed once regardless of how many chunks depend on it.

3. **Inverted overlap index — O(L*k) instead of O(C*L)**
   A naive approach iterates all L loaders for every chunk (O(C*L)).
   With 100k loaders and 100k chunks that's 10^10 checks.

   Instead we invert the loop: for each loader, use ``bisect`` on the
   sorted chunk-start coordinates to find the (small) set of chunks it
   overlaps, then record that mapping.  Cost: O(L*k) where k is the
   average number of chunks a loader spans (typically 1-4 for aligned tiles).

4. **Flat loader args — no intermediate graph keys**
   Dask resolves graph keys that appear as *positional arguments* in a task
   tuple, but does **not** recurse into nested lists.  Earlier versions
   worked around this by creating ``pair-`` and ``pairs-`` helper keys to
   assemble ``(array, bounds)`` tuples at compute time — doubling the graph
   size.

   Here we flatten loader references directly into the task tuple::

       (
           func,
           chunk_bounds,
           dtype,
           fill,
           loader_key_0,
           bounds_0,
           loader_key_1,
           bounds_1,
           ...,
       )

   Dask resolves the ``loader_key_*`` entries (they are graph keys) and
   passes the ``bounds_*`` entries through as-is (they are plain tuples of
   ints, not graph keys).  This halves the number of graph entries for chunks
   with loaders.

5. **Graph-level fast paths**
   - *Single loader, full coverage*: the chunk task is just
     ``(operator.getitem, loader_key, slices)`` — no composite function
     call, no fill allocation.  For well-tiled data this is the majority of
     chunks.
   - *No loaders*: a shared ``np.full`` fill key is reused across all empty
     chunks with the same shape, avoiding redundant graph entries.
"""

import itertools
import operator
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable

import dask.array as da
import numpy as np
from dask import base as dask_base
from dask.highlevelgraph import HighLevelGraph


def _normalize_slice(s: slice, dim_size: int) -> tuple[int, int]:
    """Convert a ``slice`` to an explicit ``(start, stop)`` pair."""
    start, stop, _ = s.indices(dim_size)
    return start, stop


def _build_chunk_ranges(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> list[list[tuple[int, int]]]:
    """Return per-axis lists of ``(start, stop)`` for every chunk."""
    ranges = []
    for dim_size, chunk_size in zip(shape, chunks, strict=True):
        axis_ranges = []
        for start in range(0, dim_size, chunk_size):
            axis_ranges.append((start, min(start + chunk_size, dim_size)))
        ranges.append(axis_ranges)
    return ranges


def _build_chunk_to_loaders(
    chunk_ranges: list[list[tuple[int, int]]],
    all_loader_bounds: list[tuple[tuple[int, int], ...]],
) -> dict[tuple[int, ...], list[int]]:
    """Map each chunk index to the loader indices that overlap it.

    Uses an *inverted* approach: iterate over loaders (not chunks) and use
    ``bisect`` on the sorted chunk-start coordinates to find the chunk index
    range each loader spans per axis.  The Cartesian product of those per-axis
    ranges gives the set of chunks the loader overlaps.

    Complexity: O(L * k) where L = number of loaders and k = average number
    of chunks per loader (typically 1-4), instead of O(C * L) for the naive
    "scan all loaders per chunk" approach.
    """
    ndim = len(chunk_ranges)

    # Sorted chunk-start coordinates per axis, used for bisect lookups.
    chunk_starts: list[list[int]] = [
        [start for start, _ in axis_ranges] for axis_ranges in chunk_ranges
    ]
    n_chunks: list[int] = [len(axis_ranges) for axis_ranges in chunk_ranges]

    chunk_to_loaders: dict[tuple[int, ...], list[int]] = defaultdict(list)

    for li, lb in enumerate(all_loader_bounds):
        chunk_idx_ranges = []
        for ax in range(ndim):
            lo, hi = lb[ax]
            # bisect_right(starts, lo) - 1  →  first chunk that could contain lo
            # bisect_left(starts, hi)       →  first chunk whose start >= hi (exclusive)
            first = bisect_right(chunk_starts[ax], lo) - 1
            last = bisect_left(chunk_starts[ax], hi)
            chunk_idx_ranges.append(range(max(0, first), min(last, n_chunks[ax])))

        for chunk_idx in itertools.product(*chunk_idx_ranges):
            chunk_to_loaders[chunk_idx].append(li)

    return chunk_to_loaders


def _composite_chunk(
    chunk_bounds: tuple[tuple[int, int], ...],
    dtype: np.dtype,
    fill_value: float,
    *loader_args: np.ndarray | tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Composite multiple pre-loaded arrays into a single output chunk.

    Loader arguments are passed as a flat sequence of alternating
    ``(array, bounds, array, bounds, ...)`` pairs via ``*loader_args``.
    This flat layout avoids the need for intermediate graph keys to assemble
    ``(array, bounds)`` tuples — dask resolves graph-key positional args but
    passes plain-data args through as-is.

    Compositing uses a last-writer-wins policy: loaders that appear later in
    the *regions* list overwrite earlier ones in overlapping areas.
    """
    chunk_shape = tuple(stop - start for start, stop in chunk_bounds)
    n_loaders = len(loader_args) // 2

    if n_loaders == 0:
        return np.full(chunk_shape, fill_value, dtype=dtype)

    out = np.full(chunk_shape, fill_value, dtype=dtype)
    for i in range(n_loaders):
        data = loader_args[2 * i]
        lb = loader_args[2 * i + 1]
        # Intersection of the chunk and loader regions in global coordinates.
        inter = tuple(
            (max(cb[0], lb_ax[0]), min(cb[1], lb_ax[1]))
            for cb, lb_ax in zip(chunk_bounds, lb, strict=True)
        )
        # Map intersection back to local coordinates of the loader array …
        src_slices = tuple(
            slice(i_ax[0] - lb_ax[0], i_ax[1] - lb_ax[0])
            for i_ax, lb_ax in zip(inter, lb, strict=True)
        )
        # … and to local coordinates of the output chunk.
        dst_slices = tuple(
            slice(i_ax[0] - cb[0], i_ax[1] - cb[0])
            for i_ax, cb in zip(inter, chunk_bounds, strict=True)
        )
        # Last-writer-wins assignment.
        out[dst_slices] = data[src_slices]  # type: ignore
    return out


def lazy_array_from_regions(
    regions: list[tuple[tuple[slice, ...], Callable[[], np.ndarray]]],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str = "uint16",
    fill_value: float = 0,
) -> da.Array:
    """Build a lazy dask array from overlapping (slices, loader) regions.

    Each loader callable is invoked **at most once** during compute.
    Overlapping regions are composited with a last-writer-wins policy: for
    regions that cover the same output pixels, the region appearing later in
    *regions* takes precedence.

    Args:
        regions: Each entry is ``(per_axis_slices, loader)`` where *loader* is
            a zero-argument callable returning an ``np.ndarray``.
        shape: Output array shape.
        chunks: Chunk size per axis.
        dtype: Output dtype.
        fill_value: Value used for areas not covered by any loader.

    Returns:
        A lazy ``dask.array.Array``.
    """
    ndim = len(shape)
    if len(chunks) != ndim:
        raise ValueError(
            f"chunks length ({len(chunks)}) must match shape length ({ndim})"
        )

    # Convert per-region slices to explicit (start, stop) bounds.
    all_loader_bounds: list[tuple[tuple[int, int], ...]] = [
        tuple(_normalize_slice(s, shape[ax]) for ax, s in enumerate(slices))
        for slices, _ in regions
    ]

    chunk_ranges = _build_chunk_ranges(shape, chunks)
    chunks_normalized = tuple(
        tuple(stop - start for start, stop in axis_ranges)
        for axis_ranges in chunk_ranges
    )

    # Deterministic token — same inputs produce the same graph keys, enabling
    # dask-level caching.  (Earlier versions used ``tokenize(id(...))`` which
    # changed every run.)
    token = dask_base.tokenize(shape, chunks, dtype, fill_value, all_loader_bounds)
    output_name = f"lazy-regions-{token}"

    # --- Inverted overlap index -----------------------------------------
    # O(L*k) instead of O(C*L).  See ``_build_chunk_to_loaders`` docstring.
    chunk_to_loaders = _build_chunk_to_loaders(chunk_ranges, all_loader_bounds)

    # --- Loader layer ---------------------------------------------------
    # One graph key per loader.  Multiple output chunks that depend on the
    # same loader reference this key; the scheduler computes it only once.
    loader_layer_name = f"loader-{token}"
    loader_layer: dict = {}
    for i, (_, loader) in enumerate(regions):
        loader_layer[(loader_layer_name, i)] = (loader,)

    # --- Output layer ---------------------------------------------------
    # One graph key per output chunk.  Three cases, from cheapest to most
    # expensive:
    output_layer: dict = {}
    for chunk_idx in itertools.product(*(range(len(cr)) for cr in chunk_ranges)):
        chunk_bounds = tuple(chunk_ranges[ax][idx] for ax, idx in enumerate(chunk_idx))
        chunk_shape = tuple(stop - start for start, stop in chunk_bounds)
        loader_indices = chunk_to_loaders.get(chunk_idx, [])

        out_key = (output_name, *chunk_idx)

        if not loader_indices:
            # Case 1 — No loaders overlap this chunk.
            # Emit a shared np.full task keyed by chunk shape, so all empty
            # chunks of the same size reuse a single graph entry.
            fill_key = (f"fill-{token}", chunk_shape)
            if fill_key not in output_layer:
                output_layer[fill_key] = (np.full, chunk_shape, fill_value, dtype)
            output_layer[out_key] = fill_key

        elif len(loader_indices) == 1:
            li = loader_indices[0]
            lb = all_loader_bounds[li]
            fully_covers = all(
                lb_ax[0] <= cb[0] and cb[1] <= lb_ax[1]
                for cb, lb_ax in zip(chunk_bounds, lb, strict=True)
            )
            if fully_covers:
                # Case 2a — Single loader fully covers the chunk.
                # Emit a direct getitem slice from the loader's output.
                # No _composite_chunk call, no fill allocation — just a view.
                src_slices = tuple(
                    slice(cb[0] - lb_ax[0], cb[1] - lb_ax[0])
                    for cb, lb_ax in zip(chunk_bounds, lb, strict=True)
                )
                output_layer[out_key] = (
                    operator.getitem,
                    (loader_layer_name, li),
                    src_slices,
                )
            else:
                # Case 2b — Single loader, partial coverage.
                # Must composite onto a fill background.
                output_layer[out_key] = (
                    _composite_chunk,
                    chunk_bounds,
                    dtype,
                    fill_value,
                    (loader_layer_name, li),
                    all_loader_bounds[li],
                )
        else:
            # Case 3 — Multiple overlapping loaders.
            # Pass all loader refs and their bounds as flat positional args.
            # Dask resolves the graph-key entries (loader refs) and passes
            # the plain-tuple entries (bounds) through unchanged.
            flat_args: list = []
            for li in loader_indices:
                flat_args.append((loader_layer_name, li))
                flat_args.append(all_loader_bounds[li])
            output_layer[out_key] = (
                _composite_chunk,
                chunk_bounds,
                dtype,
                fill_value,
                *flat_args,
            )

    graph = HighLevelGraph(
        layers={loader_layer_name: loader_layer, output_name: output_layer},
        dependencies={loader_layer_name: set(), output_name: {loader_layer_name}},
    )

    return da.Array(
        graph,
        output_name,  # type: ignore
        chunks_normalized,
        dtype=dtype,
        meta=np.array([], dtype=dtype),
    )
