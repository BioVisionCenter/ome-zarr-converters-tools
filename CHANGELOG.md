# Changelog

## [v1.0.0]

First stable release. The public API is now considered frozen under semantic
versioning.

### Features
- Promote the primary entry points to the package root so they are importable
  from `ome_zarr_converters_tools` directly (in addition to their subpackages):
  `hcs_images_from_dataframe`, `single_images_from_dataframe`,
  `tiled_image_from_tiles`, `build_dummy_tile`, the `TilingStrategy` family
  (`AutoTiling`, `SnapToGridTiling`, `SnapToCornersTiling`, `InplaceTiling`,
  `NoTiling`), `StagePositionCorrections`, and `WriterMode`.
- Export `ImageLoaderInterface` (the loader extension base class) from
  `ome_zarr_converters_tools` and `...models`, and `build_dummy_tile` /
  `UrlType` from their subpackages — previously documented as extension points
  but not importably public.
- Add `CollectionInterface.set_suffix` as the supported way to set the per-FOV
  path suffix (replaces reaching into the private `_suffix` attribute).

### Fix
- `TiledImage.load_data` / `load_data_dask` now zero each region to the union
  origin before slicing, fixing dropped tile data (or a broadcast error) for
  images whose regions do not start at pixel 0.
- `setup_plates` now builds each plate's `condition_table` from only that
  plate's images, once per plate (previously it was populated from the full
  cross-plate list and rebuilt once per image, O(N²)).
- Integer plate row `26` now maps to `"Z"` instead of being rejected.
- The dask lazy-loader graph token now includes loader identity, preventing
  graph-key collisions (and silent data substitution) between arrays with
  identical geometry but different source files.
- `_color_from_wavelength_id` maps exactly `750` nm to Red instead of magenta.
- The shared snapshot `images_common` block is now applied to every image, with
  per-image values overriding the shared defaults (previously it was applied
  under the wrong condition and merged in the wrong direction).
- `build_parallelization_list` drops the unset JSON-source field from the
  emitted `init_args` instead of relying on a no-op `model_dump(exclude=None)`.

### API Breaking Changes
- `StageOrientation.swap_xy=True` now actually transposes the X and Y stage
  axes; previously it was a silent no-op (it only reordered the ROI slice list).
  Before: `swap_xy=True` left tile positions unchanged. After: the x output is
  built from the tile's y position/length and vice versa. Converters that set
  `swap_xy=True` expecting the old (no-op) behaviour will now produce swapped
  output.
- Selecting `StagePositionCorrections.align_z` or `align_t` now raises
  `NotImplementedError` instead of emitting a warning and returning the regions
  unchanged. Set them to `False` (the default) until they are implemented.
- EXTEND mode no longer overwrites an existing store when it fails to open: a
  corrupt/partial store or a permission/version error now propagates instead of
  being silently replaced with a fresh, empty store (in `setup_plates` and
  `write_tiled_image_as_zarr`).

### Chores
- Bump the `Development Status` classifier from `3 - Alpha` to
  `5 - Production/Stable`.
- Remove the unused runtime dependencies `toml` and `tqdm`, and add minimum
  version floors to `numpy`, `pillow`, `tifffile`, `fsspec`, and `s3fs`.
- Remove the `src/debug/plotting.py` scratch module (tracked in git but never
  shipped in the wheel and referenced nowhere).
- Add a CI `lint` job running `ruff check`, `ruff format --check`, and
  `ty check src`.
- Collapse the redundant outer retry loop in `generic_compute_task` (retries are
  already handled by `tiled_image_from_json`).
- Replace `resource: Any = None` / `resource: None = None` with a consistent
  `resource: Any | None = None` across the loader surface, and convert a
  user-facing `assert` in `setup_plates` into an explicit `TypeError`.
- Fix assorted typos (`plante_url`, `GripPoint`, "avoit", "less files").

### Documentation
- Correct `docs/api.md`: rename the stale `AlignmentCorrections` to
  `StagePositionCorrections`, and document the URL helpers via the public
  `models` module instead of the private `_url_utils` path.
- Fix the false "loaded as a `pytest11` entry point" claim in the `testing`
  plugin docstrings (it is loaded via `pytest_plugins` in each consumer's
  conftest, by design).
- Correct docstrings across `src/`: the inverted pixel/world description in
  `_region_to_pixel_coordinates`, the stale "50 MB" `Auto` threshold, and a
  missing module docstring in `core/_roi_utils.py`.

## [v0.10.4]

### Features
- Add a shared `ome_zarr_converters_tools.testing` subpackage that centralizes the converter snapshot tests (previously copy-pasted into each `fractal-*-converters/tests/utils.py`). Exposes `run_converter_test` plus `build_snapshot`/`compare_snapshots` and the assertion models. Snapshots are now stored as **JSON** (was YAML): stdlib, no dependency, native `null`, and no YAML implicit-typing coercion of channel/well strings. Generation and validation share one code path (`build_snapshot`), fingerprint stats are compared with a tolerance while the sha256 pixel hash stays exact, and a mismatch raises a single `AssertionError` listing every differing field with its full path. A `ome_zarr_converters_tools.testing.plugin` pytest plugin provides `--update-snapshots`, `--extended`, the `extended` marker, and the `update_snapshots` fixture; each consumer loads it with `pytest_plugins = ["ome_zarr_converters_tools.testing.plugin"]` in its `tests/conftest.py`, so converters no longer duplicate that wiring.
- Add protocol-aware URL path helpers to `_url_utils`, exported from the package root and `ome_zarr_converters_tools.models`: `parent_url`, `basename_url`, `is_absolute_url`, and `glob_url_paths`. They are the URL equivalents of `os.path.dirname`/`basename`/`isabs`/`glob`, working transparently for local paths and `s3://` URLs and robust to Windows backslash separators.

### Fix
- `join_url_paths` now resolves `.`/`..` segments and collapses redundant slashes. It uses `posixpath` (never `os.path`) so `s3://` keys keep `/` separators on Windows instead of being rewritten to `\` (an invalid S3 key).
- `join_url_paths` now raises `ValueError` when `..` segments would ascend above the network location of a protocol URL, instead of silently dropping it (e.g. `join_url_paths("s3://bucket", "..", "x")` previously returned `s3://x`, targeting a different bucket).
- `find_url_type`, `is_absolute_url`, and `local_url_to_path` now handle `~`-anchored home paths: `~/…` classifies as `LOCAL`/absolute and `local_url_to_path` expands it via `expanduser` (previously `~` was treated as a literal relative directory).
- `local_url_to_path` no longer creates the parent directory on disk (undocumented side effect); it now purely resolves the path (expanding `~`). Callers that relied on the implicit `mkdir` must create the directory explicitly.
- `is_absolute_url` now classifies Windows drive/UNC and `~` paths as absolute independently of the host OS, matching `find_url_type` (previously `is_absolute_url("C:/path")` returned `False` on POSIX).
- `parent_url` now always returns POSIX `/` separators (uses `posixpath` for the local branch too, never OS-native `pathlib`), so it no longer emits backslashes on Windows (e.g. `parent_url("/path/to/file.txt")` returned `\path\to` on Windows).

### Chores
- Add unit tests for `ome_zarr_converters_tools.testing` (comparison branches, `build_snapshot`/`run_converter_test` over tiny ngio-built OME-Zarrs, and the pytest plugin hooks). The plugin is loaded via `pytest_plugins` in each consumer's conftest rather than a `pytest11` entry point: an entry point imports the package during pytest's plugin bootstrap (before pytest-cov starts), which marks the whole package `module-not-measured` and deflated coverage from ~97% to ~62%. `testing/__init__` also imports `_snapshot` lazily via module `__getattr__` so merely loading the plugin does not pull in numpy/ngio/pydantic.
- Route the remaining ad-hoc path manipulation through the centralized `_url_utils` helpers: `DefaultImageLoader` suffix detection uses `basename_url`, JSON cleanup uses `parent_url`, and `TempJsonOptions.format_temp_url` normalizes its result via `join_url_paths`.
- Add `ipywidgets` to the `docs` environment so `tqdm.auto` progress bars find `IProgress` when notebooks are executed during the docs build, silencing the `TqdmWarning: IProgress not found` warning.

### Documentation
- Fix the tutorial notebooks (`docs/hcs_tutorial.ipynb`, `docs/images_tutorial.ipynb`, `docs/advanced_tutorial.ipynb`) so they execute against the current API: use `StagePositionCorrections` (the removed `AlignmentCorrections` name) and read the example data from `../tests/data/` (the deleted `../examples/` path).
- Rewrite docstrings across `src/` to render cleanly as Markdown under mkdocstrings/Griffe: replace RST double-backticks and `:func:`/`:class:` roles with single backticks, drop parameter types restated in `Args:`/`Returns:` (they already come from signature annotations), and convert `core/_dask_lazy_loader.py`'s RST-style module docstring (underlined headers, `::` literal block) to Markdown with a `Note:` section and a fenced `python` code block.

## [v0.10.3]

### Features
- `DefaultImageLoader` now recognizes additional TIFF extensions (`.tf2`, `.tf8`, `.btf` BigTiff variants, closing [#60](https://github.com/BioVisionCenter/ome-zarr-converters-tools/issues/60)) and, for any unrecognized extension, warns and attempts a best-effort TIFF read instead of raising immediately.

## [v0.10.2]

### Chores
- Align project tooling with `fractal-uzh-converters` standards: replace `mypy` with `ty` for type checking (drop the `[tool.mypy]` config and `mypy` dev dependency); rename ruff rule selector `TCH` → `TC` and ignore `D415`; add `docstring-code-line-length = 89` to the ruff formatter; bump pytest `minversion` to `8.0` and drop `-vv` from `addopts`; simplify `.pre-commit-config.yaml` (drop the `ci:` autoupdate block and commented-out mypy hook) and update hook pins to latest (`ruff-pre-commit` `v0.15.17`, `validate-pyproject` `v0.25`, `typos` `v1.47.2`, `nbstripout` `0.9.1`); fix the codecov upload condition in CI to run on Python `3.12` (was an unreachable `3.10`).

## [v0.10.1]

### Fix
- `write_tiled_image_as_zarr` now forwards `TiledImage.data_type` to `ngio.create_empty_ome_zarr`; output arrays preserve the source dtype (e.g. `uint8`) instead of always being written as `uint16`.

## [v0.10.0]

### Documentation
- Add "Converters" page listing projects that use `ome-zarr-converters-tools` (`fractal-uzh-converters`, `fractal-lif-converters`, `fractal-czi-converters`, `fractal-nd2-converters`).

### Features
- Add `TempJsonOptions.serialization` field (`"Auto"` / `"Memory"` / `"JSON"`) to control how tiled image data is handed off between the init and compute phases. `"Memory"` skips all filesystem I/O by embedding the JSON string directly in `ConvertParallelInitArgs`; `"Auto"` (default) uses in-memory when the total serialized payload is ≤50 MB and falls back to disk otherwise. `"JSON"` preserves the original file-based behaviour required for distributed Fractal runs.
- Add `tiled_image_from_json_str` and `dump_json_str` helpers to `ome_zarr_converters_tools.fractal` for in-memory serialization without filesystem round-trips.
- `generic_compute_task` now accepts `init_args` as either a `ConvertParallelInitArgs` instance or a plain dict (as produced by Fractal's orchestration layer), eliminating the need for callers to manually validate the args.
- Add `exec_compound_task` public API for running Fractal compound tasks with pluggable execution strategies: `SequentialRunner` (default), `ThreadedRunner` (I/O-bound parallelism via `ThreadPoolExecutor`), and `MultiprocessingRunner` (CPU-bound parallelism via `ProcessPoolExecutor`). All three runner types and `RunnerType` are now exported from the top-level package.
- Add built-in `setup_singleimage` collection setup handler for `SingleImage` outputs; registered as `"SingleImage"` in the default `_collection_setup_registry` alongside `setup_plates`. The handler enforces the `OverwriteMode` contract (raises `FileExistsError` on `NO_OVERWRITE` when the target zarr already exists) without creating an upfront skeleton, since the zarr group is created during the compute task.

### API Breaking Changes
- `ConvertParallelInitArgs.tiled_image_json_dump_url` changed from `str` to `str | None` (defaults to `None`). Exactly one of `tiled_image_json_dump_url` or the new `tiled_image_json_str` must be set; existing code that constructs `ConvertParallelInitArgs(tiled_image_json_dump_url="...", ...)` continues to work unchanged.

### Chores
- Move test fixture data from `examples/` to `tests/data/` to clarify that these datasets are for testing only.

## [v0.9.0]

### Features
- Implement automatic color assignment in `ChannelInfo`: wavelength-based lookup covers the visible spectrum (200–1500 nm valid range, returns `None` outside bounds to fall back to label matching), and label-based
fallback uses `SequenceMatcher` fuzzy matching against common fluorophore/color names (DAPI, GFP, RFP, CFP, mCherry, Cy3/5/7, FITC, etc.).

### API Breaking Changes
- `DefaultColor` / `DefaultColorConversion` renamed to `ColorMenu` / `ColorMenuBase`; update any direct imports of these names.
- `ChannelInfoUI.color` default changed from `DefaultColor.Blue` to `ColorMenu.Auto`; channels that previously defaulted to blue will now have their color auto-assigned from the label or wavelength ID.

### Fix
- Fix three pre-existing ruff violations in `tests/unit/test_registration.py` (E501 long comment, two RUF003 ambiguous `×` characters).

### Documentation
- Update `docs/pipeline.md` to use current v0.8.0 API names throughout: `AlignmentCorrections` → `StagePositionCorrections` and `StageCorrections` → `StageOrientation`.
- Add `## RuntimeSettings` section to `docs/pipeline.md` documenting `RuntimeSettings`, `DaskScheduler` variants, and `TempJsonOptions`.

### Chores
- Add missing type annotation for `alignment_corrections` parameter in `build_default_registration_pipeline` (`_registration_pipeline.py`).
- Change default `tolerance` from `0` to `1` pixel in `AutoTiling` and `SnapToGridTiling`; aligns with the v0.8.2 fix that made non-zero tolerance work correctly.

## [v0.8.2]

### Fix
- Fix `_find_offset` in `_snap_utils.py` incorrectly rejecting jittered regular grids when `tolerance > 0`: the hardcoded `1e-6` threshold used to filter intra-column near-zero diffs has been replaced with `tolerance`, so same-column position noise is correctly discarded before computing the grid step.

## [v0.8.1]

### Fix
- Fix `setup_plates` silently reusing an existing plate instead of replacing it when `overwrite_mode=OVERWRITE`; the `OVERWRITE` branch now unconditionally calls `create_empty_plate(overwrite=True)` instead of falling through the open-or-create try/except.

## [v0.8.0]

### Features
- Add `RuntimeSettings` model with an `apply()` context manager for scoped runtime configuration: Zarrs codec backend (`use_zarrs_codec`), Dask scheduler selection (`DaskScheduler`), and temporary JSON storage options (`TempJsonOptions`).
- Replace `TilingMode` enum with rich Pydantic models (`AutoTiling`, `SnapToGridTiling`, `SnapToCornersTiling`, `InplaceTiling`, `NoTiling`) unified as a `TilingStrategy` discriminated union; each strategy carries its own parameters (e.g. `tolerance`).
- Improve snap utility internals with dedicated helper functions: `_find_offset`, `_find_grid_size`, `_match_to_perfect_grid`.

### API Breaking Changes
- `TilingMode` enum and standalone `tiling_tolerance` field replaced by the `tiling_strategy: TilingStrategy` field on `ConverterOptions`.
- Stage position field renames in `ConverterOptions` and related models:
  - `alignment_correction` → `stage_position_corrections`
  - `stage_corrections` → `stage_orientation`
- Refactor temporary JSON model (`TempJsonOptions`) out of `_converter_options.py` into the new `_runtime_settings.py` module.

### Fix
- Fix minor bug in `TileRegion` region loading that caused unnecessary reference data to be loaded; `data_type` is now stored directly in the model.


### Documentation
- Improve stage position documentation in `docs/pipeline.md`.
- Update docstrings across `core`, `fractal`, and `models` modules.
