# Changelog

## [v1.0.2]

### Fix
- `apply_reindex_channels` now compacts `TiledImage.channels` when the acquired
  channel indices are already dense. Images declaring more channels than they
  acquired kept every `ChannelInfo` entry and failed at write time, so success
  depended on which channel numbers the operator used: `[Ch1, Ch3]` worked,
  `[Ch1, Ch2]` did not.
- `reindex_channels=False` now stores unacquired channels as empty planes, as its
  description has always promised. The output `c` length comes from `channels`
  instead of the tile extent, so channels declared after the last acquired one no
  longer vanish and fail the conversion. `reindex_channels=True` is unaffected.
- `apply_reindex_channels` no longer fails on images whose `axes` omit `c` (which
  broke every `c`-less converter under the default options), nor on a
  `TiledImage` with no regions.

### Features
- `apply_reindex_channels` supports tiles spanning several channels
  (`length_c > 1`, e.g. two-camera acquisitions), which `Tile` and `ChannelFilter`
  already accepted.
- `UserFacingModel` is now public (`ome_zarr_converters_tools.UserFacingModel`).
- Every concrete filter and validator is now exported from
  `ome_zarr_converters_tools` and `ome_zarr_converters_tools.pipelines`:
  `RegexFilter`, `WellFilter`, `FovNameFilter`, `AcquisitionFilter`,
  `AttributeFilter`, `ChannelFilter`, `ZRangeFilter`, `TRangeFilter` and
  `ShapeDtypeProbeValidator`. They were previously reachable only through the
  private `pipelines._filters` / `pipelines._validators` modules, which the docs
  had to instruct readers to import from, and which kept them out of the API
  reference entirely.

  ```python
  # before
  from ome_zarr_converters_tools.pipelines._filters import RegexFilter

  # after
  from ome_zarr_converters_tools.pipelines import RegexFilter
  ```

### API Breaking Changes
- The `ngff_version` argument of `setup_images_for_conversion` is deprecated and
  will be removed in v1.1.0. The NGFF version is now taken from
  `converter_options.omezarr_options.ngff_version`.

  ```python
  # before
  setup_images_for_conversion(
      tiled_images,
      zarr_dir=zarr_dir,
      collection_type="ImageInPlate",
      converter_options=converter_options,
      ngff_version=converter_options.omezarr_options.ngff_version,
  )

  # after
  setup_images_for_conversion(
      tiled_images,
      zarr_dir=zarr_dir,
      collection_type="ImageInPlate",
      converter_options=converter_options,
  )
  ```

### Chores
- `pixi.lock` is now tracked, and the `lint` CI job runs from it via
  `prefix-dev/setup-pixi` instead of `pip install`ing unpinned tools. Every linter
  version CI uses is now a reviewed fact in the repo; upgrades happen deliberately
  through `pixi update` and `prek auto-update`. Two unrelated PR failures came from
  this drift: ruff 0.16 started formatting Markdown code blocks (CI had 0.16, the
  pixi env 0.15), and ty 0.0.68 started flagging an unused blanket `type: ignore`.
- Linting moves from `pre-commit` to [`prek`](https://github.com/j178/prek), a drop-in
  reimplementation. `.pre-commit-config.yaml` is unchanged apart from its header
  comment. `pixi run -e dev lint` is the entry point (`chores` now depends on it);
  `pre-commit autoupdate` becomes `prek auto-update`.
- Pin `ruff` to `>=0.16,<0.17` in the `dev` extra, bump the `ruff-pre-commit` hook to
  `v0.16.1`, and reformat the `docs/pipeline.md` python snippets under ruff 0.16's
  Markdown formatting.
- Drop a stale `# type: ignore` in `fractal/_models.py` that newer `ty` reports as
  unused.
- The docs build moves from MkDocs + Material to [Zensical](https://zensical.org),
  matching `ngio`. `mkdocs`, `mkdocs-material`, `mkdocs-jupyter`, `ipywidgets` and the
  two git-metadata plugins leave the `docs` extra; `zensical`, `pymdown-extensions`,
  `griffe-typingdoc` and `pandas` join it. `mike` moves to
  `[tool.pixi.feature.docs.pypi-dependencies]` because Zensical needs a git-only fork
  and a direct URL reference in `optional-dependencies` would make the package
  unpublishable to PyPI. New pixi tasks: `serve_docs`, `build_docs`, `test_snippets`,
  `clean_docs_data`.
- CI builds the docs and runs every snippet on pull requests, through a new `docs` job
  in `ci.yml` (kept out of `docs.yml` so its `contents: write` is never extended to
  PRs). `deploy` now needs it, so a broken example blocks a release instead of
  surfacing as a bad deploy.
- `docs.yml` installs through pixi rather than pip, and no longer runs
  `mike set-default dev` on tag pushes — it had been flipping the published default
  back to `dev` after every release.
- Drop the `nbstripout` pre-commit hook and the `*.ipynb` rules in `.gitignore`; there
  are no notebooks left.

### Documentation
- The three tutorial notebooks are replaced by executed Python scripts under
  `docs/snippets/`, included into Markdown pages by `pymdownx.snippets` and run at
  build time by `markdown-exec`. Each script is runnable on its own from the repo root
  (`python docs/snippets/getting_started/hcs_plates.py`), so an example that breaks
  fails a script rather than silently rendering an empty block.
- Tutorials now show what the converter produces: a stage-layout figure of the parsed
  tiles and the stitched result, rendered as inline SVG through
  `docs/snippets/_render.py` so they follow the light/dark toggle.
- New design layer, `docs/stylesheets/ozct.css`, ported token-for-token from ngio's
  stylesheet — same palette, type scale, surfaces, radii and motion. Placeholder logo
  and lockup assets live in `docs/assets/`.
- Site restructured: Home, Getting started, Guides, API reference, Contributing. The
  single `api.md` becomes curated per-area pages plus whole-module dumps, and
  `pipeline.md`, `converters_as_fractal_tasks.md` and `converters.md` move under
  `docs/guides/`.
- A `CONTRIBUTING.md` is added at the repo root and single-sourced into the site,
  alongside the changelog.
- Fix a latent bug in the programmatic-tiles tutorial: it set `start_z` as a slice index
  while leaving `start_z_space` at its `"world"` default, so both Z slices resolved to
  slice 0 and the second overwrote the first. The page now declares
  `start_z_space="pixel"` and calls out the trap.
- State explicit cost of `reindex_channels=False` (raised in
  [#70](https://github.com/BioVisionCenter/ome-zarr-converters-tools/pull/70)).

## [v1.0.1]

Re-release of v1.0.0 with no changes of its own. The `v1.0.0` tag (and the
PyPI `1.0.0` package) was accidentally created from a stale commit that
predates the final pre-release fixes listed under v1.0.0 — notably the
attribute-filter fix and the Fractal UI schema/manifest text pass, so manifest
creation is broken in `1.0.0` (plain CLI usage is unaffected). `v1.0.1` is the
release v1.0.0 was meant to be; depend on `>=1.0.1`.

## [v1.0.0]

First stable release. The public API is now considered frozen under semantic
versioning.

### Features
- The names of the OME-Zarr pyramid levels can now be chosen (closes
  [#63](https://github.com/BioVisionCenter/ome-zarr-converters-tools/issues/63)):
  `OmeZarrOptions.levels` is a discriminated union `PyramidLevels =
  NumberOfLevels | NamedLevels`. The default (`NumberOfLevels`, 5 levels named
  `0`, `1`, ...) is unchanged; `NamedLevels(level_names=["s0", "s1", "s2"])`
  writes explicitly named levels (validated: unique, non-empty path segments).
  New public exports: `PyramidLevels`, `NumberOfLevels`, `NamedLevels`.
- User-facing pass over all Fractal UI schema text: every UI-exposed model and
  field now renders a complete description (previously e.g. the tiling
  strategies and the `OverwriteMode`/`WriterMode`/`BackendType`/`Scalings`/
  `ColorMenu` enums showed a literal "Missing description for X."), texts are
  written for task users instead of developers (e.g. `StagePositionCorrections`
  cases, `RuntimeSettings`), class summaries are one line so fractal-task-tools
  no longer cuts them mid-sentence, and union dropdowns show friendly titles
  ("Well Filter", "Snap to Grid") instead of raw class names. Field titles with
  wrong casing were fixed (`XY Pixel Size`, `NGFF Version`, `Wavelength ID`).
  Descriptions use light markdown (the UI renders it): multi-choice options are
  itemized lists with backticked values, and key caveats are bolded (e.g. a
  partial channel match **fails the conversion**).
- Snapshots now record an informational top-level `versions` block (this package,
  `ngio`, `zarr`, `numpy`, `dask`, `tifffile`, `pillow`, `pydantic`, and Python) captured
  at generation time. It is written to the snapshot JSON but never compared, so a
  dependency version drift never fails a test; it exists to help diagnose a snapshot
  discrepancy caused by an upstream change.
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
- Complete the "one canonical import location" rule: the package root now
  re-exports every public name from `core`, `models`, and `pipelines`,
  including `TileSlice`, `TileFOVGroup`, the pipeline extension points
  (`add_filter`, `add_registration_func`, `add_validator`,
  `add_collection_handler`, `setup_ome_zarr_collection`,
  `write_tiled_image_as_zarr`, `FilterModel`, `ImplementedFilters`,
  `RegistrationStep`, `ValidatorModel`), and the enums that appear as public
  model-field types (`Scalings` for `FovBasedChunking.xy_scaling`,
  `TempJsonOptions` for `RuntimeSettings.temp_json_options`), plus
  `BackendType`, `find_url_type`, and `local_url_to_path`. A regression test
  asserts the root `__all__` is a superset of each subpackage's `__all__`.
- Add `CollectionInterface.set_suffix` as the supported way to set the per-FOV
  path suffix (replaces reaching into the private `_suffix` attribute).
- Six new built-in filters, all selectable from the Fractal UI via
  `AcquisitionOptions.filters`: `FovNameFilter` (regex on `fov_name`),
  `AcquisitionFilter` (acquisition indices, plates only), `AttributeFilter`
  (key/value match on tile attributes; values are typed `AttributeValue`
  entries, e.g.
  `AttributeFilter(key="condition", values=[StringValue(value="control")])`),
  `ChannelFilter` (channel labels; a partial match on a multi-channel tile
  raises instead of silently dropping channels), and `ZRangeFilter` /
  `TRangeFilter` (keep tiles whose `start_z` / `start_t` falls inside an
  inclusive `[min, max]` range). All matching filters carry a
  `mode: Literal["Include", "Exclude"]` field (default `"Include"`) selecting
  whether matching tiles are kept or removed.
- Validators are now pre-flight checks that front-load compute-time failures to
  init time. New built-in `ShapeDtypeProbeValidator` (name
  `"Shape and Dtype Probe"`): runs `preflight` on every tile's loader, then
  loads one sample tile per image and raises if its shape or dtype does not
  match the declared tile geometry — catching wrong parser `length_*` /
  `data_type` before compute jobs are dispatched.
- New optional `ImageLoaderInterface.preflight(resource)` hook: cheaply verify
  a source is reachable without loading it. Warns (never raises); the default
  implementation is a no-op. `DefaultImageLoader` implements it as a
  file-existence check.
- `build_parallelization_list` and `setup_images_for_conversion` now reject
  duplicate output paths up front: two `TiledImage`s resolving to the same
  path would race on the same zarr group during parallel compute.
- Redesign `StagePositionCorrections` around per-axis stage-position handling:
  - `remove_xy_offset` / `remove_z_offset` / `remove_t_offset` control offset
    removal per axis. `"Global"` (default) translates the axis origin to 0;
    `"Keep"` keeps absolute positions (raises on negatives, left-pads on
    positives); `"Per-FOV"` (z only) zeros each FOV's z independently.
  - `remove_xy_jitter` (default `True`) snaps a FOV's sub-tiles to a shared XY
    origin (the former `align_xy`).
  - `reindex_channels` (default `True`) compacts the channel indices actually
    present to a dense `0, 1, 2, …` range (dropping filtered channels and
    reconciling channel metadata); set `False` to keep gaps as empty channels.

### Fix
- The attribute filters' `values` field no longer breaks Fractal manifest
  generation (`fractal-task-tools` `[E05] Boolean with no default`): the bare
  `str | int | float | bool` union is replaced by a discriminated union of
  typed value models (`BoolValue`, `StringValue`, `IntValue`, `FloatValue`,
  `IsNoneValue`, `IsNotNoneValue`), whose boolean carries a real default and
  which render as concrete per-type forms in the Fractal web UI. The
  `Is None` / `Is Not None` variants additionally allow matching unset
  (`None`) attribute elements, which the old union could not express.
- Pixel-grid rounding no longer opens 1-pixel gaps/overlaps at tile boundaries:
  `_region_to_pixel_coordinates` and `apply_align_to_pixel_grid` now round the
  interval endpoints together (`round(start+length) - round(start)`) instead of
  rounding `start` and `length` independently.
- `reindex_channels` no longer silently truncates channel metadata when a tile
  references a channel index beyond the provided `channels` list; the mismatch
  is now caught at tile construction (see API Breaking Changes) so it can no
  longer surface as a confusing `NgioValueError` at write time.
- `setup_plates` now raises a clear error naming the offending images when tiles
  carry heterogeneous attribute key sets (or an attribute key collides with a
  reserved condition-table column), instead of failing with an opaque polars
  shape error while building the condition table.
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
- `OmeZarrOptions.num_levels: int` is replaced by the `levels` union field.
  Before: `OmeZarrOptions(num_levels=3)`. After:
  `OmeZarrOptions(levels=NumberOfLevels(num_levels=3))`.
- All UI-exposed models now inherit from a shared `UserFacingModel` base
  (`models/_base.py`) with `extra="forbid"`: payloads containing unknown field
  names (e.g. typos in serialized parameter files) now raise a validation
  error instead of being silently ignored. Models that already forbade extras
  are unaffected.
- The Include/Exclude filter class pairs are merged into single filters with a
  `mode: Literal["Include", "Exclude"]` field (default `"Include"`):
  `RegexIncludeFilter` / `RegexExcludeFilter` → `RegexFilter` and
  `WellIncludeFilter` / `WellExcludeFilter` → `WellFilter` (whose `wells` field
  replaces both `wells_to_include` and `wells_to_remove`).
  Before: `WellIncludeFilter(wells_to_include=["A01"])`,
  `WellExcludeFilter(wells_to_remove=["B02"])`. After:
  `WellFilter(wells=["A01"])`, `WellFilter(wells=["B02"], mode="Exclude")`.
- Image grouping is now separated from the tiling strategy. `ConverterOptions.tiling_strategy`
  is replaced by `ConverterOptions.grouping`, a discriminated union
  `Grouping = MosaicGrouping | PerFovGrouping`. `MosaicGrouping` carries the
  `tiling_strategy` (only the mosaic case has an arrangement); `PerFovGrouping` has none.
  Before: `ConverterOptions(tiling_strategy=AutoTiling())`. After:
  `ConverterOptions(grouping=MosaicGrouping(tiling_strategy=AutoTiling()))`. Before:
  `ConverterOptions(tiling_strategy=NoTiling())`. After:
  `ConverterOptions(grouping=PerFovGrouping())`. New public exports: `Grouping`,
  `MosaicGrouping`, `PerFovGrouping`.
- `NoTiling` is removed from the `TilingStrategy` union and from public exports; its
  "one OME-Zarr per field of view" behavior is now `PerFovGrouping`. `TilingStrategy` now
  only covers within-mosaic arrangement (`AutoTiling`, `SnapToGridTiling`,
  `SnapToCornersTiling`, `InplaceTiling`).
- `tiled_image_from_tiles` takes `split_per_fov: bool` instead of `converter_options`
  (`core/` no longer depends on the `models` config). Callers using
  `tiles_aggregation_pipeline` are unaffected; direct callers pass
  `split_per_fov=options.grouping.split_per_fov`.
- Wire-format note: the `ConverterOptions` JSON key `tiling_strategy` becomes `grouping`
  (with the tiling strategy nested under the `Mosaic` variant); with `extra="forbid"`,
  older serialized `init_args` payloads carrying `tiling_strategy` are rejected (acceptable
  pre-v1, no on-disk fixtures affected).
- `s3fs` is no longer a hard dependency; `s3://` support moved to an optional
  `s3` extra. Install `ome-zarr-converters-tools[s3]` for object-storage
  access. Using an `s3://` URL without it now raises an `ImportError` naming the
  extra. Before: `pip install ome-zarr-converters-tools`. After (for s3):
  `pip install "ome-zarr-converters-tools[s3]"`.
- `Tile` now validates at construction that `start_c + length_c` fits within
  `acquisition_details.channels` (when channels are provided) and that
  `start_c >= 0`; `TiledImage` enforces the same coverage when rebuilt from JSON
  at the fractal init/compute boundary. Supply one `ChannelInfo` per instrument
  channel index (padding unused slots) or set `channels=None`. Previously an
  out-of-range channel index was accepted and failed later inside ngio.
- `build_dummy_tile` is now keyword-only, matching the other tile/image
  builders. Before: `build_dummy_tile("FOV_0", start, shape, coll, acq)`. After:
  `build_dummy_tile(fov_name="FOV_0", start=start, shape=shape, collection=coll,
  acquisition_details=acq)`.
- `add_registration_func` and `add_validator` are now keyword-only with an
  optional `name` (defaulting to `function.__name__`), matching `add_filter` and
  `add_collection_handler`. Before: `add_validator(fn, "my_step")`. After:
  `add_validator(function=fn, name="my_step")`.
- The validator configuration is now a Pydantic model, aligned with filters:
  `ValidatorStep` (TypedDict with a `params` dict) is replaced by
  `ValidatorModel`, and validator functions receive the whole model plus the
  pipeline `resource` instead of unpacked params. Before:
  `apply_validator_pipeline(images, [{"name": "my_step", "params": {"k": 1}}])`
  with `def my_step(image, k): ...`. After:
  `apply_validator_pipeline(images, validators_config=[MyStepModel(name="my_step", k=1)])`
  with `def my_step(image, validator_params, resource=None): ...`.
  `tiles_aggregation_pipeline(validators=...)` takes the new models and forwards
  its `resource` to them.
- `CollectionInterface` is now an abstract base class (`abc.ABC` with an
  `@abstractmethod path()`); instantiating it directly, or a subclass that does
  not implement `path`, now raises `TypeError` instead of failing at call time.
- `StageOrientation.swap_xy=True` now actually transposes the X and Y stage
  axes; previously it was a silent no-op (it only reordered the ROI slice list).
  Before: `swap_xy=True` left tile positions unchanged. After: the x output is
  built from the tile's y position/length and vice versa. Converters that set
  `swap_xy=True` expecting the old (no-op) behaviour will now produce swapped
  output.
- `StagePositionCorrections` fields changed completely: `align_xy`, `align_z`,
  and `align_t` are removed. Migrate `align_xy=True` → `remove_xy_jitter=True`
  (now the default); `align_z`/`align_t` had no working behaviour and have no
  replacement (z/t are handled by `remove_z_offset`/`remove_t_offset`). The new
  fields are `remove_xy_offset`, `remove_z_offset`, `remove_t_offset`,
  `remove_xy_jitter`, and `reindex_channels` (see Features).
- EXTEND mode no longer overwrites an existing store when it fails to open: a
  corrupt/partial store or a permission/version error now propagates instead of
  being silently replaced with a fresh, empty store (in `setup_plates` and
  `write_tiled_image_as_zarr`).
- Pre-1.0 naming pass over the public models (four renames):
  - The `remove_xy_offset` / `remove_z_offset` / `remove_t_offset` value
    `"False"` is renamed to `"Keep"`. Before:
    `StagePositionCorrections(remove_z_offset="False")`. After:
    `StagePositionCorrections(remove_z_offset="Keep")`.
  - The scheduler models (`ThreadScheduler`, `ProcessScheduler`,
    `SynchronousScheduler`, `DefaultScheduler`) discriminate on `mode` instead
    of `type`, matching `TilingStrategy`/`ChunkingStrategy`. Before:
    `RuntimeSettings(dask_scheduler={"type": "Threads"})`. After:
    `RuntimeSettings(dask_scheduler={"mode": "Threads"})`.
  - `pixelsize` is renamed to `xy_pixel_size` on `AcquisitionDetails`,
    `TiledImage`, and `PixelSizeModel` (the `TiledImage.pixel_size` property
    returning an ngio `PixelSize` is unchanged). This also renames the
    `pixelsize` key in `acquisition_details.toml` config files. Before:
    `AcquisitionDetails(pixelsize=0.65)`. After:
    `AcquisitionDetails(xy_pixel_size=0.65)`.
  - The `AcquisitionDetails` coordinate-system fields drop the `_coo` suffix
    for `_space` (`start_{x,y,z,t}_space`, `length_{x,y,z,t}_space`), and the
    `COO_SYSTEM_TYPE` alias is renamed to `SPACE_TYPE` (`safe_to_world` takes
    `space=` instead of `coo_system=`). This also renames the `*_coo` keys in
    `acquisition_details.toml` config files. Before:
    `AcquisitionDetails(start_x_coo="world")`. After:
    `AcquisitionDetails(start_x_space="world")`.

### Chores
- Fix a `ty` type error in `_expected_region_shape` (annotate the offset
  dict as `dict[str, float]`) that failed the CI lint job.
- CI lint job installs the `test` extra alongside `dev` so `ty` can resolve
  `pytest`, imported by the shipped pytest plugin
  (`ome_zarr_converters_tools.testing.plugin`).
- Update `ty` to 0.0.58 in the pixi lockfile (matching CI, which installs the
  latest) and drop a `# type: ignore` in `_tile.py` that the newer `ty` flags
  as unused.
- UI-exposed models set `use_attribute_docstrings=True` explicitly (via the
  `UserFacingModel` base) instead of relying on fractal-task-tools patching it
  into `BaseModel` at import time — schema output no longer depends on import
  order, and the committed schema snapshot now shows the descriptions users
  actually see. A new guard test (`test_schema_text_is_complete`) fails CI if
  any schema node lacks a description or a class summary is cut mid-sentence.
- Add a JSON-schema compatibility test (`tests/unit/test_json_schema_compat.py`):
  the full set of models that downstream converter packages expose to Fractal
  manifests is rendered with `fractal-task-tools`' schema builder and compared
  against a committed snapshot (`tests/data/schemas/task_args_schema.json`),
  checked for complete coverage of the downstream `$defs` surface, and
  validated against the Fractal webui renderability rules (`E01`–`E22`).
  Regenerate the snapshot with `pytest --update-snapshots`. Adds
  `fractal-task-tools` to the `test` extra.
- Bump the `Development Status` classifier from `3 - Alpha` to
  `5 - Production/Stable`.
- Remove the unused runtime dependencies `toml` and `tqdm`, and add minimum
  version floors to `numpy`, `pillow`, `tifffile`, and `fsspec` (the `s3fs`
  floor now lives in the optional `s3` extra).
- Add `pandas` to the `test` extra (used directly by the integration tests,
  previously only resolved transitively via `ngio`).
- Consolidate the four near-identical filter/validator/registration/collection
  registries behind a shared generic `Registry` helper
  (`pipelines/_registry.py`), aligning their `add_*` signatures and giving all
  four registries error messages that list the available names and note the
  multiprocessing-visibility requirement.
- Rewrite `calculate_snap_to_corner_offset` to size its candidate grid from the
  tiles' bounding box and vectorize the nearest-corner search, replacing the
  previous O(n³)/O(n²) all-pairs scan that degraded on images with many FOVs.
- `exec_compound_task` now raises on an unrecognized runner instead of falling
  through the `match` and returning `None`.
- `tiled_image_from_json` no longer sleeps after its final retry attempt before
  raising `FileNotFoundError`.
- Add `permissions: contents: write` to the docs workflow so `mike` can push to
  the `gh-pages` branch on main/tag builds.
- Update the LICENSE copyright to `2023-2026, BioVisionCenter, University of
  Zurich`.
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
- Add a README quickstart and an S3-extra install note.
- Document the `testing` snapshot-plugin module, the `url` vs `path` field
  naming convention, and the intentionally `fractal`-namespaced JSON plumbing in
  `docs/api.md`.
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
