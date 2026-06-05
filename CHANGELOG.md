# Changelog

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
