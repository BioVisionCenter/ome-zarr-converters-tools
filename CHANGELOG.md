# Changelog

## [v0.8.1]

### Features
- Implement automatic color assignment in `ChannelInfo`: wavelength-based lookup covers the visible spectrum (200–1500 nm valid range, returns `None` outside bounds to fall back to label matching), and label-based fallback uses `SequenceMatcher` fuzzy matching against common fluorophore/color names (DAPI, GFP, RFP, CFP, mCherry, Cy3/5/7, FITC, etc.).

### Fix
- Fix syntax error in `_color_from_channel_label` that prevented the module from loading.
- Fix `ChannelInfo.validate_channel_info` model validator to mutate `self` in-place (Pydantic v2 requirement) instead of returning `model_copy`.
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
