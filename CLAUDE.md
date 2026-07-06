# CLAUDE.md

## Commands

All commands require `pixi run -e dev` prefix (never bare `python`/`pytest`/`ruff`):

- `pixi run -e dev pytest tests/` — full test suite
- `pixi run -e dev pre-commit run --all-files` — all linting/formatting hooks

## Code Style

- Ruff: line length 88, target py311
- Google-style docstrings, rendered by mkdocstrings/Griffe as Markdown (disabled for tests):
  - Inline code uses single backticks (`` `None` ``), never RST double backticks
  - Don't restate types in prose — they live in the signature (`channel: The channel to load.`, not `channel (int): ...`)
  - Sections: `Args`, `Returns`, `Raises`, `Example`, `Note`
  - One-line summary, blank line, then body
  - Code examples in fenced ` ```python ` blocks, not `>>>` doctests
  - Terse: behavior and edge cases only, don't restate the signature
- Type checking via `ty`
- Internal modules prefixed with `_`
- Spell check via typos — false positives go in `_typos.toml`
- Pydantic v2: `@field_validator` before `@classmethod`

## Changelog

- Follow the format in `CHANGELOG.md` (mirrors `../ngio/CHANGELOG.md` style).
- **Always** update `CHANGELOG.md` when making code changes — add entries under the current `## [vX.Y.Z]` section (or create one if missing).
- Use these subsections (omit empty ones):
  - `### Features` — new user-visible behaviour
  - `### Fix` — bug fixes
  - `### API Breaking Changes` — anything that breaks existing call sites (include before/after example)
  - `### Chores` — internal refactors, dependency bumps, CI changes
  - `### Documentation` — doc-only changes
- One bullet per logical change; use backticks for identifiers.
