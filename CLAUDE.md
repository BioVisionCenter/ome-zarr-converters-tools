# CLAUDE.md

## Commands

All commands require `pixi run -e dev` prefix (never bare `python`/`pytest`/`ruff`):

- `pixi run -e dev pytest tests/` — full test suite
- `pixi run -e dev pre-commit run --all-files` — all linting/formatting hooks

## Code Style

- Ruff: line length 88, target py311
- Google-style docstrings (disabled for tests)
- Strict mypy
- Internal modules prefixed with `_`
- Spell check via typos — false positives go in `_typos.toml`
- Pydantic v2: `@field_validator` before `@classmethod`
