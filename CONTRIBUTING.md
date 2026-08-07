# Contributing

Issues and pull requests are welcome, on
[GitHub](https://github.com/BioVisionCenter/ome-zarr-converters-tools).

## Development environment

The project uses [pixi](https://pixi.sh). Tool versions come from the committed
`pixi.lock`, and CI resolves from the same file, so a local run and a CI run agree by
construction.

```bash
git clone https://github.com/BioVisionCenter/ome-zarr-converters-tools.git
cd ome-zarr-converters-tools
pixi install -e dev
```

## Checks

Every command runs inside a pixi environment — never a bare `python`, `pytest` or `ruff`.

```bash
pixi run -e dev pytest tests/   # the test suite
pixi run -e dev lint            # all pre-commit hooks, via prek
pixi run -e dev ty check src    # type check the shipped package
pixi run -e dev chores          # all of the above
```

Bump tool versions deliberately: `pixi update` for the lock file, `prek auto-update` for
the hook `rev:` pins in `.pre-commit-config.yaml`.

## Documentation

The site is built by [Zensical](https://zensical.org) from `mkdocs.yml`.

```bash
pixi run -e docs serve_docs     # live preview
pixi run -e docs test_snippets  # run every executed code snippet
pixi run -e docs build_docs     # strict build, as CI runs it
pixi run -e docs clean_docs_data  # drop the OME-Zarr stores the snippets wrote
```

Code shown on a documentation page lives in a runnable script under `docs/snippets/` and
is executed at build time — there are no notebooks. `docs/CLAUDE.md` documents the
conventions and the handful of traps that are not visible from the sources.

## Code style

- Ruff, line length 88, target py311.
- Google-style docstrings, rendered by mkdocstrings/Griffe as Markdown: single backticks
  for inline code, no restating of types that are already in the signature, `Args` /
  `Returns` / `Raises` / `Example` / `Note` sections, one-line summary then a blank line.
- Internal modules are prefixed with `_`.
- Spelling is checked by [typos](https://github.com/crate-ci/typos); false positives go
  in `_typos.toml`.

## Changelog

Every code change updates `CHANGELOG.md` under the current `## [vX.Y.Z]` section, one
bullet per logical change, with identifiers in backticks. Use the subsections already in
the file — `Features`, `Fix`, `API Breaking Changes`, `Chores`, `Documentation` — and omit
the ones that do not apply. An API breaking change needs a before/after example.

## Pull requests

Branch off `main`, keep the change scoped to one thing, and make sure
`pixi run -e dev chores` passes before opening the PR. CI additionally builds the docs and
runs every snippet, so a broken example fails the PR rather than the deploy.
