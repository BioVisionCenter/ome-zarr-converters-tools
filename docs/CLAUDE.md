---
title: Docs build notes
description: Maintainer notes on how this site is built — Zensical, executed snippets, the stylesheet, and the traps.
---

# Docs build notes

The site is built by [Zensical](https://zensical.org) from `mkdocs.yml`; mkdocs-material
is not installed. Build and serve with `pixi run -e docs build_docs` / `serve_docs`.

Executed code lives in scripts under `docs/snippets/`, included by `pymdownx.snippets`
(`--8<-- "docs/snippets/<path>.py:name"`, delimited by `# --8<-- [start:name]`). Use
`source="material-block"`, and `html="1"` for figures and tables. One script per session,
each runnable standalone from the repo root — paths inside them are repo-root relative
(`tests/data/...`), and output goes to `./data/` (gitignored). Figures and tables go
through `docs/snippets/_render.py` (house style, percentile windows, scale bars, the
colour-to-CSS-variable swap); each script imports it in a hidden `plot_helpers` /
`table_helpers` section.

The whole theme is one stylesheet, `docs/stylesheets/ozct.css`, ported token-for-token
from ngio's `docs/stylesheets/ngio.css`. There are no template overrides. Changes that
are not converter-specific belong upstream in ngio first.

Five traps, none visible from the sources:

- markdown-exec does not redirect `sys.stdout`. It injects its own `print` into the
  globals of the block it executes, so only a `print` **written in the block** is
  captured; one inside an imported module resolves to the builtin, prints to the build's
  terminal, and leaves the block rendering as empty — with the build still exiting 0.
  Hence `print(figure_html(fig))` rather than a helper that prints.
- Build with `--clean --strict` (what `build_docs` does). Plain `zensical build` exits 0
  and reports "No issues found" even when a code block raised, and serves cached HTML.
- Each page gets a fresh markdown-exec session, so a page cannot use a variable bound on
  another.
- Print tables with the `table_html` helper and `html="1"`, never `.to_markdown()` —
  block-level Markdown is not run over markdown-exec output, so a pipe table stays literal
  `|---|`. The helper also strips pandas' `class`/`border` attributes, which every theme
  table rule is gated against (`table:not([class])`).
- Every `.md` under `docs/` becomes a page, this file included (`/CLAUDE/`). Zensical does
  not read `exclude_docs`, and neither an underscore prefix nor burying the file under
  `snippets/` prevents it; a post-build `rm` would not help either, because `mike deploy`
  builds its own copy of the site rather than publishing `site/`. The page is absent from
  `sitemap.xml` but present in the site search index. Accepted deliberately: the split
  between this file and the root `CLAUDE.md` is worth more than the stray page.

Declaring `markdown_extensions` in `mkdocs.yml` **replaces** Zensical's defaults instead
of extending them, so anything relied on must be listed explicitly. Dropping `tables`
makes every pipe table render as literal text; dropping `attr_list` breaks grid cards and
code annotations; dropping `abbr` breaks the glossary tooltips fed by `docs/_glossary.md`.

## Terminology and spelling

House rules for prose on the site and in docstrings; code identifiers stay as they are.

- British spelling in prose (colour, behaviour, artefact, serialise); Dask, NumPy,
  Pydantic, Zarr capitalised outside code font.
- The directory on disk is the "store" (`zarr_dir`, `zarr_url`); the objects in it are
  an "OME-Zarr image" or "OME-Zarr plate"; never "Zarr file" or "dataset". "OME-Zarr"
  is the format; "OME-NGFF 0.5" only when naming a spec version; "Zarr v3" for the
  storage format.
- `TiledImage` in code font for the class or an instance; "tiled image" in prose only
  where a plural or adjective is needed; never "TiledImages" in a heading.
- "field of view" spelled out at first use on each page, then `FOV`; hyphenate only as
  a compound modifier (per-FOV).
- "acquisition" means one multiplexing cycle inside a well, never the imaging run.
- "tile table" in prose, `tiles_table` only as the identifier; "tiling strategy", not
  "tiling mode"; a path filter matches the "output path" (`TiledImage.path`).
- "stage orientation" for `StageOrientation`, "stage position corrections" for
  `StagePositionCorrections`; never bare "stage corrections".
- "task user" runs a task in the Fractal UI; "converter author" writes one; a
  "downstream converter" is the package built on this library.
- "Z slice" (noun, no hyphen); "plane" only for an array plane; "time point";
  "pre-flight" in prose, `preflight()` as the identifier; "init task" and "compute
  task" in lower case.
- "µm" in tables and code comments, "micrometres" spelled out once per page.
- The product is "OME-Zarr Converters Tools" at most once per page, then "this
  library"; `ome-zarr-converters-tools` only in install commands; "ozct" nowhere
  user-facing.
