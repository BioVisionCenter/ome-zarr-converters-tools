"""JSON-schema compatibility tests for the models exposed to Fractal tasks.

Downstream converter packages (e.g. fractal-uzh-converters) run
fractal-task-tools over task functions whose arguments are (or nest) this
package's models. These tests regenerate that schema with the same builder
and pin it against a committed snapshot, so any schema drift (renamed
fields, changed defaults, union restructuring) fails CI here instead of
silently breaking downstream manifests.

Regenerate the snapshot deliberately with `pytest --update-snapshots`.
"""

import json
from pathlib import Path

import pytest
from fractal_task_tools._args_schemas import create_schema_for_single_task
from fractal_task_tools._specs import validate_schema

from unit import _schema_surface_task

SNAPSHOT_PATH = (
    Path(__file__).parent.parent / "data" / "schemas" / "task_args_schema.json"
)

# Every model of this package that appears in the `$defs` of the downstream
# task-argument schemas (source of truth: the $defs of
# fractal-uzh-converters' __FRACTAL_MANIFEST__.json, minus its own models).
# The filter entries are intentionally ahead of the committed downstream
# manifest: the Include/Exclude filter pairs were merged into single
# mode-carrying filters, and downstream regenerates its manifest on upgrade.
EXPECTED_DEFS = [
    "AcquisitionFilter",
    "AcquisitionOptions",
    "AttributeFilter",
    "AutoTiling",
    "BackendType",
    "BoolValue",
    "ChannelFilter",
    "ChannelInfoUI",
    "ColorMenu",
    "ConvertParallelInitArgs",
    "ConverterOptions",
    "DataTypeEnum",
    "DefaultScheduler",
    "FixedSizeChunking",
    "FloatValue",
    "FovBasedChunking",
    "FovNameFilter",
    "InplaceTiling",
    "IntValue",
    "IsNoneValue",
    "IsNotNoneValue",
    "MosaicGrouping",
    "OmeZarrOptions",
    "OverwriteMode",
    "PerFovGrouping",
    "PixelSizeModel",
    "ProcessScheduler",
    "RegexFilter",
    "RuntimeSettings",
    "Scalings",
    "SnapToCornersTiling",
    "SnapToGridTiling",
    "StageOrientation",
    "StagePositionCorrections",
    "StringValue",
    "SynchronousScheduler",
    "TRangeFilter",
    "TempJsonOptions",
    "ThreadScheduler",
    "WellFilter",
    "WriterMode",
    "ZRangeFilter",
]


@pytest.fixture(scope="module")
def surface_schema() -> dict:
    return create_schema_for_single_task(
        executable=_schema_surface_task.__file__,
        package=None,
        task_function=_schema_surface_task.schema_surface_task,
    )


def test_defs_cover_downstream_surface(surface_schema: dict) -> None:
    actual = set(surface_schema["$defs"])
    expected = set(EXPECTED_DEFS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    assert actual == expected, (
        "Generated $defs no longer match the downstream model surface.\n"
        f"Missing (expected but not generated): {missing}\n"
        f"Extra (generated but not expected): {extra}\n"
        "If the surface changed on purpose, update EXPECTED_DEFS and, if a "
        "model is nested under a new root, extend the signature of "
        "`_schema_surface_task.schema_surface_task`."
    )


def test_schema_is_webui_renderable(surface_schema: dict) -> None:
    # Raises on any E01-E22 violation (e.g. [E05] boolean without default),
    # the same checks `fractal-manifest create` applies downstream.
    validate_schema(
        schema=surface_schema,
        path="schema_surface_task",
        root_schema=surface_schema,
    )


def test_schema_text_is_complete(surface_schema: dict) -> None:
    """Every schema node users see in the Fractal UI carries a description.

    Descriptions come from class docstrings and attribute docstrings; the
    latter only reach the schema when a model sets
    `use_attribute_docstrings=True`. A failure here usually means a new
    model misses that config or a docstring.
    """
    problems = []
    for name, d in sorted(surface_schema["$defs"].items()):
        desc = d.get("description", "")
        if not desc or desc.startswith("Missing description"):
            problems.append(f"$defs.{name} has no class description")
        elif not desc.rstrip().endswith((".", "!", "?")):
            # fractal-task-tools keeps only the FIRST LINE of a class
            # docstring, so a multi-line summary gets cut mid-sentence.
            problems.append(
                f"$defs.{name} class description is cut mid-sentence "
                f"({desc!r}); make the docstring's first line a complete "
                "one-line summary"
            )
        for pname, p in (d.get("properties") or {}).items():
            if "$ref" in p:
                continue  # description lives on the referenced definition
            if not p.get("description"):
                problems.append(f"$defs.{name}.{pname} has no description")
    for pname, p in surface_schema["properties"].items():
        if not p.get("description"):
            problems.append(f"argument {pname} has no description")
    assert not problems, (
        "Schema nodes missing user-facing descriptions:\n- "
        + "\n- ".join(problems)
        + "\nAdd the missing class/attribute docstring (and make sure the "
        "model sets `use_attribute_docstrings=True` in its model_config)."
    )


def test_schema_matches_snapshot(surface_schema: dict, update_snapshots: bool) -> None:
    schema_str = json.dumps(surface_schema, indent=2, sort_keys=True)
    if update_snapshots:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(schema_str + "\n")
        return
    assert SNAPSHOT_PATH.exists(), (
        f"Schema snapshot not found at {SNAPSHOT_PATH}. "
        "Generate it with `pytest --update-snapshots`."
    )
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert surface_schema == snapshot, (
        "Generated task-arguments schema differs from the committed snapshot, "
        "i.e. the JSON schemas downstream converters generate from this "
        "package changed. If intentional, regenerate with "
        "`pytest --update-snapshots` and commit the new snapshot."
    )
