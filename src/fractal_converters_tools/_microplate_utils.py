"""Utilities for working with microplates."""

STANDARD_PLATES_LAYOUTS = {
    "24-well": {
        "rows": 4,
        "columns": 6,
    },
    "48-well": {
        "rows": 6,
        "columns": 8,
    },
    "96-well": {
        "rows": 8,
        "columns": 12,
    },
    "384-well": {
        "rows": 16,
        "columns": 24,
    },
}

STANDARD_ROWS_NAMES = "ABCDEFGHIJKLMNOP"


def get_row_column(well_id: int, layout: str) -> tuple[str, int]:
    """Get row and column from well id."""
    if layout not in STANDARD_PLATES_LAYOUTS:
        raise ValueError(f"Layout {layout} not found.")

    layout_dict = STANDARD_PLATES_LAYOUTS[layout]
    num_columns = layout_dict["columns"]
    well_id -= 1

    row = well_id // num_columns
    column = well_id % num_columns + 1

    if row >= layout_dict["rows"]:
        raise ValueError(f"Well id {well_id} is out of bounds for layout {layout}.")

    row_str = STANDARD_ROWS_NAMES[row]
    return row_str, column
