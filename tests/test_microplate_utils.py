import pytest

from fractal_converters_tools._microplate_utils import (
    wellid_to_row_column,
)


@pytest.mark.parametrize(
    "well_id, expenced_row, expenced_column, layout",
    [
        (1, "A", 1, "24-well"),
        (10, "A", 10, "96-well"),
        (15, "B", 3, "96-well"),
    ],
)
def test_get_row_column(well_id, expenced_row, expenced_column, layout):
    row, column = wellid_to_row_column(well_id, layout)
    assert row == expenced_row
    assert column == expenced_column


def test_get_row_column_out_of_bounds():
    with pytest.raises(ValueError):
        wellid_to_row_column(150, layout="96-well")


def test_layout_not_found():
    with pytest.raises(ValueError):
        wellid_to_row_column(1, layout="wrong-layout")
