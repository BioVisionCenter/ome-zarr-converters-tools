"""Tests for the error and warning classes."""

import pickle
import warnings

import pytest

from ome_zarr_converters_tools import (
    ConvertersToolsError,
    ConvertersToolsFileNotFoundError,
    ConvertersToolsInternalError,
    ConvertersToolsOutputExistsError,
    ConvertersToolsUserError,
    ConvertersToolsWarning,
)
from ome_zarr_converters_tools.models._errors import ISSUE_TRACKER_URL, REPORT_FOOTER
from ome_zarr_converters_tools.pipelines._snap_utils import (
    NotAGridError,
    NotTilableError,
)


@pytest.mark.parametrize(
    ("error", "builtin"),
    [
        (ConvertersToolsUserError, ValueError),
        (ConvertersToolsOutputExistsError, FileExistsError),
        (ConvertersToolsFileNotFoundError, FileNotFoundError),
        (ConvertersToolsInternalError, RuntimeError),
        (NotAGridError, ValueError),
        (NotTilableError, ValueError),
    ],
)
def test_hierarchy(error: type[Exception], builtin: type[Exception]) -> None:
    assert issubclass(error, ConvertersToolsError)
    assert issubclass(error, builtin)


def test_builtin_except_clauses_keep_working() -> None:
    with pytest.raises(ValueError, match="bad input"):
        raise ConvertersToolsUserError("bad input")
    with pytest.raises(FileExistsError, match="exists"):
        raise ConvertersToolsOutputExistsError("output exists")
    with pytest.raises(FileNotFoundError, match="missing"):
        raise ConvertersToolsFileNotFoundError("file missing")
    with pytest.raises(RuntimeError, match="broken"):
        raise ConvertersToolsInternalError("invariant broken")


def test_builtin_mixins_render_the_message() -> None:
    assert str(ConvertersToolsOutputExistsError("output exists")) == "output exists"
    assert str(ConvertersToolsFileNotFoundError("file missing")) == "file missing"


def test_internal_error_appends_report_footer() -> None:
    error = ConvertersToolsInternalError("invariant broken")
    message = str(error)
    assert message.startswith("invariant broken")
    assert message.endswith(REPORT_FOOTER)
    assert ISSUE_TRACKER_URL in message
    # The footer is a rendering concern only: the raw argument stays clean.
    assert error.args == ("invariant broken",)


def test_user_error_has_no_footer() -> None:
    assert str(ConvertersToolsUserError("bad input")) == "bad input"


@pytest.mark.parametrize(
    "error",
    [
        ConvertersToolsUserError("bad input"),
        ConvertersToolsInternalError("invariant broken"),
        ConvertersToolsOutputExistsError("output exists"),
    ],
)
def test_errors_survive_pickling(error: Exception) -> None:
    # Errors cross process boundaries under `MultiprocessingRunner`.
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is type(error)
    assert str(restored) == str(error)


def test_warning_is_a_user_warning() -> None:
    assert issubclass(ConvertersToolsWarning, UserWarning)
    with pytest.warns(UserWarning, match="careful"):
        warnings.warn("careful", ConvertersToolsWarning, stacklevel=1)
