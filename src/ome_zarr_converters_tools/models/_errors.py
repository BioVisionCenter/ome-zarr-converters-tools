"""Error and warning classes raised by ome-zarr-converters-tools.

Errors are split by who can act on them:

- `ConvertersToolsUserError`: the task user or the converter author can fix it
  by changing task parameters, converter options, the input data, the
  environment or the output location. The message says what is wrong, naming
  the image, tile and source file when known, with the expected and actual
  values, and how to fix it.
- `ConvertersToolsInternalError`: an invariant that a working converter can
  never violate. The message always ends with where to report the bug.

Every error subclasses `ConvertersToolsError`, so `except ConvertersToolsError`
catches all of them. Each one also subclasses the builtin it stands in for
(`ValueError`, `FileExistsError`, `FileNotFoundError`), so existing code that
catches those keeps working.

Message contract for new raise sites:

- User errors: first what is wrong, with the identifying context and the
  expected vs actual values; then how to fix it.
- Internal errors: the violated invariant with the values in scope. The report
  footer is appended automatically, never written by hand.
- Warnings: `warnings.warn(message, ConvertersToolsWarning, stacklevel=2)`,
  with the same "what is wrong, how to fix it" shape as user errors.
- Errors raised inside pydantic validators stay plain `ValueError`: pydantic
  wraps them in a `ValidationError` and the class is lost anyway.
"""

ISSUE_TRACKER_URL = "https://github.com/BioVisionCenter/ome-zarr-converters-tools/issues"
"""Where bugs in ome-zarr-converters-tools are reported."""

REPORT_FOOTER = (
    "This should not happen in a working converter and is most likely a bug in "
    "the converter or in ome-zarr-converters-tools. Please report it, including "
    "the full traceback, to the converter's issue tracker or at "
    f"{ISSUE_TRACKER_URL}"
)
"""Footer appended to every `ConvertersToolsInternalError` message."""


class ConvertersToolsError(Exception):
    """Base class for all ome-zarr-converters-tools errors."""


class ConvertersToolsUserError(ConvertersToolsError, ValueError):
    """Error the task user or converter author can fix.

    Raised for wrong task parameters or converter options, inconsistent or
    unreadable input data, a missing optional dependency, or an output
    location that already exists. The message names what is wrong and how to
    fix it.
    """


class ConvertersToolsOutputExistsError(ConvertersToolsUserError, FileExistsError):
    """Error raised when the output already exists and `overwrite_mode` forbids
    replacing it."""


class ConvertersToolsFileNotFoundError(ConvertersToolsUserError, FileNotFoundError):
    """Error raised when an input or intermediate file cannot be found."""


class ConvertersToolsInternalError(ConvertersToolsError, RuntimeError):
    """Error raised when an invariant of the conversion pipeline is violated.

    A working converter can never trigger it: it signals a bug in the converter
    implementation (e.g. a parser or loader breaking the `Tile` contract) or in
    ome-zarr-converters-tools itself. `str()` of the error ends with
    `REPORT_FOOTER`, so call sites only write the diagnostic sentence.
    """

    def __str__(self) -> str:
        """Return the message followed by the report footer."""
        return f"{super().__str__()}\n\n{REPORT_FOOTER}"


class ConvertersToolsWarning(UserWarning):
    """Base class for all warnings emitted by ome-zarr-converters-tools."""
