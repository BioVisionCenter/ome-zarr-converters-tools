"""Shared models used across multiple modules."""

from enum import StrEnum


class OverwriteMode(StrEnum):
    NO_OVERWRITE = "No Overwrite"
    OVERWRITE = "Overwrite"
    EXTEND = "Extend"
