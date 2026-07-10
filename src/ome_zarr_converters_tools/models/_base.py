"""Shared pydantic base for models exposed to the Fractal UI."""

from pydantic import BaseModel, ConfigDict


class UserFacingModel(BaseModel):
    """Base for models rendered as task parameters in the Fractal UI.

    - `use_attribute_docstrings=True` so field docstrings reach the JSON
      schema (and the UI) regardless of import order; fractal-task-tools
      only patches this in globally when it is imported before the models
      are defined.
    - `extra="forbid"` so typos in parameter files fail loudly instead of
      being silently ignored.

    Subclass `model_config` entries (e.g. `title`) merge with this config.
    """

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
