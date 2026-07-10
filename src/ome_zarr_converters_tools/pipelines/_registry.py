"""Shared name → function registry backing the pipeline extension points."""

from typing import Generic, TypeVar

F = TypeVar("F")


class Registry(Generic[F]):
    """Process-global registry for pipeline extension functions.

    Registrations only exist in the current process: under
    `MultiprocessingRunner` worker processes re-import the consumer's modules,
    so custom functions must be registered at import time of the module that
    defines them (not inside a function or `if __name__ == "__main__"` block)
    to be visible in workers.
    """

    def __init__(
        self,
        kind: str,
        register_hint: str,
        initial: dict[str, F] | None = None,
    ) -> None:
        """Create a registry.

        Args:
            kind: Human-readable name of the extension point, used in errors
                (e.g. `"Validator step"`).
            register_hint: Name of the public registration function, used in
                errors (e.g. `"add_validator"`).
            initial: Built-in functions to pre-register.
        """
        self._kind = kind
        self._register_hint = register_hint
        self._registry: dict[str, F] = dict(initial or {})

    def add(
        self, *, function: F, name: str | None = None, overwrite: bool = False
    ) -> None:
        """Register `function` under `name` (defaults to `function.__name__`)."""
        if name is None:
            name = getattr(function, "__name__")  # noqa: B009
        if not overwrite and name in self._registry:
            raise ValueError(
                f"{self._kind} '{name}' is already registered. Pass "
                f"overwrite=True to `{self._register_hint}` to replace it, or "
                "register under a different name."
            )
        self._registry[name] = function

    def get(self, name: str | None) -> F:
        """Return the function registered under `name`; raise if unknown."""
        if name is None or name not in self._registry:
            raise ValueError(
                f"{self._kind} '{name}' is not registered; available: "
                f"{sorted(self._registry)}. Register custom functions with "
                f"`{self._register_hint}` at import time of the module defining "
                "them, so that multiprocessing workers see them too."
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        """Return whether `name` is registered."""
        return name in self._registry

    def pop(self, name: str, default: F | None = None) -> F | None:
        """Remove and return the function registered under `name`."""
        return self._registry.pop(name, default)
