"""Utility functions."""

import threading
from typing import Any, ClassVar


def is_truthy(value: bool | str | int) -> bool:
    """Check if a value is truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ["true", "yes", "y"]
    if isinstance(value, int):
        return value != 0
    return False


def is_none(value: bool | str | int) -> bool:
    """Check if a value is None."""
    return value is None or str(value).lower() == "none"


def is_none_or_empty(value: bool | str | int) -> bool:
    """Check if a value is None or empty."""
    return is_none(value) or str(value) == ""


class SingletonMeta(type):
    """Metaclass for singleton classes."""

    _instances: ClassVar[dict[type, Any]] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
        """Create a new instance of the class if it doesn't exist. Otherwise, return the existing instance."""
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
