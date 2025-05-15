import threading
from typing import Any


def is_truthy(value: Any) -> bool:
    """Check if a value is truthy."""

    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() in ["true", "yes", "y"]
    elif isinstance(value, int):
        return value != 0
    else:
        raise ValueError(f"Invalid value: {value}")


def is_none(value: Any) -> bool:
    """Check if a value is None."""
    return value is None or str(value).lower() == "none"


def is_none_or_empty(value: Any) -> bool:
    """Check if a value is None or empty."""
    return is_none(value) or str(value) == ""


class SingletonMeta(type):
    """Metaclass for singleton classes."""

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
