"""Utility functions."""

import multiprocessing
import threading
from typing import Any, Callable, ClassVar


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

    def __call__(cls, *args: tuple, **kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
        """Create a new instance of the class if it doesn't exist. Otherwise, return the existing instance."""
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def run_once(
    func: Callable | None = None,
    *,
    is_exception_success: bool = True,
    passthrough_exception: bool = False,
) -> Callable:
    """
    Ensure a function is executed only once.

    After the first call, subsequent calls to the decorated function will return None.
    The first call returns the original function's return value.

    Args:
        func: The function to decorate
        is_exception_success: If True, mark as executed even if an exception occurs
        passthrough_exception: If True, re-raise any exceptions that occur

    Returns:
        A wrapped function that will only execute once

    .. note::
        This implementation is thread-safe but not multiprocessing-safe.
        For thread/process-safe version, use `run_once_multiprocessing`.

    """

    def decorator(fn: Callable) -> Callable:
        executed = False
        exception_raised = False
        lock = threading.Lock()

        def wrapper(*args: tuple, **kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
            nonlocal executed, exception_raised
            with lock:
                if not executed:
                    try:
                        result = fn(*args, **kwargs)
                        # Only set executed=True on successful execution
                        executed = True
                        return result
                    except Exception as e:
                        exception_raised = True
                        if is_exception_success:
                            executed = True
                        if passthrough_exception:
                            raise e
                return None

        return wrapper

    # Handle both @run_once and @run_once(...)
    if func is None:
        return decorator
    return decorator(func)


def run_once_multiprocessing(
    func: Callable | None = None,
    *,
    is_exception_success: bool = True,
    passthrough_exception: bool = False,
) -> Callable:
    """
    Ensure a function is executed only once in a multiprocessing and thread-safe environment.

    After the first call, every call to the decorated function will return None.
    The first call returns the original function's return value.

    Args:
        func: The function to decorate
        is_exception_success: If True, mark as executed even if an exception occurs
        passthrough_exception: If True, re-raise any exceptions that occur

    Returns:
        A wrapped function that will only execute once

    .. note::
        This implementation is both multiprocessing-safe and thread-safe.

    """

    def decorator(fn: Callable) -> Callable:
        executed = multiprocessing.Value("b", False)
        exception_raised = multiprocessing.Value("b", False)
        thread_lock = threading.Lock()

        def wrapper(*args: tuple, **kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
            with thread_lock, executed.get_lock():
                if not executed.value:
                    try:
                        result = fn(*args, **kwargs)
                        # Only set executed=True on successful execution
                        executed.value = True
                        return result
                    except Exception as e:
                        with exception_raised.get_lock():
                            exception_raised.value = True
                        if is_exception_success:
                            executed.value = True
                        if passthrough_exception:
                            raise e
                return None

        return wrapper

    # Handle both @run_once_multiprocessing and @run_once_multiprocessing(...)
    if func is None:
        return decorator
    return decorator(func)
