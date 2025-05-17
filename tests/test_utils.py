"""Test utility functions."""

import multiprocessing
import threading

import pytest

from pht.utils import run_once, run_once_multiprocessing

# Tests for run_once
# ================================================================


def test_run_once_basic_execution():
    """Test that function executes only once."""
    counter = 0

    @run_once
    def increment():
        nonlocal counter
        counter += 1

    # Call multiple times
    increment()
    increment()
    increment()

    assert counter == 1


def test_run_once_with_arguments():
    """Test that function with arguments executes only once."""
    results = []

    @run_once
    def append_args(x, y):
        results.append((x, y))

    append_args(1, 2)
    append_args(3, 4)
    append_args(5, 6)

    assert len(results) == 1
    assert results[0] == (1, 2)


def test_run_once_exception_handling():
    """Test run_once marks as executed even if an exception occurs."""
    call_count = 0

    @run_once(is_exception_success=True, passthrough_exception=True)
    def func_raises_exception():
        nonlocal call_count
        call_count += 1
        raise ValueError("Test exception")

    with pytest.raises(ValueError, match="Test exception"):
        func_raises_exception()
    assert call_count == 1

    # Subsequent calls should be no-ops and not raise the exception
    func_raises_exception()
    assert call_count == 1


def test_run_once_exception_not_success():
    """Test run_once re-executes if exception occurs and is_exception_success=False."""
    call_count = 0

    @run_once(is_exception_success=False, passthrough_exception=True)
    def func_raises_exception():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("First call exception")
        # Second call should succeed

    # First call raises exception
    with pytest.raises(ValueError, match="First call exception"):
        func_raises_exception()
    assert call_count == 1

    # Second call should execute because first call failed
    func_raises_exception()
    assert call_count == 2

    # Third call should not execute
    func_raises_exception()
    assert call_count == 2


def test_run_once_no_passthrough_exception():
    """Test run_once suppresses exceptions when passthrough_exception=False."""
    call_count = 0

    @run_once(is_exception_success=True, passthrough_exception=False)
    def func_raises_exception():
        nonlocal call_count
        call_count += 1
        raise ValueError("Test exception")

    # Should not raise exception
    func_raises_exception()
    assert call_count == 1

    # Subsequent calls should be no-ops
    func_raises_exception()
    assert call_count == 1


# Tests for run_once_multiprocessing
# ================================================================


def test_multiprocess_execution():
    """Test execution across multiple processes."""
    counter = multiprocessing.Value("i", 0)

    @run_once_multiprocessing
    def increment():
        with counter.get_lock():
            counter.value += 1

    # Create and run multiple processes
    processes = []
    for _ in range(5):
        p = multiprocessing.Process(target=increment)
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    assert counter.value == 1


def test_thread_safety():
    """Test execution across multiple threads."""
    counter = 0
    counter_lock = threading.Lock()

    @run_once_multiprocessing
    def increment():
        nonlocal counter
        with counter_lock:
            counter += 1

    # Create and run multiple threads
    threads = []
    for _ in range(5):
        t = threading.Thread(target=increment)
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    assert counter == 1


def test_with_arguments_multiprocessing():
    """Test that function with arguments executes only once across processes."""
    results = multiprocessing.Array("i", 2)  # Array to store x, y

    @run_once_multiprocessing
    def store_args(x, y):
        with results.get_lock():
            results[0] = x
            results[1] = y

    # Create and run multiple processes with different arguments
    processes = []
    args_list = [(1, 2), (3, 4), (5, 6)]
    for args in args_list:
        p = multiprocessing.Process(target=lambda: store_args(*args))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Only the first set of arguments should be stored
    assert list(results) == [1, 2]


def test_run_once_multiprocessing_exception_handling():
    """Test run_once_multiprocessing with exception handling."""
    call_count = multiprocessing.Value("i", 0)

    @run_once_multiprocessing(is_exception_success=True, passthrough_exception=True)
    def func_raises_exception():
        with call_count.get_lock():
            call_count.value += 1
        raise ValueError("Test exception")

    def worker():
        try:  # noqa: SIM105
            func_raises_exception()
        except ValueError:
            pass  # Expected exception

    # Run in multiple processes
    processes = []
    for _ in range(3):
        p = multiprocessing.Process(target=worker)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    assert call_count.value == 1  # Should only execute once


def test_run_once_multiprocessing_exception_not_success():
    """Test run_once_multiprocessing re-executes if is_exception_success=False."""
    call_attempts = multiprocessing.Value("i", 0)
    successful_executions = multiprocessing.Value("i", 0)
    manager = multiprocessing.Manager()
    log = manager.list()

    @run_once_multiprocessing(is_exception_success=False, passthrough_exception=True)
    def func_can_fail():
        with call_attempts.get_lock():
            call_attempts.value += 1
            attempt_id = call_attempts.value

        # Fail on first attempt
        if attempt_id == 1:
            log.append(
                f"Process {multiprocessing.current_process().pid} failing on attempt {attempt_id}",
            )
            raise ValueError("First attempt fails")

        # Succeed on subsequent attempts
        with successful_executions.get_lock():
            successful_executions.value += 1
        log.append(
            f"Process {multiprocessing.current_process().pid} succeeded on attempt {attempt_id}",
        )

    def worker():
        try:  # noqa: SIM105
            func_can_fail()
        except ValueError:
            pass  # Expected on first attempt

    processes = []
    for _ in range(3):
        p = multiprocessing.Process(target=worker)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # One execution should succeed after one failure
    assert call_attempts.value >= 2  # At least two attempts
    assert successful_executions.value == 1  # Only one success
    assert any("failing" in s for s in log)
    assert any("succeeded" in s for s in log)


def test_run_once_multiprocessing_combined_threads_processes():
    """Test with multiple processes, each having multiple threads."""
    call_count = multiprocessing.Value("i", 0)

    @run_once_multiprocessing
    def combined_increment():
        with call_count.get_lock():
            call_count.value += 1
        # Simulate some work
        threading.Event().wait(0.01)

    def process_worker():
        threads = []
        for _ in range(3):  # Each process spawns 3 threads
            t = threading.Thread(target=combined_increment)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    processes = []
    for _ in range(3):  # 3 processes
        p = multiprocessing.Process(target=process_worker)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    assert call_count.value == 1


def test_run_once_return_value():
    """Test that run_once preserves return values."""

    @run_once
    def func_with_return():
        return "success"

    # First call should return the value
    assert func_with_return() == "success"
    # Subsequent calls should return None
    assert func_with_return() is None


def test_run_once_thread_safety():
    """Test that run_once is thread-safe."""
    counter = 0
    counter_lock = threading.Lock()

    @run_once
    def increment():
        nonlocal counter
        with counter_lock:
            counter += 1

    threads = []
    for _ in range(5):
        t = threading.Thread(target=increment)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert counter == 1


def test_run_once_nested_calls():
    """Test nested calls to run_once decorated functions."""
    outer_count = 0
    inner_count = 0

    @run_once
    def inner_func():
        nonlocal inner_count
        inner_count += 1
        return "inner"

    @run_once
    def outer_func():
        nonlocal outer_count
        outer_count += 1
        inner_result = inner_func()
        return f"outer_{inner_result}"

    # First call should execute both functions
    assert outer_func() == "outer_inner"
    assert outer_count == 1
    assert inner_count == 1

    # Second call should execute neither
    assert outer_func() is None
    assert outer_count == 1
    assert inner_count == 1


# Module level functions for multiprocessing tests
def mp_return_value_worker(_):
    """Worker function for testing return values in multiprocessing."""
    return _mp_return_value_func()


def mp_nested_calls_worker(_):
    """Worker function for testing nested calls in multiprocessing."""
    return _mp_outer_func()


# Global variables for multiprocessing tests
_mp_success_count = None
_mp_return_value_func = None
_mp_outer_func = None
_mp_inner_count = None
_mp_outer_count = None


def test_run_once_multiprocessing_return_value():
    """Test that run_once_multiprocessing preserves return values."""
    global _mp_success_count, _mp_return_value_func  # noqa: PLW0603

    _mp_success_count = multiprocessing.Value("i", 0)

    @run_once_multiprocessing
    def func_with_return():
        with _mp_success_count.get_lock():
            _mp_success_count.value += 1
        return "success"

    _mp_return_value_func = func_with_return

    # Run in multiple processes
    with multiprocessing.Pool(3) as pool:
        results = pool.map(mp_return_value_worker, range(3))

    # Only one process should get "success", others should get None
    assert results.count("success") == 1
    assert results.count(None) == 2
    assert _mp_success_count.value == 1


def test_run_once_multiprocessing_nested_calls():
    """Test nested calls to run_once_multiprocessing decorated functions."""
    global _mp_outer_count, _mp_inner_count, _mp_outer_func  # noqa: PLW0603

    _mp_outer_count = multiprocessing.Value("i", 0)
    _mp_inner_count = multiprocessing.Value("i", 0)

    @run_once_multiprocessing
    def inner_func():
        with _mp_inner_count.get_lock():
            _mp_inner_count.value += 1
        return "inner"

    @run_once_multiprocessing
    def outer_func():
        with _mp_outer_count.get_lock():
            _mp_outer_count.value += 1
        inner_result = inner_func()
        return f"outer_{inner_result}"

    _mp_outer_func = outer_func

    # Run in multiple processes
    with multiprocessing.Pool(3) as pool:
        results = pool.map(mp_nested_calls_worker, range(3))

    # Only one process should get the full result
    assert results.count("outer_inner") == 1
    assert results.count(None) == 2
    assert _mp_outer_count.value == 1
    assert _mp_inner_count.value == 1


def test_run_once_exception_chain():
    """Test that run_once properly handles exception chains."""
    call_count = 0
    should_raise = True  # Flag to control exception raising

    @run_once(is_exception_success=False, passthrough_exception=True)
    def func_with_nested_exception():
        nonlocal call_count, should_raise
        call_count += 1
        if should_raise:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise RuntimeError("Outer error") from e
        return "success"  # Return value for successful execution

    # First call should raise the exception chain
    with pytest.raises(RuntimeError) as exc_info:
        func_with_nested_exception()
    assert "Outer error" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert "Inner error" in str(exc_info.value.__cause__)
    assert call_count == 1

    # Second call should succeed because first failed
    should_raise = False  # Disable exception for second call
    result = func_with_nested_exception()
    assert result == "success"
    assert call_count == 2

    # Third call should not execute
    result = func_with_nested_exception()
    assert result is None
    assert call_count == 2
