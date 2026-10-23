import time
import numpy as np


def benchmark(func, *args, repeats=5, **kwargs):
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    result = None
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        "result": result,
        "median_seconds": float(np.median(times)),
        "times_seconds": times
    }


def python_loop_mean(X):
    values = np.asarray(X, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("X cannot be empty.")
    total = 0.0
    for value in values:
        total += value
    return total / values.size


def numpy_mean(X):
    values = np.asarray(X, dtype=float)
    if values.size == 0:
        raise ValueError("X cannot be empty.")
    return float(np.mean(values))


def compare_means(size=100_000, repeats=5, seed=42):
    if size <= 0:
        raise ValueError("size must be positive.")

    data = np.random.default_rng(seed).normal(size=size)
    loop = benchmark(python_loop_mean, data, repeats=repeats)
    vector = benchmark(numpy_mean, data, repeats=repeats)

    return {
        "size": size,
        "python_loop_seconds": loop["median_seconds"],
        "numpy_seconds": vector["median_seconds"],
        "loop_mean": loop["result"],
        "numpy_mean": vector["result"],
        "speedup": (
            loop["median_seconds"] / vector["median_seconds"]
            if vector["median_seconds"] > 0 else float("inf")
        )
    }
