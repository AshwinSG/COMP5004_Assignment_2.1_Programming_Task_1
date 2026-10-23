import numpy as np

def logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y), axis=-1)