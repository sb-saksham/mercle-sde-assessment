import numpy as np

def unit_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize rows of X to unit L2 norm. Works in-place where possible.
    X : shape (n, d) or (d,) for a single vector
    Returns normalized copy.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        norm = np.linalg.norm(X)
        if norm < eps:
            raise ValueError("Vector norm too small to normalize.")
        return (X / norm).astype(np.float32)
    elif X.ndim == 2:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        too_small = norms.squeeze() < eps
        if np.any(too_small):
            raise ValueError(f"{np.sum(too_small)} rows have (near-)zero norm.")
        return (X / norms).astype(np.float32)
    else:
        raise ValueError("Input must be 1D or 2D numpy array.")


def batch_indices(n_items: int, batch_size: int):
    """
    Yield (start, end) indices splitting n_items into batches of given size.
    """
    for i in range(0, n_items, batch_size):
        yield i, min(i + batch_size, n_items)
