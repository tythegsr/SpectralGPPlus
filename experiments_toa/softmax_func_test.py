import numpy as np

z = np.array([[10, 0, 0], [5, -5, -5], [3, 3, 5]])


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-30, None)

print(softmax_rows(z))