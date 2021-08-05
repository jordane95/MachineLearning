import numpy as np


def orthogonal(dim):
    m = np.random.normal(0.0, 1.0, (dim, dim))
    u, _, v = np.linalg.svd(m)
    return u

