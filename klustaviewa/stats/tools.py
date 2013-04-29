"""Tools for computation of cluster statistics."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np



def matrix_of_pairs(dict):
    """Convert a dictionary (ci, cj) => value to a matrix."""
    keys = np.sort(np.unique(np.array(dict.keys()).ravel()))
    max = keys.max()
    indices_rel = np.zeros(max + 1, dtype=np.int32)
    for i, key in enumerate(keys):
        indices_rel[key] = i
    n = len(keys)
    # matrix = np.zeros((max + 1, max + 1))
    matrix = np.zeros((n, n))
    for (ci, cj), val in dict.iteritems():
        ci_rel, cj_rel = indices_rel[ci], indices_rel[cj]
        matrix[ci_rel, cj_rel] = val
    return matrix

    