"""Unit tests for stats.tools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from klustaviewa.stats.tools import matrix_of_pairs


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_matrix_of_pairs():
    dic = {(2, 2): 0.22, (3, 2): 0.10, (2, 3): .03}
    mat = matrix_of_pairs(dic)
    mat_actual = np.array([
        [.22, 0.03],
        [0.1, 0]
    ])
    assert np.array_equal(mat, mat_actual)
    
