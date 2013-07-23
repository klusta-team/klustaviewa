"""Unit tests for dataio.tools module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from klustaviewa.dataio import normalize, find_filename


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_normalize():
    data = np.array([.5, .75, 1.])
    
    # normalization
    data_normalized = normalize(data)
    assert np.array_equal(data_normalized, [-1, 0, 1])
    
    # normalization with a different range
    data_normalized = normalize(data, range=(0, 1))
    assert np.array_equal(data_normalized, [0, 0.5, 1])
    
    # symmetric normalization (0 stays 0)
    data_normalized = normalize(data, symmetric=True)
    assert np.array_equal(data_normalized, data)

    
    
    