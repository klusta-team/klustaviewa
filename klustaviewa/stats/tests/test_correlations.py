"""Unit tests for stats.correlations module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from klustaviewa.stats.correlations import compute_correlations
from klustaviewa.stats.tools import matrix_of_pairs


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_compute_correlations():
    
    n = 1000
    nspikes = 3 * n
    clusters = np.repeat([0, 1, 2],  n)
    features = np.zeros((nspikes, 2))
    masks = np.ones((nspikes, 2))
    
    # clusters 0 and 1 are close, 2 is far away from 0 and 1
    features[:n, :] = np.random.randn(n, 2)
    features[n:2*n, :] = np.random.randn(n, 2)
    features[2*n:, :] = np.array([[10, 10]]) + np.random.randn(n, 2)
    
    # compute the correlation matrix
    correlations = compute_correlations(features, clusters, masks)
    matrix = matrix_of_pairs(correlations)
    
    # check that correlation between 0 and 1 is much higher than the
    # correlation between 2 and 0/1
    assert matrix[0,1] > 100 * matrix[0, 2]
    assert matrix[0,1] > 100 * matrix[1, 2]
    
    