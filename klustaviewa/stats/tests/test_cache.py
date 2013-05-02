"""Unit tests for stats.cache module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from nose.tools import raises
import numpy as np

from klustaviewa.stats.cache import StatsCache


# -----------------------------------------------------------------------------
# Cache tests
# -----------------------------------------------------------------------------
def test_cache():
    indices = [2, 3, 5, 7]
    cache = StatsCache(ncorrbins=100)
    
    np.array_equal(cache.correlograms.not_in_indices(indices), indices)
    np.array_equal(cache.similarity_matrix.not_in_indices(indices), indices)
    
    d = {(2, i): 0 for i in indices}
    d.update({(i, 2): 0 for i in indices})
    cache.correlograms.update(2, d)
    
    np.array_equal(cache.correlograms.not_in_key_indices(indices), [3, 5, 7])
    
    cache.invalidate(2)
    
    np.array_equal(cache.correlograms.not_in_key_indices(indices), indices)
    np.array_equal(cache.similarity_matrix.not_in_key_indices(indices), 
        indices)
    
    
    