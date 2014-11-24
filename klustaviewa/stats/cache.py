"""This module implements a cache system for keeping cluster first- and
second-order statistics in memory, and updating them when necessary."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import namedtuple
from itertools import product

import numpy as np

from klustaviewa.stats.indexed_matrix import IndexedMatrix, CacheMatrix


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def is_default_slice(item):
    return (isinstance(item, slice) and item.start is None and item.stop is None
        and item.step is None)

def is_indices(item):
    return (isinstance(item, list) or isinstance(item, tuple) or 
        isinstance(item, np.ndarray) or isinstance(item, (int, long, np.integer)))
        

# -----------------------------------------------------------------------------
# Stats cache
# -----------------------------------------------------------------------------
class StatsCache(object):
    def __init__(self, ncorrbins=None):
        self.ncorrbins = ncorrbins
        self.reset()
    
    def invalidate(self, clusters):
        self.correlograms.invalidate(clusters)
        self.similarity_matrix.invalidate(clusters)
        
    def reset(self, ncorrbins=None):
        if ncorrbins is not None:
            self.ncorrbins = ncorrbins
        self.correlograms = CacheMatrix(shape=(0, 0, self.ncorrbins))
        self.similarity_matrix = CacheMatrix()
        self.similarity_matrix_normalized = None
        self.cluster_quality = None
        
    # def add(self, clusters):
        # self.correlograms.add_indices(clusters)
        # self.similarity_matrix.add_indices(clusters)
        
    # def remove(self, clusters):
        # self.correlograms.remove_indices(clusters)
        # self.similarity_matrix.remove_indices(clusters)
        
        
        