"""Robot selecting automatically the best clusters to show to the user."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np


# -----------------------------------------------------------------------------
# Robot
# -----------------------------------------------------------------------------
class Robot(object):
    """Robot object, takes the data parameters and returns propositions of
    clusters to select."""
    def __init__(self, features=None, spiketimes=None, clusters=None, 
        masks=None, cluster_groups=None, 
        correlograms=None, correlation_matrix=None):
        self.features = features
        self.spiketimes = spiketimes
        self.clusters = clusters
        self.masks = masks
        self.cluster_groups = cluster_groups
        self.correlograms = correlograms
        self.correlation_matrix = correlation_matrix
        self.best_pairs = []
        self.current = -1
    
    
    # Internal methods.
    # -----------------
    def _compute_best_pairs(self):
        if self.correlation_matrix is not None:
            matrix = self.correlation_matrix.to_array().copy()
            n = matrix.shape[0]
            if n > 0:
                self.current = -1
                matrix[np.arange(n), np.arange(n)] = 0
                indices = np.argsort(matrix.ravel())[::-1]
                # print indices[:10], matrix.ravel()[indices[:10]]
                clusters0 = self.correlation_matrix.to_absolute(indices // n)
                clusters1 = self.correlation_matrix.to_absolute(indices % n)
                self.best_pairs = zip(clusters0, clusters1)
        
    
    # Data update methods.
    # --------------------
    def set_data(self, **kwargs):
        """Set the data at the beginning of the session."""
        # Set the data.
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        # Update the unique clusters.
        if self.clusters is not None:
            self.clusters_unique = np.array(sorted(Counter(
                self.clusters).keys()))
        # Update the best pairs only if the clustering has changed.
        if 'clusters' in kwargs:
            self._compute_best_pairs()
            
    def merged(self, clusters_to_merge, cluster_new):
        """Called to signify the robot that a merge has happened.
        No data update happens here, rather, self.update needs to be called
        with the updated data."""
        pass
            
    def split(self, clusters_old, clusters_new):
        """Called to signify the robot that a split has happened."""
        pass
        
    # def data_changed(self, clusters=None):
        # self._compute_best_pairs()
        
    
    # Robot output methods.
    # ---------------------
    def previous(self):
        if len(self.best_pairs) >= 1:
            self.current = max(self.current - 1, 0)
            return self.best_pairs[self.current]
            
    def next(self):
        if len(self.best_pairs) >= 1:
            self.current = min(self.current + 1, len(self.best_pairs) - 1)
            return self.best_pairs[self.current]
    
    
    