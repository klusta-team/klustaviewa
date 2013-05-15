"""Wizard selecting automatically the best clusters to show to the user."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter, OrderedDict

import numpy as np

from klustaviewa.wizard.pair_navigator import PairNavigator


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def unique(seq):
    """Remove duplicates from a sequence whilst preserving order."""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]

    
# -----------------------------------------------------------------------------
# Wizard
# -----------------------------------------------------------------------------
class Wizard(object):
    """Wizard object, takes the data parameters and returns propositions of
    clusters to select."""
    def __init__(self, features=None, spiketimes=None, clusters=None, 
        masks=None, cluster_groups=None, 
        correlograms=None, similarity_matrix=None, 
        ):
        self.features = features
        self.spiketimes = spiketimes
        self.clusters = clusters
        self.masks = masks
        self.cluster_groups = cluster_groups
        self.correlograms = correlograms
        self.similarity_matrix = similarity_matrix
        
        # self.renaming = {}
        self.best_clusters = []
        self.best_pairs = OrderedDict()
        self.navigator = PairNavigator()
        
    
    # Computation of the best pairs.
    # ------------------------------
    def _compute_best_pairs(self):
        self.clusters_unique = np.unique(self.clusters)
        
        if (self.similarity_matrix is None or 
            self.similarity_matrix.size == 0):
            return
        
        assert len(self.clusters_unique) == self.similarity_matrix.shape[0]
            
        matrix = self.similarity_matrix
        quality = np.diag(matrix)
        n = matrix.shape[0]
    
        self.best_pairs = OrderedDict()
        
        # Sort first clusters by decreasing quality.
        # Relative indices.
        best_clusters_rel = np.argsort(quality)[-1::-1]
        # Absolute indices.
        self.best_clusters = self.clusters_unique[best_clusters_rel]
        
        for cluster_rel in best_clusters_rel:
            # Absolute cluster index.
            cluster = self.clusters_unique[cluster_rel]
            # Sort all neighbor clusters.
            clusters = np.argsort(
                np.hstack((matrix[cluster_rel, :],
                           matrix[:, cluster_rel])))[::-1] % n
            # Remove duplicates and preserve the order.
            clusters = unique(clusters)
            clusters.remove(cluster_rel)
            self.best_pairs[cluster] = self.clusters_unique[clusters]
            
    
    # Data update methods.
    # --------------------
    def set_data(self, **kwargs):
        """Set the data at the beginning of the session."""
        # Set the data.
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        # Update the best pairs only if the clustering has changed.
        if 'clusters' in kwargs:
            self._compute_best_pairs()
            
    def merged(self, clusters_to_merge, cluster_new):
        """Called to signify the wizard that a merge has happened.
        No data update happens here, rather, self.update needs to be called
        with the updated data."""
        renaming = {cluster: cluster_new for cluster in clusters_to_merge}
        self.navigator.rename(renaming)
            
    def merged_undo(self, clusters_to_merge):
        self.navigator.undo_rename(clusters_to_merge)
            
    def split(self, clusters_old, clusters_new):
        """Called to signify the wizard that a split has happened."""
        pass
        
        
    
    # Navigation methods.
    # -------------------
    def previous(self):
        pair = self.navigator.previous1()
        if pair is None:
            pair = self.previous_cluster()
        return pair
            
    def next(self):
        pair = self.navigator.next1()
        if pair is None:
            pair = self.next_cluster()
        return pair

    def previous_cluster(self):
        return self.navigator.previous0()

    def next_cluster(self):
        # Update the navigator with the updated pairs.
        pairs = self.best_pairs
        self.navigator.update(pairs)#, renaming=self.renaming)
        pair = self.navigator.next0()
        # Reset the renaming dictionary.
        # self.renaming = {}
        return pair
    
    def reset_navigation(self):
        self.navigator.reset()
    
    