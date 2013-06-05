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
        assert len(self.cluster_groups) == self.similarity_matrix.shape[0]
        
            
        matrix = self.similarity_matrix
        quality = np.diag(matrix)
        n = matrix.shape[0]
        
        # Find hidden clusters (groups 0 to 2) so that they are not taken into
        # account by the wizard.
        hidden_clusters_rel = np.nonzero(self.cluster_groups <= 2)[0]
        self.best_pairs = OrderedDict()
        
        # Sort first clusters by decreasing quality.
        # Relative indices.
        best_clusters_rel = np.argsort(quality)[-1::-1]
        # Remove hidden clusters.
        best_clusters_rel = np.array([x for x in best_clusters_rel 
            if x not in hidden_clusters_rel], dtype=np.int32)
        # Absolute indices.
        self.best_clusters = self.clusters_unique[best_clusters_rel]
        
        for cluster_rel in best_clusters_rel:
            # Absolute cluster index.
            cluster = self.clusters_unique[cluster_rel]
            # Sort all neighbor clusters.
            clusters_rel = np.argsort(
                np.hstack((matrix[cluster_rel, :],
                           matrix[:, cluster_rel])))[::-1] % n
            # Remove duplicates and preserve the order.
            clusters_rel = unique(clusters_rel)
            clusters_rel.remove(cluster_rel)
            # Remove hidden clusters.
            [clusters_rel.remove(cl) for cl in hidden_clusters_rel if cl in clusters_rel]
            self.best_pairs[cluster] = self.clusters_unique[clusters_rel]
            
    
    # Data update methods.
    # --------------------
    def set_data(self, **kwargs):
        """Set the data at the beginning of the session."""
        # Set the data.
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        # Update the best pairs only if the clustering has changed.
        if 'clusters' in kwargs or 'cluster_groups' in kwargs:
            self._compute_best_pairs()
            
    def merged(self, clusters_to_merge, cluster_merged):
        """Called to signify the wizard that a merge has happened.
        No data update happens here, rather, self.update needs to be called
        with the updated data."""
        renaming = {cluster: cluster_merged for cluster in clusters_to_merge}
        self.navigator.rename(renaming)
            
    def merged_undo(self, clusters_to_merge, cluster_merged):
        self.navigator.undo_rename(clusters_to_merge)
            
    def split(self, clusters_to_split, clusters_split, clusters_empty):
        """Called to signify the wizard that a split has happened."""
        # TODO
        pass
            
    def split_undo(self, clusters_to_split, clusters_split):
        """Called to signify the wizard that a split undo has happened."""
        # TODO
        pass
        
    def moved(self, clusters, groups_old, group):
        # Delete the cluster from the list of candidates in the navigator.
        if group <= 1:
            self.navigator.hide(clusters)
        else:
            self.navigator.unhide(clusters)
    
    def moved_undo(self, clusters, groups_old, group):
        # TODO
        pass
        
    
    # Navigation methods.
    # -------------------
    def current(self):
        return self.navigator.current()
    
    def previous(self):
        pair = self.navigator.previous1()
        # if pair is None:
            # pair = self.previous_target()
        return pair
            
    def next(self):
        pair = self.navigator.next1()
        if pair is None:
            pair = self.next_target()
        return pair

    def previous_target(self):
        return self.navigator.previous0()

    def next_target(self):
        # Update the navigator with the updated pairs.
        pairs = self.best_pairs
        self.navigator.update(pairs)
        pair = self.navigator.next0()
        return pair
    
    def reset_navigation(self):
        self.navigator.reset()
    
    