"""Wizard selecting automatically the best clusters to show to the user."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np


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
        self.best_pairs = {}
        self.propositions = []
        self.cluster_index = 0
        self.pair_index = -1
    
    
    # Internal methods.
    # -----------------
    def _compute_best_pairs(self):
        self.clusters_unique = np.unique(self.clusters)
        if (self.similarity_matrix is not None
            ):
            matrix = self.similarity_matrix
            quality = np.diag(matrix)
            n = matrix.shape[0]
            if n > 0:
                # TODO: handle ignoring pairs
                # self.current = -1
                self.best_clusters = self.clusters_unique[np.argsort(quality)[::-1]]
                    
                # Find all best pairs.
                matrix[np.arange(n), np.arange(n)] = 0
                indices = np.argsort(matrix.ravel())[::-1]
                clusters0 = self.clusters_unique[indices // n]
                clusters1 = self.clusters_unique[indices % n]
                best_pairs = zip(clusters0, clusters1)
                
                # Remove symmetric doublons.
                best_pairs = [(a, b) if a <= b else (b, a) for a, b in best_pairs]
                seen = set()
                seen_add = seen.add
                best_pairs = [x for x in best_pairs if x not in seen and not seen_add(x)]
                
                # Find the best pairs associated to the best clusters.
                for i, cluster in enumerate(self.best_clusters):
                    pairs = [(cl0, cl1) for (cl0, cl1) in best_pairs
                        if cl0 == cluster or cl1 == cluster]
                    self.best_pairs[cluster] = pairs
                
    
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
        pass
            
    def split(self, clusters_old, clusters_new):
        """Called to signify the wizard that a split has happened."""
        pass
        
        
    # Internal proposition methods.
    # -----------------------------
    def change_proposition(self, dir=1):
        pair = self.change_pair(dir)
        if pair is None:
            cluster = self.change_cluster(dir)
            if cluster is not None:
                return self.change_pair(dir)
            else:
                return None
        return pair
    
    def change_cluster(self, dir=1):
        if len(self.best_clusters) >= 1:
            self.cluster_index += dir
            self.pair_index = -1
            if self.cluster_index == len(self.best_clusters):
                return None
            cluster = int(self.best_clusters[self.cluster_index])
            # Add the cluster in propositions when changing cluster,
            # so that we know in the history 'propositions' when the cluster 
            # changed.
            if cluster not in self.propositions:
                self.propositions.append(cluster)
            return cluster
    
    def change_pair(self, dir=1):
        if len(self.best_clusters) >= 1:
            cluster = self.best_clusters[self.cluster_index]
            self.pair_index += dir
            if self.pair_index == len(self.best_pairs[cluster]):
                return None
            proposition = self.best_pairs[cluster][self.pair_index]
            if proposition not in self.propositions:
                self.propositions.append(proposition)
            return proposition
    
    
    # Wizard output methods.
    # ---------------------
    def previous(self):
        if len(self.propositions) > 0:
            self.propositions.pop()
            while not isinstance(self.propositions[-1], tuple):
                self.propositions.pop()
            return self.propositions[-1]
        return None
            
    def next(self):
        return self.change_proposition(1)

    def previous_cluster(self):
        if len(self.propositions) == 0:
            return None
        # Current cluster.
        cluster = int(self.best_clusters[self.cluster_index])
        # Move to the previous cluster.
        self.change_cluster(-1)
        # Find the latest proposition before the change to the current cluster.
        try:
            i = self.propositions.index(cluster)
        except ValueError:
            return None
        self.propositions = self.propositions[:i]
        return self.propositions[-1]

    def next_cluster(self):
        self.change_cluster(1)
        return self.next()
    
    
    