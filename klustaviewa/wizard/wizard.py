"""Wizard selecting automatically the best clusters to show to the user."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import OrderedDict

import numpy as np

from kwiklib.utils import logger as log
from kwiklib.dataio.selection import get_indices
from kwiklib.dataio.tools import get_array


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
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.target = None
        self.candidates = []
        # List of skipped candidates.
        self.skipped = []
        # List of skipped targets.
        self.skipped_targets = []
        # Current position in the candidates list.
        self.index = 0
        # Size of the candidates list.
        self.size = 0 
        
        self.quality = None
        self.matrix = None
        
        
    # Data update methods.
    # --------------------
    def set_data(self, cluster_groups=None, similarity_matrix=None):
        """Update the data."""
        
        if cluster_groups is not None:
            self.clusters_unique = get_array(get_indices(cluster_groups))
            self.cluster_groups = get_array(cluster_groups)
            
        if (similarity_matrix is not None and similarity_matrix.size > 0):

            if len(get_array(cluster_groups)) != similarity_matrix.shape[0]:
                log.warn(("Cannot update the wizard: cluster_groups "
                    "has {0:d} elements whereas the similarity matrix has {1:d}.").format(
                        len(get_array(cluster_groups)), similarity_matrix.shape[0]))
                return

            self.matrix = similarity_matrix
            self.quality = np.diag(self.matrix)
            
        
    
    # Core methods.
    # -------------
    def find_target(self):
        if self.quality is None:
            return
        # For the target, only consider the unsorted clusters, and remove
        # the skipped targets.
        kept = ((self.cluster_groups >= 3) & 
            (~np.in1d(self.clusters_unique, self.skipped_targets)))
        quality_kept = self.quality[kept]
        if len(quality_kept) == 0:
            return None
        quality_best = quality_kept.max()
        target_rel = np.nonzero(kept & (self.quality == quality_best))[0][0]
        target = self.clusters_unique[target_rel]
        return target
    
    def find_candidates(self, target):
        if target is None:
            return []
        
        # Relative target.
        try:
            target_rel = np.nonzero(self.clusters_unique == target)[0][0]
        except IndexError:
            log.debug("Target cluster {0:d} does not exist.".format(target))
            return []
        
        hidden = self.cluster_groups <= 1
        
        # Hide values in the matrix for hidden clusters.
        matrix = self.matrix.copy()
        matrix[hidden, :] = -1
        matrix[:, hidden] = -1
        n = matrix.shape[0]
        
        # Sort all neighbor clusters.
        clusters_rel = np.argsort(
            np.hstack((matrix[target_rel, :],
                       matrix[:, target_rel])))[::-1] % n
                       
        # Remove duplicates and preserve the order.
        clusters_rel = unique(clusters_rel)
        clusters_rel.remove(target_rel)
        
        # Remove hidden clusters.
        [clusters_rel.remove(cl) for cl in np.nonzero(hidden)[0] 
            if cl in clusters_rel]
        
        candidates = self.clusters_unique[clusters_rel]
        return candidates
    
    def update_candidates(self, target=None):
        # Find the target if it is not specified.
        if target is None:
            target = self.find_target()
        elif target is True:
            target = self.current_target()
        
        # Find the ordered list of candidates for the specified target.
        candidates = self.find_candidates(target)
        
        self.target = target
        self.candidates = candidates
        self.size = len(candidates)
    
        # Current position in the candidates list.
        self.index = 0
        # Skip all skipped candidates.
        if self.size >= 1:
            while self.candidates[self.index] in self.skipped:
                self.index += 1
                if self.index >= self.size - 1:
                    break
        
    def reset_skipped(self):
        self.skipped = []
        
    
    # Navigation methods.
    # -------------------
    def mark_skipped(self):
        candidate = self.current_candidate()
        if candidate is None:
            return
        if candidate not in self.skipped:
            self.skipped.append(candidate)
    
    def current_target(self):
        if self.size == 0:
            return None
        return self.target
    
    def current_candidate(self):
        if self.size == 0 or not(0 <= self.index <= self.size - 1):
            return None
        candidate = self.candidates[self.index]
        return candidate
    
    def current_pair(self):
        candidate = self.current_candidate()
        if candidate is not None:
            return self.current_target(), candidate
    
    def previous_candidate(self):
        if self.size == 0 or self.index <= 0:
            return self.current_candidate()
        self.index -= 1
        return self.current_candidate()
    
    def previous_pair(self):
        candidate = self.previous_candidate()
        if candidate is not None:
            return self.current_target(), candidate

    def next_candidate(self):
        # Return the current candidate if it is the first call to this function.
        if self.index == 0 and self.current_candidate() not in self.skipped:
            self.mark_skipped()
            return self.current_candidate()
        self.mark_skipped()
        if self.size == 0 or self.index >= self.size - 1:
            return self.current_candidate()
        self.index += 1
        return self.current_candidate()

    def next_pair(self):
        candidate = self.next_candidate()
        if candidate is not None:
            return self.current_target(), candidate
    
    def skip_target(self):
        self.skipped_targets.append(self.current_target())
    
    