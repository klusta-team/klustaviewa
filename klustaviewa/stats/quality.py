"""This module implements the evaluation of clusters quality."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np

from klustaviewa.dataio.selection import select, get_spikes_in_clusters


# -----------------------------------------------------------------------------
# Cluster quality
# -----------------------------------------------------------------------------
def cluster_quality(clusters, features, waveforms, masks, clusters_selected):
    spikes = get_spikes_in_clusters(clusters_selected, clusters)
    # clusters = select(clusters, spikes)
    # features = select(features, spikes)
    # waveforms = select(waveforms, spikes)
    # masks = select(masks, spikes)
    
    quality = {}
    
    for cluster in clusters_selected:
        pass
    
    return quality


