"""This module implements the evaluation of clusters quality."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

from kwiklib.dataio.selection import select, get_spikes_in_clusters
from kwiklib.dataio.tools import get_array
from klustaviewa.stats.correlations import (normalize,
    get_similarity_matrix)


# -----------------------------------------------------------------------------
# Cluster quality
# -----------------------------------------------------------------------------
def cluster_quality(waveforms, features, clusters, masks,
    clusters_selected=None):
    # clusters = select(clusters, spikes)
    # features = select(features, spikes)
    # waveforms = select(waveforms, spikes)
    # masks = select(masks, spikes)

    nspikes, nsamples, nchannels = waveforms.shape
    quality = {}

    for cluster in clusters_selected:
        spikes = get_spikes_in_clusters(cluster, clusters)
        w = select(waveforms, spikes)
        m = select(masks, spikes)
        q = 1. / nsamples * ((w ** 2).sum(axis=1) * 1).mean(axis=1).max()
        quality[cluster] = q

    return quality


