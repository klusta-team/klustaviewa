"""This module implements the computation of the cross-correlograms between
clusters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

from .ccg import correlograms


def compute_correlograms(spiketimes,
                         clusters,
                         clusters_to_update=None,
                         ncorrbins=None,
                         corrbin=None,
                         sample_rate=None,
                         ):

    if ncorrbins is None:
        ncorrbins = NCORRBINS_DEFAULT
    if corrbin is None:
        corrbin = CORRBIN_DEFAULT

    # Sort spiketimes for the computation of the CCG.
    ind = np.argsort(spiketimes)
    spiketimes = spiketimes[ind]
    clusters = clusters[ind]

    window_size = corrbin * ncorrbins

    # unique clusters
    clusters_unique = np.unique(clusters)

    # clusters to update
    if clusters_to_update is None:
        clusters_to_update = clusters_unique

    # Select requested clusters.
    ind = np.in1d(clusters, clusters_to_update)
    spiketimes = spiketimes[ind]
    clusters = clusters[ind]

    assert spiketimes.shape == clusters.shape
    assert np.all(np.in1d(clusters, clusters_to_update))
    assert sample_rate > 0.
    assert 0 < corrbin < window_size

    C = correlograms(spiketimes,
                     clusters,
                     cluster_ids=clusters_to_update,
                     sample_rate=sample_rate,
                     bin_size=corrbin,
                     window_size=window_size,
                     )
    dic = {(c0, c1): C[i, j, :]
           for i, c0 in enumerate(clusters_to_update)
           for j, c1 in enumerate(clusters_to_update)}
    return dic


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
NCORRBINS_DEFAULT = 101
CORRBIN_DEFAULT = .001


# -----------------------------------------------------------------------------
# Computing one correlogram
# -----------------------------------------------------------------------------
def compute_one_correlogram(spikes0, spikes1, ncorrbins, corrbin):
    clusters = np.hstack((np.zeros(len(spikes0), dtype=np.int32),
                          np.ones(len(spikes1), dtype=np.int32)))
    spikes = np.hstack((spikes0, spikes1))
    # Indices sorting the union of spikes0 and spikes1.
    indices = np.argsort(spikes)
    C = compute_correlograms(spikes[indices], clusters[indices],
        ncorrbins=ncorrbins, corrbin=corrbin)
    return C[0, 1]


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
def get_baselines(sizes, duration, corrbin):
    baselines = (sizes.reshape((-1, 1)) * sizes.reshape((1, -1))
                    * corrbin / (duration))
    return baselines





# Utility functions
def excerpt_step(nsamples, nexcerpts=None, excerpt_size=None):
    step = max((nsamples - excerpt_size) // (nexcerpts - 1),
               excerpt_size)
    return step

def excerpts(nsamples, nexcerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    step = excerpt_step(nsamples,
                        nexcerpts=nexcerpts,
                        excerpt_size=excerpt_size)
    for i in range(nexcerpts):
        start = i * step
        if start >= nsamples:
            break
        end = min(start + excerpt_size, nsamples)
        yield start, end

def get_excerpts(data, nexcerpts=None, excerpt_size=None):
    nsamples = data.shape[0]
    return np.concatenate([data[start:end,...]
                          for (start, end) in excerpts(nsamples,
                                                       nexcerpts=nexcerpts,
                                                       excerpt_size=excerpt_size)],
                          axis=-1)
