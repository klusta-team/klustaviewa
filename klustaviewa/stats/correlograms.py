"""This module implements the computation of the cross-correlograms between
clusters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from itertools import product

import numpy as np

from kwiklib.utils import logger as log

# Trying to load the Cython version.
try:
    from correlograms_cython import compute_correlograms_cython as compute_correlograms
    log.debug(("Trying to load the compiled Cython version of the correlograms"
               "computations..."))
except Exception as e:
    log.debug(e.message)
    try:
        log.debug(("failed. Trying to use Cython directly..."))
        import pyximport; pyximport.install(
            setup_args={'include_dirs': np.get_include()})
        from correlograms_cython import compute_correlograms_cython as compute_correlograms
    except Exception as e:
        log.debug(e.message)
        log.info(("Unable to load the fast Cython version of the correlograms"
                   "computations, so falling back to the pure Python version.")
                   )

        # Pure Python version.
        # --------------------
        def compute_correlograms(spiketimes, clusters, clusters_to_update=None,
            ncorrbins=None, corrbin=None):
            
            if ncorrbins is None:
                ncorrbins = NCORRBINS_DEFAULT
            if corrbin is None:
                corrbin = CORRBIN_DEFAULT
            
            # Ensure ncorrbins is an even number.
            assert ncorrbins % 2 == 0
            
            # Compute the histogram corrbins.
            # n = int(np.ceil(halfwidth / corrbin))
            n = ncorrbins // 2
            halfwidth = corrbin * n
            
            # size of the histograms
            nspikes = len(spiketimes)
            
            # unique clusters
            clusters_unique = np.unique(clusters)
            nclusters = len(clusters_unique)
            cluster_max = clusters_unique[-1]
            
            # clusters to update
            if clusters_to_update is None:
                clusters_to_update = clusters_unique
            clusters_mask = np.zeros(cluster_max + 1, dtype=np.bool)
            clusters_mask[clusters_to_update] = 1
            
            # initialize the correlograms
            correlograms = np.zeros(
        ((cluster_max + 1) ** 2, ncorrbins), dtype=np.int32)

            # loop through all spikes, across all neurons, all sorted
            for i in range(nspikes):
                t0, cl0 = spiketimes[i], clusters[i]
                # pass clusters that do not need to be processed
                if clusters_mask[cl0]:
                    # i, t0, c0: current spike index, spike time, and cluster
                    # boundaries of the second loop
                    t0min, t0max = t0 - halfwidth, t0 + halfwidth
                    j = i + 1
                    # go forward in time up to the correlogram half-width
                    while j < nspikes:
                        t1, cl1 = spiketimes[j], clusters[j]
                        # pass clusters that do not need to be processed
                        # if clusters_mask[cl1]:
                        # compute only correlograms if necessary
                        # and avoid computing symmetric pairs twice
                        if t1 < t0max:
                            d = t1 - t0
                            k = int(d / corrbin) + n
                            ind = (cluster_max + 1) * cl0 + cl1
                            correlograms[ind, k] += 1
                        else:
                            break
                        j += 1
                    j = i - 1
                    # go backward in time up to the correlogram half-width
                    while j >= 0:
                        t1, cl1 = spiketimes[j], clusters[j]
                        # pass clusters that do not need to be processed
                        # compute only correlograms if necessary
                        # and avoid computing symmetric pairs twice
                        if t0min < t1:
                            d = t1 - t0
                            k = int(d / corrbin) + n - 1
                            ind = (cluster_max + 1) * cl0 + cl1
                            correlograms[ind, k] += 1
                        else:
                            break
                        j -= 1
            dic = {(cl0, cl1): correlograms[(cluster_max + 1) * cl0 + cl1,:][::-1]
                for cl0 in clusters_to_update for cl1 in clusters_unique}
            # Add the symmetric pairs.
            dic.update({(cl1, cl0): dic[cl0, cl1]
                for cl0 in clusters_to_update for cl1 in clusters_unique})
            return dic


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
NCORRBINS_DEFAULT = 100
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
