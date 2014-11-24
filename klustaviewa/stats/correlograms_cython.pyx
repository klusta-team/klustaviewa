import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
DTYPEI = np.int32
ctypedef np.int32_t DTYPEI_t

def compute_correlograms_cython(
     np.ndarray[DTYPE_t, ndim=1] spiketimes,
     np.ndarray[DTYPEI_t, ndim=1] clusters,
     np.ndarray[DTYPEI_t, ndim=1] clusters_to_update=None,
     long ncorrbins=100,
     double corrbin=.001):
    
    # Ensure ncorrbins is an even number.
    assert ncorrbins % 2 == 0
    
    # Compute the histogram corrbins.
    cdef long n = ncorrbins // 2
    cdef double halfwidth = corrbin * n
    
    # size of the histograms
    cdef long nspikes = len(spiketimes)
    
    cdef long i, j, cl0, cl1, k, ind
    cdef double t0, t1, t0min, t0max, d

    # unique clusters
    cdef np.ndarray[DTYPEI_t, ndim=1] clusters_unique = np.unique(clusters)
    cdef long nclusters = len(clusters_unique)
    cdef long cluster_max = clusters_unique[-1]
    
    # clusters to update
    if clusters_to_update is None:
        clusters_to_update = clusters_unique
    cdef np.ndarray[DTYPEI_t, ndim=1] clusters_mask = np.zeros(cluster_max + 1, dtype=DTYPEI)
    clusters_mask[clusters_to_update] = 1
    
    # initialize the correlograms
    cdef np.ndarray[DTYPEI_t, ndim=2] correlograms = np.zeros(
        ((cluster_max + 1) ** 2, ncorrbins), dtype=DTYPEI)

    # loop through all spikes, across all neurons, all sorted
    for i in xrange(nspikes):
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
                # compute only correlograms if necessary
                # and avoid computing symmetric pairs twice
                if t1 < t0max:
                    d = t1 - t0
                    k = long(d / corrbin) + n
                    ind = (cluster_max + 1) * cl0 + cl1
                    correlograms[ind, k] += 1
                else:
                    break
                j += 1
            j = i - 1
            # go backward in time up to the correlogram half-width
            while j >= 0:
                t1, cl1 = spiketimes[j], clusters[j]
                # compute only correlograms if necessary
                # and avoid computing symmetric pairs twice
                if t0min < t1:
                    d = t1 - t0
                    k = long(d / corrbin) + n - 1
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
