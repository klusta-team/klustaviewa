"""Unit tests for stats.correlograms module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from klustaviewa.stats.correlograms import compute_correlograms


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_compute_correlograms():
    train0 = np.array(np.arange(0., 10., .1), dtype=np.float64)
    # concatenation of two spike trains
    spiketimes = np.hstack((# the first being train0
                            train0,
                            # the second a 1.5ms-shifted train0
                            train0 + .0015))
    clusters = np.hstack((np.zeros(len(train0), dtype=np.int32),
                           np.ones(len(train0), dtype=np.int32)))
    indices_sorting = np.argsort(spiketimes)
    spiketimes = spiketimes[indices_sorting]
    clusters = clusters[indices_sorting]

    correlograms = compute_correlograms(spiketimes, clusters,
                                        ncorrbins=20, corrbin=.001)

    c01 = np.zeros(20, dtype=np.int32)
    c01[11] = 100
    
    c10 = np.zeros(20, dtype=np.int32)
    c10[8] = 100
    
    assert np.array_equal(correlograms[(0, 0)], np.zeros(20))
    assert np.array_equal(correlograms[(1, 1)], np.zeros(20))
    assert np.array_equal(correlograms[(0, 1)], c01)
    assert np.array_equal(correlograms[(1, 0)], c10)
    
    # print (correlograms[(0, 1)], c01)
    