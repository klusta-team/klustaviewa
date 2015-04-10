"""Unit tests for stats.correlations module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np

from klustaviewa.stats.cache import CacheMatrix
from klustaviewa.stats.correlations import SimilarityMatrix, normalize
from klustaviewa.stats.tools import matrix_of_pairs
from kwiklib.dataio.tests.mock_data import (setup, teardown,
    nspikes, nclusters, nsamples, nchannels, fetdim, TEST_FOLDER)
from kwiklib.dataio import KlustersLoader
from kwiklib.dataio.tools import get_array
from klustaviewa.control.controller import Controller


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load():
    # Open the mock data.
    dir = TEST_FOLDER
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(filename=xmlfile)
    c = Controller(l)
    return (l, c)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_compute_correlations():

    n = 1000
    nspikes = 3 * n
    clusters = np.repeat([0, 1, 2],  n)
    features = np.zeros((nspikes, 2))
    masks = np.ones((nspikes, 2))

    # clusters 0 and 1 are close, 2 is far away from 0 and 1
    features[:n, :] = np.random.randn(n, 2)
    features[n:2*n, :] = np.random.randn(n, 2)
    features[2*n:, :] = np.array([[10, 10]]) + np.random.randn(n, 2)

    # compute the correlation matrix
    sm = SimilarityMatrix(features, masks)
    correlations = sm.compute_matrix(clusters)
    matrix = matrix_of_pairs(correlations)

    # check that correlation between 0 and 1 is much higher than the
    # correlation between 2 and 0/1
    assert matrix[0,1] > 100 * matrix[0, 2]
    assert matrix[0,1] > 100 * matrix[1, 2]

def normalize(x):
    return x

def test_recompute_correlations():
    l, c = load()

    clusters_unique = l.get_clusters_unique()

    # Select three clusters
    clusters_selected = [2, 4, 6]
    spikes = l.get_spikes(clusters=clusters_selected)
    # cluster_spikes = l.get_clusters(clusters=clusters_selected)
    # Select half of the spikes in these clusters.
    spikes_sample = spikes[::2]

    # Get the correlation matrix parameters.
    features = get_array(l.get_features('all'))
    masks = get_array(l.get_masks('all', full=True))
    clusters0 = get_array(l.get_clusters('all'))
    clusters_all = l.get_clusters_unique()

    similarity_matrix = CacheMatrix()
    sm = SimilarityMatrix(features, masks)
    correlations0 = sm.compute_matrix(clusters0)
    similarity_matrix.update(clusters_unique, correlations0)
    matrix0 = normalize(similarity_matrix.to_array().copy())



    # Merge these clusters.
    action, output = c.merge_clusters(clusters_selected)
    cluster_new = output['cluster_merged']

    # Compute the new matrix
    similarity_matrix.invalidate([2, 4, 6, cluster_new])
    clusters1 = get_array(l.get_clusters('all'))
    # sm = SimilarityMatrix(features, clusters1, masks)
    correlations1 = sm.compute_matrix(clusters1, [cluster_new])
    similarity_matrix.update([cluster_new], correlations1)
    matrix1 = normalize(similarity_matrix.to_array().copy())


    # Undo.
    assert c.can_undo()
    action, output = c.undo()


    # Compute the new matrix
    similarity_matrix.invalidate([2, 4, 6, cluster_new])
    clusters2 = get_array(l.get_clusters('all'))
    # sm = SimilarityMatrix(features, clusters2, masks,)
    correlations2 = sm.compute_matrix(clusters2)
    correlations2b = sm.compute_matrix(clusters2, clusters_selected)

    for (clu0, clu1) in correlations2b.keys():
        assert np.allclose(correlations2[clu0, clu1], correlations2b[clu0, clu1]), (clu0, clu1, correlations2[clu0, clu1], correlations2b[clu0, clu1])

    similarity_matrix.update(clusters_selected, correlations2b)
    matrix2 = normalize(similarity_matrix.to_array().copy())

    assert np.array_equal(clusters0, clusters2)
    # assert np.allclose(matrix0, matrix2)

    l.close()

