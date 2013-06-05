"""Unit tests for wizard module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import OrderedDict

from nose.tools import raises
import numpy as np

from klustaviewa.wizard.wizard import Wizard
import klustaviewa.utils.logger as log
from klustaviewa.dataio.tests.mock_data import (
    nspikes, nclusters, nsamples, nchannels, fetdim, cluster_offset,
    create_clusters, create_similarity_matrix)

    
# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def create_cluster_groups(nclusters):
    return np.array(np.ones(nclusters) * 3, dtype=np.int32)
    

# -----------------------------------------------------------------------------
# Wizard tests
# -----------------------------------------------------------------------------
def test_wizard():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters)
    cluster_groups = create_cluster_groups(nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    clusters_unique = np.unique(clusters)
    best_clusters = clusters_unique[np.argsort(quality)[::-1]]
    
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(clusters=clusters, similarity_matrix=similarity_matrix,
               cluster_groups=cluster_groups)
    
    # Check the best pairs keys and best clusters.
    cluster = w.best_pairs.keys()[0]
    assert np.array_equal(w.best_pairs.keys(), best_clusters)
    
    # Look for the best similarity value.
    m = similarity_matrix.copy()
    np.fill_diagonal(m, 0)
    best_similarity = max(m[cluster - cluster_offset, :].max(),
                          m[:, cluster - cluster_offset].max())
    
    # Check that the first proposition pair is the right one.
    pair = w.next()
    cl0, cl1 = pair[0] - cluster_offset, pair[1] - cluster_offset
    assert np.allclose(max(m[cl0, cl1], m[cl1, cl0]), best_similarity)
               
    
    # Test impossible previous.
    assert w.previous_target() is None
    assert w.previous() is None
    
    
    # Check that the first propositions contain the best cluster.
    for _ in xrange(10):
        assert best_clusters[0] in w.next()
    
    
    # Check next/previous.
    pair0 = w.next()
    pair1 = w.next()
    assert w.previous() == pair0
    assert w.next() == pair1
    
    
    # Check next cluster.
    assert best_clusters[1] in w.next_target()
    for _ in xrange(10):
        assert best_clusters[1] in w.next()
    
    
    # Check previous cluster.
    pair = w.previous_target()
    assert best_clusters[0] in pair
    
    
    # Next agaiw.
    assert best_clusters[0] in w.next()
    assert w.previous_target() is None
    
def test_wizard_merge():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters)
    cluster_groups = create_cluster_groups(nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    clusters_unique = np.unique(clusters)
    best_clusters = clusters_unique[np.argsort(quality)[::-1]]
    
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(clusters=clusters, similarity_matrix=similarity_matrix,
               cluster_groups=cluster_groups)
    
    best_cluster, cluster = w.next()
    assert best_cluster != cluster
    
    # Next best_cluster.
    best_cluster_next = best_clusters[1]
    
    # Simulate a merge: best_cluster and cluster ==> cluster_new.
    cluster_new = clusters_unique.max() + 1
    clusters[clusters == best_cluster] = cluster_new
    clusters[clusters == cluster] = cluster_new
    log.debug("Merged {0:d} and {1:d} to {2:d}".format(
        best_cluster, cluster, cluster_new))
    
    # Update the wizard.
    similarity_matrix = create_similarity_matrix(nclusters - 1)
    cluster_groups = create_cluster_groups(nclusters - 1)
    quality = np.diag(similarity_matrix)
    clusters_unique = np.unique(clusters)
    best_clusters = clusters_unique[np.argsort(quality)[::-1]]
    w.merged((best_cluster, cluster), cluster_new)
    w.set_data(clusters=clusters,
               cluster_groups=cluster_groups,
               similarity_matrix=similarity_matrix)
    
    assert best_clusters[0] in w.next_target()
    assert best_clusters[0] in w.next()
    assert best_clusters[0] in w.next()
    
def test_wizard_move():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters, cluster_offset=0)
    cluster_groups = create_cluster_groups(nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    best_clusters = np.argsort(quality)[::-1]
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(clusters=clusters, similarity_matrix=similarity_matrix,
               cluster_groups=cluster_groups)
    
    best_cluster, cluster = w.next()
    assert best_cluster != cluster
    
    # Simulate a move.
    cluster_groups[best_cluster] = 1
    w.set_data(cluster_groups=cluster_groups)
    w.moved([best_cluster], [3], 1)
    
    for _ in xrange(nclusters):
        target, candidate = w.next()
        assert (target != best_cluster and candidate != best_cluster)
    
def test_wizard2():
    n = 10
    w = Wizard()
    w.best_clusters = range(n)
    for i in xrange(n):
        w.best_pairs[i] = sorted(set(range(n)) - set([i]))
    
    # First target.
    for i in xrange(1, n):
        assert w.next() == (0, i)
    
    # New target.
    assert w.next() == (1, 0)
    assert w.next() == (1, 2)
    
    # New target.
    assert w.next_target() == (2, 0)
    
    # Merge.
    w.merged([2, 0], 10)
    assert w.next() == (10, 1)
    
    w.merged([10, 1], 12)
    assert w.next() == (12, 3)
    
    # Move candidate.
    w.moved(3, 3, 1)
    assert w.current() == (12, 4)
    
    
    # Next target.
    w.best_pairs.clear()
    w.best_pairs[4] = range(5, 13)
    w.next_target()
    assert w.current() == (4, 5)
    
    