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
    nspikes, nclusters, nsamples, nchannels, fetdim, 
    create_clusters, create_similarity_matrix)


# -----------------------------------------------------------------------------
# Wizard tests
# -----------------------------------------------------------------------------
def test_wizard():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    clusters_unique = np.unique(clusters)
    best_clusters = clusters_unique[np.argsort(quality)[::-1]]
    
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(clusters=clusters, similarity_matrix=similarity_matrix)
    
    
    # Test impossible previous.
    assert w.previous_cluster() is None
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
    assert best_clusters[1] in w.next_cluster()
    for _ in xrange(10):
        assert best_clusters[1] in w.next()
    
    
    # Check previous cluster.
    pair = w.previous_cluster()
    assert best_clusters[0] in pair
    
    
    # Next agaiw.
    assert best_clusters[0] in w.next()
    assert w.previous_cluster() is None
    
    
def test_wizard_update():
    
    # Create mock data.
    clusters = create_clusters(nspikes, nclusters)
    similarity_matrix = create_similarity_matrix(nclusters)
    quality = np.diag(similarity_matrix)
    
    # Get the best clusters.
    clusters_unique = np.unique(clusters)
    best_clusters = clusters_unique[np.argsort(quality)[::-1]]
    
    
    # Initialize the wizard.
    w = Wizard()
    w.set_data(clusters=clusters, similarity_matrix=similarity_matrix)
    
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
    quality = np.diag(similarity_matrix)
    clusters_unique = np.unique(clusters)
    best_clusters = clusters_unique[np.argsort(quality)[::-1]]
    w.merged((best_cluster, cluster), cluster_new)
    w.set_data(clusters=clusters,
               similarity_matrix=similarity_matrix)
    best_cluster_next = best_clusters[0]
    if best_cluster_next == cluster_new:
        best_cluster_next = best_clusters[1]
    
    # Go to the next best_cluster.
    pair = w.next_cluster()
    
    # return
    # The new best cluster should be there.
    [w.next() for _ in xrange(nclusters // 2)]
    assert best_cluster_next in w.next()
    
    