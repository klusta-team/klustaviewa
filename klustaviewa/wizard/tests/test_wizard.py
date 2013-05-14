"""Unit tests for wizard module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from nose.tools import raises
import numpy as np

from klustaviewa.wizard.wizard import Wizard
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
    
    
    # Check next cluster.
    assert best_clusters[1] in w.next_cluster()
    for _ in xrange(10):
        assert best_clusters[1] in w.next()
    
    
    # Check previous cluster.
    pair = w.previous_cluster()
    assert best_clusters[0] in pair
    
    
    # Next again.
    assert best_clusters[0] in w.next()
    assert w.previous_cluster() is None
    
    