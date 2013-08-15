"""Unit tests for stats.quality module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np

from kwiklib.dataio.tests.mock_data import (setup, teardown,
    nspikes, nclusters, nsamples, nchannels, fetdim, TEST_FOLDER)
from kwiklib.dataio import KlustersLoader
from klustaviewa.stats.quality import cluster_quality

                            
# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def load():
    # Open the mock data.
    dir = TEST_FOLDER
    xmlfile = os.path.join(dir, 'test.xml')
    l = KlustersLoader(filename=xmlfile)
    # c = Controller(l)
    return l


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_quality():
    l = load()
    
    clusters_selected = [2, 3, 5]
    l.select(clusters=clusters_selected)
    waveforms = l.get_waveforms()
    features = l.get_features()
    masks = l.get_masks(full=True)
    clusters = l.get_clusters()
    
    quality = cluster_quality(waveforms, features, clusters, masks, 
        clusters_selected)
    
    l.close()
    