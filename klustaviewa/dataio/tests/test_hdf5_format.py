"""Unit tests for loader module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
from collections import Counter

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil
from nose.tools import with_setup
import tables

from klustaviewa.dataio.tests.mock_data import (setup, teardown,
    freq, nspikes, nclusters, nsamples, nchannels, fetdim)
from klustaviewa.dataio import HDF5Writer
from klustaviewa.dataio import (KlustersLoader, read_clusters, save_clusters,
    find_filename, read_features,
    read_cluster_info, save_cluster_info, read_group_info, save_group_info,
    renumber_clusters, reorder, convert_to_clu, select, get_indices,
    check_dtype, check_shape, get_array, load_text)
from klustaviewa.utils.userpref import USERPREF


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_hdf5():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    filename = os.path.join(dir, 'test.xml')
    
    # Convert in HDF5.
    with HDF5Writer(filename) as writer:
        @writer.progress_report
        def report(spike, nspikes):
            if spike % 1000 == 0:
                print "{0:.1f}%\r".format(spike / float(nspikes) * 100),
        writer.convert()

    # Open the HDF5 file.
    filename = os.path.join(dir, 'test.main.h5')
    with tables.openFile(filename) as file:
        spikes = file.root.shanks.shank0.spikes
        clusters_hdf5 = spikes.col('cluster')
        features_hdf5 = spikes.col('features')
    
        # Check that the arrays correspond to the original values.
        clusters = read_clusters(os.path.join(dir, 'test.clu.1'))
        features = read_features(os.path.join(dir, 'test.fet.1'), 
            nchannels, fetdim, freq, do_process=False)
        
        np.testing.assert_equal(clusters_hdf5, clusters)
        np.testing.assert_equal(features_hdf5, features)
        
    