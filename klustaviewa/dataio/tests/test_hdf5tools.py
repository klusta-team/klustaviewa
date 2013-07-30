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

from klustaviewa.dataio.tests.mock_data import (
    freq, nspikes, nclusters, nsamples, nchannels, fetdim, duration,
    create_waveforms, create_features, create_clusters, create_cluster_colors,
    create_masks, create_xml, create_probe,
    )
from klustaviewa.dataio import HDF5Writer
from klustaviewa.dataio import (KlustersLoader, read_clusters, save_clusters,
    find_filename, read_features,
    save_binary, save_text,
    read_cluster_info, save_cluster_info, read_group_info, save_group_info,
    renumber_clusters, reorder, convert_to_clu, select, get_indices,
    check_dtype, check_shape, get_array, load_text)
from klustaviewa.utils.userpref import USERPREF


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    # log.debug("Creating mock data for dataio subpackage.")
    
    # Create mock directory if needed.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    if not os.path.exists(dir):
        os.mkdir(dir)
    # else:
        # # No need to recreate the files.
        # return
        
    # Create mock data.
    waveforms = create_waveforms(nspikes, nsamples, nchannels)
    features = create_features(nspikes, nchannels, fetdim, duration, freq)
    clusters = create_clusters(nspikes, nclusters)
    cluster_colors = create_cluster_colors(nclusters)
    masks = create_masks(nspikes, nchannels, fetdim)
    xml = create_xml(nchannels, nsamples, fetdim)
    probe = create_probe(nchannels)
    
    
    # Create mock files.
    save_binary(os.path.join(dir, 'test.spk.2'), waveforms)
    save_text(os.path.join(dir, 'test.fet.2'), features,
        header=nchannels * fetdim + 1)
    save_text(os.path.join(dir, 'test.aclu.2'), clusters, header=nclusters)
    save_text(os.path.join(dir, 'test.clu.2'), clusters, header=nclusters)
    save_text(os.path.join(dir, 'test.fmask.2'), masks, header=nclusters,
        fmt='%.6f')
    save_text(os.path.join(dir, 'test.xml'), xml)
    save_text(os.path.join(dir, 'test.probe'), probe)
    

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_hdf5():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    filename = os.path.join(dir, 'test.xml')
    
    # Convert in HDF5.
    with HDF5Writer(filename) as writer:
        writer.convert()
    
    # Open the HDF5 file.
    filename = os.path.join(dir, 'test.main.h5')
    with tables.openFile(filename) as file:
        # Shank 1.
        # --------
        spikes = file.root.shanks.shank1.spikes
        clusters_hdf5 = spikes.col('cluster')
        features_hdf5 = spikes.col('features')
    
        # Check that the arrays correspond to the original values.
        clusters = read_clusters(os.path.join(dir, 'test.clu.1'))
        features = read_features(os.path.join(dir, 'test.fet.1'), 
            nchannels, fetdim, freq, do_process=False)
        
        np.testing.assert_equal(clusters_hdf5, clusters)
        np.testing.assert_equal(features_hdf5, features)
        
        
        # Shank 2.
        # --------
        spikes = file.root.shanks.shank2.spikes
        clusters_hdf5 = spikes.col('cluster')
        features_hdf5 = spikes.col('features')
    
        # Check that the arrays correspond to the original values.
        clusters = read_clusters(os.path.join(dir, 'test.clu.2'))
        features = read_features(os.path.join(dir, 'test.fet.2'), 
            nchannels, fetdim, freq, do_process=False)
        
        np.testing.assert_equal(clusters_hdf5, clusters)
        np.testing.assert_equal(features_hdf5, features)
        
    