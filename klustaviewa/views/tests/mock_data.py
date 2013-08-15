"""Functions that generate mock data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil

from kwiklib.utils.colors import COLORS_COUNT
from kwiklib.dataio import MemoryLoader
from kwiklib.dataio.tests import *
from kwiklib.dataio.tools import normalize
from klustaviewa.stats.cache import IndexedMatrix

# -----------------------------------------------------------------------------
# Data creation methods
# -----------------------------------------------------------------------------
def create_cluster_info(nclusters, cluster_offset):
    cluster_info = np.zeros((nclusters, 3), dtype=np.int32)
    cluster_info[:, 0] = np.arange(2, nclusters + cluster_offset)
    cluster_info[:, 1] = create_cluster_colors(nclusters)
    cluster_info[:, 2] = create_cluster_groups(nclusters)
    return cluster_info
    
def create_group_info(ngroups):
    group_info = np.zeros((ngroups, 3), dtype=object)
    group_info[:, 0] = np.arange(ngroups)
    group_info[:, 1] = create_group_colors(ngroups)
    group_info[:, 2] = create_group_names(ngroups)
    return group_info
    
# def create_channel_info(nchannels, channel_offset):
#     channel_info = np.zeros((nchannels, 4), dtype=object)
#     channel_info[:, 0] = np.arange(2, nchannels + channel_offset)
#     channel_info[:, 1] = create_channel_colors(nchannels)
#     channel_info[:, 2] = create_channel_groups(nchannels)
#     channel_info[:, 3] = create_channel_names(nchannels)
#     return channel_info
#     
# def create_channel_group_info(nchannelgroups):
#     channel_group_info = np.zeros((nchannelgroups, 3), dtype=object)
#     channel_group_info[:, 0] = np.arange(nchannelgroups)
#     channel_group_info[:, 1] = create_channel_group_colors(nchannelgroups)
#     channel_group_info[:, 2] = create_channel_group_names(nchannelgroups)
#     return channel_group_info
    
def create_correlograms(clusters, ncorrbins):
    n = len(np.unique(clusters))
    shape = (n, n, ncorrbins)
    data = np.random.rand(*shape)
    data[0, 0] /= 10
    data[1, 1] *= 10
    return IndexedMatrix(clusters, shape=shape,
        data=data)
    

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
LOADER = None

def setup():
    waveforms = create_waveforms(nspikes, nsamples, nchannels)
    features = create_features(nspikes, nchannels, fetdim, duration, freq)
    clusters = create_clusters(nspikes, nclusters)
    masks = create_masks(nspikes, nchannels, fetdim)
    cluster_info = create_cluster_info(nclusters, cluster_offset)
    group_info = create_group_info(ngroups)
    # channel_info = create_channel_info(nchannels, channel_offset)
    # channel_group_info = create_channel_group_info(nchannelgroups)
    similarity_matrix = create_similarity_matrix(nclusters)
    correlograms = create_correlograms(clusters, ncorrbins)
    baselines = create_baselines(clusters)
    probe = create_probe(nchannels)
    
    global LOADER
    LOADER = MemoryLoader(
        nsamples=nsamples,
        nchannels=nchannels,
        fetdim=fetdim,
        freq=freq,
        waveforms=waveforms,
        features=features,
        clusters=clusters,
        masks=masks,
        cluster_info=cluster_info,
        group_info=group_info,
        # channel_info=channel_info,
        # channel_group_info=channel_group_info,
        probe=probe,
    )
    
    return LOADER
    
def teardown():
    pass
    
    